"""Module điều phối phiên thoại (Session & Turn Orchestrator).

Chịu trách nhiệm:
- Quản lý phiên thoại và State Machine (idle, listening, thinking, speaking, interrupted, error).
- Điều phối STT -> LLM -> TTS và Smart Turn v3.2 thời gian thực.
- Quản lý lịch sử hội thoại theo cặp user / assistant.
- Hủy tác vụ an toàn khi người dùng ngắt lời (barge-in) hoặc disconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from backend import services
from backend.config import settings
from backend.text_processing import (
    TTSTextStreamer,
    is_only_filler,
    normalize_text_for_tts,
)

logger = logging.getLogger("voice")


def _drain_queue(queue: asyncio.Queue[Any]) -> None:
    """Xả sạch hàng đợi để dừng xử lý ngay lập tức khi phát sinh lỗi."""
    while not queue.empty():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()


def _calc_rms(pcm_data: bytes) -> float:
    """Tính năng lượng hiệu dụng RMS của frame âm thanh PCM 16-bit."""
    if not pcm_data or len(pcm_data) % 2 != 0:
        return 0.0
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)) / 32768.0)


class VoiceState(StrEnum):
    """Các trạng thái hợp lệ trong State Machine của Voicebot."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class VoiceSession:
    """Phiên tương tác thoại trực tiếp gắn liền với một kết nối WebSocket."""

    session_id: str
    state: VoiceState = VoiceState.IDLE
    history: list[dict[str, str]] = field(default_factory=list)
    current_turn_id: str | None = None
    active_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    closed: bool = False

    has_speech: bool = False
    is_vad_speaking: bool = False
    last_speech_time: float = 0.0
    turn_audio_buffer: deque[bytes] = field(default_factory=deque)
    turn_audio_bytes: int = 0
    barge_in_preroll: deque[bytes] = field(default_factory=deque)
    barge_in_hits: int = 0
    stt_client: services.STTClient | None = None
    last_partial_text: str = ""
    _smart_turn_task: asyncio.Task[Any] | None = field(default=None, repr=False)

    send_json_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    send_binary_cb: Callable[[bytes], Awaitable[None]] | None = None
    _send_lock: asyncio.Lock | None = field(default=None, repr=False)
    _turn_lock: asyncio.Lock | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.barge_in_preroll.maxlen:
            self.barge_in_preroll = deque(maxlen=settings.barge_in_preroll_frames)

    @property
    def send_lock(self) -> asyncio.Lock:
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        return self._send_lock

    @property
    def turn_lock(self) -> asyncio.Lock:
        if self._turn_lock is None:
            self._turn_lock = asyncio.Lock()
        return self._turn_lock

    def buffer_audio(self, pcm: bytes, max_bytes: int = 256_000) -> None:
        """Đệm âm thanh O(1) qua deque và xả frame cũ khi vượt giới hạn."""
        self.turn_audio_buffer.append(pcm)
        self.turn_audio_bytes += len(pcm)
        while self.turn_audio_bytes > max_bytes and self.turn_audio_buffer:
            self.turn_audio_bytes -= len(self.turn_audio_buffer.popleft())

    def get_turn_audio(self) -> bytes:
        return b"".join(self.turn_audio_buffer)

    def clear_turn_audio(self) -> None:
        self.turn_audio_buffer.clear()
        self.turn_audio_bytes = 0

    async def _safe_send(self, target: Awaitable[None] | Callable[[], Awaitable[None]]) -> None:
        if self.closed:
            if asyncio.iscoroutine(target):
                target.close()
            return
        coro = target() if callable(target) else target
        try:
            async with self.send_lock:
                if self.closed:
                    if asyncio.iscoroutine(coro):
                        coro.close()
                    return
                await coro
        except (ConnectionResetError, BrokenPipeError):
            self.closed = True
        except Exception as exc:
            exc_name = type(exc).__name__
            err_msg = str(exc).lower()
            if (
                "disconnect" in exc_name.lower()
                or "closed" in exc_name.lower()
                or "disconnect" in err_msg
                or "close" in err_msg
            ):
                self.closed = True
            else:
                logger.warning("Lỗi gửi WebSocket (%s): %s", exc_name, exc)

    async def emit_event(self, event_type: str, **kwargs: Any) -> None:
        if self.closed or not self.send_json_cb:
            return
        payload = {"type": event_type, "session_id": self.session_id, **kwargs}
        await self._safe_send(lambda: self.send_json_cb(payload))

    async def emit_audio_chunk(self, turn_id: str, segment_id: int, chunk_id: int, audio_data: bytes) -> None:
        if self.closed or not self.send_json_cb or not self.send_binary_cb:
            return
        meta = {
            "type": "assistant.audio.chunk",
            "session_id": self.session_id,
            "turn_id": turn_id,
            "segment_id": segment_id,
            "chunk_id": chunk_id,
            "format": "pcm_s16le",
            "sample_rate": 48000,
            "channels": 1,
        }

        async def _send() -> None:
            if self.send_json_cb and self.send_binary_cb:
                await self.send_json_cb(meta)
                await self.send_binary_cb(audio_data)

        await self._safe_send(_send)

    async def set_state(self, new_state: VoiceState, force: bool = False) -> None:
        if self.state == new_state and not force:
            return
        if self.state == VoiceState.ERROR and not force:
            return
        self.state = new_state
        await self.emit_event("voice.state", turn_id=self.current_turn_id, state=new_state.value)

    async def cancel_active_tasks(self) -> None:
        current_task = None
        with contextlib.suppress(RuntimeError):
            current_task = asyncio.current_task()
        tasks = [t for t in self.active_tasks if t is not current_task and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active_tasks = {t for t in self.active_tasks if not t.done()}

    def add_history(self, role: str, content: str) -> None:
        max_messages = max(0, settings.voice_max_history_pairs * 2)
        if max_messages == 0:
            self.history.clear()
            return
        self.history.append({"role": role, "content": content})
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if settings.llm_system_prompt:
            messages.append({"role": "system", "content": settings.llm_system_prompt})
        messages.extend(self.history)
        return messages

    # -----------------------------------------------------------------------
    # Quản lý Turn & Điều phối sự kiện
    # -----------------------------------------------------------------------
    async def start_turn(self, turn_id: str | None = None) -> None:
        async with self.turn_lock:
            turn_id = turn_id or f"turn_{uuid.uuid4().hex[:8]}"
            await self.cancel_active_tasks()
            if self._smart_turn_task and not self._smart_turn_task.done():
                self._smart_turn_task.cancel()
                self._smart_turn_task = None
            self.current_turn_id = turn_id
            self.clear_turn_audio()
            self.barge_in_preroll.clear()
            self.has_speech = False
            self.is_vad_speaking = False
            self.last_speech_time = 0.0
            self.last_partial_text = ""
            self.barge_in_hits = 0

            if self.stt_client:
                await self.stt_client.close()
                self.stt_client = None

            try:
                stt = await services.STTClient.connect(self.session_id, turn_id)
                self.stt_client = stt
                task = asyncio.create_task(self._forward_stt_events(stt, turn_id))
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
                await self.set_state(VoiceState.LISTENING, force=True)
            except Exception as exc:
                logger.exception("Lỗi khởi tạo STT turn [%s]: %s", turn_id, exc)
                await self.emit_event("session.error", code="STT_CONNECT_FAILED", message=str(exc))
                await self.set_state(VoiceState.ERROR)

    async def _forward_stt_events(self, stt: services.STTClient, turn_id: str) -> None:
        try:
            async for event in stt.events():
                if self.current_turn_id != turn_id or self.closed:
                    break
                event_type = event.get("type")
                if event_type == "partial":
                    text = event.get("text", "")
                    if text != self.last_partial_text:
                        self.last_partial_text = text
                        if text.strip():
                            self.has_speech = True
                            self.last_speech_time = time.monotonic()
                        await self.emit_event("transcript.partial", turn_id=turn_id, text=text)
                elif event_type == "vad":
                    await self._handle_vad(turn_id, event.get("state") or event.get("vad_type") or "activity")
                elif event_type == "error":
                    await self.emit_event(
                        "session.error",
                        code=event.get("error_code") or "STT_ERROR",
                        message=event.get("message", "Lỗi STT"),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Lỗi forward STT: %s", exc)

    async def _predict_and_trigger(self, turn_id: str) -> None:
        try:
            audio = self.get_turn_audio()
            prob = await asyncio.wait_for(
                services.smart_turn_detector.predict(audio),
                timeout=settings.smart_turn_silence_timeout_seconds,
            )
            if (
                prob >= settings.smart_turn_threshold
                and self.current_turn_id == turn_id
                and not self.closed
                and not self.is_vad_speaking
                and self.state == VoiceState.LISTENING
            ):
                await self.finish_turn()
        except TimeoutError:
            logger.warning("Smart Turn inference bị quá hạn timeout turn [%s]", turn_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Lỗi Smart Turn turn [%s]: %s", turn_id, exc)
        finally:
            if self._smart_turn_task is asyncio.current_task():
                self._smart_turn_task = None

    async def _handle_vad(self, turn_id: str, vad_state: str) -> None:
        if vad_state == "speech_started":
            self.is_vad_speaking, self.has_speech = True, True
            self.last_speech_time = time.monotonic()
            if self._smart_turn_task and not self._smart_turn_task.done():
                self._smart_turn_task.cancel()
                self._smart_turn_task = None
        elif vad_state == "speech_ended":
            self.is_vad_speaking = False
            self.last_speech_time = time.monotonic()
            if self.has_speech and self.state == VoiceState.LISTENING:
                if self._smart_turn_task and not self._smart_turn_task.done():
                    return
                task = asyncio.create_task(self._predict_and_trigger(turn_id))
                self._smart_turn_task = task
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)

    async def finish_turn(self) -> None:
        turn_id = self.current_turn_id
        if not turn_id or self.state != VoiceState.LISTENING:
            return
        await self.set_state(VoiceState.THINKING)
        task = asyncio.create_task(self._process_turn(turn_id))
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    def is_silence_timeout(self) -> bool:
        if self.is_vad_speaking or not (self.has_speech and self.last_speech_time > 0):
            return False
        return (time.monotonic() - self.last_speech_time) >= settings.smart_turn_silence_timeout_seconds

    async def check_barge_in(self, pcm_data: bytes) -> bool:
        if self.state not in (VoiceState.THINKING, VoiceState.SPEAKING):
            return False

        self.barge_in_preroll.append(pcm_data)
        if _calc_rms(pcm_data) >= settings.barge_in_rms_threshold:
            self.barge_in_hits += 1
            if self.barge_in_hits >= settings.barge_in_debounce_frames:
                self.barge_in_hits = 0
                combined = b"".join(self.barge_in_preroll)
                self.barge_in_preroll.clear()

                await self.emit_event("playback.stop", turn_id=self.current_turn_id)
                await self.set_state(VoiceState.INTERRUPTED)
                await self.start_turn()
                if self.state == VoiceState.LISTENING:
                    self.has_speech = True
                    self.last_speech_time = time.monotonic()
                    self.buffer_audio(combined)
                    if self.stt_client and self.stt_client.is_active:
                        with contextlib.suppress(Exception):
                            await self.stt_client.send_audio(combined)
                return True
        else:
            if self.barge_in_hits > 0:
                self.barge_in_hits -= 1
        return False

    async def handle_audio_frame(self, pcm_data: bytes) -> None:
        if self.closed:
            return
        if len(pcm_data) % 2 != 0:
            await self.emit_event("session.error", code="INVALID_AUDIO_FORMAT", message="Byte frame phải là số chẵn.")
            return
        if await self.check_barge_in(pcm_data) or self.state != VoiceState.LISTENING or not self.current_turn_id:
            return

        self.buffer_audio(pcm_data)
        if self.stt_client and self.stt_client.is_active:
            try:
                await self.stt_client.send_audio(pcm_data)
            except services.STTError as err:
                await self.emit_event("session.error", code=err.code, message=str(err))

        if self.is_silence_timeout():
            await self.finish_turn()

    # -----------------------------------------------------------------------
    # Dual Pipeline (STT -> LLM/TTS song song)
    # -----------------------------------------------------------------------
    async def _handle_pipeline_error(self, turn_id: str, queue: asyncio.Queue[Any], code: str, msg: str) -> None:
        if self.closed or self.current_turn_id != turn_id:
            return
        _drain_queue(queue)
        await self.emit_event("playback.stop", turn_id=turn_id)
        await self.emit_event("session.error", code=code, message=msg)
        await self.set_state(VoiceState.ERROR)

    async def _commit_assistant_turn(self, turn_id: str, user_text: str, full_text: str) -> None:
        if self.current_turn_id == turn_id and not self.closed and self.state != VoiceState.ERROR:
            await self.emit_event("assistant.text.done", turn_id=turn_id, text=full_text)
            self.add_history("user", user_text)
            self.add_history("assistant", full_text)

    async def _stream_llm_to_queue(
        self, turn_id: str, queue: asyncio.Queue[str | None], streamer: TTSTextStreamer, user_text: str
    ) -> list[str]:
        chunks: list[str] = []
        messages = self.build_messages() + [{"role": "user", "content": user_text}]
        async for delta in services.llm_stream(messages):
            if self.current_turn_id != turn_id or self.closed or self.state == VoiceState.ERROR:
                return chunks
            if not chunks:
                delta = delta.lstrip("\r\n ")
                if not delta:
                    continue
            chunks.append(delta)
            await self.emit_event("assistant.text.delta", turn_id=turn_id, text=delta)
            for sentence in streamer.feed(delta):
                await queue.put(sentence)

        for sentence in streamer.flush():
            await queue.put(sentence)
        return chunks

    async def _produce_tts_text(self, turn_id: str, queue: asyncio.Queue[str | None], user_text: str) -> None:
        streamer = TTSTextStreamer(min_chars=8, max_sentences=8, max_chars=600)
        chunks: list[str] = []
        try:
            chunks = await self._stream_llm_to_queue(turn_id, queue, streamer, user_text)
        except services.LLMError as err:
            logger.error("Lỗi LLM: %s", err)
            await self._handle_pipeline_error(turn_id, queue, err.code, str(err))
            raise
        except asyncio.CancelledError:
            _drain_queue(queue)
            raise
        except Exception as exc:
            logger.exception("Lỗi LLM không xác định: %s", exc)
            await self._handle_pipeline_error(turn_id, queue, "LLM_STREAM_ERROR", str(exc))
            raise
        finally:
            if queue.full():
                _drain_queue(queue)
            with contextlib.suppress(Exception):
                queue.put_nowait(None)

        if self.state == VoiceState.ERROR:
            return

        full_text = "".join(chunks).strip()
        if not full_text:
            if self.state == VoiceState.THINKING:
                await self.set_state(VoiceState.IDLE)
            return

        await self._commit_assistant_turn(turn_id, user_text, full_text)

    async def _stream_sentence_tts(
        self, turn_id: str, sentence: str, seg_id: int, start_chunk_id: int
    ) -> tuple[int, int]:
        chunk_id = start_chunk_id
        chunk_bytes = 0
        async for pcm_chunk in services.tts_stream(sentence):
            if self.current_turn_id != turn_id or self.closed or self.state == VoiceState.ERROR:
                break
            chunk_bytes += len(pcm_chunk)
            await self.emit_audio_chunk(turn_id, seg_id, chunk_id, pcm_chunk)
            chunk_id += 1
        return chunk_id, chunk_bytes

    async def _finish_speaking(self, turn_id: str, total_bytes: int) -> None:
        await self.emit_event("assistant.audio.done", turn_id=turn_id)
        delay = (total_bytes / 96000) + 1.5 if total_bytes > 0 else 1.0

        async def _fallback() -> None:
            await asyncio.sleep(delay)
            if self.current_turn_id == turn_id and not self.closed and self.state == VoiceState.SPEAKING:
                await self.set_state(VoiceState.IDLE)

        fallback_task = asyncio.create_task(_fallback())
        self.active_tasks.add(fallback_task)
        fallback_task.add_done_callback(self.active_tasks.discard)

    async def _stream_tts_audio(self, turn_id: str, queue: asyncio.Queue[str | None]) -> None:
        seg_id, chunk_id, total_bytes = 1, 1, 0
        try:
            while True:
                sentence = await queue.get()
                if sentence is None or self.current_turn_id != turn_id or self.closed or self.state == VoiceState.ERROR:
                    break
                if self.state != VoiceState.SPEAKING:
                    await self.set_state(VoiceState.SPEAKING)

                chunk_id, sentence_bytes = await self._stream_sentence_tts(turn_id, sentence, seg_id, chunk_id)
                total_bytes += sentence_bytes
                seg_id += 1

            if self.current_turn_id == turn_id and not self.closed and self.state != VoiceState.ERROR:
                if self.state == VoiceState.SPEAKING:
                    await self._finish_speaking(turn_id, total_bytes)
                elif self.state == VoiceState.THINKING:
                    await self.set_state(VoiceState.IDLE)

        except services.TTSError as err:
            logger.error("Lỗi TTS: %s", err)
            await self._handle_pipeline_error(turn_id, queue, err.code, str(err))
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Lỗi TTS không xác định: %s", exc)
            await self._handle_pipeline_error(turn_id, queue, "TTS_STREAM_ERROR", str(exc))
            raise

    async def _extract_final_stt_text(self) -> str:
        final_text = ""
        if self.stt_client:
            stt = self.stt_client
            self.stt_client = None
            final_text = await stt.finish()
        if not final_text and self.last_partial_text:
            final_text = self.last_partial_text.strip()
        return final_text

    async def _respond_with_filler(self, turn_id: str, prompt: str) -> None:
        await self.emit_event("assistant.text.done", turn_id=turn_id, text=prompt)
        q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        q.put_nowait(normalize_text_for_tts(prompt))
        q.put_nowait(None)
        await self._stream_tts_audio(turn_id, q)

    async def _run_turn_taskgroup(self, turn_id: str, final_text: str) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        try:
            async with asyncio.TaskGroup() as tg:
                producer_task = tg.create_task(self._produce_tts_text(turn_id, queue, final_text))
                consumer_task = tg.create_task(self._stream_tts_audio(turn_id, queue))
                self.active_tasks.add(producer_task)
                self.active_tasks.add(consumer_task)
                producer_task.add_done_callback(self.active_tasks.discard)
                consumer_task.add_done_callback(self.active_tasks.discard)
        except* services.ServiceError:
            pass

    async def _handle_turn_exception(self, turn_id: str, exc: Exception) -> None:
        if self.closed or self.current_turn_id != turn_id:
            return
        if isinstance(exc, services.STTError):
            logger.error("STT lỗi turn [%s]: %s", turn_id, exc)
            await self.emit_event("session.error", code=exc.code, message=str(exc))
        else:
            logger.exception("Lỗi pipeline turn [%s]: %s", turn_id, exc)
            await self.emit_event("session.error", code="PIPELINE_ERROR", message=str(exc))
        await self.set_state(VoiceState.ERROR)

    async def _process_turn(self, turn_id: str) -> None:
        try:
            final_text = await self._extract_final_stt_text()
            if not final_text:
                if self.current_turn_id == turn_id and not self.closed:
                    await self.set_state(VoiceState.IDLE)
                return

            if self.current_turn_id != turn_id or self.closed:
                return

            await self.emit_event("transcript.final", turn_id=turn_id, text=final_text)

            if is_only_filler(final_text):
                await self._respond_with_filler(
                    turn_id, "Tôi đang lắng nghe, bạn hãy nói rõ câu hỏi hoặc yêu cầu nhé."
                )
                return

            await self._run_turn_taskgroup(turn_id, final_text)

        except services.STTError as err:
            await self._handle_turn_exception(turn_id, err)
        except services.ServiceError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_turn_exception(turn_id, exc)

    async def _handle_cancel(self, turn_id: str | None) -> None:
        if not turn_id or not self.current_turn_id or turn_id == self.current_turn_id:
            if self.stt_client:
                stt = self.stt_client
                self.stt_client = None
                with contextlib.suppress(Exception):
                    await stt.close()
            with contextlib.suppress(Exception):
                await self.cancel_active_tasks()
            if self._smart_turn_task and not self._smart_turn_task.done():
                self._smart_turn_task.cancel()
                self._smart_turn_task = None
            with contextlib.suppress(Exception):
                await self.emit_event("playback.stop", turn_id=self.current_turn_id)
            with contextlib.suppress(Exception):
                await self.set_state(VoiceState.INTERRUPTED, force=True)
                await self.set_state(VoiceState.IDLE, force=True)
            self.clear_turn_audio()
            self.barge_in_preroll.clear()
            self.barge_in_hits = 0

    async def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "session.start":
            await self.emit_event("session.ready")
            await self.set_state(VoiceState.IDLE, force=True)
        elif event_type == "turn.start":
            await self.start_turn(event.get("turn_id"))
        elif event_type == "turn.finish":
            req_turn = event.get("turn_id")
            if not req_turn or req_turn == self.current_turn_id:
                await self.finish_turn()
        elif event_type == "session.cancel":
            await self._handle_cancel(event.get("turn_id"))
        elif event_type == "session.stop":
            await self.emit_event("session.stopped")
            await orchestrator.close_session(self.session_id)
        elif event_type == "playback.done":
            req_turn = event.get("turn_id")
            if (not req_turn or req_turn == self.current_turn_id) and self.state == VoiceState.SPEAKING:
                await self.set_state(VoiceState.IDLE)


class VoiceOrchestrator:
    """Quản lý danh sách các phiên VoiceSession trong RAM."""

    def __init__(self) -> None:
        self.sessions: dict[str, VoiceSession] = {}

    def create_session(
        self,
        session_id: str | None = None,
        send_json_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        send_binary_cb: Callable[[bytes], Awaitable[None]] | None = None,
    ) -> VoiceSession:
        if len(self.sessions) >= settings.max_sessions:
            raise RuntimeError(f"Số lượng phiên đồng thời đã đạt giới hạn tối đa ({settings.max_sessions}).")
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            while session_id in self.sessions:
                session_id = f"session_{uuid.uuid4().hex[:8]}"
        elif session_id in self.sessions:
            raise ValueError(f"Phiên '{session_id}' đã tồn tại trong hệ thống.")

        session = VoiceSession(session_id=session_id, send_json_cb=send_json_cb, send_binary_cb=send_binary_cb)
        self.sessions[session_id] = session
        return session

    async def close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if not session or session.closed:
            return
        session.closed = True
        if session.stt_client:
            with contextlib.suppress(Exception):
                await session.stt_client.close()
            session.stt_client = None
        with contextlib.suppress(Exception):
            await session.cancel_active_tasks()
        if session._smart_turn_task and not session._smart_turn_task.done():
            session._smart_turn_task.cancel()
            session._smart_turn_task = None
        session.clear_turn_audio()
        session.barge_in_preroll.clear()
        session.barge_in_hits = 0
        session.history.clear()


orchestrator = VoiceOrchestrator()
