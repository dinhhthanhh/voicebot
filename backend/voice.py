"""Module điều phối phiên thoại (VoiceSession).

Chịu trách nhiệm:
- Quản lý phiên thoại và State Machine (idle, listening, thinking, speaking, error).
- Điều phối STT -> LLM -> TTS và Smart Turn v3.2 thời gian thực.
- Quản lý lịch sử hội thoại theo cặp user / assistant.
- Hủy tác vụ an toàn khi người dùng bấm dừng hoặc disconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from backend import services
from backend.config import settings
from backend.text_processing import (
    TTSTextStreamer,
    is_only_filler,
    normalize_text_for_tts,
)

logger = logging.getLogger("voice")


# =============================================================================
# Helper functions
# =============================================================================
def _drain_queue(queue: asyncio.Queue[Any]) -> None:
    """Xả sạch hàng đợi để dừng xử lý ngay lập tức khi phát sinh lỗi."""
    while not queue.empty():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()


# =============================================================================
# VoiceState
# =============================================================================
class VoiceState(StrEnum):
    """Các trạng thái hợp lệ trong State Machine của Voicebot."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


# =============================================================================
# VoiceSession
# =============================================================================
@dataclass
class VoiceSession:
    """Phiên tương tác thoại trực tiếp gắn liền với một kết nối WebSocket."""

    session_id: str  # ID duy nhất của phiên
    state: VoiceState = VoiceState.IDLE  # Trạng thái hiện tại của phiên
    history: list[dict[str, str]] = field(default_factory=list)  # Lịch sử hội thoại (gửi cho LLM)
    current_turn_id: str | None = None  # ID của lượt thoại đang diễn ra
    active_tasks: set[asyncio.Task[Any]] = field(default_factory=set)  # Tập hợp task bất đồng bộ đang chạy
    closed: bool = False  # Cờ đánh dấu phiên đã đóng
    _cleanup_done: bool = field(default=False, repr=False)  # Cờ đảm bảo chỉ giải phóng tài nguyên một lần

    has_speech: bool = False  # Đã phát hiện giọng nói trong lượt này chưa
    is_vad_speaking: bool = False  # VAD thời gian thực (True nếu người dùng đang nói)
    last_speech_time: float = 0.0  # Mốc thời gian lần cuối ghi nhận giọng nói
    turn_audio_buffer: deque[bytes] = field(default_factory=deque)  # Bộ đệm PCM cho Smart Turn
    turn_audio_bytes: int = 0  # Dung lượng byte âm thanh hiện có trong buffer
    stt_client: services.STTClient | None = None  # Client kết nối WebSocket tới STT
    last_partial_text: str = ""  # Transcript tạm thời gần nhất để tránh gửi trùng
    _smart_turn_task: asyncio.Task[Any] | None = field(default=None, repr=False)  # Task chạy suy luận Smart Turn

    send_json_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None  # Callback gửi JSON/event về client
    send_binary_cb: Callable[[bytes], Awaitable[None]] | None = None  # Callback gửi byte PCM âm thanh về client
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)  # Khóa chống xung đột khi ghi đồng thời vào WebSocket
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)  # Khóa chống gọi chồng chéo start_turn()

    # -------------------------------------------------------------------------
    # Session & Audio Buffer
    # -------------------------------------------------------------------------
    def buffer_audio(self, pcm: bytes, max_bytes: int | None = None) -> None:
        """Đệm âm thanh O(1) qua deque và xả frame cũ khi vượt giới hạn."""
        limit = max_bytes if max_bytes is not None else settings.turn_max_audio_bytes
        self.turn_audio_buffer.append(pcm)
        self.turn_audio_bytes += len(pcm)
        while self.turn_audio_bytes > limit and self.turn_audio_buffer:
            self.turn_audio_bytes -= len(self.turn_audio_buffer.popleft())

    def get_turn_audio(self) -> bytes:
        """Nối toàn bộ các mẩu PCM trong bộ đệm thành mảng byte âm thanh hoàn chỉnh."""
        return b"".join(self.turn_audio_buffer)

    def reset_audio_buffers(self) -> None:
        """Xóa sạch bộ đệm âm thanh."""
        self.turn_audio_buffer.clear()
        self.turn_audio_bytes = 0

    async def close(self) -> None:
        """Đóng toàn bộ phiên thoại và giải phóng tài nguyên."""
        if self._cleanup_done:
            return
        self.closed = True
        self._cleanup_done = True
        self.current_turn_id = None
        await self._close_stt()
        await self.cancel_active_tasks()
        self.reset_audio_buffers()
        self.history.clear()

    # -------------------------------------------------------------------------
    # WebSocket Communication
    # -------------------------------------------------------------------------
    async def _safe_send(self, send: Callable[[], Awaitable[None]]) -> None:
        """Gửi dữ liệu WebSocket an toàn qua khóa chống xung đột ghi đồng thời."""
        if self.closed:
            return
        try:
            async with self.send_lock:
                if not self.closed:
                    await send()
        except (ConnectionResetError, BrokenPipeError):
            self.closed = True
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}".lower()
            if any(k in err_msg for k in ("disconnect", "closed", "close")):
                self.closed = True
            else:
                logger.warning("Lỗi gửi WebSocket (%s): %s", type(exc).__name__, exc)

    async def emit_event(self, event_type: str, **kwargs: Any) -> None:
        """Gửi sự kiện JSON về client qua WebSocket."""
        if self.closed or not self.send_json_cb:
            return
        payload = {"type": event_type, "session_id": self.session_id, **kwargs}
        await self._safe_send(lambda: self.send_json_cb(payload))

    async def emit_audio_chunk(self, turn_id: str, chunk_id: int, audio_data: bytes) -> None:
        """Gửi gói metadata JSON kèm khối byte âm thanh PCM về client để phát ra loa."""
        if self.closed or not self.send_json_cb or not self.send_binary_cb:
            return
        meta = {
            "type": "assistant.audio.chunk",
            "session_id": self.session_id,
            "turn_id": turn_id,
            "chunk_id": chunk_id,
            "format": "pcm_s16le",
            "sample_rate": settings.tts_sample_rate,
            "channels": 1,
        }
        send_json, send_bin = self.send_json_cb, self.send_binary_cb

        async def _send() -> None:
            await send_json(meta)
            await send_bin(audio_data)

        await self._safe_send(_send)

    async def set_state(self, new_state: VoiceState, force: bool = False) -> None:
        """Chuyển trạng thái phiên thoại và gửi sự kiện voice.state về client."""
        if self.state == new_state and not force:
            return
        if self.state == VoiceState.ERROR and not force:
            return
        self.state = new_state
        if new_state == VoiceState.ERROR:
            if self._smart_turn_task and not self._smart_turn_task.done():
                self._smart_turn_task.cancel()
                self._smart_turn_task = None
            self.has_speech = False
            self.is_vad_speaking = False
        await self.emit_event("voice.state", turn_id=self.current_turn_id, state=new_state.value)

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------
    async def cancel_active_tasks(self) -> None:
        """Hủy an toàn toàn bộ các tác vụ bất đồng bộ đang chạy nền trong phiên."""
        current_task = None
        with contextlib.suppress(RuntimeError):
            current_task = asyncio.current_task()
        tasks = [t for t in self.active_tasks if t is not current_task and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active_tasks = {t for t in self.active_tasks if not t.done()}
        self._smart_turn_task = None

    def _spawn_task(self, target: Coroutine[Any, Any, Any] | asyncio.Task[Any]) -> asyncio.Task[Any]:
        """Khởi tạo và quản lý task chạy nền, tự xóa khỏi active_tasks khi hoàn thành."""
        task = target if isinstance(target, asyncio.Task) else asyncio.create_task(target)
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)
        return task

    # -------------------------------------------------------------------------
    # Conversation History
    # -------------------------------------------------------------------------
    def add_history(self, role: str, content: str) -> None:
        """Thêm tin nhắn vào lịch sử hội thoại và giới hạn số lượng theo cấu hình."""
        max_messages = max(0, settings.voice_max_history_pairs * 2)
        if max_messages == 0:
            self.history.clear()
            return
        self.history.append({"role": role, "content": content})
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def build_messages(self) -> list[dict[str, str]]:
        """Xây dựng danh sách tin nhắn hoàn chỉnh (kèm system prompt) để gửi cho LLM."""
        messages: list[dict[str, str]] = []
        if settings.llm_system_prompt:
            messages.append({"role": "system", "content": settings.llm_system_prompt})
        messages.extend(self.history)
        return messages

    # -------------------------------------------------------------------------
    # Turn Management
    # -------------------------------------------------------------------------
    def is_active_turn(self, turn_id: str | None) -> bool:
        """Kiểm tra turn có còn đang hoạt động và session chưa đóng hay không."""
        return not self.closed and bool(turn_id) and self.current_turn_id == turn_id

    def _match_turn(self, turn_id: str | None, allow_empty: bool = False) -> bool:
        """Kiểm tra sự kiện có khớp với turn hiện tại không.

        - Nếu phiên không có current_turn_id hoạt động: luôn trả về False.
        - Nếu có turn_id: bắt buộc phải khớp chính xác với current_turn_id.
        - Nếu không có turn_id: chỉ khớp khi allow_empty=True (chế độ tương thích cho cancel).
        """
        if not self.current_turn_id:
            return False
        if turn_id is not None and turn_id.strip() != "":
            return turn_id == self.current_turn_id
        return allow_empty

    async def start_turn(self) -> None:
        """Bắt đầu một lượt thoại mới: reset bộ đệm, kết nối STT và chuyển sang LISTENING."""
        async with self.turn_lock:
            turn_id = f"turn_{uuid.uuid4().hex[:8]}"
            await self.cancel_active_tasks()
            self.current_turn_id = turn_id
            self.reset_audio_buffers()
            self.has_speech = False
            self.is_vad_speaking = False
            self.last_speech_time = 0.0
            self.last_partial_text = ""
            await self._close_stt()

            try:
                self.stt_client = await services.STTClient.connect(self.session_id, turn_id)
                self._spawn_task(self._forward_stt_events(self.stt_client, turn_id))
                await self.set_state(VoiceState.LISTENING, force=True)
            except Exception as exc:
                logger.exception("Lỗi khởi tạo STT turn [%s]: %s", turn_id, exc)
                await self.emit_event("session.error", code="STT_CONNECT_FAILED", message=str(exc))
                await self.set_state(VoiceState.ERROR)

    async def finish_turn(self) -> None:
        """Kết thúc lượt nghe: chuyển sang THINKING và kích hoạt tác vụ xử lý LLM -> TTS."""
        turn_id = self.current_turn_id
        if not turn_id or self.state != VoiceState.LISTENING:
            return
        if self._smart_turn_task and not self._smart_turn_task.done():
            self._smart_turn_task.cancel()
            self._smart_turn_task = None
        await self.set_state(VoiceState.THINKING)
        self._spawn_task(self._process_turn(turn_id))

    async def _handle_cancel(self, turn_id: str | None = None) -> None:
        """Xử lý hủy lượt thoại: dừng phát loa, đóng STT, hủy task và đưa về IDLE."""
        if self._match_turn(turn_id, allow_empty=True):
            old_turn_id = self.current_turn_id
            await self._close_stt()
            await self.cancel_active_tasks()
            await self.emit_event("playback.stop", turn_id=old_turn_id)
            await self.set_state(VoiceState.IDLE, force=True)
            self.reset_audio_buffers()
            self.current_turn_id = None

    # -------------------------------------------------------------------------
    # STT & Smart Turn
    # -------------------------------------------------------------------------
    async def _close_stt(self) -> None:
        """Đóng kết nối WebSocket STT an toàn."""
        if self.stt_client:
            stt = self.stt_client
            self.stt_client = None
            with contextlib.suppress(Exception):
                await stt.close()

    async def _forward_stt_events(self, stt: services.STTClient, turn_id: str) -> None:
        """Lắng nghe sự kiện từ STTClient (partial, vad, error) và chuyển tiếp về client/hệ thống."""
        try:
            async for event in stt.events():
                if not self.is_active_turn(turn_id):
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
                    logger.info("STT VAD event: %s", event)
                    await self._handle_vad(
                        turn_id,
                        event.get("state")
                        or event.get("vad_type")
                        or "activity",
                    )
                elif event_type == "error":
                    code = event.get("error_code") or event.get("code") or "STT_ERROR"
                    msg = event.get("message", "Lỗi STT")
                    await self.emit_event("session.error", code=code, message=msg)
                    if self.is_active_turn(turn_id):
                        await self._close_stt()
                        await self.set_state(VoiceState.ERROR)
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Lỗi forward STT turn [%s]: %s", turn_id, exc)
            if self.is_active_turn(turn_id):
                await self._close_stt()
                await self.emit_event("session.error", code="STT_FORWARD_ERROR", message=str(exc))
                await self.set_state(VoiceState.ERROR)

    async def _predict_and_trigger(self, turn_id: str) -> None:
        """Chạy mô hình Smart Turn dự đoán dứt câu, nếu prob >= threshold thì gọi finish_turn()."""
        try:
            await asyncio.sleep(0.3)
            if not self.is_active_turn(turn_id) or self.is_vad_speaking:
                return
            audio = self.get_turn_audio()
            prob = await asyncio.wait_for(
                services.smart_turn_detector.predict(audio),
                timeout=settings.smart_turn_inference_timeout_seconds,
            )
            logger.info(
                "Smart Turn [%s]: %.3f (threshold=%.2f)",
                turn_id,
                prob,
                settings.smart_turn_threshold,
            )
            if (
                prob >= settings.smart_turn_threshold
                and self.is_active_turn(turn_id)
                and not self.is_vad_speaking
                and self.state == VoiceState.LISTENING
            ):
                await self.finish_turn()
        except TimeoutError:
            logger.warning("Smart Turn timeout: %s", turn_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Smart Turn error: %s", turn_id)
        finally:
            if self._smart_turn_task is asyncio.current_task():
                self._smart_turn_task = None

    async def _handle_vad(self, turn_id: str, vad_state: str) -> None:
        """Xử lý sự kiện giọng nói (speech_started / speech_ended) để hủy hoặc kích hoạt Smart Turn."""
        if vad_state == "speech_started":
            self.is_vad_speaking, self.has_speech = True, True
            self.last_speech_time = time.monotonic()
            if self._smart_turn_task and not self._smart_turn_task.done():
                self._smart_turn_task.cancel()
                self._smart_turn_task = None
        elif vad_state == "speech_ended":
            self.is_vad_speaking = False
            self.last_speech_time = time.monotonic()
            if self.has_speech and self.state == VoiceState.LISTENING and self.is_active_turn(turn_id):
                if self._smart_turn_task and not self._smart_turn_task.done():
                    return
                self._smart_turn_task = self._spawn_task(self._predict_and_trigger(turn_id))

    def is_silence_timeout(self) -> bool:
        """Kiểm tra người dùng đã nói nhưng sau đó im lặng vượt quá thời gian fallback hay chưa."""
        if self.is_vad_speaking:
            return False
        return (
            self.has_speech
            and self.last_speech_time > 0
            and time.monotonic() - self.last_speech_time
            >= settings.smart_turn_fallback_seconds
        )

    async def handle_audio_frame(self, pcm_data: bytes) -> None:
        """Xử lý frame âm thanh từ mic: đệm cho Smart Turn, stream sang STT và kiểm tra timeout im lặng."""
        if self.closed:
            return
        if len(pcm_data) % 2 != 0:
            await self.emit_event("session.error", code="INVALID_AUDIO_FORMAT", message="Byte frame phải là số chẵn.")
            return
        if self.state != VoiceState.LISTENING or not self.current_turn_id:
            return

        self.buffer_audio(pcm_data)
        if self.stt_client and self.stt_client.is_active:
            try:
                await self.stt_client.send_audio(pcm_data)
            except services.STTError as err:
                logger.error("Lỗi gửi audio tới STT turn [%s]: %s", self.current_turn_id, err)
                await self._close_stt()
                await self.emit_event("session.error", code=err.code, message=str(err))
                await self.set_state(VoiceState.ERROR)
                return

        if self.is_silence_timeout():
            await self.finish_turn()

    # -------------------------------------------------------------------------
    # LLM Producer
    # -------------------------------------------------------------------------
    async def _produce_tts_text(self, turn_id: str, queue: asyncio.Queue[str | None], user_text: str) -> None:
        """Gọi LLM streaming sinh text, stream delta về client và tách câu nạp vào queue cho TTS."""
        streamer = TTSTextStreamer()
        chunks: list[str] = []
        sentinel_sent = False
        try:
            messages = self.build_messages() + [{"role": "user", "content": user_text}]
            async for delta in services.llm_stream(messages):
                if not self.is_active_turn(turn_id) or self.state == VoiceState.ERROR:
                    break
                if not chunks:
                    delta = delta.lstrip("\r\n ")
                    if not delta:
                        continue
                chunks.append(delta)
                await self.emit_event("assistant.text.delta", turn_id=turn_id, text=delta)
                for sentence in streamer.feed(delta):
                    await queue.put(sentence)

            if self.is_active_turn(turn_id) and self.state != VoiceState.ERROR:
                for sentence in streamer.feed():
                    await queue.put(sentence)

                full_text = "".join(chunks).strip()
                if full_text:
                    await self.emit_event("assistant.text.done", turn_id=turn_id, text=full_text)
                    self.add_history("user", user_text)
                    self.add_history("assistant", full_text)
                elif self.state == VoiceState.THINKING:
                    await self.set_state(VoiceState.IDLE)

                await queue.put(None)
                sentinel_sent = True
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
            if not sentinel_sent:
                _drain_queue(queue)
                with contextlib.suppress(Exception):
                    queue.put_nowait(None)

    # -------------------------------------------------------------------------
    # TTS Consumer
    # -------------------------------------------------------------------------
    async def _finish_speaking(self, turn_id: str, total_bytes: int) -> None:
        """Báo audio.done về client và bật timer fallback để tự động chuyển về IDLE khi phát xong."""
        await self.emit_event("assistant.audio.done", turn_id=turn_id)
        bytes_per_sec = settings.tts_sample_rate * 2
        delay = (total_bytes / bytes_per_sec) + 1.5 if total_bytes > 0 else 1.0

        async def _fallback() -> None:
            await asyncio.sleep(delay)
            if self.is_active_turn(turn_id) and self.state == VoiceState.SPEAKING:
                await self.set_state(VoiceState.IDLE)

        self._spawn_task(_fallback())

    async def _stream_tts_audio(self, turn_id: str, queue: asyncio.Queue[str | None]) -> None:
        """Rút câu từ queue, gọi tts_stream tạo âm thanh và stream từng chunk PCM về client."""
        chunk_id, total_bytes = 1, 0
        try:
            while True:
                sentence = await queue.get()
                if sentence is None or not self.is_active_turn(turn_id) or self.state == VoiceState.ERROR:
                    break
                if self.state != VoiceState.SPEAKING:
                    await self.set_state(VoiceState.SPEAKING)

                async for pcm_chunk in services.tts_stream(sentence):
                    if not self.is_active_turn(turn_id) or self.state == VoiceState.ERROR:
                        break
                    total_bytes += len(pcm_chunk)
                    await self.emit_audio_chunk(turn_id, chunk_id, pcm_chunk)
                    chunk_id += 1

            if self.is_active_turn(turn_id) and self.state != VoiceState.ERROR:
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

    async def _respond_with_filler(self, turn_id: str, prompt: str) -> None:
        """Phát câu nhắc thoại mặc định (chưa nghe rõ / đang lắng nghe) qua TTS mà không gọi LLM."""
        if not self.is_active_turn(turn_id):
            return
        await self.emit_event("assistant.text.done", turn_id=turn_id, text=prompt)
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        queue.put_nowait(normalize_text_for_tts(prompt))
        queue.put_nowait(None)
        await self._stream_tts_audio(turn_id, queue)

    async def _run_turn_taskgroup(self, turn_id: str, final_text: str) -> None:
        """Chạy song song Producer (LLM) và Consumer (TTS) qua TaskGroup và hàng đợi queue."""
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._produce_tts_text(turn_id, queue, final_text))
                tg.create_task(self._stream_tts_audio(turn_id, queue))
        except* services.ServiceError as eg:
            if self.is_active_turn(turn_id) and self.state != VoiceState.ERROR:
                first_err = eg.exceptions[0] if eg.exceptions else services.ServiceError("Lỗi pipeline")
                await self._handle_turn_exception(turn_id, first_err)

    # -------------------------------------------------------------------------
    # Error Handling
    # -------------------------------------------------------------------------
    async def _handle_pipeline_error(self, turn_id: str, queue: asyncio.Queue[Any], code: str, msg: str) -> None:
        """Xử lý lỗi pipeline: xả sạch queue, dừng phát loa, báo lỗi về client và đổi state sang ERROR."""
        if not self.is_active_turn(turn_id):
            return
        _drain_queue(queue)
        await self.emit_event("playback.stop", turn_id=turn_id)
        await self.emit_event("session.error", code=code, message=msg)
        await self.set_state(VoiceState.ERROR)

    async def _handle_turn_exception(self, turn_id: str, exc: Exception) -> None:
        """Phân loại ngoại lệ trong turn, ghi log, bắn sự kiện session.error và chuyển sang ERROR."""
        if not self.is_active_turn(turn_id):
            return
        if isinstance(exc, services.STTError):
            logger.error("STT lỗi turn [%s]: %s", turn_id, exc)
            await self.emit_event("session.error", code=exc.code, message=str(exc))
        else:
            logger.exception("Lỗi pipeline turn [%s]: %s", turn_id, exc)
            await self.emit_event("session.error", code="PIPELINE_ERROR", message=str(exc))
        await self.set_state(VoiceState.ERROR)

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------
    async def _process_turn(self, turn_id: str) -> None:
        """Điều phối xử lý lượt thoại: chốt STT final_text, lọc từ đệm và kích hoạt pipeline LLM -> TTS."""
        try:
            final_text = ""
            if self.stt_client:
                stt = self.stt_client
                self.stt_client = None
                try:
                    final_text = await stt.finish()
                except services.STTError as stt_err:
                    if stt_err.code == "STT_TIMEOUT" and self.last_partial_text:
                        logger.warning(
                            "STT timeout turn [%s], dùng transcript partial dự phòng: '%s'",
                            turn_id,
                            self.last_partial_text,
                        )
                        final_text = self.last_partial_text.strip()
                    else:
                        raise
            if not final_text and self.last_partial_text:
                final_text = self.last_partial_text.strip()

            if not self.is_active_turn(turn_id):
                return

            if not final_text:
                await self.emit_event("transcript.final", turn_id=turn_id, text="")
                await self._respond_with_filler(
                    turn_id, "Tôi chưa nghe rõ, bạn vui lòng nói lại nhé."
                )
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

    async def handle_event(self, event: dict[str, Any]) -> None:
        """Điều phối và xử lý các sự kiện điều khiển nhận từ client qua WebSocket."""
        event_type = event.get("type")
        if event_type == "session.start":
            await self.emit_event("session.ready")
            await self.set_state(VoiceState.IDLE, force=True)
        elif event_type == "turn.start":
            await self.start_turn()
        elif event_type == "turn.finish" and self._match_turn(event.get("turn_id"), allow_empty=False):
            await self.finish_turn()
        elif event_type == "session.cancel":
            await self._handle_cancel(event.get("turn_id"))
        elif event_type == "session.stop":
            await self.emit_event("session.stopped")
            await self.close()
        elif (
            event_type == "playback.done"
            and self._match_turn(event.get("turn_id"), allow_empty=False)
            and self.state == VoiceState.SPEAKING
        ):
            await self.set_state(VoiceState.IDLE)
