"""server.py - Máy chủ Web Voicebot thời gian thực (FastAPI + WebSocket).
Quản lý trạng thái phiên: idle -> listening -> thinking -> speaking -> idle.
Hỗ trợ cơ chế ngắt lời (Barge-in), bộ nhớ ngữ cảnh đa lượt (RAM) và kiểm soát lỗi STT thân thiện.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from stt import (
    SttError,
    SttSession,
    VadStreamState,
    get_detector,
    get_vad_model,
    is_hesitating,
    is_only_filler,
    is_reading_sequence,
    normalize_phone_words_in_text,
    normalize_text,
)
from tts import run_turn

STT_URL = os.environ.get("STT_WS_URL", "wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream")
LLM_URL = os.environ.get("LLM_URL", "https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions")
TTS_URL = os.environ.get("TTS_BASE_URL", "https://ai.vnpost.vn/zero-tts")
VOICE = os.environ.get("TTS_VOICE", "maichi")

INDEX_HTML = Path(__file__).resolve().parent / "index.html"
MAX_HISTORY = 8

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: duy trì httpx.AsyncClient và nạp trước Smart Turn, Silero VAD."""
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    await asyncio.to_thread(get_detector)
    await asyncio.to_thread(get_vad_model)
    yield
    await app.state.http.aclose()

app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    """Trả về giao diện web đàm thoại Voicebot."""
    return FileResponse(INDEX_HTML)


def friendly_stt_error(payload: dict[str, Any]) -> str:
    """Chuyển đổi mã lỗi STT thành thông điệp thân thiện với người dùng."""
    code = payload.get("error_code", "")
    if code in ("OVERLOADED", "QUEUE_TIMEOUT"):
        return "Dịch vụ nhận dạng giọng nói đang quá tải, vui lòng thử lại sau ít giây."
    if code == "AUDIO_TOO_LONG":
        return "Câu nói quá dài (tối đa 600 giây)."
    return f"Lỗi nhận dạng giọng nói ({code or payload.get('status_code', '?')})."


class SpeculativeRun:
    """Chạy nháp run_turn() trong nền và đệm toàn bộ kết quả vào Queue trước khi commit."""

    def __init__(
        self,
        text: str,
        history: list[dict[str, str]],
        http: httpx.AsyncClient,
        send_raw_cb: Callable[[str], Awaitable[None]],
        send_audio_cb: Callable[[bytes], Awaitable[None]],
    ) -> None:
        self.text = text
        self.metrics: dict[str, float] = {}
        self._events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._committed = False
        self._lock = asyncio.Lock()
        self._send_raw_cb = send_raw_cb
        self._send_audio_cb = send_audio_cb
        self.task = asyncio.create_task(self._execute(history, http))

    async def _on_raw(self, piece: str) -> None:
        async with self._lock:
            if self._committed:
                await self._send_raw_cb(piece)
            else:
                await self._events.put(("raw", piece))

    async def _on_audio(self, pcm: bytes) -> None:
        async with self._lock:
            if self._committed:
                await self._send_audio_cb(pcm)
            else:
                await self._events.put(("audio", pcm))

    async def _execute(self, history: list[dict[str, str]], http: httpx.AsyncClient) -> str:
        return await run_turn(
            self.text,
            history=history,
            llm_url=LLM_URL,
            tts_url=TTS_URL,
            voice=VOICE,
            client=http,
            send_raw=self._on_raw,
            send_audio=self._on_audio,
            metrics=self.metrics,
        )

    async def commit_and_wait(self) -> str:
        """Xả toàn bộ dữ liệu đã chạy nháp xuống trình duyệt và chờ hoàn tất lượt."""
        async with self._lock:
            self._committed = True
            while not self._events.empty():
                kind, data = self._events.get_nowait()
                if kind == "raw":
                    await self._send_raw_cb(data)
                else:
                    await self._send_audio_cb(data)
        return await self.task

    async def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


class VoiceSession:
    """Quản lý trạng thái và luồng dữ liệu cho một kết nối WebSocket từ trình duyệt."""

    def __init__(self, ws: WebSocket, http: httpx.AsyncClient) -> None:
        self.ws = ws
        self.http = http
        self.state = "idle"
        self.auto_turn = True
        self.history: list[dict[str, str]] = []
        self.stt: SttSession | None = None
        self.turn: asyncio.Task | None = None
        self.metrics: dict[str, float] = {}
        self._send_lock = asyncio.Lock()
        self.spec_run: SpeculativeRun | None = None

        # Trạng thái đệm âm thanh phục vụ tự động phát hiện dứt câu (Smart Turn + Silero VAD)
        self.detector = get_detector()
        self.vad = VadStreamState()
        self.audio_buf = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0  # Đếm số frame thực sự có tiếng nói (1 frame = 100ms)
        self.has_speech = False
        self._detecting = False
        self.last_partial_text = ""

    def _reset_turn_audio(self) -> None:
        self.vad.reset()
        self.audio_buf.clear()
        self.silence_frames = 0
        self.speech_frames = 0
        self.has_speech = False
        self._detecting = False
        self.last_partial_text = ""

    async def _cancel_spec(self) -> None:
        spec = self.spec_run
        self.spec_run = None
        if spec:
            await spec.cancel()

    # ---- Gửi dữ liệu xuống trình duyệt ----
    async def send_json(self, obj: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_json(obj)

    async def send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(data)

    async def set_state(self, s: str) -> None:
        if self.state != s:
            self.state = s
            await self.send_json({"type": "state", "state": s})

    # ---- Xử lý dữ liệu nhận từ trình duyệt ----
    async def on_control(self, msg: dict[str, Any]) -> None:
        t = msg.get("type")
        if t == "config":
            if "auto_turn" in msg:
                self.auto_turn = bool(msg.get("auto_turn", True))
        elif t == "start_turn":
            await self.start_turn()
        elif t == "stop_turn":
            await self.stop_turn()
        elif t == "interrupt":
            await self.cancel_turn()
        elif t == "reset":
            await self.cancel_turn()
            self.history.clear()

    async def on_audio(self, data: bytes) -> None:
        if self.state != "listening" or self.stt is None:
            return
        frame = data[: len(data) // 2 * 2]
        try:
            await self.stt.send_frame(frame)
        except Exception:
            await self._abort_stt()
            await self.send_json({"type": "error", "message": "Mất kết nối dịch vụ nhận dạng giọng nói."})
            await self.set_state("idle")
            return

        # Smart Turn & Silero VAD: Tự động phát hiện dứt câu qua mô hình âm thanh
        if not self.auto_turn or not self.detector.is_available:
            return

        # Lưu tối đa 8 giây âm thanh gần nhất (8 * 16000 * 2 = 256.000 bytes)
        self.audio_buf.extend(frame)
        if len(self.audio_buf) > 256000:
            self.audio_buf = self.audio_buf[-256000:]

        # Silero VAD: Xác suất có tiếng người trên frame 100ms (chống ồn, lọc tiếng gõ phím/thở)
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        speech_prob = self.vad.speech_prob(samples)

        # Ngưỡng hysteresis: bắt đầu nói >= 0.50, duy trì nói >= 0.35
        is_speaking = speech_prob >= 0.50 if not self.has_speech else speech_prob >= 0.35

        if is_speaking:
            self.has_speech = True
            self.speech_frames += 1
            self.silence_frames = 0
            if self.spec_run:
                asyncio.create_task(self._cancel_spec())
        else:
            if self.has_speech:
                self.silence_frames += 1

        # Nếu người dùng đã nói (>= 4 frame = 400ms) và bắt đầu có khoảng lặng >= 5 frame (500ms)
        if self.has_speech and self.speech_frames >= 4 and self.silence_frames >= 5 and not self._detecting:
            self._detecting = True
            asyncio.create_task(self._check_turn_endpoint())

    async def _check_turn_endpoint(self) -> None:
        try:
            if self.state != "listening" or not self.stt:
                return

            if len(self.audio_buf) < 16000 or self.speech_frames < 4:
                return

            if not self.last_partial_text.strip():
                if self.silence_frames >= 20:
                    self._reset_turn_audio()
                return

            only_filler = is_only_filler(self.last_partial_text)
            hesitating = is_hesitating(self.last_partial_text)
            extra_frames = 5 if is_reading_sequence(self.last_partial_text) else 0

            if only_filler:
                if self.silence_frames >= 25:
                    await self.send_json({
                        "type": "auto_stop",
                        "probability": 0.0,
                        "reason": "filler_timeout",
                    })
                    await self.stop_turn()
                return

            audio_f32 = np.frombuffer(self.audio_buf, dtype=np.int16).astype(np.float32) / 32768.0
            res = await asyncio.to_thread(self.detector.predict, audio_f32, 0.75)
            prob = res.get("probability", 0.0)

            should_stop = False
            reason = ""

            if hesitating:
                if (prob >= 0.80 and self.silence_frames >= (10 + extra_frames)) or self.silence_frames >= (18 + extra_frames):
                    should_stop = True
                    reason = "hesitation_timeout"
            else:
                if prob >= 0.80 and self.silence_frames >= 5:
                    should_stop = True
                    reason = "smart_turn_confident"
                elif prob >= 0.70 and self.silence_frames >= (6 + extra_frames):  
                    should_stop = True
                    reason = "smart_turn_high"
                elif prob >= 0.50 and self.silence_frames >= (8 + extra_frames):   
                    should_stop = True
                    reason = "smart_turn_normal"
                elif self.silence_frames >= (16 + extra_frames):               
                    should_stop = True
                    reason = "silence_timeout"

            if should_stop and self.state == "listening" and self.stt is not None:
                await self.send_json({
                    "type": "auto_stop",
                    "probability": round(prob, 3),
                    "reason": reason,
                })
                await self.stop_turn()
            elif (
                not should_stop
                and self.state == "listening"
                and not hesitating
                and prob >= 0.50
                and self.silence_frames >= 4
                and self.last_partial_text.strip()
            ):
                cur_text = normalize_phone_words_in_text(self.last_partial_text.strip())
                if not is_only_filler(cur_text):
                    if self.spec_run is None:
                        self.spec_run = SpeculativeRun(
                            cur_text,
                            history=self.history[-MAX_HISTORY:],
                            http=self.http,
                            send_raw_cb=self._send_raw,
                            send_audio_cb=self._send_audio,
                        )
                    elif normalize_text(self.spec_run.text) != normalize_text(cur_text):
                        await self._cancel_spec()
                        self.spec_run = SpeculativeRun(
                            cur_text,
                            history=self.history[-MAX_HISTORY:],
                            http=self.http,
                            send_raw_cb=self._send_raw,
                            send_audio_cb=self._send_audio,
                        )
        except Exception as exc:
            print(f"[Smart Turn Error] {exc}")
        finally:
            self._detecting = False

    # ---- Vòng đời một lượt thoại ----
    async def start_turn(self) -> None:
        if self.state in ("thinking", "speaking"):
            # Bỏ barge-in: bắt buộc phải chờ bot nói xong, từ chối nhận lệnh khi bot đang nói
            await self.send_json({"type": "busy"})
            return

        await self._abort_stt()
        stt = SttSession(STT_URL, self._on_stt_event)
        try:
            await stt.open()
        except SttError as exc:
            await self.send_json({"type": "error", "message": friendly_stt_error(exc.payload)})
            return
        except Exception:
            await self.send_json({"type": "error", "message": "Không kết nối được dịch vụ nhận dạng giọng nói."})
            return

        self._reset_turn_audio()
        self.stt = stt
        await self.set_state("listening")

    async def stop_turn(self) -> None:
        if self.state != "listening" or self.stt is None:
            return
        self._reset_turn_audio()
        stt, self.stt = self.stt, None
        await self.set_state("thinking")
        self.turn = asyncio.create_task(self._turn(stt))

    async def _turn(self, stt: SttSession) -> None:
        t_stop = time.perf_counter()
        try:
            try:
                text = normalize_phone_words_in_text((await stt.finish()).strip())
            except SttError as exc:
                await self._cancel_spec()
                await self.send_json({"type": "error", "message": friendly_stt_error(exc.payload)})
                await self.send_json({"type": "bot_done"})
                return
            except Exception:
                await self._cancel_spec()
                await self.send_json({"type": "error", "message": "Nhận dạng giọng nói thất bại, hãy thử lại."})
                await self.send_json({"type": "bot_done"})
                return

            await self.send_json({
                "type": "final",
                "text": text,
                "stt_final_s": round(time.perf_counter() - t_stop, 2),
            })
            if not text or is_only_filler(text):
                await self._cancel_spec()
                msg = (
                    "Tôi đang lắng nghe, bạn hãy nói rõ câu hỏi hoặc yêu cầu nhé."
                    if text
                    else "Không nghe rõ, hãy nói lại."
                )
                await self.send_json({"type": "info", "message": msg})
                await self.send_json({"type": "bot_done"})
                return

            spec = self.spec_run
            self.spec_run = None
            if spec and normalize_text(spec.text) == normalize_text(text):
                reply = await spec.commit_and_wait()
                self.metrics.update(spec.metrics)
            else:
                if spec:
                    await spec.cancel()
                reply = await run_turn(
                    text,
                    history=self.history[-MAX_HISTORY:],
                    llm_url=LLM_URL,
                    tts_url=TTS_URL,
                    voice=VOICE,
                    client=self.http,
                    send_raw=self._send_raw,
                    send_audio=self._send_audio,
                    metrics=self.metrics,
                )
            self.history += [
                {"role": "user", "content": text},
                {"role": "assistant", "content": reply},
            ]
            await self.send_json({
                "type": "metrics",
                **{k: round(v, 2) for k, v in self.metrics.items()},
            })
            await self.send_json({"type": "bot_done"})
        except asyncio.CancelledError:
            await self._cancel_spec()
            await stt.abort()
            raise
        except Exception as exc:
            await self._cancel_spec()
            await self.send_json({"type": "error", "message": f"Lỗi khi tạo câu trả lời: {type(exc).__name__}"})
            await self.send_json({"type": "bot_done"})
        finally:
            await self._cancel_spec()
            if self.turn is asyncio.current_task():
                self.turn = None
            await self.set_state("idle")

    async def _send_raw(self, text: str) -> None:
        await self.send_json({"type": "bot_raw", "text": text})

    async def _send_audio(self, pcm: bytes) -> None:
        if self.state == "thinking":
            await self.set_state("speaking")
        await self.send_bytes(pcm)

    async def _on_stt_event(self, kind: str, payload: Any) -> None:
        if kind == "partial":
            new_text = str(payload or "").strip()
            if new_text:
                self.last_partial_text = new_text
                self.has_speech = True
            await self.send_json({"type": "partial", "text": payload})
        elif kind == "error":
            await self.send_json({"type": "error", "message": friendly_stt_error(payload)})
            if self.state == "listening":
                self.stt = None
                await self.set_state("idle")

    async def cancel_turn(self) -> None:
        """Hủy bỏ lượt thoại đang thực thi (dừng LLM, TTS và âm thanh trên trình duyệt)."""
        self._reset_turn_audio()
        await self._cancel_spec()
        task = self.turn
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.turn = None
        await self._abort_stt()
        await self.send_json({"type": "stop_audio"})
        await self.set_state("idle")

    async def _abort_stt(self) -> None:
        if self.stt:
            stt, self.stt = self.stt, None
            await stt.abort()

    async def close(self) -> None:
        """Đóng toàn bộ tác vụ nền khi người dùng ngắt kết nối WebSocket."""
        await self._cancel_spec()
        if self.turn and not self.turn.done():
            self.turn.cancel()
            await asyncio.gather(self.turn, return_exceptions=True)
        await self._abort_stt()


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket) -> None:
    """Kênh WebSocket chính giữa trình duyệt và Voicebot."""
    await ws.accept()
    sess = VoiceSession(ws, app.state.http)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await sess.on_audio(msg["bytes"])
            elif msg.get("text") is not None:
                try:
                    await sess.on_control(json.loads(msg["text"]))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    finally:
        await sess.close()
