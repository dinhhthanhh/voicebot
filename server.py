from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from stt import (
    SileroVadModel,
    SmartTurnDetector,
    SttError,
    SttSession,
    VadStreamState,
    get_detector,
    get_vad_model,
    is_hesitating,
    is_only_filler,
    is_reading_sequence,
    normalize_speech_input,
    normalize_text,
)
from tts import run_turn

# Cấu hình dịch vụ
STT_URL = os.environ.get("STT_WS_URL", "wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream")
LLM_URL = os.environ.get("LLM_URL", "https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions")
TTS_URL = os.environ.get("TTS_BASE_URL", "https://ai.vnpost.vn/zero-tts")
VOICE = os.environ.get("TTS_VOICE", "maichi")

INDEX_HTML = Path(__file__).resolve().parent / "index.html"
MAX_HISTORY = 8

# Định dạng âm thanh PCM16 mono 16 kHz
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_DURATION_MS = 100
FRAME_SAMPLES = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH


def ms_to_frames(milliseconds: int) -> int:
    return max(1, (milliseconds + FRAME_DURATION_MS - 1) // FRAME_DURATION_MS)


MIN_SPEECH_MS = 300
INITIAL_SILENCE_MS = 300
SMART_TURN_CHECK_INTERVAL_MS = 200
SMART_TURN_CONFIDENT_MS = 700
SMART_TURN_UNCERTAIN_MS = 1400
FILLER_TIMEOUT_MS = 2000
SILENCE_TIMEOUT_MS = 2000

MIN_SPEECH_FRAMES = ms_to_frames(MIN_SPEECH_MS)
INITIAL_SILENCE_FRAMES = ms_to_frames(INITIAL_SILENCE_MS)
SMART_TURN_CHECK_INTERVAL_FRAMES = ms_to_frames(SMART_TURN_CHECK_INTERVAL_MS)
SMART_TURN_CONFIDENT_FRAMES = ms_to_frames(SMART_TURN_CONFIDENT_MS)
SMART_TURN_UNCERTAIN_FRAMES = ms_to_frames(SMART_TURN_UNCERTAIN_MS)
FILLER_TIMEOUT_FRAMES = ms_to_frames(FILLER_TIMEOUT_MS)
SILENCE_TIMEOUT_FRAMES = ms_to_frames(SILENCE_TIMEOUT_MS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=60.0,
            write=30.0,
            pool=10.0,
        ),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )
    app.state.detector = await asyncio.to_thread(get_detector)
    app.state.vad_model = await asyncio.to_thread(get_vad_model)
    yield
    await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)

@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)

def friendly_stt_error(payload: dict[str, Any]) -> str:
    code = payload.get("error_code", "")
    if code in ("OVERLOADED", "QUEUE_TIMEOUT"):
        return "Dịch vụ nhận dạng giọng nói đang quá tải, vui lòng thử lại sau ít giây."
    if code == "AUDIO_TOO_LONG":
        return "Câu nói quá dài (tối đa 600 giây)."
    return f"Lỗi nhận dạng giọng nói ({code or payload.get('status_code', '?')})."

class VoiceSession:

    def __init__(
        self,
        ws: WebSocket,
        http: httpx.AsyncClient,
        detector: SmartTurnDetector | None = None,
        vad_model: SileroVadModel | None = None,
    ) -> None:
        self.ws = ws
        self.http = http
        self.state = "idle"
        self.auto_turn = True
        self.history: list[dict[str, str]] = []
        self.stt: SttSession | None = None
        self.turn: asyncio.Task | None = None
        self._start_task: asyncio.Task | None = None
        self.metrics: dict[str, float] = {}
        self._send_lock = asyncio.Lock()
        self._vad_lock = asyncio.Lock()

        self._pending_audio: deque[bytes] = deque(maxlen=50)

        self.detector = detector or get_detector()
        self.vad = VadStreamState(model=vad_model)
        self.turn_id = 0
        self._detect_task: asyncio.Task | None = None
        self.audio_buf = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.has_speech = False
        self._detecting = False
        self._last_detect_frame = 0
        self.last_partial_text = ""
        self.last_speech_time = time.monotonic()
        self.last_frame_time = time.monotonic()

    def _reset_turn_audio(self) -> None:
        self.turn_id += 1
        current = asyncio.current_task()
        if (
            self._detect_task
            and self._detect_task is not current
            and not self._detect_task.done()
        ):
            self._detect_task.cancel()
        self._detect_task = None
        self.vad.reset()
        self.audio_buf.clear()
        self._pending_audio.clear()
        self.metrics.clear()
        self.silence_frames = 0
        self.speech_frames = 0
        self.has_speech = False
        self._detecting = False
        self._last_detect_frame = 0
        self.last_partial_text = ""
        self.last_speech_time = time.monotonic()
        self.last_frame_time = time.monotonic()

    async def set_state(self, s: str, ui_state: str | None = None) -> None:
        self.state = s
        await self.send_json({"type": "state", "state": ui_state if ui_state is not None else s})

    async def cancel_turn(self) -> None:
        self._reset_turn_audio()
        if self._start_task and not self._start_task.done():
            self._start_task.cancel()
        self._start_task = None
        task = self.turn
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.turn = None
        await self._abort_stt()
        await self.send_json({"type": "stop_audio"})
        await self.set_state("idle")

    async def close(self) -> None:
        self._reset_turn_audio()
        if self._start_task and not self._start_task.done():
            self._start_task.cancel()
            await asyncio.gather(self._start_task, return_exceptions=True)
        if self.turn and not self.turn.done():
            self.turn.cancel()
            await asyncio.gather(self.turn, return_exceptions=True)
        await self._abort_stt()

    async def on_audio(self, data: bytes) -> None:
        if not data:
            return

        if len(data) > FRAME_BYTES * 2:
            await self.send_json({
                "type": "error",
                "message": "Khung âm thanh không hợp lệ (vượt quá kích thước tối đa)."
            })
            return

        frame = data[: len(data) // 2 * 2]
        if not frame:
            return

        if self.state == "starting":
            self._pending_audio.append(frame)
            return

        if self.state != "listening" or self.stt is None:
            return

        try:
            await self.stt.send_frame(frame)
        except Exception:
            await self._abort_stt()
            await self.send_json({"type": "error", "message": "Mất kết nối dịch vụ nhận dạng giọng nói."})
            await self.set_state("idle")
            return

        async with self._vad_lock:
            await self._process_frame_vad(frame, self.turn_id)

    async def _process_frame_vad(self, frame: bytes, turn_id: int | None = None) -> None:
        current_turn = self.turn_id if turn_id is None else turn_id
        if not self.auto_turn:
            return

        self.audio_buf.extend(frame)
        if len(self.audio_buf) > 256000:
            del self.audio_buf[:-256000]

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2)))

        if self.vad.model.is_available:
            speech_prob = await asyncio.to_thread(self.vad.speech_prob, samples)
            if current_turn != self.turn_id or self.state != "listening":
                return
            is_speaking = (speech_prob >= 0.50 and rms >= 0.003) if not self.has_speech else (speech_prob >= 0.30 and rms >= 0.002)
        else:
            is_speaking = rms >= 0.003

        now = time.monotonic()
        self.last_frame_time = now

        if is_speaking:
            self.has_speech = True
            self.speech_frames += 1
            self.silence_frames = 0
            self.last_speech_time = now
            self._last_detect_frame = 0
        else:
            if self.has_speech:
                self.silence_frames += 1

        silence_passed = self.silence_frames - self._last_detect_frame
        if (
            self.has_speech
            and self.speech_frames >= MIN_SPEECH_FRAMES
            and self.silence_frames >= INITIAL_SILENCE_FRAMES
            and silence_passed >= SMART_TURN_CHECK_INTERVAL_FRAMES
            and not self._detecting
        ):
            self._detecting = True
            self._last_detect_frame = self.silence_frames
            self._detect_task = asyncio.create_task(self._check_turn_endpoint(self.turn_id))

    async def _check_turn_endpoint(self, turn_id: int) -> None:
        try:
            if turn_id != self.turn_id or self.state != "listening" or not self.stt:
                return

            if len(self.audio_buf) < 12000 or self.speech_frames < MIN_SPEECH_FRAMES:
                return

            audio_silence_s = self.silence_frames * (FRAME_DURATION_MS / 1000.0)
            wall_silence_s = time.monotonic() - self.last_speech_time
            if abs(wall_silence_s - audio_silence_s) > 0.5:
                print(f"[Audio Latency Warning] Lệch nhịp frame: audio={audio_silence_s:.2f}s, wall={wall_silence_s:.2f}s")

            norm_partial = normalize_text(self.last_partial_text) if self.last_partial_text else ""
            only_filler = is_only_filler(self.last_partial_text) if self.last_partial_text else False

            if only_filler:
                if self.silence_frames >= FILLER_TIMEOUT_FRAMES and turn_id == self.turn_id:
                    await self.send_json({
                        "type": "auto_stop",
                        "probability": 0.0,
                        "reason": "filler_timeout",
                    })
                    await self.stop_turn()
                return

            should_stop = False
            reason = ""
            prob = 0.0

            if self.detector.is_available:
                audio_f32 = np.frombuffer(self.audio_buf, dtype=np.int16).astype(np.float32) / 32768.0
                res = await asyncio.to_thread(self.detector.predict, audio_f32, 0.70)

                if turn_id != self.turn_id or self.state != "listening" or not self.stt:
                    return

                prob = res.get("probability", 0.0)
                reading_seq = is_reading_sequence(norm_partial) if norm_partial else False
                hesitating = is_hesitating(norm_partial) if norm_partial else False

                if reading_seq or hesitating:
                    # Đang đọc dở số hoặc ngập ngừng: bảo vệ bằng ngưỡng im lặng dài (>= 1.4s)
                    if prob >= 0.65 and self.silence_frames >= SMART_TURN_UNCERTAIN_FRAMES:
                        should_stop = True
                        reason = "smart_turn_done"
                else:
                    # Câu nói bình thường: ngắt siêu nhạy ở 700ms nếu tự tin cao
                    if prob >= 0.65 and self.silence_frames >= SMART_TURN_CONFIDENT_FRAMES:
                        should_stop = True
                        reason = "smart_turn_done"
                    elif prob >= 0.50 and self.silence_frames >= SMART_TURN_UNCERTAIN_FRAMES:
                        should_stop = True
                        reason = "smart_turn_high"

            if not should_stop and (self.silence_frames >= SILENCE_TIMEOUT_FRAMES or wall_silence_s >= 2.5):
                should_stop = True
                reason = "silence_timeout"

            if should_stop and turn_id == self.turn_id and self.state == "listening" and self.stt is not None:
                await self.send_json({
                    "type": "auto_stop",
                    "probability": round(prob, 3),
                    "reason": reason,
                })
                await self.stop_turn()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"[Smart Turn Error] {exc}")
        finally:
            if turn_id == self.turn_id:
                self._detecting = False

    async def start_turn(self) -> None:
        if self.state in ("thinking", "speaking"):
            await self.send_json({"type": "busy"})
            return

        await self._abort_stt()
        self._reset_turn_audio()
        await self.set_state("starting", ui_state="listening")

        stt = SttSession(STT_URL, self._on_stt_event)
        try:
            await stt.open()
        except SttError as exc:
            self._pending_audio.clear()
            await self.set_state("idle")
            await self.send_json({"type": "error", "message": friendly_stt_error(exc.payload)})
            return
        except Exception:
            self._pending_audio.clear()
            await self.set_state("idle")
            await self.send_json({"type": "error", "message": "Không kết nối được dịch vụ nhận dạng giọng nói."})
            return

        self.stt = stt
        self.state = "listening"

        if self._pending_audio:
            pending_frames = list(self._pending_audio)
            self._pending_audio.clear()
            try:
                for pending_frame in pending_frames:
                    await self.stt.send_frame(pending_frame)
                    async with self._vad_lock:
                        await self._process_frame_vad(pending_frame, self.turn_id)
            except Exception:
                await self._abort_stt()
                await self.set_state("idle")
                await self.send_json({"type": "error", "message": "Lỗi kết nối khi gửi âm thanh đến dịch vụ STT."})
                return

    async def stop_turn(self) -> None:
        if self._start_task and not self._start_task.done():
            await asyncio.gather(self._start_task, return_exceptions=True)
        if self.state != "listening" or self.stt is None:
            return

        fallback_text = self.last_partial_text.strip()
        stt, self.stt = self.stt, None
        self._reset_turn_audio()
        await self.set_state("thinking")
        self.turn = asyncio.create_task(self._turn(stt, fallback_text))

    async def _turn(self, stt: SttSession, fallback_text: str = "") -> None:
        t_stop = time.perf_counter()
        try:
            try:
                raw_text = (await stt.finish()).strip()
                if not raw_text and fallback_text:
                    raw_text = fallback_text
                text = normalize_speech_input(raw_text)
            except SttError as exc:
                if fallback_text:
                    text = normalize_speech_input(fallback_text)
                else:
                    await self.send_json({"type": "error", "message": friendly_stt_error(exc.payload)})
                    await self.send_json({"type": "bot_done"})
                    return
            except Exception:
                if fallback_text:
                    text = normalize_speech_input(fallback_text)
                else:
                    await self.send_json({"type": "error", "message": "Nhận dạng giọng nói thất bại, hãy thử lại."})
                    await self.send_json({"type": "bot_done"})
                    return

            await self.send_json({
                "type": "final",
                "text": text,
                "stt_final_s": round(time.perf_counter() - t_stop, 2),
            })
            if not text or is_only_filler(text):
                msg = (
                    "Tôi đang lắng nghe, bạn hãy nói rõ câu hỏi hoặc yêu cầu nhé."
                    if text
                    else "Không nghe rõ, hãy nói lại."
                )
                await self.send_json({"type": "info", "message": msg})
                await self.send_json({"type": "bot_done"})
                return

            self.metrics.clear()
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
            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]
            await self.send_json({
                "type": "metrics",
                **{k: round(v, 2) for k, v in self.metrics.items()},
            })
            await self.send_json({"type": "bot_done"})
        except asyncio.CancelledError:
            await stt.abort()
            raise
        except Exception as exc:
            await self.send_json({"type": "error", "message": f"Lỗi khi tạo câu trả lời: {type(exc).__name__}"})
            await self.send_json({"type": "bot_done"})
        finally:
            if self.turn is asyncio.current_task():
                self.turn = None
            await self.set_state("idle")

    async def send_json(self, obj: dict[str, Any]) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_json(obj)
            except (WebSocketDisconnect, RuntimeError):
                pass

    async def send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_bytes(data)
            except (WebSocketDisconnect, RuntimeError):
                pass

    async def on_control(self, msg: dict[str, Any]) -> None:
        t = msg.get("type")
        if t == "config":
            if "auto_turn" in msg:
                self.auto_turn = bool(msg["auto_turn"])
        elif t == "start_turn":
            if self._start_task and not self._start_task.done():
                return
            self._start_task = asyncio.create_task(self.start_turn())
        elif t == "stop_turn":
            await self.stop_turn()
        elif t == "interrupt":
            await self.cancel_turn()
        elif t == "reset":
            await self.cancel_turn()
            self.history.clear()

    async def _send_raw(self, text: str) -> None:
        await self.send_json({"type": "bot_raw", "text": text})

    async def _send_audio(self, pcm: bytes) -> None:
        if self.state == "thinking":
            await self.send_json({"type": "tts_format", "sample_rate": 48000, "channels": 1})
            await self.set_state("speaking")
        await self.send_bytes(pcm)

    async def _on_stt_event(self, kind: str, payload: Any) -> None:
        if kind == "partial":
            new_text = str(payload or "").strip()
            if new_text:
                self.last_partial_text = new_text
                self.has_speech = True
                self.speech_frames = max(self.speech_frames, MIN_SPEECH_FRAMES)
            await self.send_json({"type": "partial", "text": payload})
        elif kind == "error":
            await self.send_json({"type": "error", "message": friendly_stt_error(payload)})
            if self.state in ("listening", "starting"):
                await self._abort_stt()
                await self.set_state("idle")

    async def _abort_stt(self) -> None:
        if self.stt:
            stt, self.stt = self.stt, None
            await stt.abort()

@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket) -> None:
    await ws.accept()
    sess = VoiceSession(
        ws,
        app.state.http,
        detector=getattr(app.state, "detector", None),
        vad_model=getattr(app.state, "vad_model", None),
    )
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
                except json.JSONDecodeError as exc:
                    print(f"[WebSocket] Lỗi JSON: {exc}")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[WebSocket Error] {type(exc).__name__}: {exc}")
    finally:
        await sess.close()
