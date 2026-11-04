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
from typing import Any

import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from app.pipeline import run_turn
from app.stt import SttError, SttSession

STT_URL = os.environ.get("STT_WS_URL", "wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream")
LLM_URL = os.environ.get("LLM_URL", "https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions")
TTS_URL = os.environ.get("TTS_BASE_URL", "https://ai.vnpost.vn/zero-tts")
VOICE = os.environ.get("TTS_VOICE", "maichi")

STATIC = Path(__file__).parent / "static"
MAX_HISTORY = 8  # Số tin nhắn gần nhất lưu trữ trong bộ nhớ RAM phục vụ ngữ cảnh đa lượt


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: duy trì httpx.AsyncClient dùng chung để tái sử dụng kết nối."""
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    yield
    await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    """Trả về giao diện web đàm thoại Voicebot."""
    return FileResponse(STATIC / "index.html")


def friendly_stt_error(payload: dict[str, Any]) -> str:
    """Chuyển đổi mã lỗi STT thành thông điệp thân thiện với người dùng."""
    code = payload.get("error_code", "")
    if code in ("OVERLOADED", "QUEUE_TIMEOUT"):
        return "Dịch vụ nhận dạng giọng nói đang quá tải, vui lòng thử lại sau ít giây."
    if code == "AUDIO_TOO_LONG":
        return "Câu nói quá dài (tối đa 600 giây)."
    return f"Lỗi nhận dạng giọng nói ({code or payload.get('status_code', '?')})."


class VoiceSession:
    """Quản lý trạng thái và luồng dữ liệu cho một kết nối WebSocket từ trình duyệt."""

    def __init__(self, ws: WebSocket, http: httpx.AsyncClient) -> None:
        self.ws = ws
        self.http = http
        self.state = "idle"
        self.barge_in = True
        self.history: list[dict[str, str]] = []
        self.stt: SttSession | None = None
        self.turn: asyncio.Task | None = None
        self.metrics: dict[str, float] = {}
        self._send_lock = asyncio.Lock()

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
            self.barge_in = bool(msg.get("barge_in", True))
        elif t == "start_turn":
            await self.start_turn()
        elif t == "stop_turn":
            await self.stop_turn()
        elif t == "interrupt":
            await self.cancel_turn()
        elif t == "reset":
            self.history.clear()

    async def on_audio(self, data: bytes) -> None:
        if self.state != "listening" or self.stt is None:
            return
        try:
            await self.stt.send_frame(data[: len(data) // 2 * 2])
        except Exception:
            await self._abort_stt()
            await self.send_json({"type": "error", "message": "Mất kết nối dịch vụ nhận dạng giọng nói."})
            await self.set_state("idle")

    # ---- Vòng đời một lượt thoại ----
    async def start_turn(self) -> None:
        if self.state in ("thinking", "speaking"):
            if not self.barge_in:
                await self.send_json({"type": "busy"})
                return
            await self.cancel_turn()  # Barge-in: ngắt bot ngay để chuyển sang nghe lượt mới

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

        self.stt = stt
        await self.set_state("listening")

    async def stop_turn(self) -> None:
        if self.state != "listening" or self.stt is None:
            return
        stt, self.stt = self.stt, None
        await self.set_state("thinking")
        self.turn = asyncio.create_task(self._turn(stt))

    async def _turn(self, stt: SttSession) -> None:
        t_stop = time.perf_counter()
        try:
            try:
                text = (await stt.finish()).strip()
            except SttError:
                return
            except Exception:
                await self.send_json({"type": "error", "message": "Nhận dạng giọng nói thất bại, hãy thử lại."})
                return

            await self.send_json({
                "type": "final",
                "text": text,
                "stt_final_s": round(time.perf_counter() - t_stop, 2),
            })
            if not text:
                await self.send_json({"type": "info", "message": "Không nghe rõ, hãy nói lại."})
                return

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
            await stt.abort()
            raise
        except Exception as exc:
            await self.send_json({"type": "error", "message": f"Lỗi khi tạo câu trả lời: {type(exc).__name__}"})
        finally:
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
            await self.send_json({"type": "partial", "text": payload})
        elif kind == "error":
            await self.send_json({"type": "error", "message": friendly_stt_error(payload)})
            if self.state == "listening":
                self.stt = None
                await self.set_state("idle")

    async def cancel_turn(self) -> None:
        """Hủy bỏ lượt thoại đang thực thi (dừng LLM, TTS và âm thanh trên trình duyệt)."""
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
    finally:
        await sess.close()
