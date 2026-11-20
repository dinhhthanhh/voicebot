"""Gateway chính của Voicebot MVP (FastAPI + WebSocket Gateway)."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from backend import services  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.voice import VoiceSession, orchestrator  # noqa: E402

logging.basicConfig(
    level=getattr(logging, settings.app_log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("gateway")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Khởi tạo Smart Turn đồng bộ trước khi nhận request và dọn dẹp HTTP client khi tắt."""
    logger.info("Khởi động Voicebot Gateway tại %s:%s", settings.app_host, settings.app_port)
    for w in settings.validate_config():
        logger.warning("Cấu hình cảnh báo: %s", w)
    await asyncio.to_thread(services.init_smart_turn)
    yield
    logger.info("Tắt Voicebot Gateway.")
    await services.close_http_client()


app = FastAPI(
    title="Voicebot MVP Gateway",
    version="1.0.0",
    description="Hệ thống trợ lý giọng nói đàm thoại thời gian thực",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return JSONResponse(
        {
            "status": "ok",
            "smart_turn_ready": services.is_smart_turn_ready(),
            "active_sessions": len(orchestrator.sessions),
        }
    )


async def _handle_ws_message(session: VoiceSession, message: Mapping[str, Any]) -> None:
    if session.closed:
        return
    if "text" in message:
        try:
            payload = json.loads(message["text"])
            if not isinstance(payload, dict):
                await session.emit_event(
                    "session.error", code="INVALID_PAYLOAD", message="Payload phải là JSON object."
                )
                return
            await session.handle_event(payload)
        except json.JSONDecodeError:
            await session.emit_event("session.error", code="INVALID_JSON", message="JSON không hợp lệ.")
    elif "bytes" in message:
        await session.handle_audio_frame(message["bytes"])


@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def send_json(data: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(data, ensure_ascii=False))

    async def send_binary(data: bytes) -> None:
        await websocket.send_bytes(data)

    try:
        session = orchestrator.create_session(send_json_cb=send_json, send_binary_cb=send_binary)
    except RuntimeError as exc:
        logger.warning("Từ chối kết nối WebSocket do quá tải: %s", exc)
        await send_json({"type": "session.error", "code": "SERVER_OVERLOADED", "message": str(exc)})
        await websocket.close(code=1008, reason="Server overloaded")
        return

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            await _handle_ws_message(session, message)
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        pass
    except Exception as exc:
        if "disconnect" not in str(exc).lower():
            logger.exception("Lỗi trong WebSocket loop: %s", exc)
    finally:
        await orchestrator.close_session(session.session_id)


if settings.frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.app_log_level.lower(),
        reload=True,
        app_dir=str(_ROOT_DIR),
    )
