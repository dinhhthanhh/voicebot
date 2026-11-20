"""Module Service Adapters (STT / LLM / TTS / Smart Turn).

Đóng gói toàn bộ giao tiếp với dịch vụ bên ngoài:
- Quản lý kết nối, timeout và parse response.
- Chuẩn hóa lỗi thành ServiceError có ý nghĩa.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import onnxruntime as ort
import websockets
from transformers import WhisperFeatureExtractor
from websockets.asyncio.client import ClientConnection

from backend.config import settings

logger = logging.getLogger("services")


# ---------------------------------------------------------------------------
# 1. Các ngoại lệ chuẩn hóa (Service Exceptions)
# ---------------------------------------------------------------------------
class ServiceError(Exception):
    """Ngoại lệ chuẩn hóa cho các dịch vụ bên ngoài."""

    def __init__(self, message: str, code: str = "SERVICE_ERROR", raw_payload: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.raw_payload = raw_payload


class STTError(ServiceError):
    """Ngoại lệ STT tự động ánh xạ mã lỗi thành thông báo thân thiện."""

    def __init__(self, message: str, code: str = "STT_ERROR", raw_payload: Any = None) -> None:
        friendly = {
            "OVERLOADED": "Dịch vụ nhận dạng giọng nói đang quá tải, vui lòng thử lại sau ít giây.",
            "QUEUE_TIMEOUT": "Dịch vụ nhận dạng giọng nói đang quá tải, vui lòng thử lại sau ít giây.",
            "AUDIO_TOO_LONG": "Câu nói quá dài (tối đa 600 giây).",
            "BAD_SAMPLE_RATE": "Lỗi định dạng âm thanh (sai sample rate).",
        }.get(code, message)
        super().__init__(friendly, code=code, raw_payload=raw_payload)


class LLMError(ServiceError):
    """Lỗi giao tiếp với dịch vụ LLM."""


class TTSError(ServiceError):
    """Lỗi giao tiếp với dịch vụ TTS."""


# ---------------------------------------------------------------------------
# 2. STT Client Adapter (Qwen3-ASR WebSocket)
# ---------------------------------------------------------------------------
class STTClient:
    """Quản lý phiên kết nối WebSocket tới Qwen3-ASR."""

    def __init__(self, session_id: str, turn_id: str, request_id: str) -> None:
        self.session_id, self.turn_id, self.request_id = session_id, turn_id, request_id
        self.is_active = True
        self.created_at = time.monotonic()
        self.websocket: ClientConnection | None = None
        self.event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.recv_task: asyncio.Task[None] | None = None
        self.is_ready = asyncio.Event()
        self.is_done = asyncio.Event()
        self.final_text = ""
        self.error_payload: dict[str, Any] | None = None

    @classmethod
    async def connect(cls, session_id: str, turn_id: str, request_id: str | None = None, max_retries: int = 2) -> STTClient:
        req_id = request_id or str(uuid.uuid4())
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            client = cls(session_id=session_id, turn_id=turn_id, request_id=req_id)
            try:
                client.websocket = await websockets.connect(settings.stt_ws_url, max_size=10 * 1024 * 1024)
                client.recv_task = asyncio.create_task(client._recv_loop())
                await client.websocket.send(
                    json.dumps({
                        "type": "start",
                        "request_id": req_id,
                        "model": settings.stt_model,
                        "language": settings.stt_language,
                        "sample_rate": settings.stt_sample_rate,
                        "encoding": "pcm_s16le",
                        "channels": 1,
                        "vad": settings.stt_vad,
                    })
                )
                await asyncio.wait_for(client.is_ready.wait(), timeout=10.0)
                if client.error_payload:
                    raise STTError(
                        client.error_payload.get("message", "Lỗi khởi tạo STT"),
                        code=client.error_payload.get("error_code", "STT_START_FAILED"),
                        raw_payload=client.error_payload,
                    )
                return client
            except STTError as stt_err:
                await client.close()
                if stt_err.code in ("OVERLOADED", "QUEUE_TIMEOUT") and attempt < max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise
            except Exception as exc:
                last_error = exc
                await client.close()
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                else:
                    break

        raise STTError(f"Không thể kết nối dịch vụ STT sau {max_retries + 1} lần thử: {last_error}", code="STT_CONNECT_FAILED")

    async def _handle_message(self, message: dict[str, Any]) -> bool:
        message_type = message.get("type")
        if message_type == "ready":
            self.is_ready.set()
        elif message_type == "final":
            self.final_text = message.get("text", "")
            self.is_done.set()
        elif message_type == "error":
            self.error_payload = message
            self.is_done.set()
            self.is_ready.set()
        if message_type in ("ready", "partial", "vad", "final", "error"):
            await self.event_queue.put(message)
        return message_type in ("final", "error")

    async def _recv_loop(self) -> None:
        if not self.websocket:
            return
        try:
            async for raw_message in self.websocket:
                if isinstance(raw_message, bytes):
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    if await self._handle_message(json.loads(raw_message)):
                        break
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed as cc:
            if not self.final_text and not self.error_payload:
                self.error_payload = {"type": "error", "error_code": "CONNECTION_CLOSED", "message": f"Mất kết nối STT (mã: {cc.code})"}
                await self.event_queue.put(self.error_payload)
        except Exception as exc:
            self.error_payload = {"type": "error", "error_code": "STT_RECV_ERROR", "message": str(exc)}
            await self.event_queue.put(self.error_payload)
        finally:
            self.is_done.set()
            self.is_ready.set()
            self.is_active = False
            await self.event_queue.put(None)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if not self.is_active or not self.websocket:
            raise STTError("STT session đã đóng hoặc không hoạt động.")
        if time.monotonic() - self.created_at > 600:
            raise STTError("Phiên nhận dạng vượt quá thời lượng tối đa 600 giây.", code="AUDIO_TOO_LONG")
        if len(pcm_bytes) % 2 != 0:
            raise STTError("Số byte âm thanh phải chẵn.", code="INVALID_AUDIO_FORMAT")
        if len(pcm_bytes) > 1024 * 1024:
            raise STTError("Frame audio vượt quá giới hạn 1 MiB.", code="AUDIO_FRAME_TOO_LARGE")
        try:
            await self.websocket.send(pcm_bytes)
        except websockets.ConnectionClosed as exc:
            self.is_active = False
            raise STTError("Kết nối STT bị ngắt khi gửi audio.") from exc
        except Exception as exc:
            self.is_active = False
            raise STTError(f"Lỗi gửi audio tới STT: {exc}", code="STT_SEND_ERROR") from exc

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self.event_queue.get()
            if event is None:
                break
            yield event
            if event.get("type") in ("final", "error"):
                break

    async def finish(self, timeout: float = 10.0) -> str:
        try:
            if self.websocket and self.is_active:
                await self.websocket.send(json.dumps({"type": "finish"}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.is_done.wait(), timeout=timeout)
            if self.error_payload and not self.final_text:
                err_code = self.error_payload.get("error_code") or self.error_payload.get("code") or "STT_ERROR"
                raise STTError(self.error_payload.get("message", "Lỗi nhận dạng."), code=err_code, raw_payload=self.error_payload)
            return self.final_text
        finally:
            await self.close()

    async def close(self) -> None:
        self.is_active = False
        self.is_done.set()
        current_task = None
        with contextlib.suppress(RuntimeError):
            current_task = asyncio.current_task()
        if self.recv_task and not self.recv_task.done() and self.recv_task is not current_task:
            self.recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.recv_task
        self.recv_task = None
        if self.websocket:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.websocket.close(code=1000, reason="Turn closed")
            self.websocket = None


# ---------------------------------------------------------------------------
# 3. HTTP Client Pool & LLM Adapter
# ---------------------------------------------------------------------------
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            timeout=httpx.Timeout(90.0, connect=10.0),
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


def _parse_sse_delta(data_str: str) -> str | None:
    try:
        chunk = json.loads(data_str)
    except Exception:
        return None
    if isinstance(chunk, dict):
        if "error" in chunk:
            err = chunk["error"]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            raise LLMError(f"LLM trả về lỗi stream: {err_msg}", code="LLM_STREAM_ERROR")
        choices = chunk.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            return choices[0].get("delta", {}).get("content")
    return None


def _build_llm_payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": settings.llm_temperature,
        "stream": True,
    }
    if settings.llm_model:
        payload["model"] = settings.llm_model
    if settings.llm_reasoning_effort:
        payload["reasoning_effort"] = settings.llm_reasoning_effort
    return payload


async def llm_stream(messages: list[dict[str, str]], timeout: float | None = None) -> AsyncIterator[str]:
    if not settings.llm_chat_completions_url:
        raise LLMError("LLM_CHAT_COMPLETIONS_URL chưa cấu hình.", code="CONFIG_ERROR")
    actual_timeout = timeout or settings.llm_timeout_seconds
    client = get_http_client()
    try:
        async with client.stream(
            "POST",
            settings.llm_chat_completions_url,
            json=_build_llm_payload(messages),
            headers={"Content-Type": "application/json"},
            timeout=actual_timeout,
        ) as resp:
            if resp.status_code != 200:
                err_body = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                raise LLMError(f"LLM trả về {resp.status_code}: {err_body}", code=f"HTTP_{resp.status_code}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                delta = _parse_sse_delta(data_str)
                if delta:
                    yield delta
    except httpx.TimeoutException as exc:
        raise LLMError(f"Hết thời gian chờ phản hồi LLM sau {actual_timeout}s.", code="LLM_TIMEOUT") from exc
    except httpx.RequestError as exc:
        raise LLMError(f"Lỗi kết nối tới LLM: {exc}", code="LLM_CONNECTION_ERROR") from exc


# ---------------------------------------------------------------------------
# 4. TTS Adapter (ZeroTTS HTTP Stream)
# ---------------------------------------------------------------------------
def _process_tts_chunk(
    raw_chunk: bytes, header_parsed: bool, header_buf: bytearray, leftover: bytes
) -> tuple[bytes, bool, bytes]:
    """Tách WAV header ở chunk đầu và căn chỉnh chẵn byte 16-bit."""
    if not header_parsed:
        header_buf.extend(raw_chunk)
        if len(header_buf) < 4:
            return b"", False, leftover
        if not header_buf.startswith(b"RIFF"):
            raw_chunk, header_parsed = bytes(header_buf), True
            header_buf.clear()
        else:
            data_idx = header_buf.find(b"data")
            if data_idx != -1 and len(header_buf) >= data_idx + 8:
                raw_chunk = bytes(header_buf[data_idx + 8 :])
                header_buf.clear()
                header_parsed = True
            elif len(header_buf) > 4096:
                raw_chunk = bytes(header_buf[44:])
                header_buf.clear()
                header_parsed = True
            else:
                return b"", False, leftover

    if leftover:
        raw_chunk, leftover = leftover + raw_chunk, b""
    if len(raw_chunk) % 2 != 0:
        return raw_chunk[:-1], header_parsed, raw_chunk[-1:]
    return raw_chunk, header_parsed, b""


async def tts_stream(text: str, voice: str | None = None, timeout: float | None = None) -> AsyncIterator[bytes]:
    clean_text = text.strip()
    if not clean_text:
        return
    url = f"{settings.tts_base_url.rstrip('/')}/{settings.tts_stream_path.lstrip('/')}"
    actual_timeout = timeout or settings.tts_timeout_seconds
    client = get_http_client()
    try:
        async with client.stream(
            "POST",
            url,
            json={"text": clean_text, "voice": voice or settings.tts_voice},
            timeout=actual_timeout,
        ) as resp:
            if resp.status_code != 200:
                err_body = (await resp.aread()).decode("utf-8", errors="replace")[:200]
                raise TTSError(f"ZeroTTS trả về HTTP {resp.status_code}: {err_body}", code=f"HTTP_{resp.status_code}")
            header_parsed, header_buf, leftover = False, bytearray(), b""
            async for raw_chunk in resp.aiter_bytes():
                if not raw_chunk:
                    continue
                aligned_data, header_parsed, leftover = _process_tts_chunk(raw_chunk, header_parsed, header_buf, leftover)
                if aligned_data:
                    yield aligned_data
    except httpx.TimeoutException as exc:
        raise TTSError(f"Hết thời gian chờ phản hồi TTS sau {actual_timeout}s.", code="TTS_TIMEOUT") from exc
    except httpx.RequestError as exc:
        raise TTSError(f"Lỗi kết nối tới TTS: {exc}", code="TTS_CONNECTION_ERROR") from exc


# ---------------------------------------------------------------------------
# 5. Smart Turn Adapter (Smart Turn v3.2 ONNX Model)
# ---------------------------------------------------------------------------
class SmartTurnDetector:
    """Bộ nhận diện dứt câu thông minh sử dụng mô hình ONNX Smart Turn v3.2."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path or settings.smart_turn_model_path)
        self.session: ort.InferenceSession | None = None
        self.feature_extractor: WhisperFeatureExtractor | None = None
        self._load_lock = threading.Lock()
        self._attempted = False

    def _load_model(self) -> None:
        with self._load_lock:
            if self._attempted:
                return
            self._attempted = True
            if not self.model_path.is_file():
                logger.warning("Không tìm thấy mô hình Smart Turn: %s", self.model_path)
                return
            try:
                so = ort.SessionOptions()
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.inter_op_num_threads = 1
                so.intra_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(str(self.model_path), sess_options=so)
                self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
                logger.info("Đã nạp mô hình Smart Turn: %s", self.model_path.name)
            except Exception as exc:
                logger.error("Lỗi khi khởi tạo Smart Turn ONNX: %s", exc)

    def warmup(self) -> None:
        self._load_model()

    def _sync_predict(self, audio_bytes: bytes, sample_rate: int = 16000) -> float:
        if self.session is None or self.feature_extractor is None:
            return 0.0
        audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        max_samples = 8 * sample_rate
        if len(audio_f32) > max_samples:
            audio_f32 = audio_f32[-max_samples:]
        if len(audio_f32) > 0:
            std = np.std(audio_f32)
            audio_f32 = (audio_f32 - np.mean(audio_f32)) / np.sqrt(std**2 + 1e-7)
        if len(audio_f32) < max_samples:
            audio_f32 = np.pad(audio_f32, (max_samples - len(audio_f32), 0), mode="constant")
        inputs = self.feature_extractor(
            audio_f32, sampling_rate=sample_rate, return_tensors="np", padding="do_not_pad", truncation=False, do_normalize=False
        )
        outputs = cast(list[np.ndarray], self.session.run(None, {"input_features": inputs.input_features.astype(np.float32)}))
        return max(0.0, min(1.0, float(outputs[0][0].item())))

    async def predict_probability(self, audio_bytes: bytes, sample_rate: int = 16000) -> float:
        if not self._attempted:
            await asyncio.to_thread(self._load_model)
        if self.session is None or self.feature_extractor is None or len(audio_bytes) < 3200:
            return 0.0
        try:
            return await asyncio.to_thread(self._sync_predict, audio_bytes, sample_rate)
        except Exception:
            return 0.0


_smart_turn_detector = SmartTurnDetector()


def init_smart_turn() -> None:
    _smart_turn_detector.warmup()


async def predict_smart_turn(audio_buffer: bytes, sample_rate: int = 16000) -> float:
    return await _smart_turn_detector.predict_probability(audio_buffer, sample_rate=sample_rate)


def is_smart_turn_ready() -> bool:
    """Kiểm tra mô hình Smart Turn ONNX đã nạp vào RAM và sẵn sàng hay chưa."""
    return _smart_turn_detector.session is not None

