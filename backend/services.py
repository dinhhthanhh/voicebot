"""Module Service Adapters (STT / LLM / TTS / Smart Turn).

Đóng gói toàn bộ giao tiếp với các dịch vụ bên ngoài và mô hình suy luận:
- `STTClient`: Quản lý kết nối WebSocket streaming tới Qwen3-ASR.
- `llm_stream`: Gọi HTTP SSE streaming tới LLM (chuẩn OpenAI Chat Completions).
- `tts_stream`: Gọi HTTP streaming tới ZeroTTS, tách header WAV và căn chẵn PCM 16-bit.
- `SmartTurnDetector`: Dự đoán điểm ngắt câu (turn-taking) cục bộ bằng ONNX Smart Turn v3.2.
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
from concurrent.futures import ThreadPoolExecutor
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

# Ánh xạ mã lỗi kỹ thuật từ Qwen3-ASR sang thông báo tiếng Việt thân thiện cho người dùng
_STT_FRIENDLY_MESSAGES = {
    "OVERLOADED": "Dịch vụ nhận dạng giọng nói đang quá tải, vui lòng thử lại sau ít giây.",
    "QUEUE_TIMEOUT": "Dịch vụ nhận dạng giọng nói đang quá tải, vui lòng thử lại sau ít giây.",
    "AUDIO_TOO_LONG": "Câu nói quá dài (tối đa 600 giây).",
    "BAD_SAMPLE_RATE": "Lỗi định dạng âm thanh (sai sample rate).",
}


# =============================================================================
# 1. Các ngoại lệ chuẩn hóa (Service Exceptions)
# =============================================================================
class ServiceError(Exception):
    """Ngoại lệ gốc cho toàn bộ lỗi giao tiếp dịch vụ bên ngoài."""

    def __init__(self, message: str, code: str = "SERVICE_ERROR", raw_payload: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.raw_payload = raw_payload


class STTError(ServiceError):
    """Ngoại lệ STT tự động ánh xạ mã lỗi phổ biến thành câu thông báo thân thiện."""

    def __init__(self, message: str, code: str = "STT_ERROR", raw_payload: Any = None) -> None:
        super().__init__(
            _STT_FRIENDLY_MESSAGES.get(code, message),
            code=code,
            raw_payload=raw_payload,
        )


class LLMError(ServiceError):
    """Ngoại lệ khi gọi hoặc đọc stream từ LLM."""


class TTSError(ServiceError):
    """Ngoại lệ khi gọi hoặc giải mã audio stream từ TTS."""


# =============================================================================
# 2. STT Client Adapter (Qwen3-ASR WebSocket)
# =============================================================================
class STTClient:
    """Quản lý vòng đời một phiên WebSocket streaming tới máy chủ STT Qwen3-ASR."""

    def __init__(self, session_id: str, turn_id: str, request_id: str) -> None:
        self.session_id, self.turn_id, self.request_id = session_id, turn_id, request_id
        self.is_active = True                              # Còn mở để nhận/gửi audio hay không
        self.created_at = time.monotonic()                 # Mốc bắt đầu để giới hạn thời lượng audio
        self.websocket: ClientConnection | None = None     # Kết nối WebSocket tới server STT
        self.event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()  # Hàng đợi event đẩy ra ngoài
        self.recv_task: asyncio.Task[None] | None = None   # Task nền chạy `_recv_loop`
        self.is_ready = asyncio.Event()                    # Bật khi server phản hồi `ready` (hoặc lỗi sớm)
        self.is_done = asyncio.Event()                     # Bật khi nhận được `final` hoặc kết nối đóng
        self.final_text = ""                               # Kết quả nhận dạng cuối cùng
        self.error_payload: dict[str, Any] | None = None   # Payload lỗi nhận từ server hoặc mất kết nối
        self._close_lock = asyncio.Lock()                  # Khóa chống đóng trùng lặp đồng thời
        self._closed = False                               # Cờ idempotent đảm bảo chỉ đóng một lần

    # --- Các hàm nhánh nội bộ ---
    def _raise_if_error(self, default_code: str, default_msg: str) -> None:
        """Ném `STTError` nếu đã ghi nhận `error_payload` từ server hoặc sự cố đường truyền."""
        if self.error_payload:
            raise STTError(
                str(self.error_payload.get("message") or default_msg),
                code=str(self.error_payload.get("error_code") or self.error_payload.get("code") or default_code),
                raw_payload=self.error_payload,
            )

    async def _send_config(self) -> None:
        """Gửi gói tin cấu hình khởi tạo phiên STT tới server Qwen3-ASR."""
        if not self.websocket:
            return
        payload = {
            "type": "start",
            "request_id": self.request_id,
            "model": settings.stt_model,
            "language": settings.stt_language,
            "sample_rate": settings.stt_sample_rate,
            "encoding": "pcm_s16le",
            "channels": 1,
            "vad": settings.stt_vad,
        }
        await self.websocket.send(json.dumps(payload))

    async def _handle_message(self, message: dict[str, Any]) -> bool:
        """Phân loại một gói tin JSON từ STT; trả về `True` nếu đã tới gói kết thúc (`final`/`error`)."""
        message_type = message.get("type")
        if message_type == "ready":
            self.is_ready.set()
        elif message_type == "final":
            self.final_text = message.get("text", "")
            self.is_done.set()
        elif message_type == "error":
            self.error_payload = message
            # Đánh thức cả 2 Event để `connect()` lẫn `finish()` không bị treo chờ timeout
            self.is_done.set()
            self.is_ready.set()
        if message_type in ("partial", "vad", "error"):
            await self.event_queue.put(message)
        return message_type in ("final", "error")

    async def _on_disconnect(self, message: str) -> None:
        """Ghi nhận lỗi mất kết nối nếu WebSocket đóng trước khi có kết quả `final` hoặc lỗi trước đó."""
        if not self.final_text and not self.error_payload:
            self.error_payload = {
                "type": "error",
                "error_code": "CONNECTION_CLOSED",
                "message": message,
            }
            await self.event_queue.put(self.error_payload)

    async def _recv_loop(self) -> None:
        """Vòng lặp chạy nền đọc JSON từ WebSocket STT và đẩy sự kiện vào `event_queue`."""
        if not self.websocket:
            return
        try:
            async for raw_message in self.websocket:
                if isinstance(raw_message, bytes):
                    continue
                try:
                    payload = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.debug("STT [%s/%s] bỏ qua gói JSON không hợp lệ", self.session_id, self.turn_id)
                    continue
                if await self._handle_message(payload):
                    return
            # Server đóng WebSocket sạch sẽ nhưng không gửi gói `final` -> coi như rớt giữa chừng
            await self._on_disconnect("Kết nối STT bị đóng trước khi nhận kết quả hoàn tất.")
        except websockets.ConnectionClosed as cc:
            await self._on_disconnect(f"Mất kết nối STT (mã: {cc.code})")
        except Exception as exc:
            logger.warning("Lỗi nhận dữ liệu STT [%s/%s]: %s", self.session_id, self.turn_id, exc)
            if not self.error_payload:
                self.error_payload = {"type": "error", "error_code": "STT_RECV_ERROR", "message": str(exc)}
                await self.event_queue.put(self.error_payload)
        finally:
            self.is_done.set()
            self.is_ready.set()
            self.is_active = False
            await self.event_queue.put(None)  # Sentinel kết thúc iterator `events()`

    # --- Các hàm chính quản lý vòng đời STT ---
    async def close(self) -> None:
        """Đóng an toàn task nhận dữ liệu và WebSocket (idempotent, an toàn khi gọi đồng thời)."""
        if self._closed:
            return
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.is_active = False
            self.is_done.set()

            current_task = asyncio.current_task()

            recv_task = self.recv_task
            self.recv_task = None
            # Tránh tự hủy nếu `close()` tình cờ được gọi từ chính task `_recv_loop`
            if recv_task and not recv_task.done() and recv_task is not current_task:
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await recv_task

            ws = self.websocket
            self.websocket = None
            if ws:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await ws.close(code=1000, reason="Turn closed")

    @classmethod
    async def connect(cls, session_id: str, turn_id: str) -> STTClient:
        """Mở kết nối STT mới tới Qwen3-ASR, hỗ trợ retry theo exponential backoff khi quá tải."""
        req_id = str(uuid.uuid4())
        retries = settings.stt_max_retries
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            client = cls(
                session_id=session_id,
                turn_id=turn_id,
                # Gắn thêm chỉ số lần thử để server STT phân biệt từng request retry
                request_id=f"{req_id}_{attempt}" if attempt > 0 else req_id,
            )
            try:
                client.websocket = await websockets.connect(settings.stt_ws_url, max_size=10 * 1024 * 1024)
                client.recv_task = asyncio.create_task(client._recv_loop())
                await client._send_config()
                await asyncio.wait_for(client.is_ready.wait(), timeout=settings.stt_connect_timeout)
                client._raise_if_error("STT_START_FAILED", "Lỗi khởi tạo STT")
                return client
            except asyncio.CancelledError:
                await client.close()
                raise
            except Exception as exc:
                await client.close()
                # Chỉ retry các lỗi tạm thời (quá tải / nghẽn hàng đợi / rớt khi bắt tay); giữ nguyên STTError ở lần cuối
                if isinstance(exc, STTError) and (
                    attempt >= retries or exc.code not in ("OVERLOADED", "QUEUE_TIMEOUT", "CONNECTION_CLOSED")
                ):
                    raise
                last_error = exc
                if attempt < retries:
                    await asyncio.sleep(0.5 * (2**attempt))

        raise STTError(
            f"Không thể kết nối dịch vụ STT sau {retries + 1} lần thử: {last_error}",
            code="STT_CONNECT_FAILED",
        )

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Gửi một khung âm thanh PCM 16-bit lên kết nối STT đang hoạt động."""
        if not self.is_active or not self.websocket:
            raise STTError("STT session đã đóng hoặc không hoạt động.")
        if time.monotonic() - self.created_at > settings.stt_max_duration_seconds:
            raise STTError(
                f"Phiên nhận dạng vượt quá thời lượng tối đa {int(settings.stt_max_duration_seconds)} giây.",
                code="AUDIO_TOO_LONG",
            )
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
        """Yield lần lượt các sự kiện `partial`, `vad`, `error` nhận từ Qwen3-ASR."""
        while True:
            event = await self.event_queue.get()
            if event is None:
                break
            yield event
            if event.get("type") == "error":
                break

    async def finish(self, timeout: float | None = None) -> str:
        """Gửi lệnh kết thúc thu âm, chờ transcript `final` từ server rồi tự động đóng kết nối."""
        wait_timeout = timeout if timeout is not None else settings.stt_finish_timeout
        timed_out = False
        disconnected_early = not (self.websocket and self.is_active)
        try:
            if not disconnected_early and self.websocket:
                try:
                    await self.websocket.send(json.dumps({"type": "finish"}))
                    await asyncio.wait_for(self.is_done.wait(), timeout=wait_timeout)
                except TimeoutError:
                    timed_out = True
                except websockets.ConnectionClosed:
                    self.is_active = False
                    disconnected_early = True

            if not self.final_text:
                # Thứ tự ưu tiên lỗi: lỗi server/mất kết nối giữa chừng -> mất kết nối trước khi finish -> timeout
                self._raise_if_error("STT_ERROR", "Lỗi nhận dạng.")
                if disconnected_early:
                    raise STTError(
                        "Kết nối STT đã mất hoặc không hoạt động trước khi hoàn tất.",
                        code="CONNECTION_CLOSED",
                    )
                if timed_out:
                    raise STTError(
                        f"Hết thời gian chờ phản hồi STT sau {wait_timeout}s.",
                        code="STT_TIMEOUT",
                    )

            return self.final_text
        finally:
            await self.close()


# =============================================================================
# 3. HTTP Client Pool & LLM Adapter
# =============================================================================
_http_client: httpx.AsyncClient | None = None


# --- Các hàm nhánh hỗ trợ ---
def get_http_client() -> httpx.AsyncClient:
    """Lấy (hoặc khởi tạo mới) HTTP AsyncClient dùng chung keep-alive pool cho LLM và TTS."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            timeout=httpx.Timeout(90.0, connect=10.0),
        )
    return _http_client


async def close_http_client() -> None:
    """Đóng HTTP AsyncClient dùng chung khi Gateway tắt."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


def _parse_sse_delta(data_str: str) -> str | None:
    """Tách chuỗi `choices[0].delta.content` từ một dòng SSE JSON và kiểm tra gói báo lỗi."""
    try:
        chunk = json.loads(data_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(chunk, dict):
        return None
    if "error" in chunk:
        err = chunk["error"]
        err_msg = err.get("message") if isinstance(err, dict) else str(err)
        raise LLMError(f"LLM trả về lỗi stream: {err_msg}", code="LLM_STREAM_ERROR")

    choices = chunk.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    delta = first.get("delta") if isinstance(first, dict) else None
    content = delta.get("content") if isinstance(delta, dict) else None
    return content if isinstance(content, str) else None


def _build_llm_payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Đóng gói JSON payload chuẩn OpenAI Chat Completions cho yêu cầu streaming."""
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


# --- Hàm chính gọi LLM ---
async def llm_stream(messages: list[dict[str, str]], timeout: float | None = None) -> AsyncIterator[str]:
    """Gọi API Chat Completions dạng SSE streaming và yield từng mẩu text trả về."""
    if not settings.llm_chat_completions_url:
        raise LLMError("LLM_CHAT_COMPLETIONS_URL chưa cấu hình.", code="CONFIG_ERROR")
    actual_timeout = timeout if timeout is not None else settings.llm_timeout_seconds
    client = get_http_client()
    try:
        async with client.stream(
            "POST",
            settings.llm_chat_completions_url,
            json=_build_llm_payload(messages),
            timeout=httpx.Timeout(actual_timeout, connect=10.0),
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
                if delta := _parse_sse_delta(data_str):
                    yield delta
    except httpx.TimeoutException as exc:
        raise LLMError(f"Hết thời gian chờ phản hồi LLM sau {actual_timeout}s.", code="LLM_TIMEOUT") from exc
    except httpx.RequestError as exc:
        raise LLMError(f"Lỗi kết nối tới LLM: {exc}", code="LLM_CONNECTION_ERROR") from exc


# =============================================================================
# 4. TTS Adapter (ZeroTTS HTTP Stream)
# =============================================================================
# --- Các hàm nhánh xử lý WAV / PCM ---
def _wav_data_offset(header_buf: bytearray) -> int | None:
    """Duyệt cấu trúc RIFF/WAVE để tìm vị trí bắt đầu payload PCM (hoặc `None` nếu chưa đủ header)."""
    buf_len = len(header_buf)
    if buf_len < 4:
        return None
    if not header_buf.startswith(b"RIFF"):
        return 0  # Stream PCM thô không có WAV header
    if buf_len < 12:
        return None
    if header_buf[8:12] != b"WAVE":
        logger.warning("Stream bắt đầu bằng RIFF nhưng thiếu marker WAVE; giữ nguyên stream.")
        return 0

    offset = 12
    while offset + 8 <= buf_len:
        chunk_id = bytes(header_buf[offset : offset + 4])
        chunk_size = int.from_bytes(header_buf[offset + 4 : offset + 8], "little")
        if chunk_id == b"data":
            return offset + 8
        # Theo chuẩn RIFF, mọi chunk kích thước lẻ đều được đệm 1 byte 0 ở cuối để căn chẵn 2 byte
        next_offset = offset + 8 + chunk_size + (chunk_size % 2)
        if next_offset <= offset:
            return 0
        if next_offset > buf_len:
            return None
        offset = next_offset
    return 0 if buf_len > 65536 else None


def _process_tts_chunk(
    raw_chunk: bytes, header_parsed: bool, header_buf: bytearray, leftover: bytes
) -> tuple[bytes, bool, bytes]:
    """Loại bỏ WAV header ở chunk đầu tiên và căn chẵn bội số 2 byte cho PCM 16-bit."""
    if not header_parsed:
        header_buf.extend(raw_chunk)
        data_offset = _wav_data_offset(header_buf)
        if data_offset is None:
            return b"", False, leftover
        raw_chunk = bytes(header_buf[data_offset:])
        header_buf.clear()
        header_parsed = True

    if leftover:
        raw_chunk = leftover + raw_chunk
    # Giữ lại 1 byte lẻ cuối cùng để ghép với chunk kế tiếp, tránh lệch ranh giới sample 16-bit
    if len(raw_chunk) % 2 != 0:
        return raw_chunk[:-1], header_parsed, raw_chunk[-1:]
    return raw_chunk, header_parsed, b""


# --- Hàm chính gọi TTS ---
async def tts_stream(text: str, voice: str | None = None, timeout: float | None = None) -> AsyncIterator[bytes]:
    """Gọi ZeroTTS HTTP stream và yield từng khối byte PCM 16-bit đã bỏ WAV header và căn chẵn byte."""
    if not settings.tts_base_url:
        raise TTSError("TTS_BASE_URL chưa cấu hình.", code="CONFIG_ERROR")
    clean_text = text.strip()
    if not clean_text:
        return
    url = f"{settings.tts_base_url.rstrip('/')}/{settings.tts_stream_path.lstrip('/')}"
    actual_timeout = timeout if timeout is not None else settings.tts_timeout_seconds
    client = get_http_client()
    try:
        async with client.stream(
            "POST",
            url,
            json={"text": clean_text, "voice": voice or settings.tts_voice},
            timeout=httpx.Timeout(actual_timeout, connect=10.0),
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

            if leftover:
                logger.warning(
                    "Dữ liệu PCM từ TTS không hợp lệ (stream kết thúc với %d byte lẻ). Căn chỉnh 1 byte null để bảo toàn sample 16-bit.",
                    len(leftover),
                )
                yield leftover + b"\x00"
    except httpx.TimeoutException as exc:
        raise TTSError(f"Hết thời gian chờ phản hồi TTS sau {actual_timeout}s.", code="TTS_TIMEOUT") from exc
    except httpx.RequestError as exc:
        raise TTSError(f"Lỗi kết nối tới TTS: {exc}", code="TTS_CONNECTION_ERROR") from exc


# =============================================================================
# 5. Smart Turn Adapter (Smart Turn v3.2 ONNX Model)
# =============================================================================
class SmartTurnDetector:
    """Bộ nhận diện dứt câu thông minh (end-of-turn) sử dụng mô hình ONNX Smart Turn v3.2."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path or settings.smart_turn_model_path)
        self.session: ort.InferenceSession | None = None
        self.feature_extractor: WhisperFeatureExtractor | None = None
        self._load_lock = threading.Lock()  # Đảm bảo chỉ nạp model ONNX đúng 1 lần
        self._attempted = False             # Đánh dấu đã thử nạp model (tránh retry lặp lại khi thiếu file)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.smart_turn_max_workers,
            thread_name_prefix="smart-turn-worker",
        )

    # --- Các hàm nhánh chuẩn bị và suy luận đồng bộ ---
    @property
    def is_ready(self) -> bool:
        """Kiểm tra mô hình ONNX và bộ trích xuất đặc trưng đã sẵn sàng hay chưa."""
        return self.session is not None and self.feature_extractor is not None

    def load_model(self) -> None:
        """Nạp mô hình ONNX Smart Turn và bộ trích xuất đặc trưng Whisper (thread-safe)."""
        with self._load_lock:
            if self._attempted:
                return
            try:
                if not self.model_path.is_file():
                    logger.warning("Không tìm thấy mô hình Smart Turn: %s", self.model_path)
                    return
                so = ort.SessionOptions()
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.intra_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(str(self.model_path), sess_options=so)
                extractor = WhisperFeatureExtractor(chunk_length=8)
                self.session, self.feature_extractor = session, extractor
                logger.info("Đã nạp mô hình Smart Turn: %s", self.model_path.name)
            except Exception as exc:
                self.session = None
                self.feature_extractor = None
                logger.error("Lỗi khi khởi tạo Smart Turn ONNX: %s", exc)
            finally:
                self._attempted = True

    def _sync_predict(self, audio_bytes: bytes, sample_rate: int = 16000) -> float:
        """Tiền xử lý tối đa 8 giây âm thanh cuối và chạy suy luận ONNX đồng bộ."""
        if self.session is None or self.feature_extractor is None:
            return 0.0
        # Chuyển PCM int16 [-32768, 32767] -> float32 [-1.0, 1.0] và chỉ giữ lại 8 giây cuối
        audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        max_samples = 8 * sample_rate
        if len(audio_f32) > max_samples:
            audio_f32 = audio_f32[-max_samples:]
        # Chuẩn hóa zero-mean, unit-variance trước khi pad trái đúng yêu cầu của Smart Turn v3.2
        if len(audio_f32) > 0:
            mean = np.mean(audio_f32)
            var = np.mean((audio_f32 - mean) ** 2)
            audio_f32 = (audio_f32 - mean) / np.sqrt(var + 1e-7)
        if len(audio_f32) < max_samples:
            audio_f32 = np.pad(audio_f32, (max_samples - len(audio_f32), 0), mode="constant")

        inputs = self.feature_extractor(
            audio_f32, sampling_rate=sample_rate, return_tensors="np", padding="do_not_pad", truncation=False, do_normalize=False
        )
        outputs = cast(
            list[np.ndarray],
            self.session.run(None, {"input_features": inputs["input_features"].astype(np.float32)}),
        )
        return max(0.0, min(1.0, float(outputs[0][0].item())))

    # --- Các hàm chính ---
    async def predict(self, audio_bytes: bytes, sample_rate: int = 16000) -> float:
        """Dự đoán xác suất người dùng đã dứt câu [0.0..1.0] trên thread pool riêng."""
        if not self._attempted:
            await asyncio.to_thread(self.load_model)
        # Bỏ qua đoạn âm thanh quá ngắn (< 0.1s ở 16kHz = 3200 bytes)
        if self.session is None or self.feature_extractor is None or len(audio_bytes) < 3200:
            return 0.0
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._sync_predict, audio_bytes, sample_rate)
        except Exception as exc:
            logger.warning(
                "Lỗi suy luận Smart Turn (audio_len=%d): %s",
                len(audio_bytes),
                exc,
                exc_info=True,
            )
            return 0.0

    def shutdown(self) -> None:
        """Giải phóng ThreadPoolExecutor khi ứng dụng tắt."""
        self._executor.shutdown(wait=False)


smart_turn_detector = SmartTurnDetector()
