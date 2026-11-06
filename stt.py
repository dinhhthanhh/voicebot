from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import onnxruntime as ort
import websockets

try:
    from websockets.asyncio.client import ClientConnection as WsClientConnection
except ImportError:
    try:
        from websockets.client import WebSocketClientProtocol as WsClientConnection
    except ImportError:
        WsClientConnection = Any

from transformers import WhisperFeatureExtractor

logger = logging.getLogger("stt")

PING_EVERY_SEC = 30
TARGET_SR = 16000
VAD_WINDOW_SAMPLES = 512
MAX_SECONDS = 8
MAX_SAMPLES = MAX_SECONDS * TARGET_SR

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "smart-turn-v3.2-cpu.onnx"
VAD_MODEL_PATH = MODEL_DIR / "silero_vad.onnx"

# ==============================================================================
# 1. Chuyển đổi âm thanh sang PCM16LE mono 16kHz
# ==============================================================================
def to_16k_mono_pcm16(raw_pcm: bytes, src_rate: int, src_channels: int) -> bytes:
    """Chuyển đổi PCM16LE bất kỳ tần số lấy mẫu và số kênh sang PCM16LE mono 16 kHz."""
    x = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32)
    if src_channels > 1:
        x = x.reshape(-1, src_channels).mean(axis=1)
    if src_rate != TARGET_SR:
        if src_rate > TARGET_SR:
            # Low-pass sinc filter để tránh hiện tượng aliasing khi downsample
            n = 255
            t = np.arange(n) - (n - 1) / 2
            h = np.sinc(2 * (7600 / src_rate) * t) * np.hamming(n)
            x = np.convolve(x, h / h.sum(), mode="same")
        pos = np.arange(0, len(x) - 1, src_rate / TARGET_SR)
        x = np.interp(pos, np.arange(len(x)), x)
    return np.clip(np.rint(x), -32768, 32767).astype("<i2").tobytes()


# ==============================================================================
# 2. Giao tiếp WebSocket Qwen3-ASR
# ==============================================================================
@dataclass
class SttResult:
    """Kết quả nhận dạng giọng nói kèm các mốc đo thời gian."""
    final_text: str = ""
    first_partial_at: float | None = None
    final_at: float | None = None
    final_abs: float | None = None
    error: dict[str, Any] | None = None
    events: list[tuple[float, str, str]] = field(default_factory=list)


class SttError(RuntimeError):
    """Ngoại lệ khi dịch vụ STT trả về thông báo lỗi (ví dụ 429 OVERLOADED)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(str(payload))
        self.payload = payload


class SttSession:
    """Quản lý một phiên STT streaming qua WebSocket cho trình duyệt."""

    def __init__(self, url: str, on_event: Callable[[str, Any], Awaitable[None]]) -> None:
        self.url = url
        self.on_event = on_event
        self.ws: WsClientConnection | None = None
        self.final_text = ""
        self.error: dict[str, Any] | None = None
        self._done = asyncio.Event()
        self._reader: asyncio.Task | None = None

    async def open(self, max_retries: int = 2) -> None:
        """Mở kết nối WebSocket và gửi gói tin khởi tạo `start`, tự động retry khi gặp lỗi 429/overload."""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                self.ws = await websockets.connect(self.url, max_size=10 * 2**20)
                start = {
                    "type": "start",
                    "request_id": str(uuid.uuid4()),
                    "language": "Vietnamese",
                    "sample_rate": TARGET_SR,
                    "encoding": "pcm_s16le",
                    "channels": 1,
                }
                await self.ws.send(json.dumps(start))
                ready = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=15.0))
                if ready.get("type") == "error":
                    code = ready.get("error_code", "")
                    if code in ("OVERLOADED", "QUEUE_TIMEOUT") and attempt < max_retries:
                        await self.ws.close()
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    raise SttError(ready)
                self._reader = asyncio.create_task(self._read())
                return
            except SttError:
                if self.ws is not None:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                raise
            except Exception as exc:
                last_exc = exc
                if self.ws is not None:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                else:
                    raise last_exc

    async def _read(self) -> None:
        """Đọc kết quả từ server STT và kích hoạt callback sự kiện."""
        ws = self.ws
        if ws is None:
            return
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "partial" and msg.get("text"):
                    await self.on_event("partial", msg["text"])
                elif kind == "final":
                    self.final_text = msg.get("text", "")
                    break
                elif kind == "error":
                    self.error = msg
                    await self.on_event("error", msg)
                    break
        except websockets.ConnectionClosed:
            pass
        finally:
            self._done.set()

    async def send_frame(self, pcm: bytes) -> None:
        """Gửi một khung dữ liệu âm thanh PCM16LE mono 16 kHz."""
        if self.ws is not None:
            await self.ws.send(pcm)

    async def finish(self, timeout: float = 10.0) -> str:
        """Gửi tín hiệu kết thúc thu âm và chờ nhận kết quả final cuối cùng."""
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({"type": "finish"}))
                await asyncio.wait_for(self._done.wait(), timeout)
            except asyncio.TimeoutError:
                logger.warning("STT finish timeout sau %.1fs", timeout)
            finally:
                try:
                    await self.ws.close()
                except Exception:
                    pass
        if self.error and not self.final_text:
            raise SttError(self.error)
        return self.final_text

    async def abort(self) -> None:
        """Hủy bỏ phiên nhận dạng lập tức."""
        if self._reader and not self._reader.done():
            self._reader.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        self._done.set()


async def transcribe_pcm16_16k(
    pcm: bytes, *, url: str, chunk_sec: float = 0.3, realtime: bool = True
) -> SttResult:
    """Gửi toàn bộ mảng dữ liệu PCM qua WebSocket STT và trả về kết quả kèm các mốc thời gian."""
    res = SttResult()
    frame_bytes = int(TARGET_SR * chunk_sec) * 2
    start = {
        "type": "start",
        "request_id": str(uuid.uuid4()),
        "language": "Vietnamese",
        "sample_rate": TARGET_SR,
        "encoding": "pcm_s16le",
        "channels": 1,
    }

    async with websockets.connect(url, max_size=10 * 2**20) as ws:
        await ws.send(json.dumps(start))
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
        if ready.get("type") == "error":
            res.error = ready
            return res

        t0 = time.perf_counter()
        finish_at: float | None = None
        stop = asyncio.Event()

        async def sender() -> None:
            nonlocal finish_at
            last_ping = time.perf_counter()
            for i in range(0, len(pcm), frame_bytes):
                if stop.is_set():
                    return
                await ws.send(pcm[i:i + frame_bytes])
                if realtime:
                    await asyncio.sleep(chunk_sec)
                if time.perf_counter() - last_ping >= PING_EVERY_SEC:
                    await ws.send(json.dumps({"type": "ping"}))
                    last_ping = time.perf_counter()
            if stop.is_set():
                return
            finish_at = time.perf_counter()
            await ws.send(json.dumps({"type": "finish"}))

        async def receiver() -> None:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                now = time.perf_counter() - t0
                if kind == "partial":
                    text = msg.get("text") or ""
                    if text and res.first_partial_at is None:
                        res.first_partial_at = now
                    res.events.append((now, "partial", text))
                elif kind == "final":
                    res.final_text = msg.get("text", "")
                    res.final_at = (time.perf_counter() - finish_at) if finish_at else None
                    res.final_abs = time.perf_counter()
                    res.events.append((now, "final", res.final_text))
                elif kind == "vad":
                    res.events.append((now, "vad", msg.get("state", "")))
                elif kind == "error":
                    res.error = msg
                    stop.set()
                    return

        total_sec = len(pcm) / 2 / TARGET_SR
        results = await asyncio.wait_for(
            asyncio.gather(sender(), receiver(), return_exceptions=True),
            timeout=total_sec + 60.0,
        )
        for r in results:
            if isinstance(r, Exception) and not (res.error and isinstance(r, websockets.ConnectionClosed)):
                raise r
    return res


# ==============================================================================
# 3. Phân tích ngữ nghĩa & Nhận diện ngập ngừng / Dãy số
# ==============================================================================
FILLER_WORDS = {
    "ừ", "ừm", "ờ", "ờm", "à", "ư", "hử", "hả", "ơ", "ớ", "ể", "à mà", "ừ thì", "ờ thì"
}

DANGLING_SUFFIXES = (
    "ừ", "ừm", "ờ", "ờm", "à", "ơ", "ê", "nè", "nha", "nhé",
    "là", "và", "hoặc", "nhưng", "mà", "thì", "vì", "do", "bởi", "nếu", "khi",
    "của", "cho", "với", "ở tại", "tại vì", "ở", "về", "đến", "tới", "trong", "ngoài",
    "từ", "đi", "sang", "qua", "lên", "xuống", "vào", "ra", "được",
    "như", "như là", "ví dụ", "chẳng hạn", "kiểu", "kiểu như",
    "tôi muốn", "cho tôi", "giúp tôi", "làm sao để", "có nghĩa là", "tức là", "tên là",
    "cho tôi hỏi", "cho em hỏi", "tôi muốn hỏi", "em cho anh hỏi", "em cho chị hỏi",
    "hỏi", "xem", "rằng", "để", "sau", "trước", "khoảng", "tầm", "cỡ",
    "gửi", "chuyển", "phát", "nhận", "lấy", "tính",
)

NUMBER_AND_ADDRESS_SUFFIXES = (
    "mười", "trăm", "nghìn", "ngàn", "triệu",
    "số", "ngõ", "ngách", "hẻm", "đường", "phố", "phường", "xã", "thị trấn",
    "quận", "huyện", "tỉnh", "thành phố", "bưu cục", "chi nhánh",
    "mã", "mã đơn", "vận đơn", "bưu phẩm", "điện thoại", "tòa", "tầng",
)

WORD_TO_DIGIT = {
    "không": "0", "một": "1", "mốt": "1", "hai": "2", "ba": "3",
    "bốn": "4", "tư": "4", "năm": "5", "lăm": "5", "sáu": "6",
    "bảy": "7", "tám": "8", "chín": "9",
}

TRAILING_DIGIT_RE = re.compile(r"(\d{3,}|\b(eb|vn|ems|cp|lp)\s*\d+)$", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s]")
SPACES_RE = re.compile(r"\s+")
TAIL_DIGITS_RE = re.compile(r"(\d[\d\s]*)$")
PHONE_START_RE = re.compile(r"0[35789]")


def _normalize_text(text: str) -> str:
    cleaned = NON_WORD_RE.sub(" ", text.lower())
    return SPACES_RE.sub(" ", cleaned).strip()


normalize_text = _normalize_text


def normalize_phone_words_in_text(text: str) -> str:
    """Tự động gom cụm từ đọc số tiếng Việt liên tiếp (>=7 số) thành chuỗi chữ số (VD: 'không chín...' -> '09...')."""
    tokens = text.split()
    converted: list[str] = []
    digit_run: list[tuple[str, str]] = []

    for tok in tokens:
        clean_tok = NON_WORD_RE.sub("", tok).lower()
        if clean_tok in WORD_TO_DIGIT:
            digit_run.append((clean_tok, WORD_TO_DIGIT[clean_tok]))
        elif re.fullmatch(r"\d+", clean_tok):
            digit_run.append((clean_tok, clean_tok))
        else:
            if digit_run:
                digits_str = "".join(d[1] for d in digit_run)
                if len(digits_str) >= 7:
                    converted.append(digits_str)
                else:
                    converted.extend(d[0] for d in digit_run)
                digit_run = []
            converted.append(tok)

    if digit_run:
        digits_str = "".join(d[1] for d in digit_run)
        if len(digits_str) >= 7:
            converted.append(digits_str)
        else:
            converted.extend(d[0] for d in digit_run)

    return " ".join(converted)


def normalize_speech_input(text: str) -> str:
    """Chuẩn hóa chuỗi nhận dạng giọng nói từ người dùng (gom từ đọc số thành chữ số). Việc nhận diện mã bưu phẩm do LLM trực tiếp xử lý."""
    return normalize_phone_words_in_text(text)


def validate_vietnam_phone(text: str) -> tuple[bool, str]:
    """Kiểm tra và chuẩn hóa số điện thoại di động Việt Nam bằng thư viện Google phonenumbers.
    Trả về (is_valid, formatted_phone).
    """
    import phonenumbers
    try:
        # Tìm các chuỗi số 10 chữ số bắt đầu bằng 0
        matches = re.findall(r"\b0[35789]\d{8}\b", text)
        if not matches:
            return False, ""
        p = phonenumbers.parse(matches[0], "VN")
        if phonenumbers.is_valid_number(p):
            formatted = phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.NATIONAL)
            return True, formatted.replace(" ", "")
    except Exception:
        pass
    return False, ""


def _ends_with_any(norm: str, phrases: tuple[str, ...]) -> bool:
    """Kiểm tra xem câu kết thúc bằng một cụm từ (so khớp theo ranh giới từ, không theo ký tự)."""
    padded = f" {norm}"
    return any(padded.endswith(f" {p}") for p in phrases)


def _tail_digit_run(tokens: list[str]) -> int:
    """Đếm số token chữ số liên tiếp ở cuối câu."""
    n = 0
    for t in reversed(tokens):
        if not t.isdigit():
            break
        n += 1
    return n


def is_only_filler(text: str) -> bool:
    """Kiểm tra xem văn bản nhận diện được mới chỉ có tiếng 'ừ, ờ, ùm' hoặc còn trống."""
    norm = _normalize_text(text)
    if not norm:
        return True
    words = norm.split()
    return all(w in FILLER_WORDS for w in words)


def is_reading_sequence(text: str) -> bool:
    """Phát hiện người dùng đang đọc dở dãy số (SĐT, mã vận đơn) hoặc địa chỉ."""
    norm = _normalize_text(text)
    if not norm:
        return False

    tokens = [WORD_TO_DIGIT.get(w, w) for w in norm.split()]
    norm_digits = " ".join(tokens)

    tail_match = TAIL_DIGITS_RE.search(norm_digits)
    if tail_match:
        tail_digits = SPACES_RE.sub("", tail_match.group(1))
        # Chỉ coi là dãy số/SĐT khi có >= 2 chữ số (loại trừ từ 'không' đơn lẻ cuối câu hỏi)
        if len(tail_digits) >= 2:
            # Nếu có tiền tố đầu số di động VN (03, 05, 07, 08, 09), lấy cụm SĐT gần nhất
            phone_starts = list(PHONE_START_RE.finditer(tail_digits))
            if phone_starts:
                last_start = phone_starts[-1].start()
                current_phone = tail_digits[last_start:]
                if len(current_phone) >= 10:
                    return False
                return True
            if len(tail_digits) >= 10:
                return False
            return True

    if TRAILING_DIGIT_RE.search(norm):
        return True
    return _ends_with_any(norm, NUMBER_AND_ADDRESS_SUFFIXES)


def is_hesitating(text: str) -> bool:
    """Phát hiện người dùng đang ngập ngừng hoặc bỏ lửng câu."""
    if is_only_filler(text):
        return True
    norm = _normalize_text(text)
    if not norm:
        return True
    if text.rstrip().endswith("..."):
        return True
    words = norm.split()
    if len(words) <= 1 and words[0] in FILLER_WORDS:
        return True
    # Câu quá ngắn (1-3 từ) chưa hoàn chỉnh nội dung (trừ câu trả lời chốt ngắn) -> coi là đang ngập ngừng/nghĩ tiếp
    if len(words) <= 3 and norm not in ("có", "không", "vâng", "ok", "được", "đúng", "đúng rồi", "rồi", "xong"):
        return True
    return _ends_with_any(norm, DANGLING_SUFFIXES)


# ==============================================================================
# 4. Mô hình Smart Turn v3.2 ONNX (Dự đoán dứt câu qua ngữ điệu)
# ==============================================================================
class SmartTurnDetector:
    """Wrapper quản lý phiên suy luận ONNX Smart Turn v3.2."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.session: ort.InferenceSession | None = None
        self.feature_extractor: WhisperFeatureExtractor | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            logger.warning("Không tìm thấy model Smart Turn tại %s", self.model_path)
            return
        try:
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.model_path), sess_options=so)
            self.feature_extractor = WhisperFeatureExtractor(chunk_length=MAX_SECONDS)
            logger.info("Đã tải thành công Smart Turn model: %s", self.model_path.name)
        except Exception as exc:
            logger.error("Lỗi khi nạp Smart Turn model: %s", exc)
            self.session = None

    @property
    def is_available(self) -> bool:
        return self.session is not None and self.feature_extractor is not None

    def predict(self, audio_f32: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
        if self.session is None or self.feature_extractor is None:
            return {"prediction": 0, "probability": 0.0, "available": False}

        # Pad trước, rồi mới chuẩn hóa (khớp code tham chiếu pipecat-ai/smart-turn)
        if len(audio_f32) > MAX_SAMPLES:
            audio = audio_f32[-MAX_SAMPLES:].copy()
        elif len(audio_f32) < MAX_SAMPLES:
            audio = np.pad(audio_f32, (MAX_SAMPLES - len(audio_f32), 0), mode="constant")
        else:
            audio = audio_f32.copy()

        if len(audio) > 0:
            mean = np.mean(audio)
            std = np.std(audio)
            audio = (audio - mean) / np.sqrt(std**2 + 1e-7)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="np",
            padding=False,
            truncation=False,
            do_normalize=False,
        )

        input_features = inputs.input_features.astype(np.float32)
        outputs = self.session.run(None, {"input_features": input_features})
        prob = float(outputs[0][0].item())
        pred = 1 if prob >= threshold else 0

        return {
            "prediction": pred,
            "probability": prob,
            "available": True,
        }


_detector: SmartTurnDetector | None = None


def get_detector() -> SmartTurnDetector:
    global _detector
    if _detector is None:
        _detector = SmartTurnDetector()
    return _detector


# ==============================================================================
# 5. Mô hình Silero VAD v5 ONNX (Voice Activity Detection thời gian thực)
# ==============================================================================
class SileroVadModel:
    """Quản lý phiên suy luận ONNX Silero VAD (nhận diện giọng người / tiếng ồn)."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path or VAD_MODEL_PATH)
        self.session: ort.InferenceSession | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            logger.warning("Không tìm thấy model Silero VAD tại %s", self.model_path)
            return
        try:
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.model_path), sess_options=so)
            logger.info("Đã tải thành công Silero VAD model: %s", self.model_path.name)
        except Exception as exc:
            logger.error("Lỗi khi nạp Silero VAD model: %s", exc)
            self.session = None

    @property
    def is_available(self) -> bool:
        return self.session is not None

    def predict_chunk(
        self, chunk: np.ndarray, state: np.ndarray, sr: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Chạy suy luận 1 khối 576 mẫu âm thanh float32 mono 16kHz (kèm 64 mẫu context)."""
        if self.session is None:
            return 0.0, state
        outputs = self.session.run(None, {"input": chunk, "state": state, "sr": sr})
        prob = float(outputs[0][0][0])
        new_state = outputs[1]
        return prob, new_state


class VadStreamState:
    """Quản lý trạng thái streaming của Silero VAD cho từng phiên thoại riêng biệt."""

    def __init__(self, model: SileroVadModel | None = None) -> None:
        self.model = model or get_vad_model()
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.sr = np.array(16000, dtype=np.int64)
        self._buf = np.zeros(0, dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def reset(self) -> None:
        """Reset trạng thái bộ đệm và context khi bắt đầu lượt thoại mới."""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def speech_prob(self, samples_f32: np.ndarray) -> float:
        """Đưa audio float32 (16kHz) vào, tính xác suất tiếng người (0.0 - 1.0).
        Trả về xác suất lớn nhất trong các cửa sổ 512 mẫu (32ms).
        """
        if not self.model.is_available or len(samples_f32) == 0:
            return 0.0

        if len(self._buf) > 0:
            audio = np.concatenate([self._buf, samples_f32])
        else:
            audio = samples_f32

        probs: list[float] = []
        offset = 0
        window_size = VAD_WINDOW_SAMPLES
        while len(audio) - offset >= window_size:
            chunk = audio[offset : offset + window_size].reshape(1, window_size).astype(np.float32)
            # Ghép 64 mẫu context của frame trước với 512 mẫu mới (tổng 576 mẫu cho Silero VAD v5)
            x = np.concatenate([self._context, chunk], axis=1)
            prob, self.state = self.model.predict_chunk(x, self.state, self.sr)
            self._context = chunk[:, -64:]
            probs.append(prob)
            offset += window_size

        self._buf = audio[offset:]
        return max(probs) if probs else 0.0


_vad_model: SileroVadModel | None = None


def get_vad_model() -> SileroVadModel:
    global _vad_model
    if _vad_model is None:
        _vad_model = SileroVadModel()
    return _vad_model
