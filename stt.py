from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import websockets
from transformers import WhisperFeatureExtractor
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger("stt")

# ---------------------------------------------------------------------------
# 1. Cấu hình định dạng âm thanh & Đường dẫn mô hình
# ---------------------------------------------------------------------------
STT_SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512
MAX_SECONDS = 8
MAX_SAMPLES = MAX_SECONDS * STT_SAMPLE_RATE
MAX_AUDIO_BYTES = MAX_SAMPLES * 2  # 256,000 bytes (PCM16 mono 16 kHz)

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "smart-turn-v3.2-cpu.onnx"
VAD_MODEL_PATH = MODEL_DIR / "silero_vad.onnx"


def _ort_session_options() -> ort.SessionOptions:
    """Tạo cấu hình thực thi tối ưu dùng chung cho các mô hình ONNX Runtime đơn luồng."""
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


# ---------------------------------------------------------------------------
# 2. Kết nối dịch vụ nhận dạng giọng nói
# ---------------------------------------------------------------------------
class SttError(RuntimeError):
    """Ngoại lệ STT tự động chuyển mã lỗi (OVERLOADED, QUEUE_TIMEOUT...) thành câu tiếng Việt thân thiện."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        code = payload.get("error_code") or payload.get("code") or ""
        if code in ("OVERLOADED", "QUEUE_TIMEOUT"):
            self.message = "Dịch vụ nhận dạng giọng nói đang quá tải, vui lòng thử lại sau ít giây."
        elif code == "AUDIO_TOO_LONG":
            self.message = "Câu nói quá dài (tối đa 600 giây)."
        elif code == "BAD_SAMPLE_RATE":
            self.message = "Lỗi định dạng âm thanh (sai sample rate)."
        else:
            detail = (
                payload.get("message")
                or payload.get("error")
                or code
                or payload.get("status_code", "?")
            )
            self.message = f"Lỗi nhận dạng giọng nói ({detail})."
        super().__init__(self.message)


class SttSession:
    """Quản lý một phiên STT streaming qua WebSocket cho trình duyệt."""

    def __init__(self, url: str, on_event: Callable[[str, Any], Awaitable[None]]) -> None:
        self.url = url
        self.on_event = on_event
        self.ws: ClientConnection | None = None
        self.final_text = ""
        self.error: dict[str, Any] | None = None
        self._done = asyncio.Event()
        self._reader: asyncio.Task | None = None

    async def open(self, max_retries: int = 2) -> None:
        """Mở kết nối WebSocket và gửi gói tin khởi tạo `start`, tự động retry khi gặp lỗi 429/overload."""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                await self._handshake()
                self._reader = asyncio.create_task(self._read())
                return
            except SttError as exc:
                await self._close_ws()
                code = exc.payload.get("error_code", "")
                if code in ("OVERLOADED", "QUEUE_TIMEOUT") and attempt < max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                await self._close_ws()
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                else:
                    raise last_exc from exc
            except BaseException:
                await self._close_ws()
                raise

    async def _handshake(self) -> None:
        """Khởi tạo kết nối WebSocket và gửi gói tin bắt đầu phiên nhận dạng."""
        self.ws = await websockets.connect(self.url, max_size=10 * 2**20)
        start = {
            "type": "start",
            "request_id": str(uuid.uuid4()),
            "language": "Vietnamese",
            "sample_rate": STT_SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "channels": 1,
        }
        await self.ws.send(json.dumps(start))
        ready = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=15.0))
        if ready.get("type") == "error":
            raise SttError(ready)

    async def send_frame(self, pcm: bytes) -> None:
        """Gửi một khung dữ liệu âm thanh PCM16LE mono 16 kHz."""
        if self.ws is not None:
            await self.ws.send(pcm)

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

    async def finish(self, timeout: float = 10.0) -> str:
        """Gửi tín hiệu kết thúc thu âm và chờ nhận kết quả final cuối cùng."""
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({"type": "finish"}))
                await asyncio.wait_for(self._done.wait(), timeout)
            except TimeoutError:
                logger.warning("STT finish timeout sau %.1fs", timeout)
            finally:
                await self._close_ws()
        if self.error and not self.final_text:
            raise SttError(self.error)
        return self.final_text

    async def abort(self) -> None:
        """Hủy và giải phóng phiên nhận dạng."""
        current = asyncio.current_task()
        if self._reader and not self._reader.done() and self._reader is not current:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        await self._close_ws()
        self._done.set()

    async def _close_ws(self) -> None:
        """Đóng kết nối WebSocket an toàn và nuốt ngoại lệ dọn dẹp."""
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self.ws.close()
            self.ws = None


# ---------------------------------------------------------------------------
# 3. NLP & Chuẩn hóa văn bản tiếng Việt
# ---------------------------------------------------------------------------
FILLER_WORDS = {"ừ", "ừm", "ờ", "ờm", "à", "ư", "hử", "hả", "ơ", "ớ", "ể", "à mà", "ừ thì", "ờ thì"}
_MULTI_FILLERS = "|".join(
    re.escape(w) for w in sorted((w for w in FILLER_WORDS if " " in w), key=len, reverse=True)
)


# fmt: off
WORD_TO_DIGIT = {
    "không": "0", "một": "1", "mốt": "1", "hai": "2", "ba": "3",
    "bốn": "4", "tư": "4", "năm": "5", "lăm": "5", "sáu": "6",
    "bảy": "7", "tám": "8", "chín": "9",
}

TEN_WORDS = {
    "hai": "2", "ba": "3", "bốn": "4", "tư": "4", "năm": "5",
    "sáu": "6", "bảy": "7", "tám": "8", "chín": "9",
}

# Câu ngắn hợp lệ (chốt lượt nhanh)
SHORT_ANSWERS = {
    "có", "không", "vâng", "ok", "được", "đúng", "rồi", "xong",
    "chuẩn", "đồng ý", "alo", "xin chào", "cảm ơn", "tạm biệt",
}

# Hậu tố địa chỉ và số lượng đặc thù bưu chính VNPost
ADDRESS_AND_NUM_SUFFIXES = (
    "số", "ngõ", "ngách", "hẻm", "đường", "phố", "phường", "xã",
    "quận", "huyện", "tỉnh", "thành phố", "mười", "trăm", "nghìn", "triệu", "mã", "vận đơn", "bưu cục",
)

# Từ nối ngập ngừng tối giản (giao phần lớn phán đoán ngữ điệu cho Smart Turn)
HESITATION_SUFFIXES = ("và", "hoặc", "nhưng", "mà", "thì", "vì", "là", "về")

# Bí danh tương thích ngược
DANGLING_SUFFIXES = HESITATION_SUFFIXES
VALID_SHORT_UTTERANCES = SHORT_ANSWERS
NUMBER_AND_ADDRESS_SUFFIXES = ADDRESS_AND_NUM_SUFFIXES
# fmt: on

_TENS = "|".join(TEN_WORDS)
_UNITS = "|".join(WORD_TO_DIGIT)
_DIGITS = "|".join(WORD_TO_DIGIT)

# Cụm cấu hình biểu thức chính quy (Regex) tập trung
PATTERNS = {
    "clean": re.compile(r"[^\w\s]"),
    "spaces": re.compile(r"\s+"),
    "trailing_gap": re.compile(r"(?:\.\.\.|…|[,，])$"),
    "multi_filler": re.compile(rf"\b({_MULTI_FILLERS})\b"),
    "digit_tail": re.compile(r"(\d[\d\s]*)$"),
    "phone_prefix": re.compile(r"0[35789]"),
    "postal": re.compile(r"\b(bưu chính|bưu điện|postal|zip|mã vùng)\b"),
    "tracking": re.compile(r"(\d{3,}|\b(eb|vn|ems|cp|lp)\s*\d+)$", re.IGNORECASE),
    "compound_tens": re.compile(rf"\b({_TENS})\s+mươi\s+({_UNITS})\b", re.IGNORECASE),
    "even_tens": re.compile(rf"\b({_TENS})\s+mươi\b", re.IGNORECASE),
    "teens": re.compile(rf"\bmười\s+({_UNITS})\b", re.IGNORECASE),
    "linh": re.compile(rf"\b(?:linh|lẻ)\s+({_UNITS})\b", re.IGNORECASE),
    "tram": re.compile(
        rf"\b(\d{{2}})\s*(?:nghìn|ngàn)\s*({_DIGITS}|[1-9])\s*trăm\b", re.IGNORECASE
    ),
    "nghin": re.compile(r"\b(\d{2})\s*(?:nghìn|ngàn)\b", re.IGNORECASE),
    "muoi": re.compile(r"\bmười\b", re.IGNORECASE),
}
TextRules = PATTERNS


def normalize_compound_numbers(text: str) -> str:
    """Chuẩn hóa cách đọc số ghép tiếng Việt (hàng chục, mười, linh/lẻ, tròn nghìn) thành chữ số."""
    text = PATTERNS["compound_tens"].sub(
        lambda m: f"{TEN_WORDS[m.group(1).lower()]}{WORD_TO_DIGIT[m.group(2).lower()]}", text
    )
    text = PATTERNS["even_tens"].sub(lambda m: f"{TEN_WORDS[m.group(1).lower()]}0", text)
    text = PATTERNS["teens"].sub(lambda m: f"1{WORD_TO_DIGIT[m.group(1).lower()]}", text)
    text = PATTERNS["linh"].sub(lambda m: f"0{WORD_TO_DIGIT[m.group(1).lower()]}", text)
    text = PATTERNS["muoi"].sub("10", text)
    text = PATTERNS["tram"].sub(
        lambda m: f"{m.group(1)}{WORD_TO_DIGIT.get(m.group(2).lower(), m.group(2))}00", text
    )
    return PATTERNS["nghin"].sub(r"\g<1>000", text)


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản thành khóa so khớp (xóa dấu câu, chữ thường, gộp khoảng trắng)."""
    cleaned = PATTERNS["clean"].sub(" ", text.lower())
    return PATTERNS["spaces"].sub(" ", cleaned).strip()


def _flush_digit_run(digit_run: list[tuple[str, str]], converted: list[str]) -> None:
    """Ghép chuỗi chữ số nếu đủ dài (>=5 số) hoặc giữ nguyên từ gốc."""
    if not digit_run:
        return
    digits_str = "".join(d[1] for d in digit_run)
    if len(digits_str) >= 5:
        converted.append(digits_str)
    else:
        converted.extend(d[0] for d in digit_run)


# 1. Chuẩn hóa số đọc bằng chữ thành chữ số
def normalize_speech_input(text: str) -> str:
    """Tự động gom cụm từ đọc số tiếng Việt liên tiếp (>=5 số, hỗ trợ SĐT và mã bưu chính 5 số) thành chuỗi chữ số."""
    text = normalize_compound_numbers(text)
    tokens = text.split()
    converted: list[str] = []
    digit_run: list[tuple[str, str]] = []

    for tok in tokens:
        clean_tok = PATTERNS["clean"].sub("", tok).lower()
        if clean_tok in WORD_TO_DIGIT:
            digit_run.append((tok, WORD_TO_DIGIT[clean_tok]))
        elif re.fullmatch(r"\d+", clean_tok):
            digit_run.append((tok, clean_tok))
        else:
            _flush_digit_run(digit_run, converted)
            digit_run = []
            converted.append(tok)

    _flush_digit_run(digit_run, converted)
    return " ".join(converted)


def _ends_with_any(norm: str, phrases: tuple[str, ...]) -> bool:
    """Kiểm tra xem câu kết thúc bằng một cụm từ (so khớp theo ranh giới từ, không theo ký tự)."""
    padded = f" {norm}"
    return any(padded.endswith(f" {p}") for p in phrases)


def is_only_filler(text: str) -> bool:
    """Kiểm tra xem văn bản nhận diện được mới chỉ chứa từ đệm hoặc còn trống."""
    norm = normalize_text(text)
    if not norm:
        return True
    temp = PATTERNS["multi_filler"].sub("ừ", norm)
    words = temp.split()
    return all(word in FILLER_WORDS for word in words)


# 2. Nhận diện người dùng đang đọc số điện thoại, mã vận đơn hoặc địa chỉ
def is_reading_sequence(text: str) -> bool:
    """Phát hiện người dùng đang đọc dở dãy số (SĐT, mã bưu chính, mã vận đơn) hoặc địa chỉ."""
    norm = normalize_text(text)
    if not norm:
        return False

    norm_compound = normalize_compound_numbers(norm)
    tokens = [WORD_TO_DIGIT.get(w, w) for w in norm_compound.split()]
    norm_digits = " ".join(tokens)

    tail_match = PATTERNS["digit_tail"].search(norm_digits)
    if tail_match:
        tail_digits = PATTERNS["spaces"].sub("", tail_match.group(1))
        # Chỉ coi là dãy số/SĐT khi có >= 2 chữ số (loại trừ từ 'không' đơn lẻ cuối câu hỏi)
        if len(tail_digits) >= 2:
            # Nếu là mã bưu chính (5 chữ số) kèm từ khóa bưu chính/bưu điện/zip
            if PATTERNS["postal"].search(norm):
                return len(tail_digits) < 5

            # Nếu có tiền tố đầu số di động VN (03, 05, 07, 08, 09), lấy cụm SĐT gần nhất
            phone_starts = list(PATTERNS["phone_prefix"].finditer(tail_digits))
            if phone_starts:
                last_start = phone_starts[-1].start()
                return len(tail_digits[last_start:]) < 10
            return len(tail_digits) < 10

    if PATTERNS["tracking"].search(norm):
        return True
    return _ends_with_any(norm, ADDRESS_AND_NUM_SUFFIXES)


# 3. Chỉ nhận diện từ đệm và dấu hiệu bỏ lửng rõ ràng; nhường đánh giá ngữ nghĩa cho Smart Turn
def is_hesitating(text: str) -> bool:
    """Phát hiện từ đệm hoặc dấu hiệu bỏ lửng rõ ràng; không cố đoán ngữ nghĩa bằng danh sách dài."""
    if is_only_filler(text):
        return True
    if PATTERNS["trailing_gap"].search(text.strip()):
        return True
    norm = normalize_text(text)
    if norm in SHORT_ANSWERS:
        return False
    return _ends_with_any(norm, HESITATION_SUFFIXES)


# ---------------------------------------------------------------------------
# 4. Mô hình suy luận AI cục bộ (Silero VAD & Smart Turn v3.2)
# ---------------------------------------------------------------------------
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
            so = _ort_session_options()
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
        outputs = cast(
            list[np.ndarray], self.session.run(None, {"input": chunk, "state": state, "sr": sr})
        )
        prob = float(outputs[0][0][0])
        new_state = outputs[1]
        return prob, new_state


@lru_cache(maxsize=1)
def get_vad_model() -> SileroVadModel:
    """Trả về phiên bản Singleton của mô hình SileroVadModel."""
    return SileroVadModel()


class VadStreamState:
    """Quản lý trạng thái streaming của Silero VAD cho từng phiên thoại riêng biệt."""

    def __init__(self, model: SileroVadModel | None = None) -> None:
        self.model = model or get_vad_model()
        self.sr = np.array(STT_SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Reset trạng thái bộ đệm và context khi bắt đầu lượt thoại mới."""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def speech_prob(self, samples_f32: np.ndarray, agg: str = "max") -> float:
        """Đưa audio float32 (16kHz) vào, tính xác suất tiếng người (0.0 - 1.0).
        Hỗ trợ các phương pháp tổng hợp xác suất đa cửa sổ ('max', 'mean', 'p75').
        """
        if not self.model.is_available or len(samples_f32) == 0:
            return 0.0

        audio = np.concatenate([self._buf, samples_f32]) if len(self._buf) > 0 else samples_f32

        probs: list[float] = []
        offset = 0
        window_size = VAD_WINDOW_SAMPLES
        while len(audio) - offset >= window_size:
            chunk = audio[offset : offset + window_size].reshape(1, window_size).astype(np.float32)
            x = np.concatenate([self._context, chunk], axis=1)
            prob, self.state = self.model.predict_chunk(x, self.state, self.sr)
            self._context = chunk[:, -64:]
            probs.append(prob)
            offset += window_size

        self._buf = audio[offset:]
        if not probs:
            return 0.0
        if agg == "mean":
            return float(np.mean(probs))
        if agg == "p75":
            return float(np.percentile(probs, 75))
        return max(probs)


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
            so = _ort_session_options()
            self.session = ort.InferenceSession(str(self.model_path), sess_options=so)
            self.feature_extractor = WhisperFeatureExtractor(chunk_length=MAX_SECONDS)
            logger.info("Đã tải thành công Smart Turn model: %s", self.model_path.name)
        except Exception as exc:
            logger.error("Lỗi khi nạp Smart Turn model: %s", exc)
            self.session = None

    @property
    def is_available(self) -> bool:
        return self.session is not None and self.feature_extractor is not None

    def predict(self, audio_f32: np.ndarray) -> dict[str, Any]:
        """Chạy mô hình Smart Turn v3.2 để dự đoán xác suất dứt câu."""
        if self.session is None or self.feature_extractor is None:
            return {"probability": 0.0}

        if len(audio_f32) > MAX_SAMPLES:
            audio = audio_f32[-MAX_SAMPLES:]
        elif len(audio_f32) < MAX_SAMPLES:
            audio = np.pad(audio_f32, (MAX_SAMPLES - len(audio_f32), 0), mode="constant")
        else:
            audio = audio_f32

        mean = np.mean(audio)
        std = np.std(audio)
        audio = (audio - mean) / np.sqrt(std**2 + 1e-7)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=STT_SAMPLE_RATE,
            return_tensors="np",
            padding="do_not_pad",
            truncation=False,
            do_normalize=False,
        )

        input_features = inputs.input_features.astype(np.float32)
        outputs = cast(list[np.ndarray], self.session.run(None, {"input_features": input_features}))
        prob = float(outputs[0][0].item())

        return {"probability": prob}


@lru_cache(maxsize=1)
def get_detector() -> SmartTurnDetector:
    """Trả về phiên bản Singleton của bộ phân loại SmartTurnDetector."""
    return SmartTurnDetector()
