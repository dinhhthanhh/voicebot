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

    def __init__(self, payload: dict[str, Any] | str) -> None:
        if isinstance(payload, str):
            payload = {"message": payload}
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
                try:
                    msg = json.loads(raw)
                except Exception as json_err:
                    logger.warning("STT server trả về dữ liệu không phải JSON: %s", json_err)
                    self.error = {"type": "error", "message": f"Dữ liệu STT không hợp lệ: {json_err}"}
                    with contextlib.suppress(Exception):
                        await self.on_event("error", self.error)
                    break

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
        except Exception as exc:
            logger.exception("Lỗi kết nối trong SttSession._read: %s", exc)
            self.error = {"type": "error", "message": f"Lỗi nhận dạng giọng nói: {exc}"}
            with contextlib.suppress(Exception):
                await self.on_event("error", self.error)
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
                if not self.error:
                    self.error = {"type": "error", "message": f"STT finish timeout sau {timeout:.1f}s"}
            except Exception as exc:
                logger.warning("Lỗi gửi finish tới STT: %s", exc)
                if not self.error:
                    self.error = {"type": "error", "message": str(exc)}
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

EXTRA_DIGITS = {"mười": "10", "linh": "0", "lẻ": "0"}

TEN_WORDS = {k: v for k, v in WORD_TO_DIGIT.items() if k not in ("không", "một", "mốt", "lăm")}

# Câu ngắn hợp lệ (chốt lượt nhanh)
SHORT_ANSWERS = {
    "có", "không", "vâng", "ok", "được", "đúng", "rồi", "xong",
    "chuẩn", "đồng ý", "alo", "xin chào", "cảm ơn", "tạm biệt",
}

# Tiền tố danh từ/đơn vị đứng trước một chữ số (ví dụ: quận 1, bưu cục 3, số 5, ngõ 2, mã 9)
NUMBER_PREFIXES = (
    "số", "ngõ", "ngách", "hẻm", "đường", "phố", "phường", "xã",
    "quận", "huyện", "tỉnh", "thành phố", "mã", "vận đơn", "bưu cục",
    "căn hộ", "tầng", "phòng",
)

# Cụm từ địa chỉ bỏ lửng rõ ràng (chưa đọc tên/số địa chỉ)
DANGLING_ADDRESS_SUFFIXES = ("ở số", "tại số", "nhà số", "ngõ", "ngách", "hẻm")
ADDRESS_AND_NUM_SUFFIXES = NUMBER_PREFIXES

# Từ nối ngập ngừng tối giản (giao phần lớn phán đoán ngữ điệu cho Smart Turn)
HESITATION_SUFFIXES = ("và", "hoặc", "nhưng", "mà", "thì", "vì", "là", "về")
# fmt: on

_TENS = "|".join(TEN_WORDS)
_DIGITS = "|".join(WORD_TO_DIGIT)
_COMPOUND_UNITS = "|".join(w for w in WORD_TO_DIGIT if w not in ("không", "năm"))
_TEEN_UNITS = "|".join(w for w in WORD_TO_DIGIT if w not in ("không", "mốt", "năm"))
_NUM_HEAD = rf"(?:\d{{1,2}}|mười|{_DIGITS})"

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
    "compound_tens": re.compile(rf"\b({_TENS})\s+mươi\s+({_COMPOUND_UNITS})\b", re.IGNORECASE),
    "even_tens": re.compile(rf"\b({_TENS})\s+mươi\b", re.IGNORECASE),
    "teens": re.compile(rf"\bmười\s+({_TEEN_UNITS})\b", re.IGNORECASE),
    "nghin_ktram_linh": re.compile(
        rf"\b({_NUM_HEAD})\s*(?:nghìn|ngàn)\s*không\s*trăm\s*(?:linh|lẻ)\s*({_DIGITS})\b",
        re.IGNORECASE,
    ),
    "nghin_tram_tens": re.compile(
        rf"\b({_NUM_HEAD})\s*(?:nghìn|ngàn)\s*(\d|{_DIGITS})\s*trăm\s*(\d{{2}})\b",
        re.IGNORECASE,
    ),
    "nghin_tram": re.compile(
        rf"\b({_NUM_HEAD})\s*(?:nghìn|ngàn)\s*(\d|{_DIGITS})\s*trăm\b", re.IGNORECASE
    ),
    "nghin_linh": re.compile(
        rf"\b({_NUM_HEAD})\s*(?:nghìn|ngàn)\s*(?:linh|lẻ)\s*({_DIGITS})\b", re.IGNORECASE
    ),
    "nghin_tens": re.compile(
        rf"\b({_NUM_HEAD})\s*(?:nghìn|ngàn)\s*(\d{{2}})\b", re.IGNORECASE
    ),
    "nghin": re.compile(rf"\b({_NUM_HEAD})\s*(?:nghìn|ngàn)\b", re.IGNORECASE),
    "tram_linh": re.compile(
        rf"\b(\d|{_DIGITS})\s*trăm\s*(?:linh|lẻ)\s*({_DIGITS})\b", re.IGNORECASE
    ),
    "tram_tens": re.compile(rf"\b(\d|{_DIGITS})\s*trăm\s*(\d{{2}})\b", re.IGNORECASE),
    "tram_even": re.compile(rf"\b(\d|{_DIGITS})\s*trăm\b", re.IGNORECASE),
    "seq_lead_in": re.compile(
        r"(?:\b(?:số|số\s+điện\s+thoại|điện\s+thoại|sđt|đầu\s+số|mã\s+vận\s+đơn|mã\s+bưu\s+chính|mã\s+bưu\s+điện|mã\s+zip|mã\s+vùng|mã\s+đơn|mã\s+số|mã|vận\s+đơn|tài\s+khoản|stk|căn\s+cước|cccd|cmnd)(?:\s+(?:của\s+)?(?:tôi|em|anh|chị|mình|bạn|khách|cháu))?\s+(?:là|là\s+số|số)?|\b(?:số\s+điện\s+thoại|sđt|đầu\s+số|mã\s+vận\s+đơn|mã\s+bưu\s+chính|mã\s+bưu\s+điện|mã\s+zip|mã\s+đơn|vận\s+đơn))$",
        re.IGNORECASE,
    ),
    "question_tag": re.compile(
        r"\b(?:có|phải|được|đúng|biết|thấy|nghe|rảnh)\b.*\bkhông$",
        re.IGNORECASE,
    ),
}


def _to_digit(word_or_digit: str) -> str:
    """Chuyển đổi một từ số hoặc chữ số sang chuỗi chữ số."""
    w = word_or_digit.lower()
    if w == "mười":
        return "10"
    return WORD_TO_DIGIT.get(w, w)


def normalize_compound_numbers(text: str) -> str:
    """Chuẩn hóa cách đọc số ghép tiếng Việt (hàng chục, mười, linh/lẻ, hàng trăm, tròn nghìn) thành chữ số."""
    text = PATTERNS["compound_tens"].sub(
        lambda m: f"{TEN_WORDS[m.group(1).lower()]}{WORD_TO_DIGIT[m.group(2).lower()]}", text
    )
    text = PATTERNS["even_tens"].sub(lambda m: f"{TEN_WORDS[m.group(1).lower()]}0", text)
    text = PATTERNS["teens"].sub(lambda m: f"1{WORD_TO_DIGIT[m.group(1).lower()]}", text)
    text = PATTERNS["nghin_ktram_linh"].sub(
        lambda m: f"{_to_digit(m.group(1))}00{_to_digit(m.group(2))}", text
    )
    text = PATTERNS["nghin_tram_tens"].sub(
        lambda m: f"{_to_digit(m.group(1))}{_to_digit(m.group(2))}{m.group(3)}", text
    )
    text = PATTERNS["nghin_tram"].sub(
        lambda m: f"{_to_digit(m.group(1))}{_to_digit(m.group(2))}00", text
    )
    text = PATTERNS["nghin_linh"].sub(
        lambda m: f"{_to_digit(m.group(1))}00{_to_digit(m.group(2))}", text
    )
    text = PATTERNS["nghin_tens"].sub(
        lambda m: f"{_to_digit(m.group(1))}0{m.group(2)}", text
    )
    text = PATTERNS["nghin"].sub(lambda m: f"{_to_digit(m.group(1))}000", text)
    text = PATTERNS["tram_linh"].sub(
        lambda m: f"{_to_digit(m.group(1))}0{_to_digit(m.group(2))}", text
    )
    text = PATTERNS["tram_tens"].sub(lambda m: f"{_to_digit(m.group(1))}{m.group(2)}", text)
    return PATTERNS["tram_even"].sub(lambda m: f"{_to_digit(m.group(1))}00", text)


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


def _has_following_digit(tokens: list[str], start_idx: int) -> bool:
    """Kiểm tra xem phía sau token hiện tại có chứa từ đọc số hoặc chữ số nào không."""
    for j in range(start_idx, len(tokens)):
        c = PATTERNS["clean"].sub("", tokens[j]).lower()
        if c in ("linh", "lẻ"):
            continue
        return c in WORD_TO_DIGIT or c.isdigit() or c == "mười"
    return False


# 1. Chuẩn hóa số đọc bằng chữ thành chữ số
def normalize_speech_input(text: str) -> str:
    """Tự động gom cụm từ đọc số tiếng Việt liên tiếp (>=5 số, hỗ trợ SĐT và mã bưu chính 5 số) thành chuỗi chữ số."""
    text = normalize_compound_numbers(text)
    tokens = text.split()
    converted: list[str] = []
    digit_run: list[tuple[str, str]] = []

    for i, tok in enumerate(tokens):
        clean_tok = PATTERNS["clean"].sub("", tok).lower()
        if clean_tok in WORD_TO_DIGIT:
            digit_run.append((tok, WORD_TO_DIGIT[clean_tok]))
        elif clean_tok == "mười":
            digit_run.append((tok, "10"))
        elif clean_tok in ("linh", "lẻ"):
            if digit_run and _has_following_digit(tokens, i + 1):
                digit_run.append((tok, "0"))
            else:
                _flush_digit_run(digit_run, converted)
                digit_run = []
                converted.append(tok)
        elif clean_tok.isdigit():
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


def _is_incomplete_digit_tail(norm: str, norm_digits: str, tail_match: re.Match[str]) -> bool:
    """Kiểm tra phần đuôi chữ số có đang đọc dở (SĐT, mã bưu chính, đầu số địa chỉ)."""
    if PATTERNS["question_tag"].search(norm):
        return False

    tail_digits = PATTERNS["spaces"].sub("", tail_match.group(1))
    if len(tail_digits) >= 2:
        if PATTERNS["postal"].search(norm):
            return len(tail_digits) < 5
        if len(tail_digits) == 10:
            return False
        phone_starts = list(PATTERNS["phone_prefix"].finditer(tail_digits))
        if phone_starts:
            last_start = phone_starts[-1].start()
            return len(tail_digits[last_start:]) < 10
        return len(tail_digits) < 10

    if len(tail_digits) == 1:
        prefix = norm_digits[: tail_match.start(1)].strip()
        return _ends_with_any(prefix, NUMBER_PREFIXES) or bool(PATTERNS["seq_lead_in"].search(prefix))

    return False


# 2. Nhận diện người dùng đang đọc số điện thoại, mã vận đơn hoặc địa chỉ
def is_reading_sequence(text: str) -> bool:
    """Phát hiện người dùng đang đọc dở dãy số (SĐT, mã bưu chính, mã vận đơn) hoặc địa chỉ."""
    norm = normalize_text(text)
    if not norm:
        return False

    norm_compound = normalize_compound_numbers(norm)
    tokens = [EXTRA_DIGITS.get(w, WORD_TO_DIGIT.get(w, w)) for w in norm_compound.split()]
    norm_digits = " ".join(tokens)

    tail_match = PATTERNS["digit_tail"].search(norm_digits)
    if tail_match and _is_incomplete_digit_tail(norm, norm_digits, tail_match):
        return True

    if PATTERNS["tracking"].search(norm):
        return True
    return _ends_with_any(norm, DANGLING_ADDRESS_SUFFIXES)


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

        audio = audio_f32[-MAX_SAMPLES:] if len(audio_f32) > MAX_SAMPLES else audio_f32

        if len(audio) > 0:
            mean = np.mean(audio)
            std = np.std(audio)
            audio = (audio - mean) / np.sqrt(std**2 + 1e-7)

        if len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (MAX_SAMPLES - len(audio), 0), mode="constant")

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
