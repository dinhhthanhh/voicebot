from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import struct
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import StrEnum
from typing import Any

import httpx
from num2words import num2words

logger = logging.getLogger("tts")

# ---------------------------------------------------------------------------
# Cấu hình âm thanh chuẩn của dịch vụ Zero-TTS (48 kHz mono 16-bit PCM)
# ---------------------------------------------------------------------------
TTS_SAMPLE_RATE = 48000
TTS_CHANNELS = 1
TTS_SAMPLE_BITS = 16
WAV_HEADER_MIN_BYTES = 12
WAV_DATA_HEADER_MIN_BYTES = 44
HTTP_STATUS_OK = 200
ERR_MSG_MAX_LEN = 200

SYSTEM_PROMPT = """
    Bạn là trợ lý giọng nói tiếng Việt của Tổng công ty Bưu điện Việt Nam (VNPost).
    Câu trả lời của bạn vừa hiển thị trên màn hình vừa được đọc thành tiếng qua hệ thống TTS.

    1. PHONG CÁCH TRẢ LỜI
    - Xưng "tôi" (hoặc "em" nếu khách xưng anh/chị), gọi khách hàng là "bạn" (hoặc "anh/chị" tương ứng), thái độ lịch sự, tự nhiên, tận tâm.
    - Mặc định trả lời tối đa 3 câu ngắn gọn, trừ khi khách hàng yêu cầu liệt kê chi tiết.
    - Câu đầu tiên bắt buộc ngắn dưới 12 từ để hệ thống phát giọng nói tức thì.
    - Khi khách chào hỏi ("alo", "chào bạn", "bạn là ai"): Phản hồi ngắn gọn, thân thiện và hỏi nhu cầu hỗ trợ (ví dụ: "Dạ, tôi là trợ lý ảo VNPost. Tôi có thể hỗ trợ gì cho bạn?").
    - Không sử dụng emoji, không dùng ký hiệu LaTeX.
    - Tuyệt đối không viết tắt các từ thông dụng hoặc địa danh (phải viết rõ "Thành phố Hồ Chí Minh" thay vì "TP. HCM", "Số điện thoại" thay vì "SĐT", "Bưu điện" thay vì "BĐ") để tránh ngắt câu sai khi đọc.

    2. QUY TẮC NGHIỆP VỤ & GIỚI HẠN DỮ LIỆU (QUAN TRỌNG)
    - Bạn có thể giải đáp các kiến thức chung về dịch vụ bưu chính (quy định hàng cấm gửi, cách đóng gói cơ bản, phân biệt chuyển phát nhanh EMS và bưu phẩm thường).
    - Khi khách cung cấp đầy đủ Mã vận đơn hoặc Số điện thoại để tra cứu đơn hàng: Hãy đọc lại chính xác mã/số điện thoại đó để xác nhận đã ghi nhận đúng định dạng, sau đó thông báo lịch sự rằng tính năng kết nối hệ thống tra cứu vận đơn trực tiếp đang được bảo trì/nâng cấp và hướng dẫn khách tra cứu trên trang vnpost.vn hoặc gọi tổng đài 1900 54 54 81.

    3. SỐ ĐIỆN THOẠI VÀ MÃ VẬN ĐƠN
    - Số điện thoại (thường 10 chữ số), CCCD (12 chữ số), Mã bưu chính (5 chữ số) phải viết liền bằng chữ số.
    - Mã bưu gửi, mã vận đơn phải giữ nguyên dạng mã chuẩn, viết liền và viết hoa (ví dụ: VN123456789VN, EB123456789VN, EMS123456789VN, CP123456789VN).
    - Tự động chuyển các cách phát âm chữ cái tiếng Việt/Anh như "vi en", "vê nờ", "i bi", "e bê", "i em ét", "xê pê" thành tiền tố/hậu tố mã tương ứng (VN, EB, EMS, CP) khi đi kèm chuỗi số.
    - Khi khách đọc nhầm rồi sửa lại (ví dụ: "không chín một... à nhầm không chín tám..."), chỉ lấy phiên bản sửa cuối cùng, không ghép đoạn đọc nhầm vào mã/số điện thoại.

    4. ĐỊNH DẠNG NỘI DUNG
    - Tuân thủ đúng định dạng người dùng yêu cầu (gạch đầu dòng, bảng biểu, biểu đồ).
    - Khi người dùng yêu cầu vẽ biểu đồ (bar chart, biểu đồ cột, biểu đồ thanh ngang):
      + Bắt buộc vẽ biểu đồ thanh ngang ASCII trực quan trong khối ```.
      + Luôn nói trước 1-2 câu tóm tắt số liệu quan trọng bằng văn xuôi tự nhiên trước khi xuất khối biểu đồ.
      + Mẫu biểu đồ thanh ngang ASCII chuẩn:
        ```
        Tên mục: [===       ] (giá trị)
        ```
    - Khi người dùng yêu cầu liệt kê hoặc gạch đầu dòng: Bắt buộc định dạng bằng dấu gạch đầu dòng ("- ") ở đầu mỗi dòng cho từng ý theo đúng số lượng yêu cầu.
    - Khi cần trình bày bảng so sánh: Luôn nói trước 1-2 câu tóm tắt bằng văn xuôi tự nhiên trước khi kẻ bảng Markdown.
    - Số tiền, khối lượng, kích thước viết bằng số kèm đơn vị chuẩn có khoảng trắng (ví dụ: 120 triệu đồng, 25.000 đồng, 2 ki lô gam).
    """.strip()


async def _stream_with_client(
    c: httpx.AsyncClient, url: str, body: dict[str, Any]
) -> AsyncIterator[str]:
    """Kết nối HTTP streaming tới server OpenAI-compatible và đọc từng delta content."""
    async with c.stream("POST", url, json=body) as r:
        if r.status_code != HTTP_STATUS_OK:
            raw = await r.aread()
            err_msg = raw.decode("utf-8", errors="replace")[:ERR_MSG_MAX_LEN]
            raise RuntimeError(f"LLM HTTP {r.status_code}: {err_msg}")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                yield piece


async def stream_reply(
    messages: list[dict[str, str]],
    *,
    url: str,
    think: bool = False,
    timeout: float = 60.0,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """Gửi yêu cầu hội thoại tới OpenAI-compatible API và phát luồng từng delta content."""
    body: dict[str, Any] = {"messages": messages, "temperature": 0, "stream": True}
    if think:
        body["reasoning_effort"] = "low"
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    if client is not None:
        async for piece in _stream_with_client(client, url, body):
            yield piece
    else:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as new_client:
            async for piece in _stream_with_client(new_client, url, body):
                yield piece


# Biểu thức chính quy làm sạch ký hiệu trước khi đưa vào TTS
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")
URL_RE = re.compile(r"https?://[^\s<>()]+")
MARK_RE = re.compile(r"(\*\*|__|\*|`|~~)")
PREFIX_RE = re.compile(r"^(#{1,6}\s+|[-*+•]\s+|\d+[.)]\s+|>\s*)")
RULE_RE = re.compile(r"^[-*_=\s]{3,}$")
SENT_RE = re.compile(r"(?<=[.!?…])(?<!\.\.)\s+")
MATH_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$")
WORD_CHAR_RE = re.compile(r"\w")
SENT_END_PUNCT_RE = re.compile(r"[.!?…:;,]$")

REMAINDER_NOTE = "Phần còn lại đã được hiển thị trên màn hình."


class LineKind(StrEnum):
    """Phân loại kiểu dòng Markdown trong quá trình lọc câu cho TTS."""

    FENCE = "fence"
    SKIP = "skip"
    TABLE = "table"
    PROSE = "prose"
    WAIT = "wait"


def clean_inline(text: str) -> str:
    """Làm sạch ký hiệu Markdown, emoji, URL và công thức toán học trong một câu."""
    s = MATH_RE.sub("công thức trên màn hình", text)
    s = LINK_RE.sub(r"\1", s)
    s = PLACEHOLDER_RE.sub("", s)
    s = URL_RE.sub("đường dẫn trên màn hình", s)
    s = EMOJI_RE.sub("", s)
    s = MARK_RE.sub("", s)
    return " ".join(s.split())


class SpeechFilter:
    """Bộ đệm nhận từng delta từ LLM và nhả ra các câu sạch sẵn sàng đưa vào TTS."""

    def __init__(self, min_chars: int = 8, max_sentences: int = 8, max_chars: int = 600) -> None:
        self.min_chars = min_chars
        self.max_sentences = max_sentences
        self.max_chars = max_chars

        self.buf = ""
        self.kind: LineKind | None = None
        self.fresh = True
        self.in_code = False
        self.hold = ""
        self.sent_count = 0
        self.total_chars = 0
        self.capped = False

    def feed(self, delta: str) -> list[str]:
        """Tiếp nhận một phần văn bản mới từ luồng LLM."""
        self.buf += delta
        out: list[str] = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self._handle(line, complete=True, out=out)
            self.kind, self.fresh = None, True
        self.buf = self._handle(self.buf, complete=False, out=out)
        return out

    def flush(self) -> list[str]:
        """Xả toàn bộ nội dung còn lại trong bộ đệm khi luồng LLM kết thúc."""
        out: list[str] = []
        self._handle(self.buf, complete=True, out=out)
        self.buf, self.kind, self.fresh = "", None, True

        if self.hold:
            remaining = self.hold
            self.hold = ""
            self._push(remaining, out, end=True, force=True)
        return out

    def _detect_line_kind(self, s: str, complete: bool) -> LineKind | None:
        """Nhận diện loại dòng Markdown: code block fence, bảng biểu, quy tắc phân cách hoặc văn xuôi."""
        if not s:
            return None
        if not complete and s.startswith("`") and len(s) < 3:
            return LineKind.WAIT
        if s.startswith("```"):
            return LineKind.FENCE
        if self.in_code:
            return LineKind.SKIP
        if s.startswith("|"):
            return LineKind.TABLE
        if complete and RULE_RE.match(s):
            return LineKind.SKIP
        return LineKind.PROSE

    def _handle_fence(self, s: str, text: str, complete: bool, out: list[str]) -> str:
        """Xử lý khối code Markdown ``` (bật/tắt trạng thái và bỏ qua nội dung code)."""
        if not complete:
            return text
        if not self.in_code:
            self.in_code = True
        else:
            self.in_code = False
            rest = s[3:].strip()
            if rest:
                for p in SENT_RE.split(rest):
                    self._push(p, out, end=complete)
        return ""

    def _handle_prose(self, text: str, s: str, complete: bool, out: list[str]) -> str:
        """Xử lý văn bản tự nhiên thông thường: loại bỏ tiền tố danh sách và tách thành từng câu."""
        t = PREFIX_RE.sub("", s, count=1) if self.fresh else text
        parts = SENT_RE.split(t)

        if complete:
            for p in parts:
                self._push(p, out, end=True)
            return ""

        if len(parts) > 1:
            for p in parts[:-1]:
                self._push(p, out)
            self.fresh = False
            return parts[-1]
        return text

    def _handle(self, text: str, complete: bool, out: list[str]) -> str:
        """Phân loại và điều phối xử lý từng dòng văn bản theo loại nội dung."""
        s = text.lstrip()

        if self.kind is None:
            detected = self._detect_line_kind(s, complete)
            if detected is None:
                return "" if complete else text
            if detected == LineKind.WAIT:
                return text
            self.kind = detected

        if self.kind == LineKind.FENCE:
            return self._handle_fence(s, text, complete, out)

        if self.kind in (LineKind.SKIP, LineKind.TABLE):
            return ""

        return self._handle_prose(text, s, complete, out)

    def _push(self, sent: str, out: list[str], end: bool = False, force: bool = False) -> None:
        """Kiểm tra và đưa câu vào danh sách xuất khẩu."""
        if self.capped:
            return

        sent = clean_inline(sent)
        if not WORD_CHAR_RE.search(sent):
            return

        if end and not SENT_END_PUNCT_RE.search(sent):
            sent += "."

        if self.hold:
            sent = f"{self.hold} {sent}"
            self.hold = ""

        if not force and len(sent) < self.min_chars:
            self.hold = sent
            return

        too_long = (self.total_chars + len(sent)) > self.max_chars
        if self.sent_count >= self.max_sentences or (too_long and self.sent_count > 0):
            self.capped = True
            self.hold = ""
            out.append(REMAINDER_NOTE)
            return

        self.sent_count += 1
        self.total_chars += len(sent)
        out.append(sent)


DEFAULT_PARAMS = {
    "cfg_scale": 1.0,
    "audio_temperature": 0.8,
    "text_temperature": 1.0,
    "audio_topk": 25,
    "audio_topp": 0.95,
    "audio_repetition_penalty": 1.2,
    "min_frames": 4,
    "max_frames": 1500,
}

MAX_TTS_CHARS = 450

DIGITS_VI = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}
LETTERS_VI = {
    "A": "a",
    "B": "bê",
    "C": "xê",
    "D": "đê",
    "E": "e",
    "F": "ép",
    "G": "gờ",
    "H": "hát",
    "I": "i",
    "J": "giây",
    "K": "ca",
    "L": "lờ",
    "M": "mờ",
    "N": "nờ",
    "O": "o",
    "P": "pê",
    "Q": "quy",
    "R": "rờ",
    "S": "ét",
    "T": "tê",
    "U": "u",
    "V": "vê",
    "W": "vê kép",
    "X": "ích",
    "Y": "y",
    "Z": "dét",
}
UNITS_VI = {
    "kg": "ki lô gam",
    "g": "gam",
    "km": "ki lô mét",
    "m": "mét",
    "cm": "xen ti mét",
    "mm": "mi li mét",
    "đ": "đồng",
    "đồng": "đồng",
    "vnd": "đồng",
    "vnđ": "đồng",
    "k": "nghìn",
    "%": "phần trăm",
}

PHONE_PAREN_RE = re.compile(r"\(0(\d{2,3})\)[\s.-]*(\d{3,4})[\s.-]*(\d{3,4})\b")
PHONE_INTL_RE = re.compile(r"(?:\+84|(?<=\s)84)[\s.-]*(\d{2,3})[\s.-]*(\d{3,4})[\s.-]*(\d{3,4})\b")
PHONE_RE = re.compile(r"\b(0\d{2,3})[\s.-]+(\d{3,4})[\s.-]+(\d{3,4})\b")
DECIMAL_SEP_RE = re.compile(r"[.,]")
UNIT_TOKEN_RE = re.compile(r"^([0-9][0-9.,/]*)([a-zA-ZđĐ%]+|đồng)$", re.IGNORECASE)
HAS_ALPHA_RE = re.compile(r"[A-Za-z]")
HAS_DIGIT_RE = re.compile(r"\d")
GROUPED_NUMBER_RE = re.compile(r"[1-9]\d{0,2}(?:[.,]\d{3})+")
DECIMAL_NUMBER_RE = re.compile(r"\d+[.,]\d+")
TTS_TOKEN_RE = re.compile(r"\b[A-Za-z0-9.,/]+(?:%|[đĐ](?:ồng)?|vnd|vnđ)?", re.IGNORECASE)


def _read_mixed_alphanumeric(tok: str) -> str:
    """Đọc từng ký tự và số cho các mã hỗn hợp chữ số (VD: VN123456789)."""
    return " ".join(
        DIGITS_VI.get(c, LETTERS_VI.get(c, c))
        for c in tok.upper()
        if c in DIGITS_VI or c in LETTERS_VI
    )


def _read_numeric_token(tok: str) -> str:
    """Đọc số nguyên: đọc từng số nếu là số điện thoại hoặc đọc theo từ vựng tiếng Việt."""
    if (tok.startswith("0") and len(tok) > 1) or len(tok) >= 8:
        return " ".join(DIGITS_VI[c] for c in tok)
    return num2words(int(tok), lang="vi")


def _read_decimal_token(tok: str) -> str:
    """Đọc số thập phân có dấu phẩy hoặc chấm (VD: 3.5 -> ba phẩy năm)."""
    parts = DECIMAL_SEP_RE.split(tok)
    dec_part = parts[1]
    zero_count = len(dec_part) - len(dec_part.lstrip("0"))
    zero_prefix = "không " * zero_count
    dec_val = dec_part.lstrip("0")
    dec_spoken = (
        (zero_prefix + num2words(int(dec_val), lang="vi")) if dec_val else zero_prefix.strip()
    )
    return num2words(int(parts[0]), lang="vi") + " phẩy " + dec_spoken


def _replace_token_for_tts(match: re.Match) -> str:
    """Quy đổi các token số, tiền tệ, đơn vị đo lường sang cách phát âm tự nhiên."""
    tok = match.group(0)
    clean_tok = tok.rstrip(".,;!?")
    punct = tok[len(clean_tok) :]
    tok = clean_tok

    unit = ""
    m_unit = UNIT_TOKEN_RE.match(tok)
    if m_unit and m_unit.group(2).lower() in UNITS_VI:
        tok = m_unit.group(1)
        unit = " " + UNITS_VI[m_unit.group(2).lower()]

    if HAS_ALPHA_RE.search(tok) and HAS_DIGIT_RE.search(tok):
        spoken = _read_mixed_alphanumeric(tok)
        return spoken + unit + punct

    if tok.isdigit():
        spoken = _read_numeric_token(tok)
        return spoken + unit + punct

    if GROUPED_NUMBER_RE.fullmatch(tok):
        clean_digits = DECIMAL_SEP_RE.sub("", tok)
        spoken = num2words(int(clean_digits), lang="vi")
        return spoken + unit + punct

    if DECIMAL_NUMBER_RE.fullmatch(tok):
        spoken = _read_decimal_token(tok)
        return spoken + unit + punct

    return match.group(0)


def normalize_text_for_tts(text: str) -> str:
    """Chuẩn hóa số điện thoại, đơn vị, tiền tệ và số liệu trong văn bản trước khi đọc thành tiếng."""
    text = PHONE_PAREN_RE.sub(r"0\1\2\3", text)
    text = PHONE_INTL_RE.sub(r"0\1\2\3", text)
    text = PHONE_RE.sub(r"\1\2\3", text)
    return TTS_TOKEN_RE.sub(_replace_token_for_tts, text)


def split_for_tts(text: str, limit: int = MAX_TTS_CHARS) -> list[str]:
    """Tách câu văn bản dài tại dấu phẩy hoặc khoảng trắng gần nhất để vừa giới hạn ký tự của TTS."""
    parts: list[str] = []
    while len(text) > limit:
        cut = max(text.rfind(",", 0, limit), text.rfind(" ", 0, limit))
        if cut <= 0:
            cut = limit
        parts.append(text[: cut + 1].strip())
        text = text[cut + 1 :].strip()
    if text:
        parts.append(text)
    return parts


class IncompleteHeaderError(ValueError):
    """Header WAV chưa nhận đủ số byte để phân tích cấu trúc."""


def _parse_fmt_chunk(
    buf: bytes | bytearray, offset: int, chunk_len: int
) -> tuple[int, int, int, int]:
    """Trích xuất thông số định dạng âm thanh từ chunk 'fmt '."""
    if chunk_len < 14:
        return -1, TTS_CHANNELS, TTS_SAMPLE_RATE, TTS_SAMPLE_BITS
    audio_fmt, channels, sample_rate = struct.unpack("<HHI", buf[offset : offset + 8])
    if audio_fmt != 1:
        raise RuntimeError(f"WAV format code là {audio_fmt}, chỉ hỗ trợ PCM tiêu chuẩn (code 1).")
    bits = (
        struct.unpack("<H", buf[offset + 14 : offset + 16])[0]
        if chunk_len >= 16
        else TTS_SAMPLE_BITS
    )
    return audio_fmt, channels, sample_rate, bits


def parse_wav_header(buf: bytes | bytearray) -> tuple[int, int, int, int]:
    """Phân tích header RIFF/WAVE để trích xuất sample_rate, channels, bits và vị trí bắt đầu chunk data."""
    if len(buf) < WAV_HEADER_MIN_BYTES:
        raise IncompleteHeaderError(
            f"Header chưa đủ {WAV_HEADER_MIN_BYTES} byte để kiểm tra định dạng RIFF WAVE."
        )
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise RuntimeError(
            f"Header dữ liệu không phải định dạng RIFF WAVE (nhận được {buf[:4]!r}..{buf[8:12]!r})."
        )

    offset = 12
    channels = TTS_CHANNELS
    sample_rate = TTS_SAMPLE_RATE
    bits = TTS_SAMPLE_BITS
    audio_fmt = -1
    data_offset = -1

    while offset + 8 <= len(buf):
        chunk_id = bytes(buf[offset : offset + 4])
        chunk_len = struct.unpack("<I", buf[offset + 4 : offset + 8])[0]
        offset += 8
        if chunk_id == b"fmt ":
            if offset + chunk_len > len(buf):
                raise IncompleteHeaderError("Chunk 'fmt ' chưa nhận đủ dữ liệu.")
            audio_fmt, channels, sample_rate, bits = _parse_fmt_chunk(buf, offset, chunk_len)
        elif chunk_id == b"data":
            data_offset = offset
            break
        offset += chunk_len + (chunk_len % 2)

    if data_offset == -1:
        raise IncompleteHeaderError("Chưa nhận đủ chunk 'data' trong header WAV.")
    if audio_fmt == -1:
        raise RuntimeError("Không tìm thấy chunk 'fmt ' trong header WAV.")

    return sample_rate, channels, bits, data_offset


def _try_parse_initial_wav_header(head: bytearray, chunk: bytes) -> tuple[bytes, bool]:
    """Cố gắng phân tích header WAV từ các khối dữ liệu đầu tiên nhận từ TTS stream."""
    head.extend(chunk)
    if len(head) < WAV_DATA_HEADER_MIN_BYTES or b"data" not in head:
        return b"", False
    try:
        sample_rate, channels, bits, data_offset = parse_wav_header(head)
        if (sample_rate, channels, bits) != (TTS_SAMPLE_RATE, TTS_CHANNELS, TTS_SAMPLE_BITS):
            raise RuntimeError(
                f"Định dạng TTS không đúng yêu cầu: {sample_rate} Hz, {channels} channels, {bits} bit "
                f"(kỳ vọng {TTS_SAMPLE_RATE} Hz, {TTS_CHANNELS} channel, {TTS_SAMPLE_BITS} bit PCM tiêu chuẩn)."
            )
        audio_data = bytes(head[data_offset:])
        head.clear()
        return audio_data, True
    except IncompleteHeaderError:
        return b"", False


def _align_pcm16_chunk(chunk: bytes, pending: bytes) -> tuple[bytes, bytes]:
    """Căn chỉnh luồng byte PCM16: ghép byte lẻ trước đó và giữ lại byte lẻ cuối nếu có."""
    data = pending + chunk
    if len(data) % 2 != 0:
        return data[:-1], data[-1:]
    return data, b""


async def synth_stream(
    text: str, *, base_url: str, voice: str, client: httpx.AsyncClient
) -> AsyncIterator[bytes]:
    """Phát luồng trực tiếp các chunk PCM16 (48 kHz mono) nhận từ server TTS,
    bóc header WAV động và đảm bảo mỗi chunk phát đi có số byte chẵn (căn chỉnh 16-bit).
    """
    body = {**DEFAULT_PARAMS, "text": text, "voice": voice}
    head = bytearray()
    header_parsed = False
    pending_odd_byte = b""

    async with client.stream("POST", base_url.rstrip("/") + "/tts/stream", json=body) as r:
        if r.status_code != HTTP_STATUS_OK:
            raw = await r.aread()
            raise RuntimeError(f"TTS HTTP {r.status_code}: {raw[:ERR_MSG_MAX_LEN]!r}")
        async for chunk in r.aiter_bytes():
            if not header_parsed:
                audio_data, header_parsed = _try_parse_initial_wav_header(head, chunk)
                if not header_parsed:
                    continue
                chunk = audio_data

            data, pending_odd_byte = _align_pcm16_chunk(chunk, pending_odd_byte)
            if data:
                yield data


_END = object()


async def _produce_tts_sentences(
    messages: list[dict[str, str]],
    *,
    llm_url: str,
    client: httpx.AsyncClient,
    sentence_queue: asyncio.Queue,
    send_raw: Callable[[str], Awaitable[None]],
    mark: Callable[[str], None],
    raw_parts: list[str],
) -> None:
    """Đọc stream từ LLM, gửi raw text lên giao diện và đẩy các câu đã lọc vào hàng đợi."""
    speech_filter = SpeechFilter()
    end_sent = False
    try:
        async for piece in stream_reply(messages, url=llm_url, client=client):
            mark("first_content")
            raw_parts.append(piece)
            await send_raw(piece)
            for s in speech_filter.feed(piece):
                mark("first_sentence")
                await sentence_queue.put(s)
        for s in speech_filter.flush():
            mark("first_sentence")
            await sentence_queue.put(s)
        await sentence_queue.put(_END)
        end_sent = True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Lỗi trong quá trình sinh câu trả lời LLM: %s", exc)
        raise
    finally:
        if not end_sent and not sentence_queue.full():
            with contextlib.suppress(asyncio.QueueFull):
                sentence_queue.put_nowait(_END)


async def _consume_tts_audio(
    sentence_queue: asyncio.Queue,
    *,
    tts_url: str,
    voice: str,
    client: httpx.AsyncClient,
    send_audio: Callable[[bytes], Awaitable[None]],
    mark: Callable[[str], None],
) -> None:
    """Lấy từng câu từ hàng đợi, gửi qua Zero-TTS stream và đẩy chunk PCM tới client."""
    try:
        while (sent := await sentence_queue.get()) is not _END:
            norm_sent = normalize_text_for_tts(sent)
            for piece in split_for_tts(norm_sent):
                async for pcm in synth_stream(piece, base_url=tts_url, voice=voice, client=client):
                    mark("first_audio")
                    await send_audio(pcm)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Lỗi trong quá trình tổng hợp hoặc gửi âm thanh TTS: %s", exc)
        raise


async def run_turn(
    user_text: str,
    *,
    history: list[dict[str, str]],
    llm_url: str,
    tts_url: str,
    voice: str,
    client: httpx.AsyncClient,
    send_raw: Callable[[str], Awaitable[None]],
    send_audio: Callable[[bytes], Awaitable[None]],
    metrics: dict[str, Any],
) -> str:
    """Xử lý song song một lượt thoại:
    - Kênh màn hình: đẩy nguyên văn văn bản từ LLM (kể cả bảng, code) qua send_raw.
    - Kênh âm thanh: lọc từng câu hoàn chỉnh và phát luồng chunk PCM qua send_audio.
    - Trả về toàn bộ nội dung văn bản thô của câu trả lời.
    """
    t0 = time.perf_counter()
    metrics.clear()
    sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    raw_parts: list[str] = []

    def mark(name: str) -> None:
        metrics.setdefault(name, time.perf_counter() - t0)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]

    producer_task = asyncio.create_task(
        _produce_tts_sentences(
            messages,
            llm_url=llm_url,
            client=client,
            sentence_queue=sentence_queue,
            send_raw=send_raw,
            mark=mark,
            raw_parts=raw_parts,
        )
    )
    consumer_task = asyncio.create_task(
        _consume_tts_audio(
            sentence_queue,
            tts_url=tts_url,
            voice=voice,
            client=client,
            send_audio=send_audio,
            mark=mark,
        )
    )
    try:
        await asyncio.gather(producer_task, consumer_task)
    finally:
        for t in (producer_task, consumer_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(producer_task, consumer_task, return_exceptions=True)

    return "".join(raw_parts)
