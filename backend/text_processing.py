"""Module xử lý văn bản và chuẩn hóa phát âm tiếng Việt cho Voicebot.

Nhiệm vụ:
- Lọc từ đệm đơn thuần (ừ, à, ờ...) để tránh gọi LLM vô ích.
- Tách câu streaming từ LLM (SpeechFilter) cho Zero-TTS.
- Chuẩn hóa văn bản tiếng Việt sang lời nói tự nhiên cho TTS (ngày tháng, số tiền, đơn vị, mã).
"""

from __future__ import annotations

import re
from typing import Any

from num2words import num2words

from backend.config import settings

# ---------------------------------------------------------------------------
# 1. Lọc từ đệm nhận dạng giọng nói (ASR Filler Filter)
# ---------------------------------------------------------------------------
FILLER_WORDS = {
    "ừ", "ừm", "ờ", "ờm", "à", "ư", "hử", "hả", "ơ", "ớ", "ể", "à mà", "ừ thì", "ờ thì",
}
_CLEAN_RE = re.compile(r"[^\w\s]")
_SPACES_RE = re.compile(r"\s+")
_MULTI_FILLERS_RE = re.compile(r"\b(à mà|ừ thì|ờ thì)\b", re.IGNORECASE)


def is_only_filler(text: str) -> bool:
    """Kiểm tra văn bản nhận diện được mới chỉ chứa từ đệm hoặc còn trống."""
    cleaned = _CLEAN_RE.sub(" ", text.lower())
    norm = _MULTI_FILLERS_RE.sub("ừ", _SPACES_RE.sub(" ", cleaned).strip())
    return not norm or all(w in FILLER_WORDS for w in norm.split())


# ---------------------------------------------------------------------------
# 2. Bộ đệm tách câu từ LLM Streaming cho TTS (SpeechFilter)
# ---------------------------------------------------------------------------
REMAINDER_NOTE = "Phần còn lại đã được hiển thị trên màn hình."

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_URL_RE = re.compile(r"https?://\S+")
_INLINE_MARK_RE = re.compile(r"[*_`~#]+")
_PREFIX_BULLET_RE = re.compile(r"^[-*+•]\s+|\d+[.)]\s+")
_SENT_SPLIT_RE = re.compile(
    r"(?<!\bTP\.)(?<!\bTp\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)"
    r"(?<!\bThS\.)(?<!\bTS\.)(?<!\bBS\.)(?<!\bGS\.)(?<!\bPGS\.)"
    r"(?<!\b[A-Z]\.)(?<!\bv\.v\.)(?<!\.\.)"
    r"(?<=[.!?…])\s+|\n+"
)


def clean_inline_markdown(text: str) -> str:
    """Làm sạch ký hiệu Markdown, link, URL khỏi văn bản thoại."""
    if not text:
        return ""
    s = _MD_LINK_RE.sub(r"\1", text)
    s = _URL_RE.sub(" ", s)
    s = _INLINE_MARK_RE.sub("", s)
    s = _PREFIX_BULLET_RE.sub("", s)
    return " ".join(s.split())


class SpeechFilter:
    """Bộ tách câu gọn nhẹ từ luồng LLM streaming cho TTS."""

    def __init__(
        self,
        min_chars: int | None = None,
        max_sentences: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        self.min_chars = min_chars if min_chars is not None else settings.tts_min_chars
        self.max_sentences = max_sentences if max_sentences is not None else settings.tts_max_sentences
        self.max_chars = max_chars if max_chars is not None else settings.tts_max_chars
        self.buf = ""
        self.hold = ""
        self.sent_count = 0
        self.total_chars = 0
        self.capped = False

    def feed(self, delta: str) -> list[str]:
        if self.capped:
            return []
        self.buf += delta
        out: list[str] = []
        parts = _SENT_SPLIT_RE.split(self.buf) # Quét tìm dấu ngắt câu (., !, ?, …, \n).
        if len(parts) > 1:
            for part in parts[:-1]:
                self._emit(part, out)
            self.buf = parts[-1]
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        if self.buf:
            self._emit(self.buf, out, end=True)
            self.buf = ""
        if self.hold:
            self._emit(self.hold, out, end=True, force=True)
            self.hold = ""
        return out

    def _emit(self, text: str, out: list[str], end: bool = False, force: bool = False) -> None:
        if self.capped:
            return
        cleaned = clean_inline_markdown(text) # Xóa các ký hiệu [link](url), URL, bullet points (-, *), in đậm **...
        if not re.search(r"\w", cleaned):
            return
        if self.hold:
            cleaned, self.hold = f"{self.hold} {cleaned}", ""
        if not force and len(cleaned) < self.min_chars and not end:
            self.hold = cleaned
            return
        if self.sent_count >= self.max_sentences or (
            (self.total_chars + len(cleaned)) > self.max_chars and self.sent_count > 0
        ):
            self.capped, self.hold = True, ""
            out.append(REMAINDER_NOTE)
            return
        self.sent_count += 1
        self.total_chars += len(cleaned)
        out.append(cleaned)


# ---------------------------------------------------------------------------
# 3. Chuẩn hóa phát âm tự nhiên cho TTS (TTS Normalization)
# ---------------------------------------------------------------------------
DIGITS_VI = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}
LETTERS_VI = {
    "A": "a", "B": "bê", "C": "xê", "D": "đê", "E": "e", "F": "ép",
    "G": "gờ", "H": "hát", "I": "i", "J": "giây", "K": "ca", "L": "lờ",
    "M": "mờ", "N": "nờ", "O": "o", "P": "pê", "Q": "quy", "R": "rờ",
    "S": "ét", "T": "tê", "U": "u", "V": "vê", "W": "vê kép", "X": "ích",
    "Y": "y", "Z": "dét",
}
UNITS_VI = {
    "kg": "ki lô gam", "g": "gam", "km": "ki lô mét", "m": "mét",
    "cm": "xen ti mét", "mm": "mi li mét", "m2": "mét vuông", "m²": "mét vuông",
    "m3": "mét khối", "m³": "mét khối", "ha": "héc ta", "đ": "đồng",
    "đồng": "đồng", "vnd": "đồng", "vnđ": "đồng", "k": "nghìn", "%": "phần trăm",
    "h": "giờ", "hr": "giờ",
}
SHORT_TECH_TERMS = {
    "5G": "năm gờ", "4G": "bốn gờ", "3G": "ba gờ", "2G": "hai gờ",
    "3D": "ba đê", "2D": "hai đê", "4K": "bốn ca", "8K": "tám ca",
}
_UNIT_PAT = "|".join(re.escape(u) for u in sorted(UNITS_VI, key=len, reverse=True))

HOTLINE_RE = re.compile(r"\b(1[89]00)(?:[\s.-]+(\d{2,6})){1,3}\b")
PHONE_INTL_RE = re.compile(r"(?:\+84|(?<=\s)84)[\s.-]*(\d{2,3})[\s.-]*(\d{3,4})[\s.-]*(\d{3,4})\b")
PHONE_LOCAL_RE = re.compile(r"(?<![\d.,])(?:\(0(\d{2,3})\)|\b(0[1-9]\d{1,2}))[\s.-]+(\d{3,4})[\s.-]+(\d{3,4})\b")
DATE_DMY_RE = re.compile(
    r"(?:\bngày\s+)?(0?[1-9]|[12]\d|3[01])\s*[/.-]\s*(0?[1-9]|1[0-2])\s*[/.-]\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(r"\btháng\s+(\d{1,2})\s*[/.-]\s*(\d{4})\b", re.IGNORECASE)
DATE_PREP_DM_RE = re.compile(
    r"\b(ngày|vào|từ|đến|hôm|hạn|lễ|đợt|trước|sau|giao|dự kiến)\s+(0?[1-9]|[12]\d|3[01])\s*[/.-]\s*(0?[1-9]|1[0-2])\b",
    re.IGNORECASE,
)
STANDALONE_DATE_DM_RE = re.compile(r"\b(1[3-9]|2\d|3[01])\s*/\s*(0?[1-9]|1[0-2])\b")
DECIMAL_SEP_RE = re.compile(r"[.,]")
UNIT_TOKEN_RE = re.compile(r"^([0-9][0-9.,]*)([a-zA-ZđĐ%²³]+[23]?|đồng)$", re.IGNORECASE)
TTS_TOKEN_RE = re.compile(r"\b(?=[A-Za-z0-9.,²³]*\d)[A-Za-z0-9.,²³]+(?:%|[đĐ](?:ồng)?|vnd|vnđ)?", re.IGNORECASE)


def _read_numeric_token(tok: str) -> str:
    if (tok.startswith("0") and len(tok) > 1) or len(tok) >= 8:
        return " ".join(DIGITS_VI[c] for c in tok)
    return num2words(int(tok), lang="vi")


def _read_decimal_token(tok: str) -> str:
    parts = DECIMAL_SEP_RE.split(tok)
    dec_val = parts[1].lstrip("0")
    zero_prefix = "không " * (len(parts[1]) - len(dec_val))
    dec_spoken = (zero_prefix + num2words(int(dec_val), lang="vi")) if dec_val else zero_prefix.strip()
    return f"{num2words(int(parts[0]), lang='vi')} phẩy {dec_spoken}"


def _replace_token_for_tts(match: re.Match[str]) -> str:
    tok = match.group(0)
    clean_tok = tok.rstrip(".,;!?")
    punct = tok[len(clean_tok):]
    tok = clean_tok

    if tok.upper() in SHORT_TECH_TERMS:
        return SHORT_TECH_TERMS[tok.upper()] + punct

    unit = ""
    m_unit = UNIT_TOKEN_RE.match(tok)
    if m_unit and m_unit.group(2).lower() in UNITS_VI:
        tok = m_unit.group(1)
        unit = " " + UNITS_VI[m_unit.group(2).lower()]

    if re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok):
        digits_or_letters = [DIGITS_VI.get(c, LETTERS_VI.get(c, c)) for c in tok.upper() if c in DIGITS_VI or c in LETTERS_VI]
        return " ".join(digits_or_letters) + unit + punct

    if tok.isdigit():
        return _read_numeric_token(tok) + unit + punct

    if re.fullmatch(r"[1-9]\d{0,2}(?:\.\d{3})+|[1-9]\d{0,2}(?:,\d{3}){2,}", tok) or (
        unit.strip() in ("đồng", "nghìn") and re.fullmatch(r"[1-9]\d{0,2}[.,]\d{3}", tok)
    ):
        return num2words(int(DECIMAL_SEP_RE.sub("", tok)), lang="vi") + unit + punct

    if re.fullmatch(r"\d+[.,]\d+", tok):
        return _read_decimal_token(tok) + unit + punct

    return match.group(0)


def _normalize_dates(text: str) -> str:
    text = re.sub(r"\b24\s*/\s*7\b", "hai tư trên bảy", text)
    text = re.sub(r"\b(số|ngõ|ngách|hẻm|địa chỉ)\s+(\d+)\s*/\s*(\d+)\b", r"\1 \2 xuyệt \3", text, flags=re.I)
    text = DATE_DMY_RE.sub(lambda m: f"ngày {int(m.group(1))} tháng {int(m.group(2))} năm {m.group(3)}", text)
    text = MONTH_YEAR_RE.sub(lambda m: f"tháng {int(m.group(1))} năm {m.group(2)}", text)
    text = DATE_PREP_DM_RE.sub(lambda m: f"{m.group(1) if m.group(1).lower() != 'ngày' else ''} ngày {int(m.group(2))} tháng {int(m.group(3))}".strip(), text)
    return STANDALONE_DATE_DM_RE.sub(lambda m: f"ngày {int(m.group(1))} tháng {int(m.group(2))}", text)


def normalize_text_for_tts(text: str) -> str:
    """Chuẩn hóa văn bản thành câu phát âm tự nhiên trước khi đưa vào mô hình TTS."""
    text = HOTLINE_RE.sub(lambda m: " ".join(DIGITS_VI[c] for c in re.sub(r"\D", "", m.group(0))), text)
    text = PHONE_INTL_RE.sub(r"0\1\2\3", text)
    text = PHONE_LOCAL_RE.sub(lambda m: f"{'0' + m.group(1) if m.group(1) else m.group(2)}{m.group(3)}{m.group(4)}", text)
    text = _normalize_dates(text)
    text = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " trên ", text)
    text = re.sub(rf"\b(\d{{1,3}}(?:\s\d{{3}})+)(?=\s*(?:{_UNIT_PAT})(?=[.,;!?]|\s|$))", lambda m: m.group(1).replace(" ", ""), text, flags=re.I)
    text = re.sub(rf"\b([0-9][0-9.,]*)\s+({_UNIT_PAT})(?=[.,;!?]|\s|$)", r"\1\2", text, flags=re.I)
    return TTS_TOKEN_RE.sub(_replace_token_for_tts, text)


def split_for_tts(text: str, limit: int | None = None) -> list[str]:
    """Tách câu văn bản dài tại dấu phẩy hoặc khoảng trắng gần nhất."""
    max_len = limit if limit is not None else settings.tts_split_limit
    parts: list[str] = []
    while len(text) > max_len:
        cut = max(text.rfind(",", 0, max_len), text.rfind(" ", 0, max_len))
        cut = cut if cut > 0 else max_len
        parts.append(text[: cut + 1].strip())
        text = text[cut + 1 :].strip()
    if text:
        parts.append(text)
    return parts


class TTSTextStreamer:
    """Đóng gói toàn bộ luồng xử lý văn bản thoại từ LLM đến TTS:

    Lọc ký hiệu Markdown -> Tách câu (SpeechFilter) -> Chuẩn hóa phát âm (normalize_text_for_tts)
    -> Cắt ngắn theo giới hạn TTS (split_for_tts).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.speech_filter = SpeechFilter(*args, **kwargs)

    def feed(self, delta: str | None = None) -> list[str]:
        raw = self.speech_filter.feed(delta) if delta is not None else self.speech_filter.flush()
        return [
            chunk
            for s in raw
            for chunk in split_for_tts(normalize_text_for_tts(s))
        ]

