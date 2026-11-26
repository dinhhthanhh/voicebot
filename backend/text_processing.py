"""Module xử lý văn bản và chuẩn hóa phát âm tiếng Việt cho Voicebot.

Nhiệm vụ chính:
- Lọc từ đệm đơn thuần (`is_filler`) để tránh gọi LLM vô ích.
- Làm sạch ký hiệu Markdown (`clean_md`) giữ văn bản đàm thoại tự nhiên.
- Chuẩn hóa văn bản sang lời nói tự nhiên (`normalize_tts`, `split_tts`) cho Zero-TTS.
- Tách câu và gom nhịp streaming từ LLM (`TTSStreamer`).
"""

from __future__ import annotations

import re

from num2words import num2words

from backend.config import settings

# =============================================================================
# 1. Lọc từ đệm nhận dạng giọng nói (ASR Filler Filter)
# =============================================================================
FILLER_WORDS = {
    "ừ", "ừm", "ờ", "ờm", "à", "ư", "hử", "hả", "ơ", "ớ", "ể",
}
_CLEAN_RE = re.compile(r"[^\w\s]")
_MULTI_FILLERS_RE = re.compile(r"\b(à mà|ừ thì|ờ thì)\b", re.IGNORECASE)


def is_filler(text: str) -> bool:
    """Kiểm tra transcript đầu vào có rỗng hoặc chỉ toàn từ đệm (ừ, à, ờ...) hay không.

    Dùng sau bước STT để trả câu nhắc ngắn qua TTS thay vì tốn lượt gọi LLM.
    """
    # Loại bỏ dấu câu, gom khoảng trắng và quy các cụm từ đệm ghép về một token chuẩn
    collapsed = " ".join(_CLEAN_RE.sub(" ", text.lower()).split())
    return not collapsed or all(
        w in FILLER_WORDS for w in _MULTI_FILLERS_RE.sub("ừ", collapsed).split()
    )


# =============================================================================
# 2. Làm sạch Markdown cho kênh thoại (Markdown Cleaner)
# =============================================================================
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")                  # [text](url) -> text
_URL_RE = re.compile(r"\(\s*https?://[^\s)]+\s*\)|https?://\S+")  # URL độc lập (gồm cả cặp ngoặc bao ngoài nếu có) -> khoảng trắng
_INLINE_MARK_RE = re.compile(r"[*_`~#]+")                   # Ký hiệu định dạng *, _, `, ~, #
_PREFIX_BULLET_RE = re.compile(r"^(?:[-*+•]|\d+[.)])\s+")  # Gạch đầu dòng hoặc số thứ tự đầu câu


def clean_md(text: str) -> str:
    """Làm sạch ký hiệu Markdown, link, URL và gạch đầu dòng để tạo câu thoại tự nhiên cho TTS.

    Lưu ý: Chỉ xóa số thứ tự ở đầu chuỗi (`^`) để tránh mất số ở giữa câu (vd: 'Mục 1. Gửi hàng').
    """
    if not text:
        return ""
    s = _MD_LINK_RE.sub(r"\1", text)          # Giữ lại phần chữ hiển thị của link Markdown
    s = _URL_RE.sub(" ", s)                   # Bỏ URL thô vì TTS không đọc tự nhiên được
    s = _INLINE_MARK_RE.sub("", s)            # Bỏ ký tự in đậm, nghiêng, code, tiêu đề
    s = _PREFIX_BULLET_RE.sub("", s.lstrip()) # Bỏ bullet đầu dòng sau khi đã xóa dấu * inline
    return " ".join(s.split())


# =============================================================================
# 3. Chuẩn hóa phát âm tiếng Việt cho TTS (TTS Normalization & Splitting)
# =============================================================================
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
# Chỉ khớp đúng chữ hoa để tránh nhầm đơn vị viết thường ("5g" = 5 gam, "4k" = 4 nghìn)
SHORT_TECH_TERMS = {
    "5G": "năm gờ", "4G": "bốn gờ", "3G": "ba gờ", "2G": "hai gờ",
    "3D": "ba đê", "2D": "hai đê", "4K": "bốn ca", "8K": "tám ca",
}
_ALNUM_VI = {**DIGITS_VI, **LETTERS_VI}
_UNIT_PAT = "|".join(re.escape(u) for u in sorted(UNITS_VI, key=len, reverse=True))

# Các đơn vị thời lượng/đo lường/tiền/số lượng đứng sau sẽ chặn nhận diện nhầm thành ngày tháng
_NON_DATE_UNITS_PAT = (
    r"ngày|giờ|h|hr|phút|giây|tuần|tháng|năm|"
    r"kg|g|gam|km|m|cm|mm|m2|m²|m3|m³|ha|lít|ml|"
    r"đồng|đ|vnd|vnđ|k|nghìn|ngàn|triệu|tỷ|tỉ|%|"
    r"đơn|kiện|hộp|gói|lần|cái|chiếc|bưu|tấn|tạ|yến"
)
_NOT_DATE_FOLLOWERS = rf"(?!\s*(?:{_NON_DATE_UNITS_PAT})(?=[\s.,;!?)]|$))(?!\s*[/.-]\s*\d)"
_RANGE_NUM = r"(?:[1-9]\d{0,2}(?:[.,]\d{3})+|[1-9]\d*(?:[.,]\d+)?|0[.,]\d+|0)"

# Regex nhận diện SĐT & hotline (chạy trước ngày tháng để không cắt nhầm SĐT có dấu chấm/gạch)
HOTLINE_RE = re.compile(r"\b(1[89]00)(?:[\s.-]+(\d{2,6})){1,3}\b")
PHONE_INTL_RE = re.compile(r"(?:\+84|(?<=\s)84)[\s.-]*(\d{2,3})[\s.-]*(\d{3,4})[\s.-]*(\d{3,4})\b")
PHONE_LOCAL_RE = re.compile(r"(?<![\d.,])(?:\(0(\d{2,3})\)|\b(0[1-9]\d{1,2}))[\s.-]+(\d{3,4})[\s.-]+(\d{3,4})\b")

# Regex nhận diện ngày tháng và khoảng ngày (tách rõ với số thập phân "." và khoảng giá trị "-")
DATE_DMY_RE = re.compile(
    rf"(?:\bngày\s+)?(0?[1-9]|[12]\d|3[01])\s*[/.-]\s*(0?[1-9]|1[0-2])\s*[/.-]\s*((?:19|20)\d{{2}})\b{_NOT_DATE_FOLLOWERS}",
    re.IGNORECASE,
)
DATE_RANGE_DM_RE = re.compile(
    rf"(?:\bngày\s+)?(0?[1-9]|[12]\d|3[01])\s*[-–]\s*(0?[1-9]|[12]\d|3[01])\s*/\s*(0?[1-9]|1[0-2])(?:\s*/\s*((?:19|20)\d{{2}}))?\b{_NOT_DATE_FOLLOWERS}",
    re.IGNORECASE,
)
DATE_PAIR_RANGE_RE = re.compile(
    rf"(?:\bngày\s+)?(0?[1-9]|[12]\d|3[01])\s*/\s*(0?[1-9]|1[0-2])\s*[-–]\s*(0?[1-9]|[12]\d|3[01])\s*/\s*(0?[1-9]|1[0-2])(?:\s*/\s*((?:19|20)\d{{2}}))?\b{_NOT_DATE_FOLLOWERS}",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(r"\btháng\s+(\d{1,2})\s*[/.-]\s*(\d{4})\b", re.IGNORECASE)
# Dấu "-" chỉ coi là ngày/tháng khi đứng sau "ngày"/"hôm"; sau "từ/giao/dự kiến..." chỉ nhận "/" để tránh nhầm "2-3 ngày", "5-10 kg"
DATE_PREP_DM_RE = re.compile(
    rf"\b(?:(ngày|hôm)\s+(0?[1-9]|[12]\d|3[01])\s*[/-]\s*(0?[1-9]|1[0-2])|(vào|từ|đến|hạn|lễ|đợt|trước|sau|giao|dự kiến)\s+(0?[1-9]|[12]\d|3[01])\s*/\s*(0?[1-9]|1[0-2]))\b{_NOT_DATE_FOLLOWERS}",
    re.IGNORECASE,
)
STANDALONE_DATE_DM_RE = re.compile(
    rf"\b(1[3-9]|2\d|3[01])\s*/\s*(0?[1-9]|1[0-2])\b{_NOT_DATE_FOLLOWERS}"
)

# Regex khoảng giá trị/thời gian (2-3 ngày, 1.5-3.5 kg, 15.000đ - 20.000đ) và xử lý token số
_OPT_UNIT_PAT = rf"(?:\s*(?:{_UNIT_PAT}))?"
VALUE_RANGE_RE = re.compile(
    rf"(?<![A-Za-z0-9/.,-])({_RANGE_NUM}{_OPT_UNIT_PAT})\s*[-–]\s*({_RANGE_NUM})(?=$|[\s.,;!?)]|(?:{_UNIT_PAT})(?=[\s.,;!?)]|$))",
    re.IGNORECASE,
)
DECIMAL_SEP_RE = re.compile(r"[.,]")
UNIT_TOKEN_RE = re.compile(r"^([0-9][0-9.,]*)([a-zA-ZđĐ%²³]+[23]?|đồng)$", re.IGNORECASE)
TTS_TOKEN_RE = re.compile(r"\b(?=[A-Za-z0-9.,²³]*\d)[A-Za-z0-9.,²³]+(?:%|[đĐ](?:ồng)?|vnd|vnđ)?", re.IGNORECASE)

_ALNUM_MIX_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)")
_THOUSANDS_RE = re.compile(r"[1-9]\d{0,2}(?:\.\d{3})+|[1-9]\d{0,2}(?:,\d{3}){2,}")
_THOUSANDS_CURRENCY_RE = re.compile(r"[1-9]\d{0,2}[.,]\d{3}")
_DECIMAL_RE = re.compile(r"\d+[.,]\d+")
_SUPPORT_24_7_RE = re.compile(r"\b24\s*/\s*7\b")
_ADDRESS_SLASH_RE = re.compile(r"\b(số|ngõ|ngách|hẻm|địa chỉ)\s+(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
_NON_DIGIT_RE = re.compile(r"\D")
_NUM_SLASH_NUM_RE = re.compile(r"(?<=\d)\s*/\s*(?=\d)")
_SPACED_THOUSANDS_UNIT_RE = re.compile(
    rf"\b(\d{{1,3}}(?:\s\d{{3}})+)(?=\s*(?:{_UNIT_PAT})(?=[.,;!?]|\s|$))",
    re.IGNORECASE,
)
_NUM_SPACE_UNIT_RE = re.compile(
    rf"\b([0-9][0-9.,]*)\s+({_UNIT_PAT})(?=[.,;!?]|\s|$)",
    re.IGNORECASE,
)


# --- Các hàm nhánh nhỏ hỗ trợ cho normalize_tts ---
def _read_int(tok: str, has_unit: bool = False) -> str:
    """Đọc số nguyên: tách từng chữ số nếu là SĐT/mã dài không đơn vị, ngược lại đọc thành số đếm."""
    if (tok.startswith("0") and len(tok) > 1) or (len(tok) >= 8 and not has_unit):
        return " ".join(DIGITS_VI.get(c, c) for c in tok)
    return num2words(int(tok), lang="vi")


def _read_dec(tok: str) -> str:
    """Đọc số thập phân (giữ cả các số 0 ngay sau dấu phẩy, vd: 2.05 -> 'hai phẩy không năm')."""
    parts = DECIMAL_SEP_RE.split(tok)
    dec_val = parts[1].lstrip("0")
    zero_prefix = "không " * (len(parts[1]) - len(dec_val))
    dec_spoken = (zero_prefix + num2words(int(dec_val), lang="vi")) if dec_val else zero_prefix.strip()
    return f"{num2words(int(parts[0]), lang='vi')} phẩy {dec_spoken}"


def _read_token(match: re.Match[str]) -> str:
    """Chuyển đổi một token chứa số (5G, đơn vị đo, mã vận đơn, tiền tệ, số thập phân) sang chữ đọc."""
    raw_tok = match.group(0)
    # Tách dấu câu cuối câu khỏi token số/thập phân (vd: "3.14." -> "3.14" + "."), giữ dấu chấm/phẩy bên trong
    tok = raw_tok.rstrip(".,;!?")
    punct = raw_tok[len(tok):]

    if tok in SHORT_TECH_TERMS:
        return SHORT_TECH_TERMS[tok] + punct

    # Tách phần đơn vị đo nếu dính liền số (vd: "2.5kg", "25.000đồng")
    unit = ""
    if m_unit := UNIT_TOKEN_RE.match(tok):
        unit_key = m_unit.group(2).lower()
        if unit_key in UNITS_VI:
            tok = m_unit.group(1)
            unit = " " + UNITS_VI[unit_key]

    # Mã lẫn chữ và số (mã vận đơn VN123...) -> đánh vần từng ký tự
    if _ALNUM_MIX_RE.search(tok):
        return " ".join(_ALNUM_VI[c] for c in tok.upper() if c in _ALNUM_VI) + unit + punct

    # Số nguyên ASCII thuần (tránh nhầm chữ số mũ ², ³ của Python str.isdigit)
    if tok.isascii() and tok.isdigit():
        return _read_int(tok, has_unit=bool(unit)) + unit + punct

    # Số phân tách hàng nghìn (vd: 25.000 đồng, 1.000.000.000)
    if _THOUSANDS_RE.fullmatch(tok) or (
        unit in (" đồng", " nghìn") and _THOUSANDS_CURRENCY_RE.fullmatch(tok)
    ):
        return num2words(int(DECIMAL_SEP_RE.sub("", tok)), lang="vi") + unit + punct

    if _DECIMAL_RE.fullmatch(tok):
        return _read_dec(tok) + unit + punct

    return raw_tok


def _read_prep_date(match: re.Match[str]) -> str:
    """Đọc cụm ngày/tháng có giới từ đi trước mà không lặp từ 'ngày'."""
    prep = match.group(1) or match.group(4)
    day = int(match.group(2) or match.group(5))
    month = int(match.group(3) or match.group(6))
    prefix = "" if prep.lower() == "ngày" else f"{prep} "
    return f"{prefix}ngày {day} tháng {month}"


# --- Các hàm chính xử lý TTS ---
def normalize_tts(text: str) -> str:
    """Chuẩn hóa toàn bộ câu văn bản sang chữ phát âm tự nhiên cho Zero-TTS.

    Thứ tự ưu tiên:
    1. Hotline & SĐT (+84, nội hạt) để tránh bị quy tắc ngày tháng hoặc khoảng số cắt nhầm.
    2. Cụm đặc biệt (24/7, địa chỉ ngõ/hẻm có dấu xuyệt).
    3. Khoảng ngày và ngày tháng hợp lệ (chặn trùng với khoảng thời gian và giá trị giao hàng).
    4. Khoảng giá trị/thời gian (A-B -> A đến B) và phân số (A/B -> A trên B).
    5. Gom số cách hàng nghìn, ghép đơn vị đo và đọc từng token số qua `_read_token`.
    """
    text = HOTLINE_RE.sub(lambda m: " ".join(DIGITS_VI[c] for c in _NON_DIGIT_RE.sub("", m.group(0))), text)
    text = PHONE_INTL_RE.sub(r"0\1\2\3", text)
    text = PHONE_LOCAL_RE.sub(lambda m: f"{'0' + m.group(1) if m.group(1) else m.group(2)}{m.group(3)}{m.group(4)}", text)
    text = _SUPPORT_24_7_RE.sub("hai tư trên bảy", text)
    text = _ADDRESS_SLASH_RE.sub(r"\1 \2 xuyệt \3", text)
    text = DATE_PAIR_RANGE_RE.sub(
        lambda m: f"ngày {int(m.group(1))} tháng {int(m.group(2))} đến ngày {int(m.group(3))} tháng {int(m.group(4))}"
        + (f" năm {m.group(5)}" if m.group(5) else ""),
        text,
    )
    text = DATE_RANGE_DM_RE.sub(
        lambda m: f"ngày {int(m.group(1))} đến ngày {int(m.group(2))} tháng {int(m.group(3))}"
        + (f" năm {m.group(4)}" if m.group(4) else ""),
        text,
    )
    text = DATE_DMY_RE.sub(lambda m: f"ngày {int(m.group(1))} tháng {int(m.group(2))} năm {m.group(3)}", text)
    text = MONTH_YEAR_RE.sub(lambda m: f"tháng {int(m.group(1))} năm {m.group(2)}", text)
    text = DATE_PREP_DM_RE.sub(_read_prep_date, text)
    text = STANDALONE_DATE_DM_RE.sub(lambda m: f"ngày {int(m.group(1))} tháng {int(m.group(2))}", text)
    text = VALUE_RANGE_RE.sub(r"\1 đến \2", text)
    text = _NUM_SLASH_NUM_RE.sub(" trên ", text)
    text = _SPACED_THOUSANDS_UNIT_RE.sub(lambda m: m.group(1).replace(" ", ""), text)
    text = _NUM_SPACE_UNIT_RE.sub(r"\1\2", text)
    return TTS_TOKEN_RE.sub(_read_token, text)


def split_tts(text: str, limit: int | None = None) -> list[str]:
    """Cắt câu quá dài thành nhiều đoạn ngắn vừa ngưỡng đọc của Zero-TTS.

    Ưu tiên cắt tại dấu phẩy hoặc khoảng trắng gần `limit` nhất; nếu chuỗi liền không có
    khoảng trắng thì cắt cứng đúng bằng `limit` ký tự.
    """
    max_len = limit if limit is not None else settings.tts_split_limit
    parts: list[str] = []
    while len(text) > max_len:
        cut = max(text.rfind(",", 0, max_len), text.rfind(" ", 0, max_len))
        split_at = cut + 1 if cut > 0 else max_len
        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        parts.append(text)
    return parts


# =============================================================================
# 4. Bộ đệm tách câu từ LLM Streaming cho TTS (TTSStreamer)
# =============================================================================
REMAINDER_NOTE = "Phần còn lại đã được hiển thị trên màn hình."
_SENT_SPLIT_RE = re.compile(
    r"(?<!\bTP\.)(?<!\bTp\.)"               # Tránh cắt nhầm tại "TP." (Thành phố)
    r"(?<!\b[A-Z]\.)(?<!\bv\.v\.)(?<!\.\.)"  # Tránh cắt nhầm tại tên viết tắt, "v.v." và "..."
    r"(?<!\b\d\.)(?<!\b\d\d\.)"              # Tránh cắt nhầm tại số thứ tự danh sách (1., 2., 10.)
    r"(?<=[.!?…])\s+|\n+"                    # Cắt câu khi gặp dấu kết câu + khoảng trắng hoặc xuống dòng
)
_HAS_WORD_RE = re.compile(r"\w")


class TTSStreamer:
    """Bộ tách câu và chuẩn hóa luồng văn bản thoại từ LLM sang TTS theo thời gian thực.

    Quy trình cho mỗi câu hoàn chỉnh: `clean_md` -> ghép câu quá ngắn -> kiểm tra giới hạn
    (`max_sentences`, `max_chars`) -> `normalize_tts` -> `split_tts`.
    """

    def __init__(
        self,
        min_chars: int | None = None,
        max_sentences: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        self.min_chars = min_chars if min_chars is not None else settings.tts_min_chars
        self.max_sentences = max_sentences if max_sentences is not None else settings.tts_max_sentences
        self.max_chars = max_chars if max_chars is not None else settings.tts_max_chars
        self.buf = ""         # Đoạn văn bản dở dang chưa gặp dấu ngắt câu
        self.hold = ""        # Câu quá ngắn (< min_chars) tạm giữ để ghép với câu kế tiếp
        self.sent_count = 0   # Số câu đã xuất cho TTS trong lượt hiện tại
        self.total_chars = 0  # Tổng số ký tự đã xuất cho TTS trong lượt hiện tại
        self.capped = False   # Cờ báo đã chạm ngưỡng đọc tối đa

    # --- Hàm nhánh hỗ trợ ---
    @staticmethod
    def _format(sentences: list[str]) -> list[str]:
        """Chuẩn hóa phát âm và chia đoạn đọc cho danh sách câu đã chốt."""
        return [
            chunk
            for s in sentences
            for chunk in split_tts(normalize_tts(s))
        ]

    def _emit(self, text: str, out: list[str], end: bool = False) -> None:
        """Làm sạch Markdown, ghép câu ngắn bị giữ (`hold`) và kiểm tra giới hạn capping."""
        if self.capped:
            return
        cleaned = clean_md(text)
        if not _HAS_WORD_RE.search(cleaned):
            return
        if self.hold:
            cleaned, self.hold = f"{self.hold} {cleaned}", ""
        # Giữ lại câu quá ngắn khi chưa kết thúc stream để TTS không bị gọi cho câu 1-2 từ
        if not end and len(cleaned) < self.min_chars:
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

    # --- Các hàm chính ---
    def flush(self) -> list[str]:
        """Xả toàn bộ phần chữ còn nằm trong `buf` và `hold` khi LLM kết thúc stream."""
        if self.capped:
            self.buf, self.hold = "", ""
            return []
        out: list[str] = []
        if buf := self.buf:
            self.buf = ""
            self._emit(buf, out, end=True)
        # Xóa self.hold trước khi gọi _emit để tránh _emit tự nối held với chính nó
        if held := self.hold:
            self.hold = ""
            self._emit(held, out, end=True)
        return self._format(out)

    def feed(self, delta: str | None = None) -> list[str]:
        """Nạp thêm mẩu text `delta` từ LLM và trả về các đoạn đã chuẩn hóa sẵn sàng cho TTS.

        Truyền `delta=None` khi kết thúc stream để tự động gọi `flush()`.
        """
        if delta is None:
            return self.flush()
        if self.capped or not delta:
            return []
        self.buf += delta
        out: list[str] = []
        parts = _SENT_SPLIT_RE.split(self.buf)
        if len(parts) > 1:
            for part in parts[:-1]:
                self._emit(part, out)
                if self.capped:
                    self.buf = ""
                    break
            else:
                self.buf = parts[-1]
        return self._format(out)
