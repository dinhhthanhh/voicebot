"""app/text_clean.py - Bộ lọc văn bản từ luồng LLM sang TTS.
Nhiệm vụ:
1. Tách dòng văn bản thành từng câu hoàn chỉnh khi đủ dấu ngắt.
2. Bỏ qua các khối định dạng không phù hợp đọc thành tiếng (bảng Markdown, khối mã nguồn).
3. Làm sạch ký hiệu nội dòng: emoji, Markdown formatting, công thức toán, URL, placeholder.
4. Phát hiện ký tự CJK (Hán/Hàn/Nhật) để thông báo thay vì đọc sai.
5. Giới hạn số câu và số ký tự đọc (capping) để tránh đọc quá dài trên kênh thoại.
"""
from __future__ import annotations

import re

# Regex làm sạch nội dòng
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")
URL_RE = re.compile(r"https?://[^\s<>()]+")
MARK_RE = re.compile(r"(\*\*|__|\*|`|~~)")
PREFIX_RE = re.compile(r"^(#{1,6}\s+|[-*+•]\s+|\d+[.)]\s+|>\s*)")
RULE_RE = re.compile(r"^[-*_=\s]{3,}$")
SENT_RE = re.compile(r"(?<=[.!?…])(?<!\.\.)\s+")
MATH_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$")
CJK_RE = re.compile("[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\([^()\n]*\)")

# Câu thông báo thay thế cho kênh thoại khi gặp định dạng đặc biệt
CJK_NOTE = "Một phần câu trả lời không đọc được, xem chi tiết trên màn hình."
TABLE_NOTE = "Chi tiết dạng bảng đã được hiển thị trên màn hình."
CODE_NOTE = "Phần mã đã được hiển thị trên màn hình."
REMAINDER_NOTE = "Phần còn lại đã được hiển thị trên màn hình."

# Danh sách ngôn ngữ lập trình phổ biến nhận diện trong khối fence ```
CODE_LANGS = {
    "python", "py", "javascript", "js", "typescript", "ts",
    "java", "cpp", "c", "csharp", "cs", "go", "rust",
    "php", "ruby", "sql", "bash", "sh", "powershell", "ps1",
    "json", "yaml", "yml", "xml", "html", "css", "dockerfile", "toml",
}


def has_cjk(text: str) -> bool:
    """Kiểm tra sự xuất hiện của ký tự CJK (Hán, Nhật, Triều Tiên)."""
    return CJK_RE.search(text) is not None


def clean_inline(text: str) -> str:
    """Làm sạch ký hiệu Markdown, emoji, URL và biểu thức mã trong một câu."""
    s = MATH_RE.sub("công thức trên màn hình", text)
    s = CALL_RE.sub(r"hàm \1", s)
    s = re.sub(r"\bhàm\s+hàm\b", "hàm", s, flags=re.IGNORECASE)
    s = LINK_RE.sub(r"\1", s)
    s = PLACEHOLDER_RE.sub("", s)
    s = URL_RE.sub("đường dẫn trên màn hình", s)
    s = EMOJI_RE.sub("", s)
    s = MARK_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


class SpeechFilter:
    """Bộ đệm nhận từng delta từ LLM và nhả ra các câu sạch sẵn sàng đưa vào TTS."""

    def __init__(self, min_chars: int = 8, max_sentences: int = 8, max_chars: int = 600) -> None:
        self.min_chars = min_chars
        self.max_sentences = max_sentences
        self.max_chars = max_chars

        self.buf = ""
        self.kind: str | None = None
        self.fresh = True
        self.in_code = False
        self.in_table = False
        self.hold = ""
        self.last_sent = ""
        self.sent_count = 0
        self.total_chars = 0
        self.capped = False
        self.cjk_hits: list[str] = []

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
            self._push(self.hold, out, end=True, force=True)
            self.hold = ""
        return out

    def _push_note(self, note: str, out: list[str]) -> None:
        """Đẩy câu thông báo màn hình, tránh lặp lại nếu câu trước đã có nội dung tương tự."""
        last = self.hold or (out[-1] if out else self.last_sent)
        last_lower = last.lower()
        if "màn hình" in last_lower or "hiển thị" in last_lower:
            return
        self._push(note, out, end=True, force=True)

    def _handle(self, text: str, complete: bool, out: list[str]) -> str:
        s = text.lstrip()

        # Xác định loại nội dung của dòng hiện tại
        if self.kind is None:
            if not s:
                return "" if complete else text
            if not complete and s.startswith("`") and len(s) < 3:
                return text
            if s.startswith("```"):
                self.kind = "fence"
            elif self.in_code:
                self.kind = "skip"
            elif s.startswith("|"):
                self.kind = "table"
            elif complete and RULE_RE.match(s):
                self.kind = "skip"
            else:
                self.kind = "prose"

        # Khối rào mã nguồn (fence)
        if self.kind == "fence":
            if not complete:
                return text
            if not self.in_code:
                info = s[3:].strip().lower()
                lang = info.split()[0] if info else ""
                self.in_code = True
                if lang in CODE_LANGS:
                    self._push_note(CODE_NOTE, out)
            else:
                self.in_code = False
                rest = s[3:].strip()
                if rest:
                    for p in SENT_RE.split(rest):
                        self._push(p, out, end=complete)
            return ""

        if self.kind == "skip":
            return ""

        # Khối bảng Markdown
        if self.kind == "table":
            if not self.in_table:
                self.in_table = True
                self._push_note(TABLE_NOTE, out)
            return ""

        # Văn bản thông thường (prose)
        self.in_table = False
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

    def _push(self, sent: str, out: list[str], end: bool = False, force: bool = False) -> None:
        """Kiểm tra và đưa câu vào danh sách xuất khẩu."""
        if self.capped:
            return

        # 1. Phát hiện ký tự CJK
        if has_cjk(sent):
            self.cjk_hits.append(sent)
            self._push_note(CJK_NOTE, out)
            return

        # 2. Làm sạch nội dòng
        sent = clean_inline(sent)
        if not re.search(r"\w", sent):
            return

        # 3. Chuẩn hoá dấu câu kết thúc
        if end and not re.search(r"[.!?…:;,]$", sent):
            sent += "."

        # 4. Ghép với phần dở trước đó
        if self.hold:
            sent = f"{self.hold} {sent}"
            self.hold = ""

        # 5. Nếu quá ngắn và chưa phải cuối luồng, tạm giữ lại để ghép câu sau
        if not force and len(sent) < self.min_chars:
            self.hold = sent
            return

        # 6. Kiểm tra giới hạn số câu và độ dài (capping)
        too_long = (self.total_chars + len(sent)) > self.max_chars
        if self.sent_count >= self.max_sentences or (too_long and self.sent_count > 0):
            self.capped = True
            self.hold = ""
            out.append(REMAINDER_NOTE)
            return

        # 7. Nhận câu hợp lệ
        self.last_sent = sent
        self.sent_count += 1
        self.total_chars += len(sent)
        out.append(sent)


if __name__ == "__main__":
    sample = (
        "\n\nXin chào! 👋 Đây là **so sánh** nhanh:\n\n"
        "| Dịch vụ | Thời gian |\n|---|---|\n| Nhanh | 1 ngày |\n| Thường | 3 ngày |\n\n"
        "- Giao tận nơi\n- Có theo dõi\n\n"
        "```python\nprint(1)\n```\n"
        "Chi tiết xem tại https://example.com nhé."
    )
    f, result = SpeechFilter(), []
    for i in range(0, len(sample), 7):
        result += f.feed(sample[i:i + 7])
    result += f.flush()
    for s in result:
        print("-", s)
