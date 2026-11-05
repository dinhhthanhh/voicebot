"""tts.py - Hợp nhất xử lý tạo câu trả lời và tổng hợp giọng nói đầu ra.
Bao gồm:
1. LLM streaming (vLLM Qwen 2.5): SYSTEM_PROMPT, stream_reply.
2. SpeechFilter: Lọc sạch markdown, emoji, bảng, code, tách câu cho TTS.
3. Zero-TTS streaming: split_for_tts, synth, synth_stream (PCM16 48kHz mono).
4. Điều phối luồng đàm thoại: run_turn (LLM Producer -> SpeechFilter -> Queue -> TTS Consumer).
"""
from __future__ import annotations

import asyncio
import json
import re
import struct
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

# ==============================================================================
# 1. Cấu hình LLM & Prompt
# ==============================================================================
SYSTEM_PROMPT = (
    "Bạn là trợ lý giọng nói tiếng Việt. Câu trả lời của bạn vừa được đọc to thành tiếng, "
    "vừa hiển thị trên màn hình. Viết ngắn gọn như văn nói, đủ câu, không dùng emoji. "
    "Tuân thủ đúng yêu cầu định dạng của người dùng: nếu họ yêu cầu gạch đầu dòng, bảng, "
    "danh sách, mã nguồn hay biểu đồ thì phải làm đúng như vậy, không từ chối. "
    "Xử lý nhận dạng giọng nói, số điện thoại và mã vận đơn: "
    "- Người dùng nói qua micro nên văn bản có thể có từ ngữ ngập ngừng, lặp từ, hoặc tự sửa lời. "
    "- Số điện thoại di động Việt Nam chuẩn gồm đúng 10 chữ số (bắt đầu bằng 03, 05, 07, 08, 09). "
    "Khi người dùng đọc số điện thoại hoặc mã bưu phẩm mà có hiện tượng nói ngập ngừng, đọc lặp đầu số hoặc đọc lại từ đầu "
    "(ví dụ: 'số điện thoại của tôi là 09 093 45 43 22 69', '098... à 0981234567'): Hãy thông minh hiểu đây là người dùng đọc lại/sửa lại, "
    "tự động trích xuất đúng số điện thoại 10 chữ số hoàn chỉnh (trong ví dụ trên là 09345432269), "
    "tuyệt đối không ghép nối các đoạn lặp lại thành dãy số thừa (12, 13 chữ số). "
    "- Các chữ số giống nhau đứng liền kề (ví dụ: '77', '88', '11', '00', 'bảy bảy', 'tám tám', 'một một', 'không không') là các chữ số thực sự hợp lệ của số điện thoại, tuyệt đối không được coi là lặp từ để xóa bỏ. "
    "- Nếu chuỗi số điện thoại người dùng cung cấp không đủ 10 số hoặc không rõ ràng, hãy nhắc lại lịch sự và hỏi lại để người dùng xác nhận. "
    "BẮT BUỘC: Khi tạo bảng, mã nguồn hoặc biểu đồ, luôn phải nói trước 1 đến 2 câu tóm tắt ý chính "
    "hoặc kết luận quan trọng bằng lời văn tự nhiên (đặc biệt khi dùng bảng thì phải tóm tắt nội dung so sánh trước khi vẽ bảng, "
    "vì hệ thống âm thanh chỉ đọc câu tóm tắt này và bỏ qua nội dung chi tiết dạng bảng hay mã); "
    "tuyệt đối không tự viết các câu như 'Nội dung đã được hiển thị trên màn hình', 'Bảng đã được hiển thị trên màn hình' hay 'Mã nguồn đã được hiển thị trên màn hình'. "
    "Không đọc tên hàm hay biểu thức code trong câu nói, chỉ mô tả bằng lời. "
    "Quy cách vẽ bảng và biểu đồ: "
    "- Bảng dùng định dạng bảng markdown. "
    "- Khi người dùng yêu cầu biểu đồ: vẽ biểu đồ thanh ngang ngắn gọn trong khối ``` theo dạng: "
    "Tên mục: [===] (giá trị), dùng từ 3 đến 10 dấu '=' hoặc '#' trong cặp ngoặc vuông [], "
    "tuyệt đối không in chuỗi ký tự đặc biệt quá dài và không vẽ cột dọc ASCII; "
    "hoặc có thể trình bày thành bảng số liệu markdown. "
    "Viết công thức bằng lời, không dùng ký hiệu LaTeX. "
    "Chỉ dùng chữ cái tiếng Việt và tiếng Anh, không dùng chữ Hán, Nhật hay Hàn."
)


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

    async def _stream_with_client(c: httpx.AsyncClient) -> AsyncIterator[str]:
        async with c.stream("POST", url, json=body) as r:
            if r.status_code != 200:
                await r.aread()
                raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:200]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    yield piece

    if client is not None:
        async for piece in _stream_with_client(client):
            yield piece
    else:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as new_client:
            async for piece in _stream_with_client(new_client):
                yield piece


# ==============================================================================
# 2. Bộ lọc văn bản sang TTS (SpeechFilter)
# ==============================================================================
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")
URL_RE = re.compile(r"https?://[^\s<>()]+")
MARK_RE = re.compile(r"(\*\*|__|\*|`|~~)")
PREFIX_RE = re.compile(r"^(#{1,6}\s+|[-*+•]\s+|\d+[.)]\s+|>\s*)")
RULE_RE = re.compile(r"^[-*_=\s]{3,}$")
SENT_RE = re.compile(r"(?<=[.!?…])(?<!\.\.)\s+")
MATH_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$")
CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\([^()\n]*\)")

CJK_NOTE = "Một phần câu trả lời không đọc được, xem chi tiết trên màn hình."
TABLE_NOTE = "Chi tiết dạng bảng đã được hiển thị trên màn hình."
CODE_NOTE = "Phần mã đã được hiển thị trên màn hình."
REMAINDER_NOTE = "Phần còn lại đã được hiển thị trên màn hình."

CODE_LANGS = {
    "python", "py", "javascript", "js", "typescript", "ts",
    "java", "cpp", "c", "csharp", "cs", "go", "rust",
    "php", "ruby", "sql", "bash", "sh", "powershell", "ps1",
    "json", "yaml", "yml", "xml", "html", "css", "dockerfile", "toml",
}


DOUBLE_HAM_RE = re.compile(r"\bhàm\s+hàm\b", re.IGNORECASE)
SPACES_RE = re.compile(r"\s+")


def has_cjk(text: str) -> bool:
    """Kiểm tra sự xuất hiện của ký tự CJK (Hán, Nhật, Triều Tiên)."""
    return CJK_RE.search(text) is not None


def clean_inline(text: str) -> str:
    """Làm sạch ký hiệu Markdown, emoji, URL và biểu thức mã trong một câu."""
    s = MATH_RE.sub("công thức trên màn hình", text)
    s = CALL_RE.sub(r"hàm \1", s)
    s = DOUBLE_HAM_RE.sub("hàm", s)
    s = LINK_RE.sub(r"\1", s)
    s = PLACEHOLDER_RE.sub("", s)
    s = URL_RE.sub("đường dẫn trên màn hình", s)
    s = EMOJI_RE.sub("", s)
    s = MARK_RE.sub("", s)
    return SPACES_RE.sub(" ", s).strip()


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

        if self.kind == "table":
            if not self.in_table:
                self.in_table = True
                self._push_note(TABLE_NOTE, out)
            return ""

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

        if has_cjk(sent):
            self.cjk_hits.append(sent)
            self._push_note(CJK_NOTE, out)
            return

        sent = clean_inline(sent)
        if not re.search(r"\w", sent):
            return

        if end and not re.search(r"[.!?…:;,]$", sent):
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

        self.last_sent = sent
        self.sent_count += 1
        self.total_chars += len(sent)
        out.append(sent)


# ==============================================================================
# 3. Tổng hợp giọng nói (Zero-TTS)
# ==============================================================================
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


def split_for_tts(text: str, limit: int = MAX_TTS_CHARS) -> list[str]:
    """Tách câu văn bản dài tại dấu phẩy hoặc khoảng trắng gần nhất để vừa giới hạn ký tự của TTS."""
    parts: list[str] = []
    while len(text) > limit:
        cut = max(text.rfind(",", 0, limit), text.rfind(" ", 0, limit))
        if cut <= 0:
            cut = limit
        parts.append(text[:cut + 1].strip())
        text = text[cut + 1:].strip()
    if text:
        parts.append(text)
    return parts


@dataclass
class TtsResult:
    """Kết quả tổng hợp âm thanh dạng PCM thô kèm thông số audio và mốc thời gian."""
    pcm: bytes = b""
    sample_rate: int = 48000
    channels: int = 1
    bits: int = 16
    first_byte_at: float | None = None
    total_sec: float = 0.0


async def synth(text: str, *, base_url: str, voice: str, client: httpx.AsyncClient) -> TtsResult:
    """Gửi yêu cầu tổng hợp giọng nói, bóc bỏ header WAV 44 byte và trả về toàn bộ mảng PCM."""
    if len(text) > 500:
        raise ValueError(f"Đoạn văn đưa vào TTS quá dài ({len(text)} ký tự); cần qua split_for_tts() trước.")

    body = {**DEFAULT_PARAMS, "text": text, "voice": voice}
    buf = bytearray()
    t0 = time.perf_counter()
    t_first = None

    async with client.stream("POST", base_url.rstrip("/") + "/tts/stream", json=body) as r:
        if r.status_code != 200:
            raw = await r.aread()
            raise RuntimeError(f"TTS HTTP {r.status_code}: {raw[:200]!r}")
        async for chunk in r.aiter_bytes():
            buf += chunk
            if t_first is None and len(buf) > 44:
                t_first = time.perf_counter() - t0
    total = time.perf_counter() - t0

    if buf[:4] != b"RIFF" or buf[36:40] != b"data":
        raise RuntimeError("Header WAV không đúng định dạng chuẩn 44 byte.")

    channels, rate = struct.unpack("<HI", buf[22:28])
    bits = struct.unpack("<H", buf[34:36])[0]
    pcm = bytes(buf[44:])
    pcm = pcm[: len(pcm) // 2 * 2]
    return TtsResult(
        pcm=pcm, sample_rate=rate, channels=channels, bits=bits,
        first_byte_at=t_first, total_sec=total
    )


async def synth_stream(text: str, *, base_url: str, voice: str, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    """Phát luồng trực tiếp các chunk PCM16 (48 kHz mono) nhận từ server TTS, bóc bỏ 44 byte header."""
    body = {**DEFAULT_PARAMS, "text": text, "voice": voice}
    head = bytearray()

    async with client.stream("POST", base_url.rstrip("/") + "/tts/stream", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"TTS HTTP {r.status_code}")
        async for chunk in r.aiter_bytes():
            if len(head) < 44:
                take = 44 - len(head)
                head += chunk[:take]
                chunk = chunk[take:]
                if len(head) == 44 and (head[:4] != b"RIFF" or head[36:40] != b"data"):
                    raise RuntimeError("Header WAV không đúng định dạng chuẩn.")
            if chunk:
                yield chunk


# ==============================================================================
# 4. Điều phối quy trình lượt thoại (Pipeline Producer - Consumer)
# ==============================================================================
_END = object()


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
    q: asyncio.Queue = asyncio.Queue()
    flt = SpeechFilter()
    raw_parts: list[str] = []

    def mark(name: str) -> None:
        metrics.setdefault(name, time.perf_counter() - t0)

    async def producer() -> None:
        """Đọc stream từ LLM, gửi raw text lên giao diện và đẩy các câu đã lọc vào hàng đợi."""
        try:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": user_text},
            ]
            async for piece in stream_reply(msgs, url=llm_url, client=client):
                mark("first_content")
                raw_parts.append(piece)
                await send_raw(piece)
                for s in flt.feed(piece):
                    mark("first_sentence")
                    await q.put(s)
            for s in flt.flush():
                mark("first_sentence")
                await q.put(s)
        finally:
            await q.put(_END)

    async def consumer() -> None:
        """Lấy từng câu từ hàng đợi, gửi qua Zero-TTS stream và đẩy chunk PCM tới client."""
        while (sent := await q.get()) is not _END:
            for piece in split_for_tts(sent):
                async for pcm in synth_stream(piece, base_url=tts_url, voice=voice, client=client):
                    mark("first_audio")
                    await send_audio(pcm)

    prod = asyncio.create_task(producer())
    cons = asyncio.create_task(consumer())
    try:
        await asyncio.gather(prod, cons)
    finally:
        for t in (prod, cons):
            if not t.done():
                t.cancel()
        await asyncio.gather(prod, cons, return_exceptions=True)

    return "".join(raw_parts)

