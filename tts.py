from __future__ import annotations

import asyncio
import json
import logging
import re
import struct
import time
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from num2words import num2words

logger = logging.getLogger("tts")

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
                raw = await r.aread()
                err_msg = raw.decode("utf-8", errors="replace")[:200]
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

    if client is not None:
        async for piece in _stream_with_client(client):
            yield piece
    else:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as new_client:
            async for piece in _stream_with_client(new_client):
                yield piece


EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")
URL_RE = re.compile(r"https?://[^\s<>()]+")
MARK_RE = re.compile(r"(\*\*|__|\*|`|~~)")
PREFIX_RE = re.compile(r"^(#{1,6}\s+|[-*+•]\s+|\d+[.)]\s+|>\s*)")
RULE_RE = re.compile(r"^[-*_=\s]{3,}$")
SENT_RE = re.compile(r"(?<=[.!?…])(?<!\.\.)\s+")
MATH_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$")

REMAINDER_NOTE = "Phần còn lại đã được hiển thị trên màn hình."


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
        self.kind: str | None = None
        self.fresh = True
        self.in_code = False
        self.in_table = False
        self.hold = ""
        self.last_sent = ""
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
                self.in_code = True
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

DIGITS_VI = {"0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn", "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"}
LETTERS_VI = {
    "A": "a", "B": "bê", "C": "xê", "D": "đê", "E": "e", "F": "ép", "G": "gờ", "H": "hát", "I": "i",
    "J": "giây", "K": "ca", "L": "lờ", "M": "mờ", "N": "nờ", "O": "o", "P": "pê", "Q": "quy", "R": "rờ",
    "S": "ét", "T": "tê", "U": "u", "V": "vê", "W": "vê kép", "X": "ích", "Y": "y", "Z": "dét"
}
UNITS_VI = {
    "kg": "ki lô gam", "g": "gam", "km": "ki lô mét", "m": "mét", "cm": "xen ti mét", "mm": "mi li mét",
    "đ": "đồng", "vnd": "đồng", "vnđ": "đồng", "k": "nghìn", "%": "phần trăm"
}

PHONE_PAREN_RE = re.compile(r"\(0(\d{2,3})\)[\s.-]*(\d{3,4})[\s.-]*(\d{3,4})\b")
PHONE_INTL_RE = re.compile(r"(?:\+84|(?<=\s)84)[\s.-]*(\d{2,3})[\s.-]*(\d{3,4})[\s.-]*(\d{3,4})\b")
PHONE_RE = re.compile(r"\b(0\d{2,3})[\s.-]+(\d{3,4})[\s.-]+(\d{3,4})\b")


def normalize_text_for_tts(text: str) -> str:
    text = PHONE_PAREN_RE.sub(r"0\1\2\3", text)
    text = PHONE_INTL_RE.sub(r"0\1\2\3", text)
    text = PHONE_RE.sub(r"\1\2\3", text)

    def _replace_token(match: re.Match) -> str:
        tok = match.group(0)
        clean_tok = tok.rstrip(".,;!?")
        punct = tok[len(clean_tok):]
        tok = clean_tok

        unit = ""
        m_unit = re.match(r"^([0-9][0-9.,/]*)\s*([a-zA-ZđĐ%]+)$", tok)
        if m_unit and m_unit.group(2).lower() in UNITS_VI:
            tok = m_unit.group(1)
            unit = " " + UNITS_VI[m_unit.group(2).lower()]

        if re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok):
            spoken = " ".join(
                DIGITS_VI.get(c, LETTERS_VI.get(c, c))
                for c in tok.upper()
                if c in DIGITS_VI or c in LETTERS_VI
            )
            return spoken + unit + punct

        if tok.isdigit():
            if (tok.startswith("0") and len(tok) > 1) or len(tok) >= 8:
                spoken = " ".join(DIGITS_VI[c] for c in tok)
            else:
                spoken = num2words(int(tok), lang="vi")
            return spoken + unit + punct

        if re.fullmatch(r"[1-9]\d{0,2}(?:[.,]\d{3})+", tok):
            clean_digits = re.sub(r"[.,]", "", tok)
            spoken = num2words(int(clean_digits), lang="vi")
            return spoken + unit + punct

        if re.fullmatch(r"\d+[.,]\d+", tok):
            parts = re.split(r"[.,]", tok)
            dec_part = parts[1]
            zero_count = len(dec_part) - len(dec_part.lstrip("0"))
            zero_prefix = "không " * zero_count
            dec_val = dec_part.lstrip("0")
            dec_spoken = (zero_prefix + num2words(int(dec_val), lang="vi")) if dec_val else zero_prefix.strip()
            spoken = num2words(int(parts[0]), lang="vi") + " phẩy " + dec_spoken
            return spoken + unit + punct

        return match.group(0)

    return re.sub(r"\b[A-Za-z0-9.,/]+(?:%|đ|Đ|kg|g|km|m|cm|mm|vnd|vnđ)?", _replace_token, text)


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


class IncompleteHeaderError(ValueError):
    """Header WAV chưa nhận đủ số byte để phân tích cấu trúc."""


def parse_wav_header(buf: bytes | bytearray) -> tuple[int, int, int, int]:
    """Phân tích RIFF WAV header và trả về (sample_rate, channels, bits, data_offset).
    Hỗ trợ cả standard 44-byte header và header có chunk phụ (JUNK, LIST, fact...).
    Kiểm tra chặt chẽ WAV format code (phải là PCM tiêu chuẩn code 1).
    """
    if len(buf) < 12:
        raise IncompleteHeaderError("Header chưa đủ 12 byte để kiểm tra định dạng RIFF WAVE.")
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise RuntimeError(f"Header dữ liệu không phải định dạng RIFF WAVE (nhận được {buf[:4]!r}..{buf[8:12]!r}).")

    offset = 12
    channels = 1
    sample_rate = 48000
    bits = 16
    audio_fmt = -1
    data_offset = -1

    while offset + 8 <= len(buf):
        chunk_id = bytes(buf[offset:offset + 4])
        chunk_len = struct.unpack("<I", buf[offset + 4:offset + 8])[0]
        offset += 8
        if chunk_id == b"fmt ":
            if offset + chunk_len > len(buf):
                raise IncompleteHeaderError("Chunk 'fmt ' chưa nhận đủ dữ liệu.")
            if chunk_len >= 14:
                audio_fmt, channels, sample_rate = struct.unpack("<HHI", buf[offset:offset + 8])
                if audio_fmt != 1:
                    raise RuntimeError(f"WAV format code là {audio_fmt}, chỉ hỗ trợ PCM tiêu chuẩn (code 1).")
                if chunk_len >= 16:
                    bits = struct.unpack("<H", buf[offset + 14:offset + 16])[0]
        elif chunk_id == b"data":
            data_offset = offset
            break
        offset += chunk_len + (chunk_len % 2)

    if data_offset == -1:
        raise IncompleteHeaderError("Chưa nhận đủ chunk 'data' trong header WAV.")
    if audio_fmt == -1:
        raise RuntimeError("Không tìm thấy chunk 'fmt ' trong header WAV.")

    return sample_rate, channels, bits, data_offset


async def synth_stream(text: str, *, base_url: str, voice: str, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    """Phát luồng trực tiếp các chunk PCM16 (48 kHz mono) nhận từ server TTS,
    bóc header WAV động và đảm bảo mỗi chunk phát đi có số byte chẵn (căn chỉnh 16-bit).
    """
    body = {**DEFAULT_PARAMS, "text": text, "voice": voice}
    head = bytearray()
    header_parsed = False
    pending_odd_byte = b""

    async with client.stream("POST", base_url.rstrip("/") + "/tts/stream", json=body) as r:
        if r.status_code != 200:
            raw = await r.aread()
            raise RuntimeError(f"TTS HTTP {r.status_code}: {raw[:200]!r}")
        async for chunk in r.aiter_bytes():
            if not header_parsed:
                head.extend(chunk)
                if len(head) >= 44 and b"data" in head:
                    try:
                        sample_rate, channels, bits, data_offset = parse_wav_header(head)
                        if (sample_rate, channels, bits) != (48000, 1, 16):
                            raise RuntimeError(
                                f"Định dạng TTS không đúng yêu cầu: {sample_rate} Hz, {channels} channels, {bits} bit "
                                "(kỳ vọng 48000 Hz, 1 channel, 16 bit PCM tiêu chuẩn)."
                            )
                        audio_data = bytes(head[data_offset:])
                        head.clear()
                        header_parsed = True
                        if audio_data:
                            if len(audio_data) % 2 != 0:
                                pending_odd_byte = audio_data[-1:]
                                audio_data = audio_data[:-1]
                            if audio_data:
                                yield audio_data
                        continue
                    except IncompleteHeaderError:
                        continue
                continue

            data = pending_odd_byte + chunk
            if len(data) % 2 != 0:
                pending_odd_byte = data[-1:]
                data = data[:-1]
            else:
                pending_odd_byte = b""
            if data:
                yield data


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
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    flt = SpeechFilter()
    raw_parts: list[str] = []

    def mark(name: str) -> None:
        metrics.setdefault(name, time.perf_counter() - t0)

    end_sent = False

    async def producer() -> None:
        """Đọc stream từ LLM, gửi raw text lên giao diện và đẩy các câu đã lọc vào hàng đợi."""
        nonlocal end_sent
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
            await q.put(_END)
            end_sent = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Lỗi trong quá trình sinh câu trả lời LLM: %s", exc)
            raise
        finally:
            if not end_sent and not q.full():
                try:
                    q.put_nowait(_END)
                except asyncio.QueueFull:
                    pass

    async def consumer() -> None:
        """Lấy từng câu từ hàng đợi, gửi qua Zero-TTS stream và đẩy chunk PCM tới client."""
        try:
            while (sent := await q.get()) is not _END:
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

