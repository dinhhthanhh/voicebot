"""app/tts.py - Giao tiếp với dịch vụ Zero-TTS qua HTTP streaming.
Hỗ trợ:
- Tách câu dài thành các đoạn nhỏ dưới 450 ký tự (split_for_tts).
- Tổng hợp toàn bộ câu một lần cho CLI (synth).
- Phát luồng từng chunk PCM16 48 kHz cho Web client (synth_stream).
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

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
    pcm = pcm[: len(pcm) // 2 * 2]  # Đảm bảo số byte là chẵn
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
