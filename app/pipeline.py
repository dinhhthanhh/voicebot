"""app/pipeline.py - Điều phối quy trình xử lý một lượt thoại:
LLM Stream (Producer) -> SpeechFilter (Lọc câu) -> Queue -> TTS Stream (Consumer).
Hỗ trợ cơ chế ngắt lời (Barge-in) thông qua việc hủy asyncio.Task.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

import httpx

from app.llm import SYSTEM_PROMPT, stream_reply
from app.text_clean import SpeechFilter
from app.tts import split_for_tts, synth_stream

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
            # Luôn gửi tín hiệu kết thúc để consumer thoát an toàn, kể cả khi producer gặp lỗi
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
        # Nếu task bên ngoài bị hủy (Barge-in), dọn dẹp hủy cả producer và consumer
        for t in (prod, cons):
            if not t.done():
                t.cancel()
        await asyncio.gather(prod, cons, return_exceptions=True)

    return "".join(raw_parts)
