"""app/stt.py - Xử lý chuyển đổi âm thanh và giao tiếp với Qwen3-ASR WebSocket.
Hỗ trợ:
- Chuyển đổi định dạng âm thanh sang PCM16LE mono 16 kHz.
- Phiên làm việc streaming theo thời gian thực (SttSession) cho Web client.
- Gọi nhận dạng đơn lẻ (transcribe_pcm16_16k) cho CLI và các bài test.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
import websockets

PING_EVERY_SEC = 30
TARGET_SR = 16000


def to_16k_mono_pcm16(raw_pcm: bytes, src_rate: int, src_channels: int) -> bytes:
    """Chuyển đổi PCM16LE bất kỳ tần số lấy mẫu và số kênh sang PCM16LE mono 16 kHz."""
    x = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32)
    if src_channels > 1:
        x = x.reshape(-1, src_channels).mean(axis=1)
    if src_rate != TARGET_SR:
        if src_rate > TARGET_SR:
            # Low-pass sinc filter để tránh hiện tượng aliasing khi downsample
            n = 255
            t = np.arange(n) - (n - 1) / 2
            h = np.sinc(2 * (7600 / src_rate) * t) * np.hamming(n)
            x = np.convolve(x, h / h.sum(), mode="same")
        pos = np.arange(0, len(x) - 1, src_rate / TARGET_SR)
        x = np.interp(pos, np.arange(len(x)), x)
    return np.clip(np.rint(x), -32768, 32767).astype("<i2").tobytes()


@dataclass
class SttResult:
    """Kết quả nhận dạng giọng nói kèm các mốc đo thời gian."""
    final_text: str = ""
    first_partial_at: float | None = None
    final_at: float | None = None
    final_abs: float | None = None
    error: dict[str, Any] | None = None
    events: list[tuple[float, str, str]] = field(default_factory=list)


class SttError(RuntimeError):
    """Ngoại lệ khi dịch vụ STT trả về thông báo lỗi (ví dụ 429 OVERLOADED)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(str(payload))
        self.payload = payload


class SttSession:
    """Quản lý một phiên STT streaming qua WebSocket cho trình duyệt."""

    def __init__(self, url: str, on_event: Callable[[str, Any], Awaitable[None]]) -> None:
        self.url = url
        self.on_event = on_event
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.final_text = ""
        self.error: dict[str, Any] | None = None
        self._done = asyncio.Event()
        self._reader: asyncio.Task | None = None

    async def open(self) -> None:
        """Mở kết nối WebSocket và gửi gói tin khởi tạo `start`."""
        self.ws = await websockets.connect(self.url, max_size=10 * 2**20)
        try:
            start = {
                "type": "start",
                "request_id": str(uuid.uuid4()),
                "language": "Vietnamese",
                "sample_rate": TARGET_SR,
                "encoding": "pcm_s16le",
                "channels": 1,
            }
            await self.ws.send(json.dumps(start))
            ready = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=15.0))
            if ready.get("type") == "error":
                raise SttError(ready)
        except Exception:
            await self.ws.close()
            raise
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        """Đọc kết quả từ server STT và kích hoạt callback sự kiện."""
        try:
            async for raw in self.ws:
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

    async def send_frame(self, pcm: bytes) -> None:
        """Gửi một khung dữ liệu âm thanh PCM16LE mono 16 kHz."""
        if self.ws is not None:
            await self.ws.send(pcm)

    async def finish(self, timeout: float = 10.0) -> str:
        """Gửi tín hiệu kết thúc thu âm và chờ nhận kết quả final cuối cùng."""
        if self.ws is not None:
            await self.ws.send(json.dumps({"type": "finish"}))
            await asyncio.wait_for(self._done.wait(), timeout)
            await self.ws.close()
        if self.error and not self.final_text:
            raise SttError(self.error)
        return self.final_text

    async def abort(self) -> None:
        """Hủy bỏ phiên nhận dạng lập tức (khi ngắt lời/barge-in)."""
        if self._reader and not self._reader.done():
            self._reader.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        self._done.set()


async def transcribe_pcm16_16k(
    pcm: bytes, *, url: str, chunk_sec: float = 0.3, realtime: bool = True
) -> SttResult:
    """Gửi toàn bộ mảng dữ liệu PCM qua WebSocket STT và trả về kết quả kèm các mốc thời gian."""
    res = SttResult()
    frame_bytes = int(TARGET_SR * chunk_sec) * 2
    start = {
        "type": "start",
        "request_id": str(uuid.uuid4()),
        "language": "Vietnamese",
        "sample_rate": TARGET_SR,
        "encoding": "pcm_s16le",
        "channels": 1,
    }

    async with websockets.connect(url, max_size=10 * 2**20) as ws:
        await ws.send(json.dumps(start))
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
        if ready.get("type") == "error":
            res.error = ready
            return res

        t0 = time.perf_counter()
        finish_at: float | None = None
        stop = asyncio.Event()

        async def sender() -> None:
            nonlocal finish_at
            last_ping = time.perf_counter()
            for i in range(0, len(pcm), frame_bytes):
                if stop.is_set():
                    return
                await ws.send(pcm[i:i + frame_bytes])
                if realtime:
                    await asyncio.sleep(chunk_sec)
                if time.perf_counter() - last_ping >= PING_EVERY_SEC:
                    await ws.send(json.dumps({"type": "ping"}))
                    last_ping = time.perf_counter()
            if stop.is_set():
                return
            finish_at = time.perf_counter()
            await ws.send(json.dumps({"type": "finish"}))

        async def receiver() -> None:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                now = time.perf_counter() - t0
                if kind == "partial":
                    text = msg.get("text") or ""
                    if text and res.first_partial_at is None:
                        res.first_partial_at = now
                    res.events.append((now, "partial", text))
                elif kind == "final":
                    res.final_text = msg.get("text", "")
                    res.final_at = (time.perf_counter() - finish_at) if finish_at else None
                    res.final_abs = time.perf_counter()
                    res.events.append((now, "final", res.final_text))
                elif kind == "vad":
                    res.events.append((now, "vad", msg.get("state", "")))
                elif kind == "error":
                    res.error = msg
                    stop.set()
                    return

        total_sec = len(pcm) / 2 / TARGET_SR
        results = await asyncio.wait_for(
            asyncio.gather(sender(), receiver(), return_exceptions=True),
            timeout=total_sec + 60.0,
        )
        for r in results:
            if isinstance(r, Exception) and not (res.error and isinstance(r, websockets.ConnectionClosed)):
                raise r
    return res
