import argparse
import asyncio
import json
import os
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.stt import to_16k_mono_pcm16 as resample_to_16k

PING_EVERY_SEC = 30  # heartbeat mức ứng dụng, theo tài liệu
TARGET_SR = 16000


def to_16k_mono_pcm16(path: str) -> bytes:
    """Đọc WAV 16-bit (tần số/số kênh bất kỳ) -> PCM16LE mono 16 kHz."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit(f"Chỉ hỗ trợ WAV 16-bit; file này {8 * w.getsampwidth()}-bit")
        ch, sr = w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    return resample_to_16k(raw, sr, ch)


async def run(args) -> int:
    pcm = to_16k_mono_pcm16(args.wav)
    # số mẫu * 2 byte => luôn chẵn (tài liệu yêu cầu frame có số byte chẵn)
    frame_bytes = int(16000 * args.chunk_sec) * 2
    st = {"first_partial": None, "partials": 0, "final": None,
          "final_at": None, "finish_at": None, "error": None}
    t0 = time.perf_counter()

    def now() -> float:
        return time.perf_counter() - t0

    # 'start' PHẢI là message đầu tiên; chỉ gửi các giá trị tài liệu cho phép
    start = {"type": "start", "request_id": str(uuid.uuid4()),
             "language": "Vietnamese", "sample_rate": 16000,
             "encoding": "pcm_s16le", "channels": 1}
    if args.vad:
        start["vad"] = args.vad  # ở start, vad là STRING (silero/webrtc/off)

    async with websockets.connect(args.url, max_size=10 * 2**20) as ws:
        await ws.send(json.dumps(start))
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        if ready.get("type") == "error":
            print("LỖI khi start:", json.dumps(ready, ensure_ascii=False))
            return 1
        # ở 'ready', vad là OBJECT cấu hình thực tế -> xem server có silero không
        print("ready:", json.dumps(ready, ensure_ascii=False))
        t0 = time.perf_counter()  # mốc 0 = lúc bắt đầu gửi audio

        async def sender():
            last_ping = time.perf_counter()
            for i in range(0, len(pcm), frame_bytes):
                await ws.send(pcm[i:i + frame_bytes])  # binary frame, không base64
                if not args.fast:
                    await asyncio.sleep(args.chunk_sec)  # giả lập nói realtime
                if time.perf_counter() - last_ping >= PING_EVERY_SEC:
                    await ws.send(json.dumps({"type": "ping"}))
                    last_ping = time.perf_counter()
            st["finish_at"] = time.perf_counter()
            await ws.send(json.dumps({"type": "finish"}))  # luôn finish rồi chờ final
            print(f"[{now():6.2f}s] >> finish")

        async def receiver():
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "vad":
                    print(f"[{now():6.2f}s] vad     {msg.get('state')}")
                elif kind == "partial": 
                    st["partials"] += 1
                    if msg.get("text") and st["first_partial"] is None:
                        st["first_partial"] = now()
                    print(f"[{now():6.2f}s] partial #{msg.get('chunk_id')}: {msg.get('text')}")
                elif kind == "final":
                    st["final"], st["final_at"] = msg.get("text", ""), time.perf_counter()
                    print(f"[{now():6.2f}s] FINAL: {st['final']}")
                elif kind == "pong":
                    print(f"[{now():6.2f}s] pong")
                elif kind == "error":
                    st["error"] = msg
                    print(f"[{now():6.2f}s] ERROR {msg.get('status_code')} "
                          f"{msg.get('error_code')}: {msg.get('message')}")
                    return

        total_sec = len(pcm) / 2 / 16000
        results = await asyncio.wait_for(
            asyncio.gather(sender(), receiver(), return_exceptions=True),
            timeout=total_sec + 60)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, websockets.ConnectionClosed):
                print("Ngoại lệ:", type(r).__name__, r)
        print(f"close_code = {getattr(ws, 'close_code', None)}")

    print("---- TỔNG KẾT ----")
    print(f"audio: {total_sec:.1f}s | số partial: {st['partials']}")
    if st["first_partial"] is not None:
        print(f"partial đầu tiên sau {st['first_partial']:.2f}s kể từ khi gửi audio")
    if st["final_at"] and st["finish_at"]:
        print(f"final đến sau finish: {st['final_at'] - st['finish_at']:.2f}s")
    if st["error"]:
        code = st["error"].get("error_code")
        print("Có lỗi:", code, "(429 => thử lại sau)" if st["error"].get("status_code") == 429 else "")
        return 1
    return 0 if st["final"] is not None else 1


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # in tiếng Việt đúng trên Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--url", default=os.environ.get("STT_WS_URL"))
    p.add_argument("--vad", choices=["silero", "webrtc", "off"])
    p.add_argument("--chunk-sec", type=float, default=0.3)
    p.add_argument("--fast", action="store_true", help="gửi nhanh, không giả lập realtime")
    args = p.parse_args()
    if not args.url:
        sys.exit("Thiếu --url (hoặc biến môi trường STT_WS_URL)")
    try:
        sys.exit(asyncio.run(run(args)))
    except Exception as exc:  # lỗi handshake (404/403...) hiện rõ ở đây
        sys.exit(f"Không kết nối được: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
