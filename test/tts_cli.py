# tts_cli.py — Kiểm chứng ZeroTTS /tts/stream: lưu WAV hợp lệ + đo độ trễ
import argparse
import os
import struct
import sys
import time
import wave

import httpx

# Theo mẫu body /tts/stream trong Postman collection
DEFAULT_BODY = {
    "cfg_scale": 1.0, "audio_temperature": 0.8, "text_temperature": 1.0,
    "audio_topk": 25, "audio_topp": 0.95, "audio_repetition_penalty": 1.2,
    "min_frames": 4, "max_frames": 1500,
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--base-url", default=os.environ.get("TTS_BASE_URL"))
    p.add_argument("--voice", default="maichi")  # giọng trong mẫu Postman
    p.add_argument("--out", default="data/tts_out.wav")
    a = p.parse_args()
    if not a.base_url:
        sys.exit("Thiếu --base-url (hoặc biến môi trường TTS_BASE_URL)")

    body = {**DEFAULT_BODY, "text": a.text, "voice": a.voice}
    buf = bytearray()
    t0, t_first = time.perf_counter(), None
    with httpx.Client(timeout=httpx.Timeout(60.0)) as c:
        with c.stream("POST", a.base_url.rstrip("/") + "/tts/stream", json=body) as r:
            if r.status_code != 200:
                r.read()
                sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
            for chunk in r.iter_bytes():
                buf += chunk
                if t_first is None and len(buf) > 44:  # byte PCM đầu tiên (sau header)
                    t_first = time.perf_counter() - t0
    total = time.perf_counter() - t0

    # Header 44 byte, size giả: tự đọc thông số, kiểm tra chunk 'data' ở offset 36
    if buf[:4] != b"RIFF" or buf[36:40] != b"data":
        sys.exit("Header không phải WAV 44 byte như dự kiến, dừng để kiểm tra")
    channels, rate = struct.unpack("<HI", buf[22:28])
    bits = struct.unpack("<H", buf[34:36])[0]
    pcm = bytes(buf[44:])
    pcm = pcm[: len(pcm) // 2 * 2]  # số byte phải chẵn
    with wave.open(a.out, "wb") as w:  # wave tự ghi lại size đúng
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(rate)
        w.writeframes(pcm)

    dur = len(pcm) / 2 / channels / rate
    print(f"{rate} Hz, {channels} kênh, {bits}-bit | audio {dur:.2f}s")
    print(f"byte PCM đầu: {t_first:.3f}s | tổng: {total:.2f}s | RTF {total / dur:.2f}")
    print("Đã lưu:", a.out)


if __name__ == "__main__":
    main()