"""voicebot_cli.py - Chạy thử nghiệm một file WAV qua quy trình tuần tự:
STT (nhận dạng) -> LLM (sinh câu trả lời) -> SpeechFilter (lọc câu) -> TTS (tổng hợp tiếng).
Xuất ra file WAV kết quả và bảng đo độ trễ các mốc thời gian.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import wave

import httpx

from stt import to_16k_mono_pcm16, transcribe_pcm16_16k
from tts import SYSTEM_PROMPT, SpeechFilter, split_for_tts, stream_reply, synth


def load_wav_as_16k_mono_pcm16(path: str) -> bytes:
    """Đọc file WAV 16-bit và chuyển đổi sang chuẩn PCM16LE mono 16 kHz."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit(f"Chỉ hỗ trợ WAV 16-bit; file này {8 * w.getsampwidth()}-bit")
        raw = w.readframes(w.getnframes())
        rate, ch = w.getframerate(), w.getnchannels()
    return to_16k_mono_pcm16(raw, rate, ch)


def save_wav(path: str, pcm: bytes, rate: int, channels: int, bits: int) -> None:
    """Ghi dữ liệu PCM ra file WAV."""
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(rate)
        w.writeframes(pcm)


async def run(args: argparse.Namespace) -> None:
    t_mark: dict[str, float] = {}

    def mark(name: str) -> None:
        t_mark[name] = time.perf_counter()

    mark("T_start")
    pcm16k = load_wav_as_16k_mono_pcm16(args.wav)

    # 1. Nhận dạng giọng nói (STT)
    print("== STT ==")
    stt_res = await transcribe_pcm16_16k(pcm16k, url=args.stt_url, realtime=not args.fast)
    t_mark["T1_final"] = stt_res.final_abs or time.perf_counter()
    if stt_res.error:
        sys.exit(f"STT lỗi: {stt_res.error}")
    user_text = stt_res.final_text
    print("Người dùng nói:", user_text)
    if not user_text.strip():
        sys.exit("STT không nhận được nội dung nào (final rỗng)")

    # 2. Sinh câu trả lời (LLM) và lọc câu (SpeechFilter)
    print("\n== LLM + lọc câu + TTS ==")
    flt = SpeechFilter()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    out_pcm, out_rate, out_ch, out_bits = bytearray(), 48000, 1, 16
    first_tts_byte_at = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        sentences: list[str] = []
        async for piece in stream_reply(messages, url=args.llm_url, client=client):
            sentences += flt.feed(piece)
        sentences += flt.flush()
        mark("T2_llm_done")

        # 3. Tổng hợp giọng nói (TTS) từng câu
        idx = 0
        for sent in sentences:
            for piece in split_for_tts(sent):
                idx += 1
                print(f"  câu/đoạn {idx}: {piece}")
                t_call = time.perf_counter()
                tts_res = await synth(piece, base_url=args.tts_url, voice=args.voice, client=client)
                if first_tts_byte_at is None:
                    first_tts_byte_at = t_call + (tts_res.first_byte_at or 0.0)
                    t_mark["T3_first_audio"] = first_tts_byte_at
                out_rate, out_ch, out_bits = tts_res.sample_rate, tts_res.channels, tts_res.bits
                out_pcm += tts_res.pcm
                print(
                    f"    -> TTS byte đầu {tts_res.first_byte_at:.2f}s, "
                    f"tổng {tts_res.total_sec:.2f}s, {len(tts_res.pcm) / 2 / out_rate:.2f}s audio"
                )

    save_wav(args.out, bytes(out_pcm), out_rate, out_ch, out_bits)
    bot_text = " ".join(sentences)
    print("\nBot trả lời (đã lọc, sẽ đọc):", bot_text)
    print("Đã lưu:", args.out)

    # 4. In bảng đo độ trễ
    print("\n---- BẢNG ĐỘ TRỄ (giây, kể từ T_start) ----")
    for name in ("T1_final", "T2_llm_done", "T3_first_audio"):
        if name in t_mark:
            dt = t_mark[name] - t_mark["T_start"]
            print(f"  {name}: {dt:.2f}s")
    if "T3_first_audio" in t_mark and "T1_final" in t_mark:
        print(f"  Time-to-first-audio (từ khi STT final): {t_mark['T3_first_audio'] - t_mark['T1_final']:.2f}s")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Chạy thử nghiệm Voicebot qua dòng lệnh với file WAV.")
    p.add_argument("wav", help="Đường dẫn file WAV 16-bit đầu vào")
    p.add_argument("--stt-url", default=os.environ.get("STT_WS_URL", "wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream"))
    p.add_argument("--llm-url", default=os.environ.get("LLM_URL", "https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions"))
    p.add_argument("--tts-url", default=os.environ.get("TTS_BASE_URL", "https://ai.vnpost.vn/zero-tts"))
    p.add_argument("--voice", default="maichi", help="Giọng đọc TTS (mặc định maichi)")
    p.add_argument("--out", default="data/voicebot_out.wav", help="File WAV kết quả")
    p.add_argument("--fast", action="store_true", help="Gửi audio nhanh, không giả lập thời gian thực")
    args = p.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
