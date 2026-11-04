import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # để import được app/
from app.llm import SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2, stream_reply  # noqa: E402
from app.text_clean import SpeechFilter  # noqa: E402


async def one(prompt: str, args, log, client: httpx.AsyncClient | None = None) -> None:
    messages = [{"role": "user", "content": prompt}]
    if not args.no_system:
        sp = SYSTEM_PROMPT_V1 if args.v1 else SYSTEM_PROMPT_V2
        messages.insert(0, {"role": "system", "content": sp})
    flt, raw, sents = SpeechFilter(), [], []
    t0, t_first = time.perf_counter(), None
    pieces = []
    async for piece in stream_reply(messages, url=args.url, think=args.think, client=client):
        t = time.perf_counter() - t0
        pieces.append(t)
        if t_first is None:
            t_first = t
        raw.append(piece)
        sents += [(t, s) for s in flt.feed(piece)]
    t_end = time.perf_counter() - t0
    sents += [(t_end, s) for s in flt.flush()]

    log(f"\n=== PROMPT: {prompt}")
    log("--- THÔ (chỉ content) ---")
    log("".join(raw))
    log("--- CÂU ĐƯA VÀO TTS (thời điểm sẵn sàng) ---")
    for t, s in sents:
        log(f"[{t:5.2f}s] {s}")
    tf = f"{t_first:.2f}s" if t_first is not None else "-"
    ts = f"{sents[0][0]:.2f}s" if sents else "-"
    log(f"--- content đầu: {tf} | câu đầu: {ts} | tổng: {t_end:.2f}s | "
        f"{len(sents)} câu, {len(''.join(raw))} ký tự")
    if pieces:
        log(f"--- số mẩu={len(pieces)} | đầu={pieces[0]:.2f}s | cuối={pieces[-1]:.2f}s")
    if flt.cjk_hits:
        log(f"--- CJK phát hiện ({len(flt.cjk_hits)}): {flt.cjk_hits}")


async def amain(args) -> None:
    lines: list[str] = []

    def log(s: str) -> None:
        print(s)
        lines.append(s)

    if args.file:
        prompts = [ln.strip() for ln in
                   Path(args.file).read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    else:
        prompts = [args.prompt]

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for p in prompts:
            await one(p, args, log, client=client)

    if args.out:  # ghi UTF-8 trực tiếp, tránh lỗi mã hoá khi chuyển hướng trong PowerShell
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print("Đã ghi", args.out)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?")
    p.add_argument("--file", help="file .txt, mỗi dòng một prompt (lưu UTF-8)")
    p.add_argument("--url", default=os.environ.get("LLM_URL"))
    p.add_argument("--think", action="store_true", help="bật suy luận (reasoning_effort=low)")
    p.add_argument("--no-system", action="store_true", help="không gửi system prompt")
    p.add_argument("--out")
    p.add_argument("--v1", action="store_true", help="dùng system prompt v1 (không markdown)")
    a = p.parse_args()
    if not a.url:
        sys.exit("Thiếu --url (hoặc biến môi trường LLM_URL)")
    if not a.prompt and not a.file:
        sys.exit("Cần một prompt hoặc --file")
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
