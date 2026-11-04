"""app/llm.py - Giao tiếp với dịch vụ vLLM (Qwen 2.5) qua SSE stream.
Bao gồm:
- SYSTEM_PROMPT: Hướng dẫn văn phong, quy tắc tóm tắt trước bảng/mã, vẽ biểu đồ thanh ngang.
- stream_reply: Generator bất đồng bộ nhận từng mẩu delta văn bản (bỏ qua reasoning tokens).
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

# System prompt chuẩn cho Voicebot (văn phong đàm thoại, tóm tắt trước bảng/code, không tự báo màn hình)
SYSTEM_PROMPT = (
    "Bạn là trợ lý giọng nói tiếng Việt. Câu trả lời của bạn vừa được đọc to thành tiếng, "
    "vừa hiển thị trên màn hình. Viết ngắn gọn như văn nói, đủ câu, không dùng emoji. "
    "Tuân thủ đúng yêu cầu định dạng của người dùng: nếu họ yêu cầu gạch đầu dòng, bảng, "
    "danh sách, mã nguồn hay biểu đồ thì phải làm đúng như vậy, không từ chối. "
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

# Bản v1 ngắn gọn cũ để phục vụ kiểm thử so sánh
SYSTEM_PROMPT_V1 = (
    "Bạn là trợ lý giọng nói tiếng Việt. Trả lời ngắn gọn, tự nhiên như văn nói. "
    "Không dùng emoji và không dùng định dạng markdown. Nếu nội dung cần bảng hoặc "
    "danh sách dài, chỉ tóm tắt trong một đến hai câu."
)
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT


async def stream_reply(
    messages: list[dict[str, str]],
    *,
    url: str,
    think: bool = False,
    timeout: float = 60.0,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """Gửi yêu cầu hội thoại tới OpenAI-compatible API và phát luồng từng delta content."""
    body: dict = {"messages": messages, "temperature": 0, "stream": True}
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
