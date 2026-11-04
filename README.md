# Voicebot (STT → LLM → SpeechFilter → TTS)

Hệ thống trợ lý giọng nói đàm thoại thời gian thực (Full-duplex / Half-duplex với cơ chế ngắt lời Barge-in), tối ưu độ trễ thấp và tích hợp bộ lọc văn bản thông minh.

## 🌟 Tính năng nổi bật

- **Kiến trúc luồng kép (Dual-channel Streaming):**
  - **Kênh hiển thị (Màn hình):** Hiển thị tức thì toàn văn câu trả lời từ LLM (kể cả bảng Markdown, mã nguồn, danh sách).
  - **Kênh âm thanh (Thoại):** Xử lý song song qua hàng đợi Producer-Consumer, nhả âm thanh ngay khi câu đầu tiên sẵn sàng, đạt **Time-to-First-Audio ~0.7s**.
- **Cơ chế ngắt lời (Barge-in):** Cho phép người dùng chen ngang ngắt lời bot bất cứ lúc nào; hệ thống lập tức hủy luồng LLM và TTS đang phát dở trên cả server và trình duyệt.
- **Bộ lọc văn bản thông minh (`SpeechFilter`):**
  - Tách câu hoàn chỉnh thời gian thực theo dấu câu.
  - Bỏ qua khối code và bảng biểu chi tiết, thay thế bằng câu thông báo điều hướng ngắn gọn.
  - Tự động chuẩn hoá biểu thức toán học, đường dẫn URL, tên hàm (`range()` → `hàm range`).
  - Phát hiện và xử lý ký tự CJK (chữ Hán/Hàn/Nhật) để tránh đọc lỗi.
  - Giới hạn độ dài và số lượng câu đọc (capping) chống dông dài.
- **Giao diện Web nhẹ & Mạch chống dội âm (Anti-feedback):**
  - Sử dụng Web Audio API thuần, Linear Resample 16 kHz.
  - Bộ lọc `GainNode(gain=0)` chống dội tiếng micro ra loa máy tính.
- **Bản CLI độc lập:** Hỗ trợ kiểm thử file WAV, đo đạc chi tiết các mốc độ trễ ($T_1, T_2, T_3$).

---

## 🏗️ Cấu trúc dự án

```text
voice-bot/
├── app/
│   ├── __init__.py
│   ├── llm.py             # Prompt V2 & SSE Stream từ LLM (vLLM)
│   ├── pipeline.py        # Producer-Consumer điều phối luồng thoại và Barge-in
│   ├── stt.py             # Resample 16kHz & WebSocket streaming tới Qwen3-ASR
│   ├── text_clean.py      # SpeechFilter làm sạch và tách câu cho TTS
│   └── tts.py             # Split text & streaming PCM từ Zero-TTS
├── data/                  # Mẫu âm thanh test và logs kiểm nghiệm
├── static/
│   └── index.html         # Web UI Voicebot (Web Audio API, streaming player 48kHz)
├── test/                  # Các kịch bản kiểm thử tải song song và WebSocket
├── server.py              # FastAPI WebSocket server chính
├── voicebot_cli.py        # CLI chạy kiểm thử file WAV tuần tự
├── pyproject.toml         # Cấu hình dự án (uv / pip)
├── requirements.txt       # Dependencies
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Cài đặt môi trường

Sử dụng `uv` (khuyên dùng):
```powershell
uv sync
```
Hoặc dùng `pip`:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Chạy Web Server

Khởi động server tại cổng 8000:
```powershell
uv run uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
Truy cập trình duyệt: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 3. Chạy qua CLI với file WAV

```powershell
uv run python voicebot_cli.py data/test_3s.wav --out data/voicebot_out.wav
```
