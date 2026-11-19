# Voicebot MVP (STT → LLM → SpeechFilter → TTS)

Hệ thống trợ lý giọng nói đàm thoại thời gian thực độ trễ thấp, hỗ trợ tiếng Việt hoàn chỉnh với cơ chế phát hiện ngắt lượt nói hai tầng (Silero VAD + Smart Turn v3.2 ONNX), luồng kép Dual-channel Streaming và đàm thoại hai chiều liên tục.

---

## 🌟 Tính năng nổi bật

### 1. Kiến trúc phân tầng chuẩn hóa & Luồng kép (Dual-channel Streaming)

- **Kênh hiển thị (Màn hình):** Stream toàn văn câu trả lời từ LLM tức thì (hỗ trợ đầy đủ Markdown, bảng biểu, danh sách, mã nguồn).
- **Kênh âm thanh (Thoại):** Xử lý song song qua mô hình Producer-Consumer không đồng bộ (`asyncio.Queue`). Tách câu thời gian thực và stream audio PCM 48 kHz (Mono 16-bit) lập tức từ Zero-TTS tới người dùng.
- **Connection Pool tái sử dụng:** Persistent HTTP Client dùng chung kết nối keep-alive cho cả LLM và TTS, tối ưu latency tối đa.

### 2. Xử lý & Chuẩn hóa văn bản Tiếng Việt 3 tầng

- **Tiền xử lý đầu vào STT (`text_processing.py`):**
  - **Chuẩn hóa số ghép tiếng Việt (`normalize_compound_numbers`):** Xử lý linh hoạt số ghép (*"hai mươi mốt"* → `21`, *"mười hai nghìn năm trăm"* → `12500`).
  - **Gom chuỗi số liên tiếp (`normalize_speech_input`):** Tự động nhận diện và gom các từ đọc số rời rạc liên tiếp thành chuỗi số hoàn chỉnh (hỗ trợ số điện thoại, CCCD, mã vận đơn bưu chính VNPost).
- **Bộ lọc luồng LLM (`SpeechFilter` trong `text_processing.py`):**
  - **Tách câu thời gian thực:** Nhận diện ranh giới câu tự nhiên theo dấu ngắt câu (`SENT_RE` bỏ qua các từ viết tắt `TP.`, `ThS.`, `BS.`, `Mr.`...).
  - **Lược bỏ thành phần trực quan:** Bỏ qua code block (` ``` `), bảng Markdown (`|...|`), làm sạch link Markdown, emoji, URL, công thức KaTeX/LaTeX.
  - **Capping an toàn:** Giới hạn tối đa 8 câu hoặc 600 ký tự thoại; tự động thêm thông báo thoại *"Phần còn lại đã được hiển thị trên màn hình."* khi đạt ngưỡng.
- **Hậu xử lý văn bản đọc TTS (`normalize_text_for_tts` trong `text_processing.py`):**
  - Phân tách hàng nghìn (`1.500`), số thập phân (`0,007` → *"không phẩy không không bảy"*), tiền tệ (`VNĐ`, `đồng`, `k` → *"nghìn"*).
  - Chuẩn hóa số điện thoại, ngày tháng `DD/MM/YYYY`, dịch vụ `24/7` (*"hai tư trên bảy"*), địa chỉ ngõ ngách `số 10/2` (*"số 10 xuyệt 2"*), đơn vị đo lường (`km`, `kg`, `ha`, `m²`...).

### 3. Cơ chế kết thúc lượt nói thông minh (Smart Turn & VAD)

- **Silero VAD (ASR Side / Local):** Phát hiện tiếng nói trong frame âm thanh thời gian thực.
- **Smart Turn v3.2 ONNX:** Phân tích ngữ nghĩa âm học (cao độ, ngữ điệu kết thúc) trên cửa sổ trượt âm thanh 8s gần nhất.
- **Heuristics bảo vệ ngữ cảnh:** Tự động tăng thời gian chờ ngắt khi người dùng ngập ngừng (*"ừ"*, *"à"*, *"ờ"*), nói từ nối (*"và"*, *"thì"*, *"nhưng"*...) hoặc đọc dở dãy số điện thoại/mã vận đơn.

### 4. Giao diện Web Audio Hiện đại

- **Web Audio API & AudioWorklet:** Thu âm micro, resample 16 kHz PCM16 gửi STT; phát stream PCM16 48 kHz mượt mà với cơ chế zero-copy.
- **Anti-feedback & Barge-in:** Hỗ trợ ngắt lời tức thì (Barge-in) và cơ chế chống dội âm (Anti-feedback) tự động.

---

## 🏗️ Cấu trúc dự án

```text
voice-bot/
├── backend/                    # Backend FastAPI & Voice Gateway
│   ├── config.py               # Quản lý cấu hình & biến môi trường
│   ├── main.py                 # FastAPI app, static mount & WebSocket endpoint
│   ├── patterns.py             # Regular expressions xử lý số, câu, đơn vị
│   ├── services.py             # Client kết nối STT (ASR), LLM, Zero-TTS, Smart Turn
│   ├── text_processing.py      # Chuẩn hóa văn bản STT/TTS và SpeechFilter
│   ├── voice.py                # State machine, session, pipeline xử lý turn & barge-in
│   └── .env.example            # File mẫu cấu hình biến môi trường
├── frontend/                   # Giao diện Web demo
│   ├── index.html              # HTML giao diện chat & voice controls
│   ├── app.js                  # AudioWorklet, WebSocket client, Web Audio API
│   └── style.css               # Giao diện trực quan hiện đại
├── models/                     # Thư mục chứa mô hình ONNX
│   └── smart-turn-v3.2-cpu.onnx# Smart Turn v3.2 (ONNX runtime CPU)
├── docs/                       # Tài liệu chi tiết kiến trúc & giao thức
│   ├── 01_architecture.md      # Cấu trúc hệ thống & nguyên tắc thiết kế
│   ├── 02_voice_flow.md        # Luồng hội thoại, Smart Turn và barge-in
│   ├── 03_api_protocol.md      # Hợp đồng WebSocket & API
│   └── 04_implementation.md    # Checklist triển khai & nghiệm thu
├── tests/                      # Bộ kiểm thử tự động
│   └── test_backend.py         # 18 test cases kiểm thử toàn diện backend
├── requirements.txt            # Danh sách thư viện phụ thuộc (pip)
├── pyproject.toml              # Cấu hình dự án & linter (ruff)
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Cấu hình biến môi trường

Tạo file `.env` tại thư mục `backend/` (hoặc thư mục gốc dự án) dựa trên `backend/.env.example`:

| Biến môi trường       | Mặc định                                                  | Mô tả                                           |
| :------------------------ | :----------------------------------------------------------- | :------------------------------------------------ |
| `STT_WS_URL`            | `wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream`              | WebSocket endpoint của Qwen3-ASR                 |
| `LLM_URL`               | `https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions` | OpenAI-compatible endpoint của LLM               |
| `TTS_BASE_URL`          | `https://ai.vnpost.vn/zero-tts`                            | Endpoint dịch vụ Zero-TTS                       |
| `TTS_VOICE`             | `maichi`                                                   | Giọng đọc mặc định của Zero-TTS            |
| `SMART_TURN_MODEL_PATH` | `models/smart-turn-v3.2-cpu.onnx`                          | Đường dẫn file mô hình Smart Turn v3.2 ONNX |

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt môi trường

Sử dụng `uv` (khuyên dùng):

```powershell
uv sync
```

Hoặc sử dụng `pip` với Python 3.11+:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Kiểm tra bộ kiểm thử (Unit tests & Linting)

Chạy kiểm tra cú pháp và linter:

```powershell
.\.venv\Scripts\ruff check backend tests
```

Chạy bộ 18 bài kiểm tra tự động:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend.py -v
```

### 3. Khởi động Web Server

Chạy server FastAPI uvicorn:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Hoặc chạy trực tiếp qua module main:

```powershell
python -m backend.main
```

Truy cập giao diện Web trên trình duyệt: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
