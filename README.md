# Voicebot MVP (STT → LLM → TTSStreamer → TTS)

Hệ thống trợ lý giọng nói đàm thoại thời gian thực độ trễ thấp, hỗ trợ tiếng Việt với cơ chế phát hiện ngắt lượt nói hai tầng (Silero VAD + Smart Turn v3.2 ONNX), luồng kép Dual-channel Streaming và quản lý lượt thoại liên tục.

---

## 🌟 Tính năng nổi bật

### 1. Kiến trúc phân tầng chuẩn hóa & Luồng kép (Dual-channel Streaming)

- **Kênh hiển thị (Màn hình):** Stream toàn văn câu trả lời từ LLM tức thì đến giao diện người dùng.
- **Kênh âm thanh (Thoại):** Xử lý song song qua mô hình Producer-Consumer không đồng bộ (`asyncio.Queue`). Tách câu thời gian thực và stream audio PCM 48 kHz (Mono 16-bit) lập tức từ Zero-TTS tới người dùng.
- **Connection Pool tái sử dụng:** Persistent HTTP Client dùng chung kết nối keep-alive cho cả LLM và TTS, tối ưu latency tối đa.

### 2. Xử lý & Chuẩn hóa văn bản Tiếng Việt

- **Phát hiện từ đệm & transcript rỗng (`is_filler` trong `text_processing.py`):**
  - Tự động nhận diện khi người dùng chỉ ngập ngừng phát ra từ đệm (*"ừ"*, *"ừm"*, *"ờ"*, *"à"*...) hoặc STT không nhận diện được văn bản (chuỗi rỗng).
  - Phản hồi câu thoại ngắn nhắc nhở (*"Tôi đang lắng nghe..."* hoặc *"Tôi chưa nghe rõ, bạn vui lòng nói lại nhé."*) qua TTS và tự động kích hoạt lượt nghe tiếp theo mà không cần gọi LLM, giúp tiết kiệm tài nguyên và giữ luồng đàm thoại thông suốt.
- **Bộ lọc làm sạch Markdown nội dòng (`clean_md` trong `text_processing.py`):**
  - Làm sạch các link Markdown (`[text](url)` → `text`) và loại bỏ URL (`https?://...`).
  - Xóa các ký hiệu định dạng văn bản inline (`*`, `_`, `` ` ``, `~`, `#`) và các dấu đầu dòng (bullet points: `-`, `*`, `+`, `•`, `1.`, `1)`) để câu thoại được phát âm tự nhiên.
- **Bộ lọc luồng LLM (`TTSStreamer` trong `text_processing.py`):**
  - **Tách câu thời gian thực:** Nhận diện ranh giới câu tự nhiên theo dấu ngắt câu (`.`, `!`, `?`, `…`, xuống dòng), bỏ qua các từ viết tắt phổ biến (`TP.`, tên viết tắt `A.`, `v.v.`, số thứ tự `1.`).
  - **Capping an toàn:** Giới hạn tối đa số câu (mặc định 8 câu) hoặc độ dài (mặc định 600 ký tự thoại); tự động gắn thêm câu thông báo thoại *"Phần còn lại đã được hiển thị trên màn hình."* khi đạt ngưỡng.
- **Hậu xử lý văn bản đọc TTS (`normalize_tts` trong `text_processing.py`):**
  - Chuẩn hóa số điện thoại (hotline, quốc tế +84, số nội hạt), ngày tháng & khoảng ngày (phân biệt với khoảng thời gian/giá trị như `2-3 ngày`, `15.000đ - 20.000đ`), dịch vụ `24/7` (*"hai tư trên bảy"*), phân số/ngõ ngách `10/2`.
  - Đọc chuẩn tiền tệ (`đồng`, `tỷ`, `nghìn`), phân tách hàng nghìn, số thập phân (*"không phẩy..."*), đơn vị đo (`km`, `kg`, `ha`, `m²`...), thuật ngữ ngắn (`5G`, `4K`) và đánh vần mã vận đơn (`VN123456789VN`).

### 3. Cơ chế kết thúc lượt nói thông minh (Smart Turn & VAD)

- **Silero VAD (ASR Side):** Nhận diện tín hiệu bắt đầu/kết thúc giọng nói (`speech_started`, `speech_ended`) trong từng frame âm thanh truyền tới dịch vụ STT.
- **Smart Turn v3.2 ONNX:** Khi VAD báo kết thúc giọng nói, hệ thống đưa đoạn âm thanh vào mô hình ONNX [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) để dự đoán xác suất người dùng đã dứt câu hay còn nói tiếp.
- **Cơ chế Fallback thời gian im lặng:** Tự động hoàn tất lượt nói khi thời gian im lặng vượt ngưỡng cấu hình (`SMART_TURN_FALLBACK_SECONDS`, mặc định `5.0s`) hoặc khi mô hình Smart Turn xác nhận dứt lời (`SMART_TURN_THRESHOLD`, mặc định `0.70`).

### 4. Giao diện Web Audio Hiện đại

- **Web Audio API & AudioWorklet:** Thu âm micro, resample 16 kHz PCM 16-bit gửi backend STT; phát stream PCM 16-bit 48 kHz từ Zero-TTS với cơ chế hàng đợi mượt mà.
- **Chống dội âm (AEC Loopback):** Thiết lập loopback RTCPeerConnection để hỗ trợ Acoustic Echo Cancellation trên trình duyệt, giảm thiểu dội âm từ loa vào micro.
- **Quản lý thu âm theo trạng thái & Ngắt lời thủ công:**
  - Frontend tự động ngắt gửi dữ liệu micro khi bot đang trong trạng thái xử lý hoặc phát tiếng (`thinking` / `speaking`), tránh nhiễu âm.
  - Người dùng có thể nhấn nút **Dừng nói** bất cứ lúc nào để dừng phát âm thanh ngay lập tức, hủy lượt hiện tại và giải phóng trạng thái hệ thống.
  - Tự động kích hoạt nghe lượt tiếp theo sau khi bot đọc xong câu trả lời.

---

## 🏗️ Cấu trúc dự án

```text
voice-bot/
├── backend/                    # Backend FastAPI & Voice Gateway
│   ├── config.py               # Quản lý cấu hình tập trung & biến môi trường
│   ├── main.py                 # FastAPI app, static mount & WebSocket endpoint
│   ├── prompt.txt              # System prompt mặc định cho trợ lý VNPost
│   ├── services.py             # Client kết nối STT (ASR), LLM, Zero-TTS, Smart Turn ONNX
│   ├── text_processing.py      # Chuẩn hóa văn bản STT/TTS, TTSStreamer & regex
│   ├── voice.py                # State machine, session, pipeline quản lý lượt thoại (turn)
│   └── .env.example            # File mẫu cấu hình biến môi trường
├── frontend/                   # Giao diện Web demo
│   ├── index.html              # HTML giao diện chat & voice controls
│   ├── app.js                  # AudioWorklet, WebSocket client, Web Audio API
│   └── style.css               # Giao diện trực quan hiện đại
├── models/                     # Thư mục chứa mô hình ONNX
│   └── smart-turn-v3.2-cpu.onnx# Smart Turn v3.2 (tải từ Hugging Face pipecat-ai/smart-turn-v3)
├── docs/                       # Tài liệu chi tiết kiến trúc & giao thức
│   ├── 01_architecture.md      # Cấu trúc hệ thống & nguyên tắc thiết kế
│   ├── 02_voice_flow.md        # Luồng hội thoại, Smart Turn và ngắt lời
│   ├── 03_api_protocol.md      # Hợp đồng WebSocket & API
│   ├── 04_implementation.md    # Checklist triển khai & nghiệm thu
│   └── 05_runtime_flow_report.md # Báo cáo luồng chạy chi tiết CMD -> Runtime & kiểm định docs
├── tests/                      # Bộ kiểm thử tự động
│   └── test_backend.py         # Unit tests kiểm thử toàn diện backend
├── requirements.txt            # Danh sách thư viện phụ thuộc (pip)
├── pyproject.toml              # Cấu hình dự án & linter (ruff)
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Cấu hình biến môi trường

Tạo file `.env` tại thư mục `backend/` (hoặc thư mục gốc dự án) dựa trên `backend/.env.example`:

| Biến môi trường              | Mặc định                                                     | Mô tả                                                         |
| :--------------------------- | :----------------------------------------------------------- | :------------------------------------------------------------ |
| `APP_HOST`                   | `127.0.0.1`                                                  | Host chạy FastAPI server                                      |
| `APP_PORT`                   | `8000`                                                       | Port chạy FastAPI server                                      |
| `APP_LOG_LEVEL`              | `INFO`                                                       | Mức log hệ thống (`DEBUG`, `INFO`, `WARNING`, `ERROR`)        |
| `APP_RELOAD`                 | `false`                                                      | Tự động reload uvicorn khi phát triển (`true`/`false`)        |
| `LLM_CHAT_COMPLETIONS_URL`   | `https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions`   | OpenAI-compatible chat completions endpoint của LLM           |
| `LLM_MODEL`                  | `""`                                                         | Tên model LLM (để trống nếu vLLM tự định tuyến)               |
| `LLM_TEMPERATURE`            | `0`                                                          | Độ sáng tạo của câu trả lời LLM                               |
| `LLM_REASONING_EFFORT`       | `low`                                                        | Mức reasoning effort gửi kèm payload LLM                      |
| `LLM_TIMEOUT_SECONDS`        | `90`                                                         | Timeout tối đa cho lượt stream LLM                            |
| `STT_WS_URL`                 | `wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream`                | WebSocket streaming endpoint của Qwen3-ASR                    |
| `STT_MODEL`                  | `qwen3-asr-1p7b-ft`                                          | Tên mô hình nhận diện giọng nói                               |
| `STT_LANGUAGE`               | `Vietnamese`                                                 | Ngôn ngữ nhận diện giọng nói                                  |
| `STT_SAMPLE_RATE`            | `16000`                                                      | Tần số lấy mẫu âm thanh đầu vào cho STT (Hz)                  |
| `STT_VAD`                    | `silero`                                                     | Cơ chế VAD phía STT server                                    |
| `STT_MAX_RETRIES`            | `2`                                                          | Số lần retry tối đa khi kết nối STT bị quá tải/lỗi tạm        |
| `STT_CONNECT_TIMEOUT`        | `10.0`                                                       | Timeout (giây) chờ handshake `ready` từ STT                   |
| `STT_FINISH_TIMEOUT`         | `5.0`                                                        | Timeout (giây) chờ gói `final` sau khi gửi `finish`           |
| `STT_MAX_DURATION_SECONDS`   | `600`                                                        | Thời lượng tối đa (giây) cho một turn STT                     |
| `TTS_BASE_URL`               | `https://ai.vnpost.vn/zero-tts`                              | Base URL dịch vụ Zero-TTS                                     |
| `TTS_STREAM_PATH`            | `/tts/stream`                                                | Đường dẫn API stream TTS                                      |
| `TTS_VOICE`                  | `maichi`                                                     | Giọng đọc mặc định của Zero-TTS                               |
| `TTS_TIMEOUT_SECONDS`        | `90`                                                         | Timeout tối đa (giây) cho request TTS stream                  |
| `TTS_SAMPLE_RATE`            | `48000`                                                      | Tần số âm thanh trả về từ TTS (Hz)                            |
| `TTS_SPLIT_LIMIT`            | `450`                                                        | Giới hạn ký tự tối đa mỗi đoạn gửi sang TTS                   |
| `TTS_MIN_CHARS`              | `8`                                                          | Ngưỡng ký tự tối thiểu trước khi xuất câu TTS                 |
| `TTS_MAX_SENTENCES`          | `8`                                                          | Số câu thoại tối đa trước khi thực hiện capping               |
| `TTS_MAX_CHARS`              | `600`                                                        | Số ký tự thoại tối đa trước khi thực hiện capping             |
| `VOICE_MAX_HISTORY_PAIRS`    | `8`                                                          | Số cặp hội thoại user/assistant tối đa lưu trong RAM          |
| `TURN_MAX_AUDIO_BYTES`       | `256000`                                                     | Dung lượng bộ đệm vòng PCM tối đa (~8s) cho Smart Turn        |
| `SMART_TURN_MODEL_PATH`      | `models/smart-turn-v3.2-cpu.onnx`                            | Đường dẫn file mô hình Smart Turn v3.2 ONNX                   |
| `SMART_TURN_THRESHOLD`       | `0.70`                                                       | Ngưỡng xác suất kết thúc câu của mô hình Smart Turn (0.0 - 1.0)|
| `SMART_TURN_INFERENCE_TIMEOUT_SECONDS` | `3.0`                                              | Thời gian tối đa (giây) chờ mô hình Smart Turn chạy suy luận   |
| `SMART_TURN_FALLBACK_SECONDS`| `5.0`                                                        | Thời gian im lặng (giây) tối đa trước khi tự động kết thúc lượt|
| `SMART_TURN_MAX_WORKERS`     | `2`                                                          | Số worker threads trong ThreadPool cho Smart Turn             |

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

### 2. Tải mô hình Smart Turn v3.2 ONNX

Mô hình phát hiện ngắt lượt thông minh được cung cấp tại repo Hugging Face: [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3).

Dự án sử dụng mặc định file `smart-turn-v3.2-cpu.onnx` và đặt tại thư mục `models/`. Bạn có thể tải theo một trong các cách sau:

#### Cách 1: Sử dụng PowerShell (Khuyên dùng trên Windows)

```powershell
# Tạo thư mục models nếu chưa có
if (!(Test-Path -Path "models")) { New-Item -ItemType Directory -Path "models" }

# Tải file mô hình CPU (khoảng 8.7 MB)
Invoke-WebRequest -Uri "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx" -OutFile "models/smart-turn-v3.2-cpu.onnx"
```

#### Cách 2: Sử dụng `curl` (Linux / macOS / Git Bash / Command Prompt)

```bash
mkdir -p models
curl -L -o models/smart-turn-v3.2-cpu.onnx https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
```

#### Cách 3: Sử dụng Python (`huggingface_hub`)

Nếu môi trường đã cài đặt `transformers` / `huggingface_hub`:

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='pipecat-ai/smart-turn-v3', filename='smart-turn-v3.2-cpu.onnx', local_dir='models')"
```

#### Cách 4: Sử dụng Hugging Face CLI

```powershell
huggingface-cli download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx --local-dir models
```

#### Cách 5: Tải thủ công từ giao diện Hugging Face

1. Truy cập [pipecat-ai/smart-turn-v3 Files](https://huggingface.co/pipecat-ai/smart-turn-v3/tree/main).
2. Nhấn vào file **`smart-turn-v3.2-cpu.onnx`** và chọn **Download**.
3. Lưu file vào đường dẫn `models/smart-turn-v3.2-cpu.onnx` trong thư mục dự án.

> [!NOTE]
> - **Phiên bản GPU:** Nếu muốn chạy inference trên GPU với `smart-turn-v3.2-gpu.onnx` (~32.4 MB), tải file tương ứng và khai báo trong file `.env`: `SMART_TURN_MODEL_PATH=models/smart-turn-v3.2-gpu.onnx`.
> - **Cơ chế Fallback:** Nếu file mô hình không tồn tại trong thư mục `models/`, hệ thống sẽ cảnh báo trong log và tự động fallback hoàn toàn sang cơ chế đếm ngược im lặng (`is_silence_timeout()` theo `SMART_TURN_FALLBACK_SECONDS`, tính từ mốc tiếng nói/partial gần nhất), đảm bảo bot vẫn tự ngắt lượt bình thường.

### 3. Kiểm tra bộ kiểm thử (Unit tests & Linting)

Chạy kiểm tra cú pháp và linter:

```powershell
.\.venv\Scripts\ruff check backend tests
```

Chạy bộ kiểm tra tự động:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend.py -v
```

### 4. Khởi động Web Server

Sử dụng `uv` (khuyên dùng):

```powershell
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Hoặc sử dụng `python`:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Hoặc chạy trực tiếp qua module main:

```powershell
python -m backend.main
```

Truy cập giao diện Web trên trình duyệt: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
