# Voicebot (STT → LLM → SpeechFilter → TTS)

Hệ thống trợ lý giọng nói đàm thoại thời gian thực, độ trễ thấp (Time-to-First-Audio ~0.7s) tích hợp bộ lọc văn bản thông minh và cơ chế phát hiện ngắt lượt nói hai tầng (Silero VAD + Smart Turn v3.2).

---

## 🌟 Tính năng nổi bật

### 1. Kiến trúc luồng kép (Dual-channel Streaming)
- **Kênh hiển thị (Màn hình):** Stream toàn văn câu trả lời từ LLM tức thì (bao gồm đầy đủ Markdown, bảng biểu, mã nguồn, danh sách).
- **Kênh âm thanh (Thoại):** Xử lý song song qua hàng đợi Producer-Consumer. Tổng hợp và phát âm thanh ngay khi câu đầu tiên sẵn sàng qua Zero-TTS (PCM 48 kHz, Mono, 16-bit), đảm bảo **Time-to-First-Audio ~0.7s**.

### 2. Bộ lọc văn bản thông minh (`SpeechFilter`)
- **Tách câu thời gian thực:** Nhận diện ranh giới câu tự nhiên theo dấu ngắt câu và ngữ cảnh hội thoại.
- **Lược bỏ thành phần trực quan:** Tự động bỏ qua các khối mã nguồn (` ``` `) và bảng Markdown (`|...|`) để tránh việc bot đọc cú pháp thô (LLM được định hướng diễn đạt tóm tắt tự nhiên trước phần bảng/mã).
- **Chuẩn hóa văn bản Tiếng Việt nâng cao:**
  - **Số điện thoại & Mã bưu chính:** Nhận dạng số di động 10 số, số cố định dạng `(024) 3xxx`, mã quốc gia `+84` và mã bưu chính VNPost (5 số).
  - **Cách đọc số ghép tự nhiên:** Chuyển đổi mượt mà các cụm chục (*"mươi"*, *"mười"*), lẻ (*"linh"*, *"lẻ"*), và tròn nghìn/trăm (*"mười nghìn"*, *"mười hai nghìn năm trăm"*).
  - **Số và tiền tệ:** Phân biệt chính xác phân cách hàng nghìn (`1.500`, `1,500`) và số thập phân (`0,007` → đọc là "không phẩy không không bảy"); hỗ trợ ký hiệu tiền tệ (`VNĐ`, `$`, `€`).
  - **Đơn vị & ký hiệu:** Đổi tự động các đơn vị thông dụng (`km`, `cm`, `kg`, `%`, `°C`...).
  - **Biểu thức & đường dẫn:** Chuẩn hóa biểu thức toán học, đường dẫn URL sang văn phong giao tiếp trực tiếp.
  - **Capping:** Giới hạn độ dài và số lượng câu đọc tối đa nhằm giữ câu trả lời thoại súc tích, mạch lạc.

### 3. Cơ chế kết thúc lượt nói thông minh (Smart Turn & VAD)
- **Kiến trúc phát hiện 2 tầng:** Kết hợp **Silero VAD v5** (phát hiện giọng nói / khoảng lặng) và **Smart Turn v3.2 ONNX** (phân tích ngữ nghĩa âm học xem người dùng đã dứt ý hay chưa thay vì hardcode danh sách từ ngắn).
- **Bộ lọc năng lượng RMS:** Ngăn chặn tiếng ồn nền và hơi thở gây kích hoạt giả (`RMS Energy Gate > 100`).
- **Ngưỡng quyết định linh hoạt:** Tự tin cao ($\ge 0.65$) ngắt nhạy ngay sau $0.7\text{s}$; ngưỡng mở rộng ($\ge 0.50$) khi im lặng $\ge 1.4\text{s}$ đối với các câu không ngập ngừng. Chờ tối đa $2.0\text{s}$ đối với trường hợp im lặng tuyệt đối.
- **Bảo vệ ngữ cảnh dở dang:** Tự động tăng thời gian chờ nếu người dùng đang đọc dở số điện thoại/dãy số (`\d{2,4}$`) hoặc kết thúc bằng các từ nối/ngập ngừng (*"và"*, *"hoặc"*, *"thì"*, *"nhưng"*, *"là"*...).
- **Watchdog chống lệch nhịp:** Tự động thu hồi và reset phiên an toàn nếu quá trình nhận dạng hoặc kết nối bị gián đoạn.

### 4. Giao diện Web & Trải nghiệm đàm thoại liên tục
- **Web Audio API:** Thu âm và Linear Resample về 16 kHz gửi STT; phát stream âm thanh PCM16 48 kHz mượt mà, không giật hạt.
- **Mạch chống dội âm (Anti-feedback):** Sử dụng `GainNode(gain=0)` triệt tiêu hoàn toàn hiện tượng âm từ loa dội ngược vào micro.
- **Đàm thoại tự nhiên:** Bấm **"Bắt đầu nói"** để kích hoạt chế độ đàm thoại liên tục (các nút chuyển sang **"Gửi"** để ngắt thủ công và **"Dừng nói"** để kết thúc phiên). Bot và người dùng trò chuyện qua lại tự động mà không cần bấm nút lại từng câu.

---

## 🏗️ Cấu trúc dự án

```text
voice-bot/
├── data/                       # Dữ liệu âm thanh test và logs kiểm nghiệm
├── models/                     # Các mô hình ONNX phục vụ VAD & Smart Turn
│   ├── silero_vad.onnx         # Silero VAD v5 (ONNX runtime CPU)
│   └── smart-turn-v3.2-cpu.onnx# Smart Turn v3.2 (ONNX runtime CPU)
├── index.html                  # Giao diện Web Voicebot (Web Audio API, visualizer)
├── server.py                   # FastAPI WebSocket server chính (/ws/voice)
├── stt.py                      # Qwen3-ASR client, Silero VAD, Smart Turn v3.2
├── tts.py                      # LLM Client, SpeechFilter, Zero-TTS stream client
├── pyproject.toml              # Cấu hình dự án (uv / pip)
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Cấu hình biến môi trường

Các tham số có thể cấu hình qua biến môi trường (hoặc đặt giá trị mặc định trong code):

| Biến môi trường | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `STT_WS_URL` | `ws://127.0.0.1:8765/ws/asr` | WebSocket endpoint của Qwen3-ASR |
| `LLM_URL` | `http://127.0.0.1:8001/v1/chat/completions` | OpenAI-compatible endpoint của LLM |
| `TTS_BASE_URL` | `http://127.0.0.1:8002` | Endpoint dịch vụ Zero-TTS |
| `TTS_VOICE` | `bich_ngoc` | Giọng đọc TTS |
| `VAD_MODEL_PATH` | `models/silero_vad.onnx` | Đường dẫn mô hình Silero VAD ONNX |
| `SMART_TURN_MODEL_PATH` | `models/smart-turn-v3.2-cpu.onnx` | Đường dẫn mô hình Smart Turn ONNX |
| `VOICEBOT_HTML_PATH` | `index.html` | Đường dẫn file giao diện web |

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Chuẩn bị môi trường & Cài đặt thư viện

Sử dụng `uv` (khuyên dùng):
```powershell
uv sync
```

Hoặc sử dụng `pip`:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Kiểm tra mô hình ONNX

Đảm bảo hai file mô hình đã nằm trong thư mục `models/`:
- `models/silero_vad.onnx`
- `models/smart-turn-v3.2-cpu.onnx`

### 3. Khởi động Web Server

Chạy server FastAPI uvicorn tại cổng 8000:
```powershell
uv run uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Hoặc khi dùng môi trường `.venv`:
```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Truy cập giao diện Web trên trình duyệt: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
