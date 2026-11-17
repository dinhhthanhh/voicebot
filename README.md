# Voicebot (STT → LLM → SpeechFilter → TTS)

Hệ thống trợ lý giọng nói đàm thoại thời gian thực độ trễ thấp, tích hợp bộ lọc văn bản thông minh tiếng Việt và cơ chế phát hiện ngắt lượt nói hai tầng (Silero VAD v5 + Smart Turn v3.2).

---

## 🌟 Tính năng nổi bật

### 1. Kiến trúc luồng kép (Dual-channel Streaming)
- **Kênh hiển thị (Màn hình):** Stream toàn văn câu trả lời từ LLM tức thì (bao gồm đầy đủ Markdown, bảng biểu, mã nguồn, danh sách).
- **Kênh âm thanh (Thoại):** Xử lý song song qua hàng đợi Producer-Consumer. Tách câu thời gian thực và đẩy ngay câu hoàn chỉnh đầu tiên qua Zero-TTS (PCM 48 kHz, Mono, 16-bit) để phát âm thanh lập tức tới người dùng.

### 2. Xử lý & Chuẩn hóa văn bản Tiếng Việt 3 tầng
- **Tiền xử lý đầu vào STT (`stt.py`):**
  - **Chuẩn hóa số ghép tiếng Việt (`normalize_compound_numbers`):** Chuyển đổi linh hoạt các cụm từ đọc số thành chữ số (hàng chục *"mươi"*, *"mười"*, lẻ *"linh"*, *"lẻ"*, và hàng trăm/nghìn tròn số như *"hai mươi mốt"* → `21`, *"mười hai nghìn năm trăm"* → `12500`).
  - **Gom chuỗi số liên tiếp (`normalize_speech_input`):** Tự động nhận diện và gom các từ đọc số rời rạc liên tiếp ($\ge 5$ chữ số) thành chuỗi số hoàn chỉnh (hỗ trợ số điện thoại 10 số, mã bưu chính VNPost 5 số, số CCCD, mã vận đơn).
- **Bộ lọc luồng LLM (`SpeechFilter` trong `tts.py`):**
  - **Tách câu thời gian thực:** Nhận diện ranh giới câu tự nhiên theo dấu ngắt câu (`SENT_RE` hỗ trợ lookbehind bỏ qua các từ viết tắt thông dụng như `TP.`, `ThS.`, `BS.`, `Mr.`...) để đẩy ngay câu hoàn chỉnh đầu tiên xuống Zero-TTS.
  - **Lược bỏ thành phần trực quan:** Tự động bỏ qua các khối mã nguồn (` ``` `), bảng Markdown (`|...|`), vạch phân cách và làm sạch inline formatting (emoji, link Markdown, công thức toán LaTeX `$`, đường dẫn URL) nhằm tránh việc bot đọc cú pháp thô.
  - **Capping an toàn:** Giới hạn tối đa 8 câu hoặc 600 ký tự thoại; tự động thêm câu thông báo thoại *"Phần còn lại đã được hiển thị trên màn hình."* khi đạt ngưỡng giới hạn để giữ câu trả lời thoại súc tích, mạch lạc.
- **Hậu xử lý văn bản đọc TTS (`normalize_text_for_tts` trong `tts.py`):**
  - **Số và tiền tệ:** Phân biệt chính xác phân cách hàng nghìn (`1.500`, `1,500`) và số thập phân (`0,007` → đọc là *"không phẩy không không bảy"* qua `num2words`); hỗ trợ các ký hiệu tiền tệ thông dụng (`đồng`, `VNĐ`, `đ`, `k` → *"nghìn"*).
  - **Số điện thoại & Ngày tháng:** Chuẩn hóa số điện thoại cố định `(024) 3xxx`, số di động, mã quốc tế `+84` đọc tách từng chữ số; định dạng ngày tháng `DD/MM/YYYY`, dịch vụ `24/7` (*"hai tư trên bảy"*), địa chỉ ngõ ngách `số 10/2` (*"số 10 xuyệt 2"*).
  - **Đơn vị & ký hiệu:** Đổi tự động các đơn vị đo lường thông dụng (`km`, `m`, `cm`, `mm`, `kg`, `g`, `ha`, `m²`, `m³`, `m2`, `m3`, `%`...) sang từ ngữ phát âm tự nhiên.

### 3. Cơ chế kết thúc lượt nói thông minh (Smart Turn & VAD)
- **Kiến trúc phát hiện 2 tầng tách bạch:**
  - **Silero VAD v5 (Voice Activity Detection):** Chuyên trách phát hiện theo thời gian thực xem người dùng **có đang nói hay không** (phân biệt giọng nói với khoảng lặng và tạp âm môi trường).
  - **Smart Turn v3.2 ONNX (Turn-Taking Prediction):** Chuyên trách dự đoán xem người dùng **đã nói xong lượt hay chưa** bằng cách phân tích ngữ nghĩa âm học (cao độ rơi, ngữ điệu, khoảng lặng dứt câu) trên cửa sổ trượt tối đa 8 giây âm thanh gần nhất.
- **Bộ lọc năng lượng RMS & Ngưỡng trễ:** Ngăn tiếng ồn nền và hơi thở gây kích hoạt giả bằng cơ chế hysteresis trễ (đã tinh chỉnh chống nhiễu ồn phòng):
  - Bắt đầu nói: Xác suất VAD $\ge 0.60$ và RMS $\ge 0.0025$.
  - Duy trì nói: Xác suất VAD $\ge 0.40$ và RMS $\ge 0.0015$.
  - Khi Silero VAD nhận diện chắc chắn là tiếng người ($\ge 0.70$): Duy trì theo ngưỡng RMS $\ge 0.0015$ (loại bỏ tiếng ồn nền tĩnh).
- **Ngưỡng quyết định ngắt câu linh hoạt (phân tầng 3 mức Smart Turn):**
  - Rất tự tin dứt câu ($\ge 0.85$): Ngắt siêu nhanh sau $0.5\text{s}$ ($500\text{ms}$) im lặng (hoặc $1.0\text{s}$ khi đọc xong dãy số).
  - Tự tin cao ($\ge 0.70$): Ngắt sau $0.7\text{s}$ ($700\text{ms}$) im lặng (câu thường), $1.2\text{s}$ (khi ngập ngừng) hoặc $1.5\text{s}$ (khi đọc xong dãy số/mã vận đơn).
  - Tự tin trung bình ($\ge 0.55$): Ngắt sau $1.2\text{s}$ im lặng đối với câu dứt khoát không ngập ngừng.
  - Hết thời gian im lặng (Timeout an toàn):
    - Câu thường: Tự động ngắt khi im lặng $\ge 2.0\text{s}$ (hoặc $\ge 2.5\text{s}$ theo đồng hồ thực).
    - Câu ngập ngừng / đọc dở dãy số: Tự động gia hạn tới $3.5\text{s}$ im lặng (hoặc $\ge 3.5\text{s}$ theo đồng hồ thực).
    - Chỉ nói từ đệm (ừ, à, ờ...): Tự động ngắt sau $2.0\text{s}$ im lặng.
- **Bảo vệ ngữ cảnh dở dang:** Tự động tăng thời gian chờ nếu người dùng đang đọc dở số điện thoại/dãy số hoặc kết thúc bằng các từ nối/ngập ngừng (*"và"*, *"hoặc"*, *"thì"*, *"nhưng"*, *"là"*...).

### 4. Giao diện Web & Trải nghiệm đàm thoại liên tục
- **Web Audio API & AudioWorklet:** Thu âm từ micro và Linear Resample về 16 kHz PCM16 gửi STT (kết hợp bộ lọc chống méo bậc 4 Butterworth 7.2 kHz); phát stream âm thanh PCM16 48 kHz mượt mà với cơ chế zero-copy qua `AudioWorklet` (có fallback `ScriptProcessor`) giảm tối đa áp lực Garbage Collector.
- **Cơ chế chống dội âm (Anti-feedback):** Kết hợp `echoCancellation: true` phần cứng/trình duyệt của Web Audio API, tự động chặn gửi frame micro ở tầng phần mềm (`isBotBusy`) trong khi bot đang suy nghĩ hoặc phát âm thanh, và thêm khoảng trễ đệm 600ms (`scheduleAutoListen`) trước khi mở lại micro đón lượt nói mới.
- **Đàm thoại tự nhiên:** Bấm **"Bắt đầu nói"** để kích hoạt chế độ đàm thoại liên tục (các nút chuyển sang **"Gửi"** để ngắt thủ công, **"Dừng nói"** để hủy lượt thoại, và **"Hội thoại mới"** để xóa lịch sử ngữ cảnh). Bot và người dùng trò chuyện qua lại tự động mà không cần bấm nút lại từng câu.

---

## 🏗️ Cấu trúc dự án

```text
voice-bot/
├── models/                     # Thư mục chứa mô hình ONNX (VAD & Smart Turn)
│   ├── silero_vad.onnx         # Silero VAD v5 (ONNX runtime CPU)
│   └── smart-turn-v3.2-cpu.onnx# Smart Turn v3.2 (ONNX runtime CPU)
├── index.html                  # Giao diện Web Voicebot (Web Audio API, điều khiển hội thoại)
├── server.py                   # FastAPI WebSocket server chính (/ws/voice)
├── stt.py                      # Qwen3-ASR client, Silero VAD, Smart Turn v3.2 ONNX
├── tts.py                      # LLM Client, SpeechFilter, Zero-TTS stream client
├── pyproject.toml              # Cấu hình dự án & linter (uv / pip / ruff)
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Cấu hình biến môi trường

Các tham số có thể cấu hình qua biến môi trường (hoặc sử dụng giá trị mặc định trong code):

| Biến môi trường | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `STT_WS_URL` | `wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream` | WebSocket endpoint của Qwen3-ASR |
| `LLM_URL` | `https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions` | OpenAI-compatible endpoint của LLM |
| `TTS_BASE_URL` | `https://ai.vnpost.vn/zero-tts` | Endpoint dịch vụ Zero-TTS |
| `TTS_VOICE` | `maichi` | Giọng đọc mặc định của Zero-TTS |
| `SMART_TURN_MODEL_PATH` | `models/smart-turn-v3.2-cpu.onnx` | Đường dẫn file mô hình Smart Turn v3.2 ONNX |
| `VAD_MODEL_PATH` | `models/silero_vad.onnx` | Đường dẫn file mô hình Silero VAD v5 ONNX |

*Lưu ý: Mặc định nếu không chỉ định biến môi trường, các mô hình ONNX và giao diện Web được nạp tự động theo đường dẫn tương đối trong thư mục dự án (`models/silero_vad.onnx`, `models/smart-turn-v3.2-cpu.onnx`, `index.html`).*

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

*Ghi chú phụ thuộc:* Thư viện `transformers` được giữ trong danh sách phụ thuộc nhằm cung cấp `WhisperFeatureExtractor` trích xuất đặc trưng âm học chuẩn hóa cho mô hình Smart Turn ONNX.

### 2. Chuẩn bị mô hình ONNX

Do quy tắc `.gitignore` loại trừ các file nhị phân lớn (`*.onnx`), bạn cần đảm bảo hai file mô hình sau đã nằm trong thư mục `models/` trước khi chạy:
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
