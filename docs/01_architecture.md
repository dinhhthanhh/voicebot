# 01 -- Thiết kế kiến trúc Voicebot MVP

**Phiên bản:** 1.0

## 1. Mục tiêu

Xây dựng Voicebot độc lập với chatbot hiện tại. Người dùng có thể nói
qua microphone, hệ thống nhận dạng tiếng Việt thời gian thực, gửi nội dung sang LLM và
phát câu trả lời bằng giọng nói trực tiếp. Hệ thống tích hợp cơ chế phát hiện ngắt lượt
tự động (Silero VAD + Smart Turn v3.2 ONNX) và hỗ trợ ngắt lời tức thì (Barge-in).

Kiến trúc phải cho phép thay thế từng nhà cung cấp STT/LLM/TTS mà không
phải viết lại toàn bộ luồng hội thoại.

## 2. Sơ đồ kiến trúc

``` mermaid
flowchart TB
    FE["Frontend Demo<br/>index.html / style.css / app.js"]
    GW["main.py<br/>FastAPI + WebSocket Gateway"]
    ORCH["voice.py<br/>Session / Turn Orchestrator"]
    NLP["text_processing.py<br/>SpeechFilter & TTS Normalizer"]
    ADAPTER["services.py<br/>STT / LLM / TTS / Smart Turn adapters"]
    CFG["config.py<br/>Settings"]
    STT["Qwen3-ASR WebSocket<br/>Silero VAD tích hợp"]
    LLM["Qwen 27B Chat Completions"]
    TTS["ZeroTTS HTTP Chunked Streaming"]
    TURN["Smart Turn v3.2"]

    FE <-->|"WebSocket: JSON + binary audio"| GW
    GW <-->|"events / audio frames"| ORCH
    ORCH <-->|"clean & segment text"| NLP
    ORCH <-->|"service interfaces"| ADAPTER
    CFG --> GW
    CFG --> ORCH
    CFG --> ADAPTER
    ADAPTER <--> STT
    ADAPTER <--> LLM
    ADAPTER <--> TTS
    ADAPTER <--> TURN
```

## 3. Ranh giới trách nhiệm

### 3.1 `main.py` -- Gateway

**Chịu trách nhiệm** - Khởi tạo ứng dụng FastAPI. - Cung cấp endpoint
WebSocket nội bộ, ví dụ `/ws/voice`. - Nhận JSON control events và
binary audio từ frontend. - Chuyển event/audio vào đối tượng điều phối
phiên trong `voice.py`. - Gửi event và audio về frontend. - Xử lý
disconnect và yêu cầu đóng session.

**Không chịu trách nhiệm** - Gọi trực tiếp API STT/LLM/TTS. - Quyết định
thời điểm kết thúc lượt nói. - Quản lý lịch sử hội thoại hoặc prompt.

### 3.2 `voice.py` -- Orchestrator

**Chịu trách nhiệm** - Tạo, quản lý và đóng session. - Duy trì state và
history trong RAM. - Điều phối STT → LLM → TTS. - Kết hợp VAD event từ STT
với Smart Turn v3.2 để tự động phát hiện dứt câu. - Quản lý ngắt lời (Barge-in)
và hủy tác vụ an toàn khi người dùng nói đè. - Kiểm tra `session_id`/`turn_id` để
loại bỏ kết quả cũ.

**Không chịu trách nhiệm** - Tự xây dựng HTTP request cụ thể cho từng
nhà cung cấp. - Parse giao thức SSE/TTS transport ở mức thấp. - Xử lý
giao diện hoặc phát audio trên trình duyệt.

### 3.3 `services.py` -- Service adapters

Đóng gói giao tiếp bên ngoài với các dịch vụ AI. Các interface chính:

``` python
class STTClient:
    @classmethod
    async def connect(cls, session_id: str, turn_id: str) -> STTClient: ...
    async def send_audio(self, pcm_bytes: bytes) -> None: ...
    async def events(self) -> AsyncIterator[dict[str, Any]]: ...
    async def finish(self, timeout: float = 10.0) -> str: ...
    async def close(self) -> None: ...

async def llm_stream(messages: list[dict[str, str]]) -> AsyncIterator[str]: ...

async def tts_stream(text: str, voice: str | None = None) -> AsyncIterator[TTSAudioChunk]: ...

async def smart_turn_predict(audio_data: bytes) -> float: ...
```

Adapter cần: - Quản lý kết nối, timeout và parse response. - Chuẩn hóa
lỗi thành exception nội bộ có ý nghĩa. - Không giữ lịch sử hội thoại. -
Không quyết định state của Voicebot.

### 3.4 `config.py` -- Configuration

-   Đọc biến môi trường.
-   Kiểm tra cấu hình bắt buộc khi khởi động.
-   Cung cấp settings cho các file còn lại.
-   Không ghi secret vào log.

### 3.5 `text_processing.py` -- Text Processing & SpeechFilter

-   Lọc từ đệm đơn thuần (`is_only_filler`) để tránh gọi LLM khi người dùng chỉ ngập ngừng (ừ, à, ờ...).
-   Làm sạch văn bản (`clean_inline_markdown`): bóc tách code block, KaTeX, Markdown link, URL, ký hiệu Markdown và bullet points.
-   Tách câu thời gian thực từ luồng LLM streaming (`SpeechFilter`) cho Zero-TTS kèm cơ chế capping an toàn tối đa 8 câu hoặc 600 ký tự.
-   Chuẩn hóa văn bản tiếng Việt sang ngữ điệu đọc tự nhiên cho TTS (`normalize_text_for_tts`: số điện thoại, tổng đài 1900/1800, ngày tháng `DD/MM/YYYY`, tiền tệ, số thập phân, đơn vị đo lường, từ công nghệ 5G/24h, mã vận đơn).

## 4. Quản lý session

Một kết nối frontend tương ứng một Voice session trong MVP.

Cấu trúc dữ liệu logic:

``` python
VoiceSession:
    session_id: str
    state: VoiceState
    history: list[Message]
    current_turn_id: str | None
    active_tasks: set[Task]
    closed: bool
```

History gồm message theo chuẩn Chat Completions:

``` python
[
    {"role": "user", "content": "Xin chào"},
    {"role": "assistant", "content": "Chào bạn"}
]
```

Quy tắc: - Chỉ lưu `transcript.final`, không lưu partial. - Chỉ lưu
assistant response khi lượt LLM hoàn tất bình thường. - Khi assistant bị
ngắt giữa chừng, không lưu phần trả lời chưa hoàn tất như một message
hoàn chỉnh. - Khi session đóng, hủy task, đóng stream và xóa session
khỏi registry RAM. - MVP không đảm bảo giữ session nếu backend restart.

`MAX_HISTORY` cần được định nghĩa rõ. Đề xuất giới hạn 8 cặp
user/assistant gần nhất, không tính system prompt. Nếu cần giới hạn
token, bổ sung cơ chế cắt theo token ở giai đoạn tối ưu.

## 5. Session, turn và request

  ID             Phạm vi                   Ý nghĩa
  -------------- ------------------------- -------------------------------------
  `session_id`   Một phiên kết nối Voice   Gắn history và trạng thái
  `turn_id`      Một lượt hỏi/đáp          Phân biệt lượt hiện tại với lượt cũ
  `request_id`   Một request STT           Đối chiếu message STT

Không dùng một ID thay cho cả ba.

## 6. State machine

Các state công khai:

-   `idle`: chưa thu âm/chưa xử lý.
-   `listening`: đang nhận audio từ người dùng.
-   `thinking`: đang chờ hoặc nhận LLM response.
-   `speaking`: đang tạo/phát audio TTS.
-   `interrupted`: lượt trả lời bị người dùng ngắt.
-   `error`: lỗi cần hiển thị.

State phải được chuyển đổi tập trung trong `voice.py`, không để frontend
tự suy đoán trạng thái backend.

## 7. Định dạng âm thanh nội bộ

Đầu vào STT theo tài liệu: - PCM signed 16-bit little-endian. - Mono. -
16,000 Hz. - Binary frames, không Base64.

Đầu ra TTS: - Tài liệu endpoint `/tts` full audio mô tả WAV, 48 kHz, PCM
16-bit mono. - Định dạng thực tế của `/tts/stream` phải xác minh trước
khi phát. - Chỉ chuẩn hóa audio sau khi xác định rõ chunk là WAV
container hay PCM raw.

Frontend nên thu bằng AudioWorklet và chuyển sample rate về 16 kHz mono
PCM16LE. Không giả định sample rate microphone luôn là 16 kHz.

## 8. Nguyên tắc tích hợp chatbot sau này

Voice module không nên phụ thuộc vào database hoặc framework backend
chatbot hiện tại. Khi tích hợp: - Thay hoặc bổ sung adapter gửi
`messages` tới chatbot backend. - Ánh xạ `session_id` Voice sang
conversation ID của chatbot. - Chuyển history sang cơ chế quản lý hội
thoại chung. - Giữ nguyên giao thức audio và frontend nếu phù hợp.

Không triển khai phần tích hợp này trong MVP.
