# 01 -- Thiết kế kiến trúc Voicebot MVP

**Phiên bản:** 1.0

## 1. Mục tiêu

Xây dựng Voicebot độc lập với chatbot hiện tại. Người dùng có thể nói
qua microphone, hệ thống nhận dạng tiếng Việt thời gian thực, gửi nội dung sang LLM và
phát câu trả lời bằng giọng nói trực tiếp. Hệ thống tích hợp cơ chế phát hiện ngắt lượt
tự động (Silero VAD + Smart Turn v3.2 ONNX) và hỗ trợ hủy lượt thoại tức thì (`session.cancel`).

Kiến trúc phải cho phép thay thế từng nhà cung cấp STT/LLM/TTS mà không
phải viết lại toàn bộ luồng hội thoại.

## 2. Sơ đồ kiến trúc

``` mermaid
flowchart TB
    FE["Frontend Demo\nindex.html / style.css / app.js"]
    GW["main.py\nFastAPI + WebSocket Gateway"]
    ORCH["voice.py\nSession / Turn Orchestrator"]
    NLP["text_processing.py\nTTSStreamer & TTS Normalizer"]
    ADAPTER["services.py\nSTT / LLM / TTS / Smart Turn adapters"]
    CFG["config.py\nSettings"]
    STT["Qwen3-ASR WebSocket\nSilero VAD tích hợp"]
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

## 2.1 Luồng xử lý chi tiết (Event Flow & Audio Bytes Flow)

Hệ thống phân tách thành hai luồng dữ liệu độc lập qua kết nối WebSocket: **Luồng sự kiện JSON (Control Plane)** và **Luồng dữ liệu byte âm thanh (Data Plane)**.

### A. Luồng nhận sự kiện JSON (Control Plane)
Điều phối trạng thái phiên thoại và kiểm soát vòng đời lượt thoại (`session.start`, `turn.start`, `turn.finish`, `playback.done`, `session.cancel`).

``` mermaid
flowchart TD
    Client["Client / Frontend (app.js)"] -- "Gửi JSON qua WebSocket" --> WS["_handle_ws_message (main.py)"]
    WS -- "json.loads(text)" --> HandleEvent["session.handle_event(payload) (voice.py)"]

    HandleEvent --> Switch{"Loại sự kiện (type)?"}

    %% session.start
    Switch -- "session.start" --> SStart["Khởi tạo phiên"]
    SStart --> EmitReady["emit_event('session.ready')\nset_state(IDLE)"]

    %% turn.start
    Switch -- "turn.start" --> StartTurn["start_turn()"]
    StartTurn --> ST1["1. Sinh turn_id mới\n2. Dọn turn cũ"]
    ST1 --> ST2["3. STTClient.connect()\n4. Bật ngầm _forward_stt"]
    ST2 --> ST3["5. set_state(LISTENING)\n(Sẵn sàng nhận tiếng nói)"]

    %% turn.finish
    Switch -- "turn.finish" --> FinishTurn["finish_turn()"]
    FinishTurn --> FT1["set_state(THINKING)\n(Khóa mic, bắt đầu xử lý AI)"]
    FT1 --> FT2["_spawn_task(_process_turn)\n(Chuyển sang luồng LLM -> TTS)"]

    %% playback.done
    Switch -- "playback.done" --> PlaybackDone["Loa Client đã phát xong"]
    PlaybackDone --> SetIdle["set_state(IDLE)\n(Mở đường cho lượt nói tiếp theo)"]

    %% session.cancel
    Switch -- "session.cancel" --> Cancel["cancel_turn()"]
    Cancel --> Can1["Hủy mọi task, đóng STT\nemit_event('playback.stop')\nset_state(IDLE)"]
```

### B. Luồng nhận byte âm thanh từ Micro (Data Plane)
Hàm `handle_audio_frame` tiếp nhận frame âm thanh thô (PCM 16kHz Mono) từ micro gửi lên, chỉ làm đúng 3 việc:
1. Lưu vào kho đệm trượt `turn_audio_buffer` (tối đa ~8s).
2. Chuyển tiếp ngay sang STT Server (Qwen3-ASR) qua WebSocket nội bộ.
3. Kiểm tra đồng hồ đếm ngược im lặng dự phòng (5.0s).

``` mermaid
flowchart TD
    Micro["Micro Client (app.js)"] -- "Gửi Binary PCM 16kHz" --> WS["_handle_ws_message (main.py)"]
    WS -- "message['bytes']" --> AudioFrame["handle_audio_frame(pcm_data) (voice.py)"]

    AudioFrame --> Check{"Kiểm tra điều kiện:\n- Byte có chẵn không?\n- State có phải LISTENING?"}
    Check -- "Sai" --> Drop["Bỏ qua frame âm thanh này"]
    Check -- "Đúng" --> Action["Thực hiện 3 tác vụ song song:"]

    %% Tác vụ 1
    Action --> Task1["1. buffer_audio(pcm_data)\n(Lưu tích lũy vào kho turn_audio_buffer)"]

    %% Tác vụ 2
    Action --> Task2["2. stt_client.send_audio(pcm_data)\n(Bắn thẳng byte sang STT Server)"]

    %% Tác vụ 3
    Action --> Task3["3. is_silence_timeout()?\n(Nếu im lặng >= 5.0s)"]
    Task3 -- "Đúng" --> FinishFallback["finish_turn()\n(Kích hoạt dừng nghe dự phòng)"]
```

### C. Luồng nhận sự kiện JSON từ STT Server (STT & VAD Event Flow)
Sau khi STT Server nhận được âm thanh từ Luồng B, mô hình AI bên trong STT Server sẽ phân tích và trả về các gói tin JSON qua kết nối WebSocket nội bộ. **Chính tại đây mới là nơi nhận sự kiện VAD (`speech_ended`) để kích hoạt Smart Turn:**

``` mermaid
flowchart TD
    STTServer["STT Server (Qwen3-ASR + Silero VAD)"] -- "Gửi gói tin JSON" --> RecvLoop["_recv_loop() (services.py)"]
    RecvLoop --> Queue["stt_client.event_queue"]
    Queue --> Forward["_forward_stt() (voice.py)"]

    Forward --> EventType{"Loại sự kiện STT?"}

    %% partial
    EventType -- "type == 'partial'" --> Partial["emit_event('transcript.partial')\n(Hiển thị chữ đang nói lên màn hình Client)"]

    %% vad
    EventType -- "type == 'vad'" --> HandleVAD["_handle_vad(turn_id, state)"]
    HandleVAD --> VADState{"state là gì?"}

    VADState -- "'speech_started'\n(Đang nói)" --> CancelST["Hủy smart_turn_task đang chờ (nếu có)\nĐánh dấu is_vad_speaking = True"]

    VADState -- "'speech_ended'\n(Vừa ngừng nói)" --> TriggerST["_run_smart_turn()"]
    TriggerST --> PullAudio["get_audio()\n(Rút toàn bộ audio đã tích lũy ở Kho Luồng B)"]
    PullAudio --> STModel["smart_turn_detector.predict(audio)"]
    STModel -- "prob >= 0.70\n(AI tự tin dứt câu)" --> FinishSmartTurn["finish_turn()\n(Chuyển sang trả lời)"]
    STModel -- "prob < 0.70\n(Ngập ngừng)" --> WaitMore["Tiếp tục giữ LISTENING để nghe tiếp"]

    %% error
    EventType -- "type == 'error'" --> STTErr["Báo lỗi session.error\nChuyển state -> ERROR"]
```

### D. Điểm hội tụ chuyển giao: Pipeline Trả lời (LLM → TTS)
Khi lượt nghe kết thúc (`finish_turn`), toàn bộ câu nói được chốt và kích hoạt đường ống xử lý song song Producer - Consumer:

``` mermaid
flowchart TD
    FT["finish_turn()"] --> PT["_process_turn(turn_id)"]
    PT --> STTFinish["1. stt.finish() chốt final_text\n2. emit_event('transcript.final')"]
    STTFinish --> TG["3. _run_pipeline(turn_id, final_text)"]

    subgraph Pipeline["Đường ống Producer - Consumer song song (TaskGroup)"]
        TG --> Q["asyncio.Queue (maxsize=8)"]
        TG --> Producer["Producer: _produce_llm()"]
        TG --> Consumer["Consumer: _stream_tts()"]

        Producer -- "1. LLM Stream sinh text\n2. TTSStreamer tách câu &\nchuẩn hóa SĐT, ngày tháng" --> Q
        Q -- "3. Rút câu hoàn chỉnh (sentence)" --> Consumer
        Consumer -- "4. tts_stream(sentence)\n(Gọt WAV header & căn chẵn 16-bit)" --> EmitChunk["emit_audio_chunk()"]
    end

    EmitChunk --> Play["Trình duyệt Client nhận PCM chunk\nAudioContext phát ra loa ngay lập tức"]
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
với Smart Turn v3.2 để tự động phát hiện dứt câu. - Quản lý hủy lượt thoại
(`session.cancel`) và dọn dẹp tác vụ an toàn. - Kiểm tra `session_id`/`turn_id` để
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
    async def finish(self, timeout: float | None = None) -> str: ...
    async def close(self) -> None: ...

async def llm_stream(messages: list[dict[str, str]], timeout: float | None = None) -> AsyncIterator[str]: ...

async def tts_stream(text: str, voice: str | None = None, timeout: float | None = None) -> AsyncIterator[bytes]: ...

class SmartTurnDetector:
    def load_model(self) -> None: ...
    @property
    def is_ready(self) -> bool: ...
    async def predict(self, audio_bytes: bytes, sample_rate: int = 16000) -> float: ...
    def shutdown(self) -> None: ...
```

Adapter cần: - Quản lý kết nối, timeout và parse response. - Chuẩn hóa
lỗi thành exception nội bộ có ý nghĩa. - Không giữ lịch sử hội thoại. -
Không quyết định state của Voicebot.

### 3.4 `config.py` -- Configuration

-   Đọc biến môi trường.
-   Kiểm tra cấu hình bắt buộc khi khởi động.
-   Cung cấp settings cho các file còn lại.
-   Không ghi secret vào log.

### 3.5 `text_processing.py` -- Text Processing & `TTSStreamer`

-   Lọc từ đệm đơn thuần (`is_filler`) để tránh gọi LLM khi người dùng chỉ ngập ngừng (ừ, à, ờ...).
-   Làm sạch văn bản (`clean_md`): giữ text từ Markdown link, loại bỏ URL, ký hiệu inline (`*`, `_`, `` ` ``, `~`, `#`) và gạch đầu dòng.
-   Tách câu thời gian thực từ luồng LLM streaming (`TTSStreamer`) cho Zero-TTS kèm cơ chế capping an toàn tối đa 8 câu hoặc 600 ký tự.
-   Chuẩn hóa văn bản tiếng Việt sang ngữ điệu đọc tự nhiên cho TTS (`normalize_tts`: số điện thoại, tổng đài 1900/1800, ngày tháng `DD/MM/YYYY`, tiền tệ, số thập phân, đơn vị đo lường, từ công nghệ 5G/24h, mã vận đơn).

## 4. Quản lý session

Một kết nối frontend tương ứng một Voice session trong MVP.

Cấu trúc dữ liệu logic:

``` python
VoiceSession:
    session_id: str
    state: VoiceState
    history: list[dict[str, str]]
    current_turn: TurnState | None  # cung cấp @property current_turn_id
    active_tasks: set[asyncio.Task[Any]]
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
hoàn chỉnh. - Khi session đóng, hủy task, đóng STT/HTTP stream và dọn session khỏi RAM.
- MVP không đảm bảo giữ session nếu backend restart.

`VOICE_MAX_HISTORY_PAIRS` giới hạn mặc định 8 cặp user/assistant gần nhất
(không tính system prompt). Nếu cần giới hạn theo token, bổ sung cơ chế
cắt theo token ở giai đoạn tối ưu.

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
-   `error`: lỗi cần hiển thị.

State phải được chuyển đổi tập trung trong `voice.py`, không để frontend
tự suy đoán trạng thái backend.

## 7. Định dạng âm thanh nội bộ

Đầu vào STT theo tài liệu: - PCM signed 16-bit little-endian. - Mono. -
16,000 Hz. - Binary frames, không Base64.

Đầu ra TTS: - `/tts/stream` trả luồng WAV (RIFF/WAVE, 48 kHz, PCM 16-bit
mono). - Adapter `services.tts_stream` duyệt chunk động qua
`_wav_data_offset` để bóc header WAV ở gói đầu (hỗ trợ cả extended
header) và căn chẵn bội số 2 byte bằng `_process_tts_chunk` trước khi
stream raw PCM16LE về frontend.

Frontend nên thu bằng AudioWorklet và chuyển sample rate về 16 kHz mono
PCM16LE. Không giả định sample rate microphone luôn là 16 kHz.

## 8. Nguyên tắc tích hợp chatbot sau này

Voice module không nên phụ thuộc vào database hoặc framework backend
chatbot hiện tại. Khi tích hợp: - Thay hoặc bổ sung adapter gửi
`messages` tới chatbot backend. - Ánh xạ `session_id` Voice sang
conversation ID của chatbot. - Chuyển history sang cơ chế quản lý hội
thoại chung. - Giữ nguyên giao thức audio và frontend nếu phù hợp.

Không triển khai phần tích hợp này trong MVP.
