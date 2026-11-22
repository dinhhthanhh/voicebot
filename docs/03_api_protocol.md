# 03 -- API tích hợp và giao thức Frontend/Backend

**Phiên bản:** 1.0

## 1. Quy ước

Tài liệu này phân biệt: - **API bên ngoài:** STT, LLM, TTS do dịch vụ AI
cung cấp. - **Protocol nội bộ:** WebSocket do dự án Voicebot tự thiết kế
giữa frontend và backend.

Không coi các event nội bộ dưới đây là API có sẵn của VNPost.

## 2. WebSocket nội bộ Frontend -- Backend

Endpoint dự kiến:

``` text
ws://localhost:8000/ws/voice
```

Production phải dùng `wss://` sau khi cấu hình TLS/reverse proxy.

Một WebSocket nội bộ tồn tại trong một Voice session. Dùng JSON text cho
control/event và binary cho audio. Vì WebSocket bảo toàn thứ tự message,
khi gửi audio TTS, backend gửi metadata JSON ngay trước binary payload
tương ứng; frontend phải đọc hai message theo cặp.

### 2.1 Client → Backend

  Event              Trường chính         Ý nghĩa
  ------------------ -------------------- ---------------------------------------------
  `session.start`    --                   Khởi tạo Voice session
  `turn.start`       `turn_id` tùy chọn   Bắt đầu lượt nói mới
  `turn.finish`      `turn_id` tùy chọn   Kết thúc lượt nói của người dùng
  `playback.done`    `turn_id` tùy chọn   Client đã phát hết audio, chuyển bot về idle
  `session.stop`     --                   Đóng session
  `session.cancel`   `turn_id` tùy chọn   Hủy lượt hiện tại (barge-in / reset)

Binary message từ frontend là microphone PCM sau khi chuẩn hóa: -
mono; - PCM16LE; - 16 kHz; - frame có số byte chẵn.

### 2.2 Backend → Client

  -------------------------------------------------------------------------
  Event                     Trường chính            Ý nghĩa
  ------------------------- ----------------------- -----------------------
  `session.ready`           `session_id`            Session sẵn sàng

  `session.stopped`         `session_id`            Xác nhận session đã dừng

  `voice.state`             `state`, `turn_id`      Trạng thái hiện tại

  `transcript.partial`      `turn_id`, `text`       STT tạm thời,
                                                    cumulative

  `transcript.final`        `turn_id`, `text`       STT cuối

  `assistant.text.delta`    `turn_id`, `text`       Phần text mới từ LLM

  `assistant.text.done`     `turn_id`               LLM hoàn tất

  `assistant.audio.chunk`   `turn_id`, `chunk_id`,  Metadata của binary
                            audio metadata          audio tiếp theo

  `assistant.audio.done`    `turn_id`               Hoàn tất audio của lượt

  `playback.stop`           `turn_id`               Dừng và xóa queue audio
                                                    cũ (khi bị ngắt lời)

  `session.error`           `code`, `message`       Lỗi có thể hiển thị
  -------------------------------------------------------------------------

### 2.3 Ví dụ control messages

Khởi tạo:

``` json
{
  "type": "session.start"
}
```

Bắt đầu lượt:

``` json
{
  "type": "turn.start"
}
```

Kết thúc lượt:

``` json
{
  "type": "turn.finish",
  "turn_id": "turn_001"
}
```

Event partial:

``` json
{
  "type": "transcript.partial",
  "session_id": "session_001",
  "turn_id": "turn_001",
  "text": "Xin chào các bạn"
}
```

Event audio metadata (binary payload kế tiếp là audio chunk):

``` json
{
  "type": "assistant.audio.chunk",
  "session_id": "session_001",
  "turn_id": "turn_001",
  "chunk_id": 1,
  "format": "pcm_s16le",
  "sample_rate": 48000,
  "channels": 1
}
```

**Lưu ý:** `format`, `sample_rate`, `channels` ở ví dụ trên chỉ là hợp
đồng nội bộ mục tiêu. Chỉ phát audio raw PCM nếu adapter đã xác minh và
loại bỏ/chuẩn hóa WAV container đúng cách. Nếu stream còn WAV header
hoặc codec khác, protocol phải mô tả chính xác format thực tế.

### 2.4 Ghép metadata và binary

Quy tắc bắt buộc: 1. Backend gửi một JSON `assistant.audio.chunk`. 2.
Backend gửi ngay sau đó đúng một binary message cho chunk đó. 3. Không
xen event JSON khác giữa metadata và binary. 4. Frontend chỉ nhận binary
khi đang chờ audio payload theo metadata. 5. Kiểm tra `turn_id`; bỏ qua
chunk thuộc lượt đã bị hủy.

Nếu sau này cần multiplex nhiều luồng audio, chuyển sang envelope nhị
phân có header hoặc mã hóa payload rõ ràng. Không cần làm phức tạp ở
MVP.

## 3. STT -- Qwen3-ASR-1.7B-ft

### 3.1 Thông tin theo tài liệu cung cấp

-   WebSocket streaming.
-   Model: `qwen3-asr-1p7b-ft`.
-   Language: `Vietnamese`.
-   Audio: PCM16LE, mono, 16 kHz.
-   Binary audio frames, không Base64.
-   Tối đa 1 MiB/frame.
-   Tối đa 600 giây/session.
-   Một kết nối WebSocket = một ASR session.
-   `start` là JSON message đầu tiên.
-   `finish` kết thúc session; server trả `final` rồi đóng connection.
-   `partial` là văn bản cộng dồn, frontend phải thay thế nội dung trước
    đó.
-   VAD chọn bằng trường `vad`, giá trị `silero`, `webrtc` hoặc `off`.

### 3.2 Start message

``` json
{
  "type": "start",
  "request_id": "request-uuid",
  "model": "qwen3-asr-1p7b-ft",
  "language": "Vietnamese",
  "sample_rate": 16000,
  "encoding": "pcm_s16le",
  "channels": 1,
  "vad": "silero"
}
```

Sau khi connect: 1. Gửi `start`. 2. Chờ `ready`. 3. Gửi binary PCM
frames. 4. Đọc `vad` và `partial`. 5. Gửi `{"type":"finish"}` khi turn
kết thúc. 6. Chờ `final`. 7. Đóng/giải phóng client socket.

### 3.3 STT response events

  ---------------------------------------------------------------------------------
  Type                    Ý nghĩa                           Xử lý
  ----------------------- --------------------------------- -----------------------
  `ready`                 Session STT sẵn sàng              Bắt đầu gửi audio

  `vad`                   `speech_started`/`speech_ended`   Phục vụ phát hiện biên thoại
                                                            và kích hoạt Smart Turn

  `partial`               Text cumulative                   Thay thế transcript tạm

  `final`                 Text hoàn chỉnh                   Commit user message và
                                                            gọi LLM

  `error`                 Lỗi API                           Map sang lỗi nội bộ
  ---------------------------------------------------------------------------------

### 3.4 Giới hạn và lỗi

-   `429 OVERLOADED`: dịch vụ quá tải.
-   `429 QUEUE_TIMEOUT`: chờ slot quá 5 giây.
-   `400 INVALID_AUDIO_FORMAT`: sai sample rate/encoding/channels hoặc
    frame byte lẻ.
-   `413 MESSAGE_TOO_LARGE`: frame lớn hơn 1 MiB.
-   `413 AUDIO_TOO_LONG`: quá 600 giây.
-   `500 MODEL_INFERENCE_FAILED`: lỗi inference/VAD.
-   Khi lỗi, server gửi `error` rồi đóng connection (1008 với lỗi nghiệp
    vụ, 1011 với lỗi nội bộ theo tài liệu).

### 3.5 URL cần xác minh

Tài liệu API có hai cách ghi: - `API.txt`:
`wss://ai.vnpost.vn/qwen3-asr-1p7b/`. - `streaming-asr-v1.md`: endpoint
chuẩn là `/v1/stream` sau reverse proxy TLS; ví dụ dùng
`wss://{domain}/v1/stream`.

Phải xác nhận URL thực tế từ môi trường đang chạy trước khi đưa vào
`.env`.

## 4. LLM -- Qwen 27B Chat Completions

Endpoint đã có trong `API.txt`:

``` text
https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions
```

Request mẫu nguồn cung cấp:

``` json
{
  "messages": [
    {"role": "user", "content": "xin chào bạn"}
  ],
  "temperature": 0,
  "reasoning_effort": "low"
}
```

API mẫu hiện chưa xác nhận: - `model` có bắt buộc không; - cách bật
streaming; - response streaming là SSE hay dạng khác; - trường JSON nào
chứa text delta; - có yêu cầu authorization/header riêng không.

Adapter chỉ được hoàn thiện parser sau khi có response thực tế từ
request streaming thành công. Nếu endpoint tương thích OpenAI Chat
Completions, kiểm tra `stream: true` và format chunk thực tế; không mặc
định parser nếu chưa thử.

## 5. TTS -- ZeroTTS

Base URL từ Postman collection:

``` text
https://ai.vnpost.vn/zero-tts
```

Streaming endpoint:

``` text
POST /tts/stream
```

Voice mẫu: `maichi`.

Request body mẫu:

``` json
{
  "text": "Xin chào, tôi có thể giúp gì cho bạn?",
  "voice": "maichi",
  "cfg_scale": 1.0,
  "audio_temperature": 0.8,
  "text_temperature": 1.0,
  "audio_topk": 25,
  "audio_topp": 0.95,
  "audio_repetition_penalty": 1.2,
  "min_frames": 4,
  "max_frames": 1500
}
```

Postman collection mô tả HTTP Chunked Streaming, WAV chunks và audio bắt
đầu có thể từ khoảng 0.3 giây. Endpoint `/tts` full audio được mô tả trả
`audio/wav`, 48 kHz, PCM16 mono. Không được suy diễn cấu trúc chunk của
`/tts/stream` từ endpoint `/tts`.

Trước khi phát: 1. Gọi thử `/tts/stream` với câu ngắn. 2. Lưu response
headers và bytes đầu của stream. 3. Xác định chunk có WAV header hay
không. 4. Xác định sample rate, channels, bit depth, codec. 5. Xác định
cách ghép hoặc tách chunk. 6. Viết adapter chuyển về format frontend hỗ
trợ. 7. Kiểm tra audio không bị lặp header, rè, lệch tốc độ hoặc mất
đoạn.

## 6. Smart Turn v3.2

Smart Turn chạy ở backend; không phải endpoint STT/LLM/TTS được mô tả
trong các tài liệu API trên.

Cần xác minh trước khi implement: - package/model source và phiên bản
chính xác; - cách load model; - sample rate và định dạng input; - độ dài
audio đầu vào; - output schema/threshold; - thiết bị chạy CPU/GPU; -
latency khi inference.

`services.py` cung cấp adapter `smart_turn_predict(...)`; `voice.py` chỉ
nhận kết quả dự đoán, không phụ thuộc chi tiết thư viện.

## 7. Cấu hình môi trường

`.env.example` đề xuất:

``` env
APP_HOST=127.0.0.1
APP_PORT=8000
APP_LOG_LEVEL=INFO

LLM_CHAT_COMPLETIONS_URL=https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions
LLM_MODEL=
LLM_TEMPERATURE=0
LLM_REASONING_EFFORT=low
LLM_TIMEOUT_SECONDS=90

STT_WS_URL=
STT_MODEL=qwen3-asr-1p7b-ft
STT_LANGUAGE=Vietnamese
STT_SAMPLE_RATE=16000
STT_VAD=silero

TTS_BASE_URL=https://ai.vnpost.vn/zero-tts
TTS_STREAM_PATH=/tts/stream
TTS_VOICE=maichi
TTS_TIMEOUT_SECONDS=90

VOICE_MAX_HISTORY_PAIRS=8
```

Không commit `.env`; commit `.env.example` với secret/URL nhạy cảm đã
loại bỏ hoặc để placeholder. Tên model và STT URL để trống đến khi xác
minh.

## 8. Bảo mật và vận hành

-   Không gửi API secret xuống frontend.
-   Chỉ backend gọi dịch vụ AI.
-   Giới hạn kích thước audio frame và thời lượng session.
-   Cấu hình timeout cho mọi dịch vụ.
-   Không log nội dung audio hoặc transcript nhạy cảm mặc định.
-   Kiểm tra Origin/WebSocket access khi đưa ra môi trường dùng chung.
-   Production dùng TLS (`wss://`, `https://`).
