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
  `turn.start`       --                   Bắt đầu lượt nói mới (backend tự sinh `turn_id`)
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

  `assistant.text.done`     `turn_id`, `text`         LLM hoàn tất toàn văn

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

**Lưu ý:** Endpoint `/tts/stream` của ZeroTTS trả dữ liệu trong container WAV (`audio/wav`). Adapter `services.tts_stream` tự động bóc RIFF/WAVE header qua `_wav_data_offset()` và căn chẵn sample 16-bit qua `_process_tts_chunk()` trước khi gửi raw `pcm_s16le` 48 kHz mono về frontend.

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

### 3.5 Endpoint thực tế đã xác minh

Endpoint WebSocket mặc định trong cấu hình:

``` text
wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream
```

## 4. LLM -- Qwen 27B Chat Completions

Endpoint thực tế (tương thích OpenAI Chat Completions SSE):

``` text
https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions
```

Request payload do `services._build_llm_payload()` gửi:

``` json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "xin chào bạn"}
  ],
  "temperature": 0,
  "stream": true,
  "reasoning_effort": "low"
}
```

Trường `model` được đính kèm nếu `LLM_MODEL` khác rỗng. Adapter `services.llm_stream()` đọc từng dòng `data:` chuẩn SSE, gọi `_parse_sse_delta()` để tách `choices[0].delta.content` (bỏ qua `reasoning_content`) và dừng tại `[DONE]`.

## 5. TTS -- ZeroTTS

Base URL và endpoint streaming đang dùng:

``` text
POST https://ai.vnpost.vn/zero-tts/tts/stream
```

Request body do `services.tts_stream()` gửi:

``` json
{
  "text": "Xin chào, tôi có thể giúp gì cho bạn?",
  "voice": "maichi"
}
```

Response trả stream WAV (`48 kHz`, `pcm_s16le`, `mono`). Adapter `services.tts_stream()` bóc header RIFF/WAVE ở chunk đầu tiên qua `_wav_data_offset()` (hỗ trợ cả extended WAV chunks), giữ/ghép byte lẻ qua `_process_tts_chunk()` để đảm bảo mỗi chunk truyền về frontend đều là raw PCM 16-bit chẵn byte.

## 6. Smart Turn v3.2

Smart Turn v3.2 ONNX chạy cục bộ ở backend qua `services.SmartTurnDetector` (singleton `smart_turn_detector`):
- Đường dẫn model mặc định: `models/smart-turn-v3.2-cpu.onnx` (`SMART_TURN_MODEL_PATH`).
- Đầu vào: tối đa 8 giây PCM 16-bit 16 kHz cuối cùng, chuẩn hóa zero-mean/unit-variance, pad trái và trích xuất đặc trưng log-mel qua `WhisperFeatureExtractor(chunk_length=8)`.
- Đầu ra: xác suất `[0.0 .. 1.0]`; `voice.py` tự động gọi `finish_turn()` khi `prob >= SMART_TURN_THRESHOLD` (`0.70`), giới hạn suy luận `3.0s` trên `ThreadPoolExecutor` riêng.

## 7. Cấu hình môi trường

Đồng bộ với `backend/.env.example`:

``` env
APP_HOST=127.0.0.1
APP_PORT=8000
APP_LOG_LEVEL=INFO
APP_RELOAD=false

LLM_CHAT_COMPLETIONS_URL=https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions
LLM_MODEL=
LLM_TEMPERATURE=0
LLM_REASONING_EFFORT=low
LLM_TIMEOUT_SECONDS=90

STT_WS_URL=wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream
STT_MODEL=qwen3-asr-1p7b-ft
STT_LANGUAGE=Vietnamese
STT_SAMPLE_RATE=16000
STT_VAD=silero
STT_MAX_RETRIES=2
STT_CONNECT_TIMEOUT=10.0
STT_FINISH_TIMEOUT=5.0
STT_MAX_DURATION_SECONDS=600

TTS_BASE_URL=https://ai.vnpost.vn/zero-tts
TTS_STREAM_PATH=/tts/stream
TTS_VOICE=maichi
TTS_TIMEOUT_SECONDS=90
TTS_SAMPLE_RATE=48000
TTS_SPLIT_LIMIT=450
TTS_MIN_CHARS=8
TTS_MAX_SENTENCES=8
TTS_MAX_CHARS=600

VOICE_MAX_HISTORY_PAIRS=8
TURN_MAX_AUDIO_BYTES=256000

SMART_TURN_MODEL_PATH=models/smart-turn-v3.2-cpu.onnx
SMART_TURN_THRESHOLD=0.70
SMART_TURN_INFERENCE_TIMEOUT_SECONDS=3.0
SMART_TURN_FALLBACK_SECONDS=5.0
SMART_TURN_MAX_WORKERS=2
```

Không commit `.env`; chỉ commit `backend/.env.example` không chứa thông tin nhạy cảm.

## 8. Bảo mật và vận hành

-   Không gửi API secret xuống frontend.
-   Chỉ backend gọi dịch vụ AI.
-   Giới hạn kích thước audio frame và thời lượng session.
-   Cấu hình timeout cho mọi dịch vụ.
-   Không log nội dung audio hoặc transcript nhạy cảm mặc định.
-   Kiểm tra Origin/WebSocket access khi đưa ra môi trường dùng chung.
-   Production dùng TLS (`wss://`, `https://`).
