# 05 -- Báo Cáo Chi Tiết Luồng Vận Hành & Kiểm Định Tài Liệu Voicebot MVP

Tài liệu này tổng hợp đầy đủ luồng chạy thực tế theo từng hàm và file của hệ thống Voicebot MVP từ lúc gõ lệnh khởi chạy trên `CMD` / `PowerShell`, mở trình duyệt cho đến mọi kịch bản tương tác, kèm bảng đối chiếu tình trạng đồng bộ của bộ tài liệu dự án (`docs/01..04` và `README.md`).

---

## Phần I. Luồng Chạy Từ Lệnh CMD Đến Khi Trình Duyệt Sẵn Sàng (Bootstrap)

```mermaid
flowchart TD
    CMD["1. CMD: uv run uvicorn backend.main:app\nhoặc python -m backend.main"] --> Entry{"Cách khởi chạy?"}
    Entry -- "python -m backend.main" --> UviRun["main.py (__main__): uvicorn.run('backend.main:app')"]
    Entry -- "uvicorn backend.main:app" --> ImpMain["Nạp module backend/main.py"]
    UviRun --> ImpMain

    ImpMain --> ImpCfg["config.py: load_dotenv(.env), đọc prompt.txt,\nkhởi tạo singleton settings = Settings()"]
    ImpCfg --> ImpSrv["services.py: tạo singleton smart_turn_detector\n(ThreadPoolExecutor cho ONNX, chưa nạp file)"]
    ImpSrv --> ImpTxt["voice.py -> text_processing.py:\nbiên dịch sẵn regex TTSStreamer, normalize_tts, clean_md"]
    ImpTxt --> FastAPIApp["main.py: tạo FastAPI(lifespan=lifespan),\ngắn CORSMiddleware + StaticFiles('frontend/' tại '/')"]

    FastAPIApp --> Life["main.py: lifespan() trước khi mở cổng"]
    Life --> VCfg["settings.validate_config() kiểm tra cấu hình"]
    VCfg --> LoadONNX["asyncio.to_thread(smart_turn_detector.load_model)\nnạp smart-turn-v3.2-cpu.onnx + WhisperFeatureExtractor"]
    LoadONNX --> PortOpen["Server mở cổng http://127.0.0.1:8000\n(Khi Ctrl+C: shutdown() + close_http_client())"]

    PortOpen --> OpenWeb["2. Browser mở http://127.0.0.1:8000"]
    OpenWeb --> LoadUI["index.html -> style.css -> chạy IIFE trong app.js"]
    LoadUI --> InitJS["app.js: ánh xạ DOM, đọc DEFAULT_YOU_HINT / DEFAULT_BOT_HINT,\ngắn onclick 4 nút + pagehide -> gọi connect()"]
    InitJS --> WSOpen["app.js -> main.py: websocket_voice_endpoint()\ntạo VoiceSession(session_id) -> vòng while not session.closed"]
    WSOpen --> Handshake["ws.onopen gửi JSON session.start\n-> handle_event() trả session.ready + voice.state(idle)"]
```

### Thứ tự gọi hàm chi tiết khi khởi động

| Bước | File                                                                               | Hàm / Thành phần           | Nhiệm vụ thực hiện                                                                                                                                                                                                                                                                     |
| :----- | :--------------------------------------------------------------------------------- | :---------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | [`main.py`](../backend/main.py#L125-L135)                                         | Khối `__main__`            | Nếu chạy bằng `python -m backend.main`, gọi `uvicorn.run("backend.main:app", host, port, reload, app_dir)`.                                                                                                                                                                        |
| 2      | [`config.py`](../backend/config.py#L17-L174)                                      | Module init & `Settings()`  | Tìm nạp `.env` bằng `load_dotenv()`, đọc [`prompt.txt`](../backend/prompt.txt) vào `DEFAULT_SYSTEM_PROMPT`, tạo singleton `settings`.                                                                                                                                      |
| 3      | [`services.py`](../backend/services.py#L505-L597)                                 | `SmartTurnDetector()`       | Khởi tạo singleton `smart_turn_detector` kèm `ThreadPoolExecutor` (chưa nạp mô hình ONNX).                                                                                                                                                                                      |
| 4      | [`text_processing.py`](../backend/text_processing.py#L1-L380)                     | Module init                   | Được [`voice.py`](../backend/voice.py#L25) import; biên dịch sẵn toàn bộ regex lọc từ đệm, Markdown, chuẩn hóa số/ngày/tiền và tách câu.                                                                                                                              |
| 5      | [`main.py`](../backend/main.py#L37-L46)                                           | `lifespan()`                | Trước `yield`: gọi `settings.validate_config()`, chạy `smart_turn_detector.load_model()` trong `asyncio.to_thread` để nạp ONNX + `WhisperFeatureExtractor(chunk_length=8)`. Sau `yield` (lúc `Ctrl+C` tắt server): gọi `shutdown()` và `close_http_client()`. |
| 6      | [`main.py`](../backend/main.py#L49-L123)                                          | `FastAPI` + `StaticFiles` | Gắn CORS middleware, đăng ký `/health`, `/ws/voice` và mount thư mục `frontend/` tại `/`.                                                                                                                                                                                  |
| 7      | [`index.html`](../frontend/index.html) & [`app.js`](../frontend/app.js#L45-L179) | `connect()`                 | Trình duyệt tải HTML/CSS/JS, đọc `data-placeholder` mặc định từ HTML, mở WebSocket `/ws/voice` → backend tạo `VoiceSession` tại `websocket_voice_endpoint()`. `ws.onopen` gửi `session.start` → backend trả `session.ready` + `voice.state = "idle"`.      |

---

## Phần II. Luồng Nghiệp Vụ Chuẩn: Nói Bình Thường (Smart Turn → LLM → TTS → Loa)

```mermaid
sequenceDiagram
    autonumber
    participant UI as "app.js (Browser / Worklet)"
    participant GW as "main.py (FastAPI WS)"
    participant VO as "voice.py (VoiceSession)"
    participant SV as "services.py (Adapters & ONNX)"
    participant TP as "text_processing.py"

    Note over UI,SV: Bấm 'Bắt đầu nói' -> Khởi tạo Audio và mở STT
    UI->>UI: "btnTalk.onclick -> ensureAudio() -> setupAecLoopback() + MicProcessor"
    UI->>GW: "startListeningTurn() -> turn.start"
    GW->>VO: "_handle_ws_message() -> handle_event() -> start_turn()"
    VO->>SV: "STTClient.connect() -> _send_config() + _recv_loop()"
    VO->>VO: "spawn _forward_stt() + set_state(listening)"
    VO-->>UI: "voice.state(listening, turn_id) -> setCallButtons(true)"

    loop Mỗi 100ms khi đang nói
        UI->>GW: "MicProcessor._sendPcm() -> ws.send(PCM 16kHz)"
        GW->>VO: "_handle_ws_message() -> handle_audio_frame()"
        VO->>VO: "turn.buffer_audio() (đệm 8s cho Smart Turn)"
        VO->>SV: "STTClient.send_audio(pcm)"
        SV-->>VO: "_handle_message() -> event_queue (partial / vad)"
        VO-->>UI: "_forward_stt() -> transcript.partial -> cập nhật #you"
    end

    Note over UI,VO: Phát hiện dứt câu bằng VAD + Smart Turn (hoặc Fallback 5s)
    SV-->>VO: "vad = speech_ended -> _handle_vad()"
    VO->>VO: "_run_smart_turn() đợi 0.3s -> turn.get_audio()"
    VO->>SV: "smart_turn_detector.predict() -> _sync_predict()"
    Note over VO: "prob >= 0.70 -> gọi finish_turn()"
    VO->>VO: "finish_turn(): tách stt -> turn.clear() -> set_state(thinking)"
    VO-->>UI: "voice.state(thinking) -> recording=false, resetAudioInputState()"

    Note over VO,TP: Chốt STT và chạy song song Producer (LLM) - Consumer (TTS)
    VO->>SV: "_process_turn() -> _finish_stt() -> STTClient.finish()"
    VO-->>UI: "emit_event(transcript.final)"
    VO->>TP: "is_filler(final_text) == False"
    VO->>VO: "_run_pipeline() mở TaskGroup: _produce_llm || _stream_tts"

    par Producer (_produce_llm)
        VO->>SV: "build_messages() -> llm_stream() + _build_llm_payload() + _parse_sse_delta()"
        SV-->>VO: "delta text"
        VO-->>UI: "assistant.text.delta -> nối chữ vào #bot"
        VO->>TP: "TTSStreamer.feed(delta) -> clean_md() -> normalize_tts() -> split_tts()"
        TP-->>VO: "đẩy câu sạch vào asyncio.Queue(maxsize=8)"
    and Consumer (_stream_tts)
        VO->>SV: "tts_stream() -> _wav_data_offset() + _process_tts_chunk()"
        SV-->>VO: "PCM 48kHz 16-bit chẵn byte"
        VO-->>UI: "voice.state(speaking) + emit_audio_chunk(JSON + Binary PCM)"
        UI->>UI: "onMessage() -> playPcm() -> playHead -> Loa"
    end

    VO->>VO: "_finalize_llm(): lưu history, emit assistant.text.done"
    VO->>VO: "_finish_tts(): emit assistant.audio.done + _fallback()"
    UI->>UI: "scheduleAutoListen() đợi playingSources rỗng"
    UI->>GW: "playback.done"
    GW->>VO: "handle_event() -> set_state(idle)"
    UI->>UI: "queueNextTurn() đợi 600ms -> startListeningTurn()"
```

### Chi tiết xử lý ở từng tầng

1. **Thu âm & chống dội âm ([`app.js`](../frontend/app.js#L65-L121), [`L288-L384`](../frontend/app.js#L288-L384)):**
   - `ensureAudio()` tạo `AudioContext`, dựng `setupAecLoopback()` (2 `RTCPeerConnection` nối audio TTS vào thẻ `<audio>` để kích hoạt Acoustic Echo Cancellation của trình duyệt), lọc mic qua `highpass 85Hz` + `lowpass Butterworth 7.2kHz`, rồi `AudioWorklet` (`MicProcessor`) resample xuống **16 kHz Mono PCM16LE** và gửi mỗi 100ms (`1600` samples = `3200` bytes).
2. **Đệm audio & đánh giá dứt câu ([`voice.py`](../backend/voice.py#L479-L605)):**
   - `handle_audio_frame()` thực hiện 3 tác vụ: lưu đệm vòng tối đa ~8s (`buffer_audio`), đẩy frame sang Qwen3-ASR (`send_audio`), và kiểm tra timeout im lặng dự phòng `is_silence_timeout()` (mặc định 5s).
   - Khi nhận VAD `speech_ended`, `_handle_vad()` chạy `_run_smart_turn()` (ngủ 0.3s debounce → gọi `smart_turn_detector.predict()` chuẩn hóa zero-mean/unit-variance + pad trái 8s trên ONNX → nếu `prob >= 0.70` tự động gọi `finish_turn()`).
3. **Chuẩn hóa giọng đọc & Streaming song song ([`text_processing.py`](../backend/text_processing.py), [`voice.py`](../backend/voice.py#L296-L418)):**
   - `_produce_llm()` đẩy nguyên văn token `assistant.text.delta` về UI, đồng thời đưa qua `TTSStreamer`: `clean_md()` (xóa ký hiệu Markdown/URL) → ghép câu quá ngắn (`< 8` ký tự) → capping (`max_sentences = 8`, `max_chars = 600`) → `normalize_tts()` (đọc hotline, SĐT, ngày tháng, khoảng giá trị/thời gian, đơn vị đo, thuật ngữ công nghệ, mã vận đơn) → `split_tts()` (cắt đoạn ≤ 450 ký tự) → đẩy vào `queue`.
   - `_stream_tts()` rút từng câu gọi `tts_stream()`, bóc RIFF/WAVE header động qua `_wav_data_offset()` + căn chẵn sample 16-bit qua `_process_tts_chunk()`, rồi gọi `emit_audio_chunk()` gửi liền cặp `[JSON metadata, Binary PCM]` dưới một `send_lock`.
4. **Lập lịch phát loa & tự nghe lại ([`app.js`](../frontend/app.js#L410-L494)):**
   - `playPcm()` tạo `Int16Array` view chẵn byte (zero-copy), chuyển sang `AudioBuffer` 48 kHz và xếp nối tiếp theo `playHead`.
   - Khi nhận `assistant.audio.done`, `scheduleAutoListen()` chờ `playingSources` rỗng → gửi `playback.done` → gọi `queueNextTurn()` đợi 600ms rồi gọi `startListeningTurn()` mở lượt nghe kế tiếp.

---

## Phần III. Các Kịch Bản Điều Khiển & Tình Huống Đặc Biệt

```mermaid
flowchart TD
    C2["Case 2: Đang nói bấm 'Gửi' (btnFinish)"] --> C2S["app.js stopTalking():\ngửi 'flush' cho MicProcessor -> nhận flushed\n-> gửi turn.finish(turn_id)"]
    C2S --> C2B["voice.py handle_event() -> finish_turn(require_turn_id=True)\nhủy task listening -> set_state(THINKING) -> _process_turn()"]

    C3["Case 3: Bot đang nói bấm 'Dừng nói' (btnStop)"] --> C3S["app.js teardownCallState():\n- stopPlayback(): dừng AudioBufferSourceNode, discarding=true\n- releaseMicrophone(), callActive=false, setCallButtons(false)\ngửi session.cancel(turn_id)"]
    C3S --> C3B["voice.py cancel_turn():\n- current_turn.close(), cancel_tasks() ngắt TaskGroup LLM/TTS\n- Không commit history dở dang\n- emit playback.stop -> set_state(IDLE, force=True)"]

    C4["Case 4: Bot đang nói bấm 'Hội thoại mới' (btnReset)"] --> C4S["app.js teardownCallState()\nreset #you/#bot về DEFAULT_HINT -> gọi connect()\nđóng WS cũ và mở WS mới"]
    C4S --> C4B["WS cũ -> main.py finally:\nsession.close() xóa turn, tasks, history.clear()"]
    C4S --> C4C["WS mới -> VoiceSession mới (session_id mới, history=[])\nsession.start -> session.ready + idle"]

    C5["Case 5: Nói từ đệm ('ừ', 'à') / STT rỗng"] --> C5B["voice.py _process_turn():\nnot final_text hoặc is_filler() == True\n-> bỏ qua LLM, _speak_prompt() -> normalize_tts() -> _stream_tts()"]

    C6["Case 6: TTS không tạo audio (0 byte / chỉ Markdown/URL)"] --> C6B["voice.py _finish_tts(): state == THINKING -> set_state(IDLE)\napp.js: voice.state(idle) với prevState=='thinking' -> queueNextTurn()"]

    C7["Case 7: Lỗi STT / LLM / TTS"] --> C7B["voice.py _fail_turn():\n_drain_queue(), current_turn.close() (giữ turn_id),\nplayback.stop + session.error -> set_state(ERROR)"]
```

| Kịch bản                                                    | Hành vi nổi bật & Bảo vệ an toàn                                                                                                                                                                                                      |
| :------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Case 2 — Đang nói bấm "Gửi"**                    | `stopTalking()` flush nốt phần audio lẻ `< 100ms` trong `MicProcessor` (kèm timer fallback 500ms); nếu Smart Turn vừa tự chốt turn trùng lúc bấm, `turn_lock` + kiểm tra `VoiceState.LISTENING` chặn xử lý trùng. |
| **Case 3 — Đang nghe/xử lý/nói bấm "Dừng nói"** | Frontend ngắt ngay mọi `AudioBufferSourceNode` và đặt `discarding = true` để bỏ các chunk trôi giữa mạng. Backend hủy `TaskGroup` LLM/TTS và giữ nguyên quy tắc không ghi lượt dở dang vào `history`.         |
| **Case 4 — Bấm "Hội thoại mới"**                   | Frontend dừng mic/loa, khôi phục placeholder từ HTML, đóng WS cũ (kích hoạt `finally: await session.close()` dọn RAM trên backend) và mở WS + `VoiceSession` mới.                                                         |
| **Case 5 — Chỉ nói từ đệm / im lặng**            | `is_filler()` chặn gọi LLM, gọi `_speak_prompt()` phát câu nhắc ngắn qua ZeroTTS và không đưa vào `history`.                                                                                                              |
| **Case 6 — TTS rỗng (`THINKING -> IDLE`)**          | `app.js` nhận `voice.state = "idle"` khi `prevState === "thinking"` sẽ tự gọi `queueNextTurn()` mở mic tiếp sau 600ms, tránh kẹt `"Chờ nghe tiếp..."`.                                                                  |
| **Case 7 — Lỗi dịch vụ**                            | `_fail_turn()` xả queue, đóng tài nguyên trong turn nhưng giữ `turn_id` cho event lỗi, và khóa state ở `ERROR` cho đến khi người dùng bấm nói lại hoặc Reset.                                                     |