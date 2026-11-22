# 02 -- Luồng hội thoại thời gian thực, Smart Turn và ngắt lời

**Phiên bản:** 2.0 (Kiến trúc hội thoại hợp nhất)

## 1. Các khái niệm cốt lõi

- **Session:** Phiên kết nối Voice giữa client và backend gateway, duy trì qua nhiều lượt hỏi/đáp liên tục trong RAM.
- **Turn:** Một lượt tương tác hoàn chỉnh: người dùng nói và Voicebot xử lý phản hồi bằng văn bản lẫn giọng nói.
- **STT session:** Một kết nối WebSocket độc lập tới API ASR (Qwen3-ASR) cho từng turn cụ thể (`start` → gửi binary audio → `finish` → `final` → server đóng kết nối).
- **VAD (Voice Activity Detection):** Phát hiện biên đoạn tiếng nói thời gian thực (sử dụng Silero VAD tích hợp trong luồng STT).
- **Turn detection:** Đánh giá người dùng đã kết thúc ý định nói hay chưa bằng mô hình âm học **Smart Turn v3.2 ONNX** kết hợp cơ chế timeout im lặng an toàn.

---

## 2. Luồng hội thoại thời gian thực (Real-time Conversational Flow)

Hệ thống hoạt động theo cơ chế đàm thoại hai chiều liên tục. Người dùng có thể đàm thoại tự nhiên (hệ thống tự động phát hiện dứt câu) hoặc sử dụng các nút tương tác nhanh trên giao diện để điều khiển lượt nói.

``` mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend (Web Audio)
    participant GW as main.py (Gateway)
    participant VC as voice.py (Orchestrator)
    participant STT as Qwen3-ASR API
    participant TURN as Smart Turn v3.2
    participant LLM as Qwen 27B Chat
    participant TTS as Zero-TTS Streaming

    U->>FE: Bắt đầu nói (hoặc bấm "Bắt đầu nói")
    FE->>GW: turn.start
    GW->>VC: _start_turn()
    VC->>STT: WebSocket Connect + start JSON
    STT-->>VC: ready
    Note over VC: Chuyển trạng thái sang LISTENING

    loop Thu và truyền âm thanh
        U->>FE: Giọng nói liên tục
        FE->>GW: Binary PCM16LE frames (16 kHz mono)
        GW->>VC: handle_audio_frame()
        VC->>STT: Chuyển tiếp Binary audio frames
        STT-->>VC: transcript.partial (văn bản cộng dồn)
        VC-->>FE: transcript.partial
        FE-->>U: Hiển thị văn bản tạm thời
    end

    STT-->>VC: VAD event (speech_ended)
    VC->>TURN: smart_turn_predict(turn_audio_buffer)
    TURN-->>VC: Xác suất dứt câu (prob >= 0.70)
    Note over VC: Tự động chốt lượt (hoặc nhận turn.finish từ UI)

    VC->>STT: finish JSON
    STT-->>VC: final transcript
    VC-->>FE: transcript.final (chốt câu hỏi)
    Note over VC: Chuyển trạng thái sang THINKING

    par LLM Streaming & Dual-channel processing
        VC->>LLM: Gửi messages kèm lịch sử hội thoại
        loop Stream text deltas
            LLM-->>VC: Chunk text delta
            VC-->>FE: assistant.text.delta (Hiển thị toàn văn ngay lập tức)
            VC->>VC: SpeechFilter tách câu & chuẩn hóa tiếng Việt
        end
        LLM-->>VC: Kết thúc LLM stream
        VC-->>FE: assistant.text.done
    and TTS Chunked Streaming (Producer - Consumer)
        loop Với từng câu hoàn chỉnh từ SpeechFilter
            VC->>TTS: POST /tts/stream (văn bản đã chuẩn hóa)
            loop Stream audio chunks
                TTS-->>VC: Chunk audio (WAV bóc header -> PCM 48 kHz)
                VC-->>FE: assistant.audio.chunk (JSON) + Binary Payload
                FE-->>U: Lập lịch phát âm thanh mượt mà (Zero-latency)
            end
        end
        VC-->>FE: assistant.audio.done
    end

    Note over VC: Chuyển trạng thái sang IDLE (sẵn sàng cho lượt tiếp theo)
```

---

## 3. Cơ chế phát hiện kết thúc lượt hai tầng (VAD + Smart Turn)

Để mang lại trải nghiệm đàm thoại tự nhiên mà không cần người dùng phải bấm nút dừng:

1. **Tầng 1 -- Silero VAD (Real-time Speech Framing):**
   - Nhận diện sự kiện `speech_started` và `speech_ended` từ luồng ASR.
   - Khi có `speech_started`, hệ thống ghi nhận thời điểm bắt đầu nói và reset bộ đếm im lặng.
   - Khi có `speech_ended`, hệ thống nhận biết người dùng tạm ngưng nghỉ âm thanh.

2. **Tầng 2 -- Smart Turn v3.2 ONNX (Semantic Turn Completion):**
   - VAD chỉ nhận biết khoảng im lặng chứ không biết người dùng đã nói hết câu hay chỉ ngập ngừng giữa câu.
   - Ngay khi VAD báo `speech_ended`, backend trích xuất `turn_audio_buffer` (tối đa 8 giây âm thanh gần nhất) và đưa vào mô hình **Smart Turn v3.2 ONNX**.
   - Nếu xác suất dứt câu `prob >= settings.smart_turn_threshold` (mặc định 0.70): Hệ thống kích hoạt kết thúc lượt nói ngay lập tức mà không cần chờ thêm.

3. **Cơ chế Timeout dự phòng (Safety Silence Fallback):**
   - Nếu Smart Turn đánh giá chưa hết câu nhưng người dùng im lặng kéo dài vượt ngưỡng `smart_turn_fallback_seconds` (mặc định 5.0 giây), hệ thống sẽ tự động kích hoạt kết thúc lượt để tránh treo phiên. Quá trình suy luận của Smart Turn có timeout riêng `smart_turn_inference_timeout_seconds` (mặc định 3.0 giây).
   - Tuyệt đối không kích hoạt timeout khi VAD vẫn đang báo trạng thái có tiếng nói (`is_vad_speaking = True`).

---

## 4. Pipeline xử lý LLM Streaming và SpeechFilter TTS

Hệ thống triển khai kiến trúc **Dual-channel Streaming** độc lập giữa màn hình và kênh âm thanh:

``` mermaid
flowchart LR
    A["LLM Text Deltas"] --> B["Kênh hiển thị màn hình<br/>assistant.text.delta"]
    A --> C["SpeechFilter<br/>(Lọc markdown, code, KaTeX, URL)"]
    C --> D["Tách câu theo ranh giới câu tự nhiên"]
    D --> E["Hậu xử lý tiếng Việt<br/>normalize_text_for_tts"]
    E --> F["Queue đệm bất đồng bộ<br/>asyncio.Queue"]
    F --> G["Zero-TTS HTTP Stream Adapter"]
    G --> H["Bóc WAV header chunk 0<br/>Căn chỉnh byte chẵn"]
    H --> I["Kênh âm thanh<br/>Metadata JSON + Binary Payload"]
```

### Quy tắc an toàn:
- **Lược bỏ thành phần trực quan:** Code block, bảng Markdown, link Markdown, emoji, URL và công thức toán học được SpeechFilter lược bỏ khỏi kênh đọc, nhưng vẫn giữ đầy đủ trên kênh hiển thị màn hình.
- **Giới hạn độ dài phát âm (Capping):** Giới hạn tối đa 8 câu hoặc 600 ký tự đọc để tránh bot nói quá dài làm nghẽn hội thoại. Khi vượt ngưỡng, tự động bổ sung câu thông báo: *"Phần còn lại đã được hiển thị trên màn hình."*
- **Chuẩn hóa tiếng Việt cho giọng đọc:** Tự động đọc đúng số ghép, tiền tệ (`VNĐ`, `đồng`), số điện thoại, ngày tháng, địa chỉ và đơn vị đo lường.

---

## 5. Cơ chế ngắt lời tức thì (Barge-in)

Khi bot đang phản hồi (trạng thái `THINKING` hoặc `SPEAKING`), do kết nối STT của lượt cũ đã đóng sau lệnh `finish`, backend không thể phụ thuộc vào VAD từ ASR server. Thay vào đó, backend trực tiếp phân tích năng lượng âm thanh hiệu dụng (RMS) trên từng frame microphone nhận được kết hợp gom đệm các frame trong giai đoạn debounce:

``` mermaid
flowchart TD
    A["Người dùng nói đè khi Bot đang nói/suy nghĩ"] --> B["Backend tính RMS frame audio<br/>Gom các frame trong barge_in_frames<br/>(Debounce N frame liên tiếp đạt ngưỡng RMS)"]
    B --> C["Hủy ngay lập tức các tác vụ LLM/TTS đang stream"]
    C --> D["Gửi playback.stop về frontend"]
    D --> E["Frontend dừng phát tức thì & xóa hàng đợi audio"]
    C --> F["Đánh dấu turn cũ: INTERRUPTED"]
    F --> G["Cấp phát turn_id mới"]
    G --> H["Nối toàn bộ barge_in_frames gửi sang STT mới & chuyển LISTENING"]
```

### Nguyên tắc bảo đảm:
- **Không mất âm tiết đầu:** Toàn bộ các frame âm thanh trong thời gian debounce được lưu trong `barge_in_frames` và chuyển tiếp trọn vẹn sang STT session mới ngay khi xác nhận ngắt lời, tránh mất 120–180ms âm tiết đầu tiên của người dùng.
- **Frontend phản hồi tức thì:** Dừng phát âm thanh và làm rỗng hàng đợi phát ngay khi nhận được `playback.stop`.
- **Chặn rò rỉ dữ liệu cũ:** Backend kiểm tra chặt chẽ `turn_id`: mọi audio chunk hoặc text delta còn sót lại của turn cũ trên đường truyền đều bị bỏ qua, đảm bảo không có hiện tượng phát lồng tiếng cũ vào lượt mới.

---

## 6. Quản lý lịch sử hội thoại (History Management)

- **Nguyên tắc đồng bộ theo cặp (Pairwise Commit):**
  - Không ghi `user` vào lịch sử trước khi LLM hoàn thành.
  - Chỉ commit đồng thời cả cặp tin nhắn `user` và `assistant` vào `session.history` khi lượt phản hồi LLM hoàn tất trọn vẹn (`assistant.text.done`).
  - Nếu LLM gặp lỗi mạng/API, trả về kết quả rỗng hoặc người dùng ngắt lời giữa chừng (Barge-in), cả câu hỏi của user lẫn câu trả lời dở dang của assistant đều không được lưu, tránh tuyệt đối lỗi lệch cặp hội thoại (hai tin nhắn `user` liên tiếp nhau).
- **Không ghi dữ liệu tạm:** Transcript dạng `partial` không bao giờ được đưa vào lịch sử chính thức.
- **Giới hạn cửa sổ ngữ cảnh:** Giới hạn tối đa 8 cặp hội thoại gần nhất (`VOICE_MAX_HISTORY_PAIRS`) trong bộ nhớ RAM để đảm bảo latency của LLM luôn tối ưu.

---

## 7. Vòng đời và dọn dẹp tài nguyên (Cleanup Lifecycle)

Khi kết thúc cuộc gọi hoặc khi người dùng ngắt kết nối WebSocket:
1. Hủy toàn bộ các tác vụ nền đang chạy (`session.cancel_active_tasks()`).
2. Đóng kết nối WebSocket tới STT server và giải phóng HTTP stream của TTS.
3. Xóa bộ đệm âm thanh `turn_audio_buffer` và giải phóng cấu trúc phiên khỏi bộ nhớ RAM.
4. Cơ chế dọn dẹp bảo đảm tính **idempotent**: thực thi an toàn nhiều lần mà không gây lỗi biệt lệ hay rò rỉ tài nguyên.
