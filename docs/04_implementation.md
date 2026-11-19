# 04 -- Kế hoạch triển khai, kiểm thử và nghiệm thu

**Phiên bản:** 1.0

## 1. Nguyên tắc triển khai

-   Làm từng lớp, kiểm thử trước khi ghép pipeline.
-   Xây dựng luồng đàm thoại thời gian thực vững chắc từ đầu vào audio đến các mô hình AI.
-   Chạy riêng từng API STT, LLM, TTS để xác nhận contract.
-   Không triển khai Smart Turn trước khi xác minh model input/output.
-   Mọi stream phải có đường đóng/hủy rõ ràng.
-   Không mở rộng thành RAG hoặc nghiệp vụ trong MVP.

## 2. Thứ tự công việc

### Bước 0 -- Xác minh API và môi trường [x]

Việc cần làm:
- [x] 1. Xác nhận STT WebSocket URL đang hoạt động (`wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream`).
- [x] 2. Thử STT `start → ready → binary → finish → final`.
- [x] 3. Gửi LLM request có streaming và lưu raw response.
- [x] 4. Thử TTS `/tts/stream`, lưu headers và bytes để phân tích WAV/chunk.
- [x] 5. Xác nhận Smart Turn v3.2 runtime, input và output.
- [x] 6. Xác nhận có cần API key/token hoặc custom headers không (STT không yêu cầu auth).

**Điều kiện qua bước:** Có mẫu request/response thực tế và cấu hình đã kiểm chứng cho các thành phần.

### Bước 1 -- Khởi tạo backend và frontend [x]

Backend:
- [x] Tạo `main.py`, `voice.py`, `services.py`, `config.py`.
- [x] Tạo FastAPI app.
- [x] Tạo `/ws/voice`.
- [x] Xử lý connect/disconnect.
- [x] Trao đổi JSON control event.
- [x] Thử gửi binary audio giả lập qua WebSocket.

Frontend:
- [x] Tạo giao diện tối thiểu, tinh gọn, đúng chuẩn màu sắc và bố cục gốc.
- [x] Kết nối WebSocket 2 chiều.
- [x] Các nút điều khiển (`btnTalk`, `btnFinish`, `btnStop`, `btnReset`).
- [x] Hiển thị trạng thái kết nối và trạng thái đàm thoại.
- [x] Thử gửi/nhận binary payload.

**Điều kiện qua bước:** [x] Frontend và backend trao đổi được event và binary theo đúng thứ tự.

### Bước 2 -- Tích hợp STT độc lập [x]

- [x] Tạo STT adapter (`STTSession`, `stt_open`, `stt_send_audio`, `stt_events`, `stt_finish`, `stt_cancel`).
- [x] Gửi `start` trước audio.
- [x] Chờ `ready`.
- [x] Gửi PCM16LE mono 16 kHz.
- [x] Nhận `partial`, `vad`, `final`, `error`.
- [x] Đảm bảo partial cumulative được thay thế, không append.
- [x] Đóng STT session đúng lifecycle.

**Kiểm thử:** Đã kiểm thử probe thực tế và end-to-end với endpoint `wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream`.

**Điều kiện qua bước:** [x] Nhận được transcript final ổn định qua nhiều lượt.

### Bước 3 -- Tích hợp LLM streaming [x]

- [x] Tạo adapter gọi Chat Completions (`services.llm_stream`).
- [x] Xác nhận stream flag (`stream: True`) và parser từ response thực tế (`text/event-stream`).
- [x] Truyền history đúng role (`user`, `assistant`).
- [x] Parse text delta (trích xuất `content`, lọc bỏ `reasoning`).
- [x] Xử lý finish, timeout, HTTP error và cancellation.
- [x] Ghép vào luồng pipeline `voice.py`: phát sự kiện `assistant.text.delta` và `assistant.text.done`.
- [x] Commit tin nhắn assistant vào RAM history sau khi stream thành công theo đúng thiết kế.

**Kiểm thử:** Đã kiểm thử câu đơn tiếng Việt, multi-turn history, hủy giữa stream, và tích hợp pipeline thành công 100%.

**Điều kiện qua bước:** [x] Text delta hiển thị đúng thứ tự và không lặp/mất nội dung.

### Bước 4 -- Tích hợp TTS streaming [x]

- [x] Xác minh format response (`audio/wav`, `48000 Hz`, `pcm_s16le` mono).
- [x] Tạo adapter đọc HTTP stream (`services.tts_stream`).
- [x] Bóc tách 44-byte WAV header ở chunk đầu tiên để chuyển sang raw PCM16LE mono 48 kHz.
- [x] Đảm bảo căn chỉnh 16-bit (số byte chẵn cho mỗi chunk).
- [x] Gửi từng audio chunk về frontend theo quy tắc bắt buộc: JSON metadata đi trước, binary payload liền kề ngay sau dưới `send_lock`.
- [x] Frontend buffer và phát tuần tự qua Web Audio API không có độ trễ.

**Kiểm thử:** Đã kiểm thử câu ngắn, câu dài, hủy giữa stream, và ghép pipeline STT -> LLM -> TTS thành công 100%.

**Điều kiện qua bước:** [x] Audio nghe liên tục, đúng tốc độ, không lặp header, không mất chunk.

### Bước 5 -- Hoàn thiện luồng đàm thoại thời gian thực (Audio Streaming) [x]

- [x] Nhấn bắt đầu → tạo turn (`turn.start`).
- [x] Thu microphone bằng AudioWorklet (`MicProcessor` 16 kHz Mono).
- [x] Resample về 16 kHz mono (bộ lọc khử răng cưa Butterworth 4th-order).
- [x] Chuyển Float32 sample sang PCM16LE.
- [x] Gửi binary frames về backend.
- [x] Nhấn dừng hoặc dứt câu → gửi `turn.finish`.
- [x] Chờ STT final → LLM stream → TTS stream.
- [x] Hiển thị transcript và câu trả lời trực quan trên giao diện.

**Điều kiện qua bước:** [x] Đã hoàn thiện toàn bộ luồng đàm thoại từ client đến model và phát âm thanh.

### Bước 6 -- Quản lý history và session [x]

- [x] Tạo session registry trong RAM (`VoiceOrchestrator.sessions`).
- [x] Mỗi session có history và active tasks riêng biệt.
- [x] Chỉ commit user final và assistant response hoàn chỉnh.
- [x] Giới hạn history 8 cặp gần nhất (`VOICE_MAX_HISTORY_PAIRS`).
- [x] Xóa session khi disconnect/stop.
- [x] Cleanup idempotent, an toàn.

**Điều kiện qua bước:** [x] Hai session demo độc lập không dùng lẫn history; đóng session giải phóng dữ liệu.

### Bước 7 -- Tự động ngắt lượt với Smart Turn v3.2 [x]

- [x] Giữ microphone capture liên tục cho đàm thoại thời gian thực hai chiều.
- [x] Tích hợp STT VAD (`speech_started`, `speech_ended`).
- [x] Buffer audio cho turn hiện tại (`turn_audio_buffer`).
- [x] Tích hợp mô hình ONNX Smart Turn v3.2 (`services.smart_turn_predict`).
- [x] Tự động kích hoạt đánh giá Smart Turn khi vừa dứt lời (`prob >= 0.70`).
- [x] Tự động kết thúc lượt khi im lặng vượt ngưỡng tối đa (`smart_turn_silence_timeout_seconds`).
- [x] Khởi tạo lượt thoại mới tự động sau khi bot hoàn tất phát ngôn (`scheduleAutoListen`).

**Điều kiện qua bước:** [x] Có thể nói nhiều câu liên tiếp mà không cần thao tác dừng thủ công; không cắt câu quá sớm.

### Bước 8 -- Barge-in và ổn định [x]

- [x] Phát hiện người dùng bắt đầu nói khi bot đang xử lý (`THINKING`) hoặc phát âm thanh (`SPEAKING`).
- [x] Hủy tức thì các tác vụ LLM/TTS đang stream (`session.cancel_active_tasks()`).
- [x] Gửi lệnh dừng playback khẩn cấp về client (`playback.stop`).
- [x] Đánh dấu turn cũ bị ngắt (`VoiceState.INTERRUPTED`).
- [x] Cấp phát `turn_id` mới và mở STT session mới ngay lập tức.
- [x] Frontend xóa queue và dừng phát ngay khi nhận `playback.stop`.

**Điều kiện qua bước:** [x] Người dùng ngắt lời thì audio cũ dừng ngay lập tức; không phát lại chunk cũ sau khi lượt mới bắt đầu.

## 3. Ma trận kiểm thử

| ID | Tình huống | Kết quả mong đợi | Trạng thái |
|:---|:---|:---|:---:|
| T01 | Kết nối frontend/backend | Nhận `session.ready` | [x] Đạt |
| T02 | Ngắt WebSocket frontend | Backend cleanup session | [x] Đạt |
| T03 | STT một câu ngắn | Có partial và final | [x] Đạt |
| T04 | STT câu dài | Không mất phần cuối | [x] Đạt |
| T05 | STT gửi frame byte lẻ | API báo lỗi định dạng, backend xử lý an toàn | [x] Đạt |
| T06 | STT quá tải 429 | Hiển thị lỗi có thể thử lại | [x] Đạt |
| T07 | LLM stream bình thường | Text delta đúng thứ tự | [x] Đạt |
| T08 | LLM timeout | Báo lỗi, không treo session | [x] Đạt |
| T09 | Hủy LLM giữa stream | Không nhận/hiển thị delta cũ | [x] Đạt |
| T10 | TTS câu ngắn | Audio phát được | [x] Đạt |
| T11 | TTS nhiều chunk | Chunk phát đúng thứ tự | [x] Đạt |
| T12 | TTS stream lỗi giữa chừng | Báo lỗi, giải phóng response | [x] Đạt |
| T13 | Đàm thoại qua nhiều lượt | Mỗi lượt có turn_id riêng biệt | [x] Đạt |
| T14 | Hai session cùng lúc | History không lẫn nhau | [x] Đạt |
| T15 | Đàm thoại tự nhiên có khoảng nghỉ | Smart Turn quyết định đúng theo ngưỡng (prob >= 0.70 hoặc timeout) | [x] Đạt |
| T16 | Người dùng ngắt bot | Dừng audio cũ và hủy task (Barge-in `playback.stop`) | [x] Đạt |
| T17 | Đóng session | History/audio buffer được xóa | [x] Đạt |
| T18 | Backend restart | Session RAM cũ mất; hành vi được thông báo rõ | [x] Đạt |

## 4. Checklist chất lượng

### Kiến trúc

-   [x] `main.py` không chứa logic gọi model.
-   [x] `voice.py` không chứa chi tiết transport API.
-   [x] `services.py` không quản lý history.
-   [x] `config.py` không chứa business logic.

### Audio

-   [x] Input STT đúng PCM16LE mono 16 kHz.
-   [x] Frame byte length luôn chẵn.
-   [x] AudioWorklet được dùng để capture.
-   [x] Resampling được kiểm thử, không chỉ đổi nhãn sample rate.
-   [x] TTS stream format được xác minh trước khi playback.
-   [x] Audio chunks có thứ tự và không phát chunk cũ.

### Hội thoại

-   [x] Mỗi lượt có `turn_id`.
-   [x] Partial không commit vào history.
-   [x] Assistant response chỉ commit khi hoàn tất.
-   [x] History tách biệt theo session.
-   [x] Cancel task có hiệu lực cả backend và frontend.

### Vận hành

-   [x] Timeout cho STT/LLM/TTS.
-   [x] Cleanup dùng `try/finally`.
-   [x] Không log secret.
-   [x] Không gửi API credentials xuống frontend.
-   [x] Có thông báo lỗi rõ ràng.

## 5. Rủi ro và cách giảm thiểu

  -----------------------------------------------------------------------
  Rủi ro                  Ảnh hưởng               Giảm thiểu
  ----------------------- ----------------------- -----------------------
  STT URL chưa thống nhất Không kết nối được      Xác minh endpoint trước
                                                  khi code

  LLM stream format chưa  Parser sai              Lưu raw response và
  rõ                                              viết test parser

  TTS chunk chứa WAV      Audio lỗi khi ghép      Phân tích bytes/headers
  header                                          trước playback

  Smart Turn input không  Cắt lượt sai            Xác minh model
  đúng                                            contract, test audio
                                                  mẫu

  Cancel không dừng       Tốn tài nguyên, output  Hủy task và đóng
  provider                cũ                      stream/response

  History RAM mất khi     Mất hội thoại           Chấp nhận trong MVP,
  restart                                         database ở giai đoạn
                                                  sau

  Nhiều session vượt      429                     Giới hạn concurrency và
  quota STT                                       báo retry
  -----------------------------------------------------------------------

## 6. Định nghĩa hoàn thành MVP [x]

MVP được nghiệm thu hoàn tất khi đạt 10 tiêu chuẩn:
- [x] 1. Luồng đàm thoại thời gian thực hoạt động ổn định qua nhiều lượt.
- [x] 2. STT trả partial/final đúng lifecycle.
- [x] 3. LLM trả lời streaming (SSE trích xuất delta, lọc reasoning).
- [x] 4. TTS phát được streaming audio không lỗi định dạng (bóc WAV header ở chunk 0, căn chỉnh 16-bit).
- [x] 5. History tách biệt theo session và được xóa khi đóng session.
- [x] 6. Tự động nhận diện dứt câu bằng VAD + Smart Turn v3.2 ONNX để kết thúc lượt thoại thông minh.
- [x] 7. Barge-in dừng được audio (`playback.stop`) và hủy tác vụ cũ ngay lập tức.
- [x] 8. Lỗi timeout, disconnect và provider overload được xử lý an toàn.
- [x] 9. Frontend demo độc lập, backend Voice độc lập.
- [x] 10. Toàn bộ cấu hình endpoint, models, thresholds đặt tại `backend/config.py`.
