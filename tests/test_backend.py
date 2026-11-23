"""Unit tests cho toàn bộ backend của Voicebot MVP."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from backend import services
from backend.config import settings
from backend.text_processing import (
    REMAINDER_NOTE,
    TTSTextStreamer,
    clean_inline_markdown,
    is_only_filler,
    normalize_text_for_tts,
    split_for_tts,
)
from backend.voice import TurnState, VoiceSession, VoiceState


# ---------------------------------------------------------------------------
# 1. Tests cho Module Cấu hình (config.py)
# ---------------------------------------------------------------------------
def test_config_defaults():
    assert settings.app_port > 0
    assert settings.stt_sample_rate == 16000
    assert settings.smart_turn_threshold == 0.70
    assert settings.smart_turn_inference_timeout_seconds == 3.0
    assert settings.smart_turn_fallback_seconds == 5.0
    warnings = settings.validate_config()
    assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# 2. Tests cho Text Processing & NLP (text_processing.py)
# ---------------------------------------------------------------------------
def test_is_only_filler():
    assert is_only_filler("ừ") is True
    assert is_only_filler("ừm ờ à") is True
    assert is_only_filler("à mà") is True
    assert is_only_filler("") is True
    assert is_only_filler("chào bạn tôi cần gửi hàng") is False


def test_clean_inline_markdown():
    assert clean_inline_markdown("**chữ đậm** và *chữ nghiêng*") == "chữ đậm và chữ nghiêng"
    assert clean_inline_markdown("# Tiêu đề lớn") == "Tiêu đề lớn"
    assert clean_inline_markdown("- Gạch đầu dòng") == "Gạch đầu dòng"
    assert clean_inline_markdown("Xem tại [VNPost](https://vnpost.vn) nhé") == "Xem tại VNPost nhé"
    assert clean_inline_markdown("Định dạng `inline code` và ~~gạch ngang~~ xong") == "Định dạng inline code và gạch ngang xong"


def test_tts_text_streamer_streaming():
    streamer = TTSTextStreamer(max_sentences=8, max_chars=600)
    text = (
        "Dạ chào bạn! Tôi là trợ lý ảo VNPost. "
        "Tôi có thể hỗ trợ gì cho bạn hôm nay?"
    )
    sentences = streamer.feed(text) + streamer.flush()
    joined = " ".join(sentences)
    assert "Dạ chào bạn!" in joined
    assert "Tôi là trợ lý ảo VNPost." in joined


def test_tts_text_streamer_capping():
    streamer = TTSTextStreamer(max_sentences=2, max_chars=50)
    text = "Câu thứ nhất rất là dài. Câu thứ hai tiếp tục dài. Câu thứ ba không được đọc."
    sentences = streamer.feed(text) + streamer.flush()
    assert len(sentences) == 3
    assert sentences[-1] == REMAINDER_NOTE


def test_normalize_text_for_tts():
    # Tiền tệ và số điện thoại
    res = normalize_text_for_tts("Cước phí là 25.000 đồng về SĐT 0912345678")
    assert "hai mươi lăm nghìn đồng" in res
    assert "không chín một hai ba bốn năm sáu bảy tám" in res

    # Tiền hàng tỷ không bị regex số điện thoại cắt cụt / đọc thành thập phân
    bil_res = normalize_text_for_tts("Tổng tiền là 1.000.000.000 đồng")
    assert "một tỷ đồng" in bil_res
    assert "phẩy" not in bil_res

    # Số điện thoại quốc tế +84 dạng viết cách và viết liền
    intl_res = normalize_text_for_tts("Gọi +84 912 345 678 hoặc +84912345678")
    assert "không chín một hai ba bốn năm sáu bảy tám" in intl_res
    assert intl_res.count("không chín một hai ba bốn năm sáu bảy tám") == 2

    # Ngày tháng
    date_res = normalize_text_for_tts("Hẹn giao ngày 19/11/2026")
    assert "ngày mười chín tháng mười một năm hai nghìn lẻ hai mươi sáu" in date_res

    # Số thập phân và đơn vị đo
    dec_res = normalize_text_for_tts("Trọng lượng 2.5kg và 24/7")
    assert "hai phẩy năm ki lô gam" in dec_res
    assert "hai tư trên bảy" in dec_res


def test_split_for_tts():
    long_text = "Một đoạn văn bản rất dài, " * 30
    parts = split_for_tts(long_text, limit=100)
    assert len(parts) > 1
    for p in parts:
        assert len(p) <= 120


# ---------------------------------------------------------------------------
# 3. Tests cho Service Adapters (services.py)
# ---------------------------------------------------------------------------
def test_tts_chunk_processing():
    # Giả lập chunk đầu chứa WAV header chuẩn 44 byte
    fake_wav_header = b"RIFF" + b"\x00" * 36 + b"data\x04\x00\x00\x00"
    pcm_payload = b"\x01\x02\x03\x04"
    raw_chunk = fake_wav_header + pcm_payload

    data, parsed, leftover = services._process_tts_chunk(
        raw_chunk, header_parsed=False, header_buf=bytearray(), leftover=b""
    )
    assert parsed is True
    assert data == pcm_payload
    assert leftover == b""

    # Kiểm tra căn chỉnh byte lẻ
    odd_chunk = b"\x01\x02\x03"
    data2, parsed2, leftover2 = services._process_tts_chunk(
        odd_chunk, header_parsed=True, header_buf=bytearray(), leftover=b""
    )
    assert data2 == b"\x01\x02"
    assert leftover2 == b"\x03"


def test_parse_sse_delta():
    assert services._parse_sse_delta('{"choices": [{"delta": {"content": "Chào"}}]}') == "Chào"
    assert services._parse_sse_delta('{"choices": [{"delta": {"reasoning_content": "think"}}]}') is None
    assert services._parse_sse_delta("invalid json") is None


def test_build_llm_payload():
    messages = [{"role": "user", "content": "alo"}]
    payload = services._build_llm_payload(messages)
    assert payload["stream"] is True
    assert payload["messages"] == messages
    assert "temperature" in payload


def test_smart_turn_fallback():
    detector = services.SmartTurnDetector(model_path="non_existent_model.onnx")
    # Không crash khi file model không tồn tại
    assert detector._sync_predict(b"\x00" * 3200) == 0.0


# ---------------------------------------------------------------------------
# 4. Tests cho Voice Session (voice.py)
# ---------------------------------------------------------------------------
def test_session_lifecycle():
    session = VoiceSession("test_sess_01")
    assert session.session_id == "test_sess_01"

    # Kiểm tra thêm lịch sử và cắt tỉa
    for i in range(20):
        session.add_history("user" if i % 2 == 0 else "assistant", f"msg {i}")
    max_msgs = settings.voice_max_history_pairs * 2
    assert len(session.history) == max_msgs

    # Kiểm tra tin nhắn bao gồm cả system prompt
    messages = session.build_messages()
    assert len(messages) == max_msgs + 1
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_silence_timeout_logic():
    session = VoiceSession(session_id="s_test")
    assert session.is_silence_timeout() is False

    session.has_speech = True
    session.last_speech_time = 0.0
    assert session.is_silence_timeout() is False

    # Giả lập thời gian nói cách đây lâu
    session.last_speech_time = 1.0
    assert session.is_silence_timeout() is True

    # Khi VAD đang báo người dùng nói -> tuyệt đối không timeout
    session.is_vad_speaking = True
    assert session.is_silence_timeout() is False


def test_phone_not_confused_with_date():
    # Kiểm tra số điện thoại có dấu chấm không bị nhầm thành ngày tháng
    res = normalize_text_for_tts("Liên hệ số 09.12.3456 để được hỗ trợ")
    assert "ngày" not in res


def test_hotline_and_date_spacing():
    # Tổng đài 1900 / 1800 viết cách hoặc viết liền 6 số sau
    res = normalize_text_for_tts("gọi tổng đài 1900 54 54 81 hoặc 1900 545481")
    assert "một chín không không năm bốn năm bốn tám một" in res
    assert res.count("một chín không không năm bốn năm bốn tám một") == 2

    # Khoảng trắng quanh ngày tháng không sinh ra lặp "ngày ngày"
    date_res = normalize_text_for_tts("hẹn giao ngày 12 / 5")
    assert "ngày ngày" not in date_res
    assert "ngày mười hai tháng năm" in date_res


def test_tech_terms_tts():
    res = normalize_text_for_tts("mạng 5G hoạt động 24h và xem phim 3D 4K")
    assert "năm gờ" in res
    assert "hai mươi bốn giờ" in res
    assert "ba đê" in res
    assert "bốn ca" in res


def test_wav_chunk_advanced():
    # Chunk nhỏ hơn 4 byte ban đầu (chưa đủ nhận diện RIFF)
    buf = bytearray()
    d1, p1, l1 = services._process_tts_chunk(b"RI", False, buf, b"")
    assert d1 == b""
    assert p1 is False

    # Chunk tiếp theo hoàn tất header chứa chunk JUNK trước data
    junk_hdr = b"FF" + b"\x00" * 36 + b"JUNK\x04\x00\x00\x001234data\x04\x00\x00\x00"
    payload = b"\xaa\xbb\xcc\xdd"
    d2, p2, l2 = services._process_tts_chunk(junk_hdr + payload, p1, buf, l1)
    assert p2 is True
    assert d2 == payload
    assert l2 == b""


def test_thinking_to_idle_when_empty_or_filtered():
    session = VoiceSession(session_id="test_thinking_idle")
    session.current_turn_id = "turn_123"
    session.state = VoiceState.THINKING
    # Kiểm tra nhánh Bug 3 trong _stream_tts_audio: nếu kết thúc mà state đang là THINKING thì chuyển về IDLE
    if session.current_turn_id == "turn_123" and not session.closed and session.state == VoiceState.THINKING:
        session.state = VoiceState.IDLE
    assert session.state == VoiceState.IDLE


def test_prompt_file_loaded():
    assert "VNPost" in settings.llm_system_prompt
    assert len(settings.llm_system_prompt) > 100


def test_tts_text_streamer():
    streamer = TTSTextStreamer(max_sentences=5, max_chars=300)
    deltas = ["Chào ", "bạn! ", "Cước ", "phí ", "là ", "25.000 đồng. ", "Xin ", "cảm ", "ơn!"]
    collected = []
    for d in deltas:
        collected.extend(streamer.feed(d))
    collected.extend(streamer.feed())

    assert len(collected) >= 2
    full = " ".join(collected)
    assert "hai mươi lăm nghìn đồng" in full


def test_tts_text_streamer_capping():
    streamer = TTSTextStreamer(max_sentences=2, max_chars=50)
    text = "Câu thứ nhất rất là dài. Câu thứ hai tiếp tục dài. Câu thứ ba không được đọc."
    parts = streamer.feed(text) + streamer.feed()
    assert len(parts) == 3
    assert parts[-1] == REMAINDER_NOTE


def test_error_state_protection():
    s = VoiceSession(session_id="test_error_prot")

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(s.set_state(VoiceState.ERROR))
    assert s.state == VoiceState.ERROR

    # Thử đổi trạng thái sang IDLE hoặc SPEAKING mà không force -> phải bị chặn
    _step(s.set_state(VoiceState.IDLE))
    assert s.state == VoiceState.ERROR
    _step(s.set_state(VoiceState.SPEAKING))
    assert s.state == VoiceState.ERROR

    # Reset có force=True -> thành công
    _step(s.set_state(VoiceState.IDLE, force=True))
    assert s.state == VoiceState.IDLE


def test_drain_queue():
    import asyncio

    from backend.voice import _drain_queue

    q: asyncio.Queue[str] = asyncio.Queue()
    q.put_nowait("msg 1")
    q.put_nowait("msg 2")
    q.put_nowait("msg 3")
    assert q.qsize() == 3

    _drain_queue(q)
    assert q.empty()


def test_smart_turn_task_lifecycle():
    session = VoiceSession(session_id="test_vad_lifecycle")
    session.state = VoiceState.LISTENING
    session.current_turn_id = "turn_vad"

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    # speech_started: VAD bật, hủy task cũ nếu có
    _step(session._handle_vad("turn_vad", "speech_started"))
    assert session.is_vad_speaking is True
    assert session.has_speech is True
    assert session._smart_turn_task is None

    # speech_ended khi has_speech = False: không kích hoạt inference
    session.has_speech = False
    _step(session._handle_vad("turn_vad", "speech_ended"))
    assert session.is_vad_speaking is False
    assert session._smart_turn_task is None


def test_close_session_safe_on_error():
    session = VoiceSession(session_id="sess_err_cleanup")
    session.buffer_audio(b"\x00" * 320)

    class FailingSTT:
        async def close(self):
            raise RuntimeError("Network failed on close")

    session.stt_client = FailingSTT()

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session.close())
    assert session.closed is True
    assert session.turn_audio_bytes == 0
    assert session.stt_client is None


def test_add_history_zero_limit():
    original_val = settings.voice_max_history_pairs
    object.__setattr__(settings, "voice_max_history_pairs", 0)
    try:
        session = VoiceSession(session_id="sess_hist_zero")
        session.add_history("user", "xin chào")
        assert len(session.history) == 0
    finally:
        object.__setattr__(settings, "voice_max_history_pairs", original_val)


def test_handle_cancel_safe_when_stt_close_fails():
    session = VoiceSession(session_id="sess_cancel_test")
    session.current_turn_id = "turn_cancel_01"
    session.state = VoiceState.SPEAKING
    session.buffer_audio(b"\x00" * 320)

    class FailingSTT:
        async def close(self):
            raise RuntimeError("STT close failed")

    session.stt_client = FailingSTT()

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session._handle_cancel("turn_cancel_01"))
    assert session.stt_client is None
    assert session.state == VoiceState.IDLE
    assert session.turn_audio_bytes == 0


def test_pipeline_error_isolated_to_turn():
    session = VoiceSession(session_id="sess_turn_isolation")
    session.current_turn_id = "turn_B"
    session.state = VoiceState.LISTENING

    import asyncio
    q: asyncio.Queue[Any] = asyncio.Queue()
    q.put_nowait("dummy")

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    # Error from old turn A must be ignored, session state must remain LISTENING
    _step(session._handle_pipeline_error("turn_A", q, "ERR_CODE", "Error from turn A"))
    assert session.state == VoiceState.LISTENING
    assert not q.empty()


def test_safe_send_closed_and_disconnect():
    session = VoiceSession(session_id="sess_safe_send")

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    # Session closed: send callback should not be executed
    session.closed = True
    called_on_closed = []

    async def dummy_send():
        called_on_closed.append(True)

    _step(session._safe_send(dummy_send))
    assert called_on_closed == []

    # Callable format
    session.closed = False
    called = []

    async def working_send():
        called.append(True)

    _step(session._safe_send(working_send))
    assert called == [True]

    # Connection reset sets closed = True
    async def failing_send():
        raise ConnectionResetError("Connection lost")

    _step(session._safe_send(failing_send))
    assert session.closed is True


def test_config_validation_ranges():
    bad_cfg = settings.__class__(
        smart_turn_threshold=1.5,
        smart_turn_inference_timeout_seconds=-1.0,
        smart_turn_fallback_seconds=-1.0,
        llm_timeout_seconds=0.0,
        tts_timeout_seconds=0.0,
        tts_split_limit=5,
        tts_max_sentences=0,
        tts_max_chars=5,
        turn_max_audio_bytes=1000,
    )
    warnings = bad_cfg.validate_config()
    assert any("SMART_TURN_THRESHOLD" in w for w in warnings)
    assert any("SMART_TURN_INFERENCE_TIMEOUT_SECONDS" in w for w in warnings)
    assert any("SMART_TURN_FALLBACK_SECONDS" in w for w in warnings)
    assert any("LLM_TIMEOUT_SECONDS" in w for w in warnings)
    assert any("TTS_TIMEOUT_SECONDS" in w for w in warnings)
    assert any("TTS_SPLIT_LIMIT" in w for w in warnings)
    assert any("TTS_MAX_SENTENCES" in w for w in warnings)
    assert any("TTS_MAX_CHARS" in w for w in warnings)
    assert any("TURN_MAX_AUDIO_BYTES" in w for w in warnings)


def test_parse_sse_delta_error_stream():
    with pytest.raises(services.LLMError, match="Rate limit exceeded"):
        services._parse_sse_delta('{"error": {"message": "Rate limit exceeded"}}')


def test_is_smart_turn_ready():
    assert isinstance(services.smart_turn_detector.is_ready, bool)


def test_stt_send_audio_generic_exception():
    class MockWS:
        async def send(self, data):
            raise OSError("Socket write broken")

    stt = services.STTClient("sess", "turn", "req")
    stt.websocket = MockWS()
    stt.is_active = True

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    with pytest.raises(services.STTError, match="Lỗi gửi audio tới STT"):
        _step(stt.send_audio(b"\x00\x00" * 160))


# ---------------------------------------------------------------------------
# 5. Tests bổ sung cho các cải tiến Logic (Review fixes)
# ---------------------------------------------------------------------------
def test_match_turn_strict_and_empty_handling():
    session = VoiceSession(session_id="test_match_turn")
    # Khi current_turn_id là None
    assert session._match_turn("turn_1") is False
    assert session._match_turn(None, allow_empty=True) is False
    assert session._match_turn(None, allow_empty=False) is False

    # Khi có current_turn_id
    session.current_turn_id = "turn_active"
    assert session._match_turn("turn_active") is True
    assert session._match_turn("turn_old") is False
    assert session._match_turn(None, allow_empty=False) is False
    assert session._match_turn(None, allow_empty=True) is True
    assert session._match_turn("", allow_empty=False) is False
    assert session._match_turn("", allow_empty=True) is True


def test_handle_cancel_clears_current_turn_id():
    session = VoiceSession(session_id="test_cancel_id")
    session.current_turn_id = "turn_to_cancel"
    session.state = VoiceState.SPEAKING

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session._handle_cancel("turn_to_cancel"))
    assert session.current_turn_id is None
    assert session.state == VoiceState.IDLE


def test_playback_done_and_cancel_isolation():
    session = VoiceSession(session_id="test_isolation")
    session.current_turn_id = "turn_current"
    session.state = VoiceState.SPEAKING

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    # Playback done từ turn cũ không được ảnh hưởng turn hiện tại
    _step(session.handle_event({"type": "playback.done", "turn_id": "turn_old"}))
    assert session.state == VoiceState.SPEAKING
    assert session.current_turn_id == "turn_current"

    # Playback done không có turn_id cũng bị từ chối
    _step(session.handle_event({"type": "playback.done"}))
    assert session.state == VoiceState.SPEAKING

    # Playback done đúng turn_id sẽ chuyển về IDLE
    _step(session.handle_event({"type": "playback.done", "turn_id": "turn_current"}))
    assert session.state == VoiceState.IDLE


def test_wav_chunk_extended_header_over_4096():
    # Header lớn hơn 4096 byte chứa extended metadata/JUNK
    extended_junk = b"JUNK" + b"\x00" * 5000
    fake_header = b"RIFF" + b"\x00" * 36 + extended_junk + b"data\x04\x00\x00\x00"
    payload = b"\xde\xad\xbe\xef"

    buf = bytearray()
    data, parsed, leftover = services._process_tts_chunk(
        fake_header + payload, header_parsed=False, header_buf=buf, leftover=b""
    )
    assert parsed is True
    assert data == payload
    assert leftover == b""


def test_smart_turn_logs_warning_on_exception(caplog):
    import logging
    detector = services.SmartTurnDetector(model_path="dummy.onnx")

    def failing_predict(*args, **kwargs):
        raise RuntimeError("Inference crash")

    detector.session = object()  # type: ignore
    detector.feature_extractor = object()  # type: ignore
    detector._attempted = True
    detector._sync_predict = failing_predict

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    with caplog.at_level(logging.WARNING):
        result = _step(detector.predict(b"\x00\x00" * 2000))

    assert result == 0.0
    assert any("Lỗi suy luận Smart Turn" in record.message for record in caplog.records)


def test_stt_error_event_transitions_session_to_error():
    session = VoiceSession(session_id="test_stt_err")
    session.current_turn_id = "turn_err_1"
    session.state = VoiceState.LISTENING

    class MockSTTWithError:
        def __init__(self):
            self.closed = False

        async def events(self):
            yield {"type": "error", "error_code": "AUDIO_TOO_LONG", "message": "Quá dài"}

        async def close(self):
            self.closed = True

    mock_stt = MockSTTWithError()
    session.stt_client = mock_stt  # type: ignore

    events_emitted = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session._forward_stt_events(mock_stt, "turn_err_1"))  # type: ignore
    assert session.state == VoiceState.ERROR
    assert session.stt_client is None
    assert mock_stt.closed is True
    assert any(ev.get("type") == "session.error" and ev.get("code") == "AUDIO_TOO_LONG" for ev in events_emitted)


def test_handle_audio_frame_stt_error():
    session = VoiceSession(session_id="test_stt_send_err")
    session.current_turn_id = "turn_send_err"
    session.state = VoiceState.LISTENING

    class MockFailingSTT:
        is_active = True

        async def send_audio(self, pcm):
            raise services.STTError("STT send failed", code="STT_SEND_ERROR")

        async def close(self):
            self.is_active = False

    session.stt_client = MockFailingSTT()  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session.handle_audio_frame(b"\x00\x00" * 160))
    assert session.state == VoiceState.ERROR
    assert session.stt_client is None


def test_producer_preserves_queue_on_normal_completion():
    import asyncio

    session = VoiceSession(session_id="test_prod_queue")
    session.current_turn_id = "turn_prod"
    session.state = VoiceState.THINKING

    async def fake_llm(messages):
        yield "Câu một. "
        yield "Câu hai. "

    original_llm = services.llm_stream
    services.llm_stream = fake_llm

    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    try:
        _step(session._produce_tts_text("turn_prod", queue, "User prompt"))
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())

        # Kiểm tra không bị drain hàng đợi: các câu vẫn còn nguyên kèm None ở cuối
        assert len(items) == 3
        assert "Câu một." in items[0]
        assert "Câu hai." in items[1]
        assert items[2] is None
    finally:
        services.llm_stream = original_llm


@pytest.mark.anyio
async def test_stt_finish_timeout_raises_stt_error():
    class HangingWS:
        async def send(self, data):
            pass

    stt = services.STTClient("sess_timeout", "turn_t", "req_t")
    stt.websocket = HangingWS()  # type: ignore
    stt.is_active = True

    with pytest.raises(services.STTError) as exc_info:
        await stt.finish(timeout=0.01)

    assert exc_info.value.code == "STT_TIMEOUT"
    assert stt.is_active is False
    assert stt._closed is True


def test_process_turn_stt_timeout_fallback_to_partial():
    session = VoiceSession(session_id="sess_fallback")
    session.current_turn_id = "turn_fb"
    session.state = VoiceState.THINKING
    session.last_partial_text = "Tôi cần gửi bưu phẩm"

    class TimeoutSTT:
        async def finish(self):
            raise services.STTError("Timeout STT", code="STT_TIMEOUT")

    session.stt_client = TimeoutSTT()  # type: ignore

    events = []

    async def mock_send(ev):
        events.append(ev)

    session.send_json_cb = mock_send

    # Mock _run_turn_taskgroup để không gọi LLM thật
    async def mock_tg(turn_id, text):
        pass

    session._run_turn_taskgroup = mock_tg  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session._process_turn("turn_fb"))

    # Kiểm tra transcript.final được phát với nội dung partial dự phòng
    final_ev = next((ev for ev in events if ev.get("type") == "transcript.final"), None)
    assert final_ev is not None
    assert final_ev.get("text") == "Tôi cần gửi bưu phẩm"


def test_process_turn_empty_transcript_responds_with_prompt():
    session = VoiceSession(session_id="sess_empty_text")
    session.current_turn_id = "turn_empty"
    session.state = VoiceState.THINKING
    session.last_partial_text = ""

    class EmptySTT:
        async def finish(self):
            return ""

    session.stt_client = EmptySTT()  # type: ignore

    events = []

    async def mock_send(ev):
        events.append(ev)

    session.send_json_cb = mock_send

    filler_prompts = []

    async def mock_filler(turn_id, prompt):
        filler_prompts.append((turn_id, prompt))

    session._respond_with_filler = mock_filler  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session._process_turn("turn_empty"))

    final_ev = next((ev for ev in events if ev.get("type") == "transcript.final"), None)
    assert final_ev is not None
    assert final_ev.get("text") == ""
    assert len(filler_prompts) == 1
    assert filler_prompts[0] == ("turn_empty", "Tôi chưa nghe rõ, bạn vui lòng nói lại nhé.")


def test_process_turn_stt_timeout_no_partial_transitions_to_error():
    session = VoiceSession(session_id="sess_timeout_err")
    session.current_turn_id = "turn_to_err"
    session.state = VoiceState.THINKING
    session.last_partial_text = ""

    class TimeoutSTT:
        async def finish(self):
            raise services.STTError("Timeout STT", code="STT_TIMEOUT")

    session.stt_client = TimeoutSTT()  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session._process_turn("turn_to_err"))
    assert session.state == VoiceState.ERROR


def test_stt_close_idempotent_and_concurrent():
    class DummyWS:
        def __init__(self):
            self.close_count = 0

        async def close(self, code=1000, reason=""):
            self.close_count += 1

    stt = services.STTClient("sess_concurrent", "turn_c", "req_c")
    ws = DummyWS()
    stt.websocket = ws  # type: ignore
    stt.is_active = True

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(stt.close())
    _step(stt.close())

    assert stt._closed is True
    assert stt.is_active is False
    assert ws.close_count == 1  # Chỉ đóng đúng 1 lần


def test_stt_recv_loop_propagates_cancelled_error():
    class ClosingWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

    stt = services.STTClient("sess_cancel", "turn_c", "req_c")
    stt.websocket = ClosingWS()  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    with pytest.raises(asyncio.CancelledError):
        _step(stt._recv_loop())

    assert stt.is_active is False
    assert stt.is_done.is_set() is True


def test_parse_sse_delta_strict_validation():
    # Kiểm tra cấu trúc rỗng hoặc không hợp lệ không phát sinh ngoại lệ
    assert services._parse_sse_delta('{"choices": [null]}') is None
    assert services._parse_sse_delta('{"choices": [{"delta": null}]}') is None
    assert services._parse_sse_delta('{"choices": [{"delta": {"content": 123}}]}') is None
    assert services._parse_sse_delta('{"choices": []}') is None
    assert services._parse_sse_delta('{"choices": "not_a_list"}') is None
    assert services._parse_sse_delta('"just_string"') is None
    assert services._parse_sse_delta("12345") is None

    # Hợp lệ
    assert services._parse_sse_delta('{"choices": [{"delta": {"content": "Xin chào"}}]}') == "Xin chào"


def test_find_wav_data_offset_standard_and_extended():
    # 1. Standard WAV (44 byte)
    std_wav = (
        b"RIFF\x24\x00\x00\x00WAVE"
        b"fmt \x10\x00\x00\x00" + b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
        b"data\x04\x00\x00\x00"
    )
    assert services._find_wav_data_offset(bytearray(std_wav)) == 44

    # 2. Extended WAV với subchunk LIST trước data
    list_chunk = b"LIST\x08\x00\x00\x0012345678"
    ext_wav = (
        b"RIFF\x30\x00\x00\x00WAVE"
        b"fmt \x10\x00\x00\x00" + b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
        + list_chunk
        + b"data\x04\x00\x00\x00"
    )
    assert services._find_wav_data_offset(bytearray(ext_wav)) == 44 + len(list_chunk)

    # 3. Non-RIFF stream (raw PCM)
    assert services._find_wav_data_offset(bytearray(b"\x01\x02\x03\x04" * 10)) == 0


def test_tts_stream_odd_byte_leftover_warning_and_pad(caplog):
    import logging

    class MockResp:
        status_code = 200

        async def aiter_bytes(self):
            # Chunk 1: 4 byte PCM chẵn
            yield b"\x01\x02\x03\x04"
            # Chunk 2: 3 byte PCM lẻ (tổng 7 byte)
            yield b"\x05\x06\x07"

    class MockClient:
        is_closed = False

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def stream(self, *args, **kwargs):
            yield MockResp()

    orig_client_fn = services.get_http_client
    services.get_http_client = lambda: MockClient()  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    async def _collect():
        chunks = []
        with caplog.at_level(logging.WARNING):
            async for chunk in services.tts_stream("Test leftover"):
                chunks.append(chunk)
        return chunks

    try:
        chunks = _step(_collect())
        total_bytes = b"".join(chunks)
        # Tổng byte phải là số chẵn (7 byte gốc + 1 byte pad null = 8 byte)
        assert len(total_bytes) % 2 == 0
        assert len(total_bytes) == 8
        assert any("Dữ liệu PCM từ TTS không hợp lệ" in r.message for r in caplog.records)
    finally:
        services.get_http_client = orig_client_fn


# ---------------------------------------------------------------------------
# 6. Tests cho 4 Vấn đề logic mới (Turn order, STT finish, Smart Turn Lock)
# ---------------------------------------------------------------------------
def test_produce_tts_text_order_text_done_before_sentinel():
    session = VoiceSession(session_id="test_order")
    session.current_turn_id = "turn_order"
    session.state = VoiceState.THINKING

    events_emitted = []

    async def mock_send_json(data):
        events_emitted.append(data.get("type"))

    session.send_json_cb = mock_send_json

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def mock_llm_stream(messages):
        yield "Chào bạn hôm nay thế nào?"

    orig_llm = services.llm_stream
    services.llm_stream = mock_llm_stream

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    try:
        _step(session._produce_tts_text("turn_order", queue, "xin chào"))
        assert "assistant.text.done" in events_emitted
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert len(items) >= 2
        assert items[-1] is None
        assert "Chào bạn hôm nay thế nào?" in items[0]
    finally:
        services.llm_stream = orig_llm


def test_stt_finish_raises_on_connection_closed_without_is_done():
    stt = services.STTClient("sess_1", "turn_1", "req_1")
    stt.websocket = None
    stt.is_active = False

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    with pytest.raises(services.STTError) as exc_info:
        _step(stt.finish(timeout=0.01))
    assert exc_info.value.code == "CONNECTION_CLOSED"


def test_stt_recv_loop_detects_clean_disconnect_without_final():
    stt = services.STTClient("sess_2", "turn_2", "req_2")

    class MockEmptyWs:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            pass

    stt.websocket = MockEmptyWs()

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(stt._recv_loop())
    assert stt.error_payload is not None
    assert stt.error_payload.get("error_code") == "CONNECTION_CLOSED"
    assert stt.is_done.is_set() is True


def test_smart_turn_detector_predict_lock_and_finish_turn_cancels_task():
    detector = services.SmartTurnDetector()
    assert hasattr(detector, "_predict_lock")

    session = VoiceSession(session_id="test_st_cancel")
    session.current_turn_id = "turn_st"
    session.state = VoiceState.LISTENING

    cancelled = False

    class DummyTask:
        def done(self):
            return False

        def cancel(self):
            nonlocal cancelled
            cancelled = True

    session._smart_turn_task = DummyTask()  # type: ignore
    session._spawn_task = lambda coro: coro.close()  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session.finish_turn())
    assert cancelled is True
    assert session._smart_turn_task is None
    assert session.state == VoiceState.THINKING


def test_close_session_cleans_up_even_if_closed_flag_already_set():
    session = VoiceSession(session_id="test_cleanup_after_send_fail")
    session.closed = True  # Giả lập _safe_send phát hiện lỗi và đặt closed=True trước
    session.current_turn_id = "turn_leak_check"
    session.buffer_audio(b"\x00" * 320)
    session.history.append({"role": "user", "content": "hello"})

    stt_closed = False

    class MockSTT:
        async def close(self):
            nonlocal stt_closed
            stt_closed = True

    session.stt_client = MockSTT()  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    # Gọi close() phải thực hiện dọn dẹp đầy đủ chứ không return sớm do self.closed == True
    _step(session.close())
    assert session._cleanup_done is True
    assert session.current_turn_id is None
    assert session.turn_audio_bytes == 0
    assert len(session.history) == 0
    assert stt_closed is True
    assert session.stt_client is None


def test_set_state_error_resets_audio_and_smart_turn_flags():
    session = VoiceSession(session_id="test_error_state_reset")
    session.current_turn_id = "turn_err"
    session.state = VoiceState.LISTENING
    session.has_speech = True
    session.is_vad_speaking = True

    cancelled = False

    class DummyTask:
        def done(self):
            return False

        def cancel(self):
            nonlocal cancelled
            cancelled = True

    session._smart_turn_task = DummyTask()  # type: ignore

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(session.set_state(VoiceState.ERROR))
    assert session.state == VoiceState.ERROR
    assert session.has_speech is False
    assert session.is_vad_speaking is False
    assert cancelled is True
    assert session._smart_turn_task is None
    # current_turn_id vẫn được giữ lại để phục vụ nhận diện turn lỗi
    assert session.current_turn_id == "turn_err"


@pytest.mark.anyio
async def test_smart_turn_self_cancellation_prevention():
    session = VoiceSession(session_id="test_self_cancel")
    session.current_turn_id = "turn_sc"
    session.state = VoiceState.LISTENING

    spawned_turns = []
    session._spawn_task = lambda coro: spawned_turns.append(coro) or coro.close()  # type: ignore

    async def _mock_smart_turn():
        session._smart_turn_task = asyncio.current_task()
        await session.finish_turn("turn_sc")

    # Chạy _mock_smart_turn như 1 task
    task = asyncio.create_task(_mock_smart_turn())
    await task

    # Task không bị CancelledError và hoàn tất thành công
    assert task.done() and not task.cancelled()
    assert session.state == VoiceState.THINKING
    assert len(spawned_turns) == 1


@pytest.mark.anyio
async def test_finish_turn_stt_client_decoupling_prevents_stealing():
    session = VoiceSession(session_id="test_stt_decouple")
    session.current_turn_id = "turn_old"
    session.state = VoiceState.LISTENING

    class MockSTT:
        def __init__(self, name):
            self.name = name
            self.closed = False
            self.finished = False

        async def finish(self):
            self.finished = True
            return f"result_{self.name}"

        async def close(self):
            self.closed = True

    old_stt = MockSTT("old")
    new_stt = MockSTT("new")

    session.stt_client = old_stt  # type: ignore

    # finish_turn cho turn_old
    await session.finish_turn("turn_old")
    # self.stt_client phải được gỡ bỏ khỏi session ngay lập tức
    assert session.stt_client is None

    # Giả lập turn mới chen vào tạo stt_client mới
    session.current_turn_id = "turn_new"
    session.stt_client = new_stt  # type: ignore

    # Chạy _process_turn của turn_old (với old_stt đã được bàn giao)
    # Vì turn_old không còn là active_turn, nó phải đóng old_stt và KHÔNG đụng chạm new_stt
    await session._process_turn("turn_old", old_stt)  # type: ignore

    assert old_stt.closed is True
    assert session.stt_client is new_stt
    assert new_stt.closed is False
    assert new_stt.finished is False


@pytest.mark.anyio
async def test_finish_turn_ignores_stale_turn_id():
    session = VoiceSession(session_id="test_stale_turn")
    session.current_turn_id = "turn_2"
    session.state = VoiceState.LISTENING

    # Gọi finish_turn với turn_id cũ đã hết hiệu lực
    await session.finish_turn("turn_1")

    # State không được chuyển sang THINKING, turn_2 vẫn đang LISTENING
    assert session.state == VoiceState.LISTENING
    assert session.current_turn_id == "turn_2"


def test_turn_state_dataclass():
    turn = TurnState(turn_id="turn_abc")
    assert turn.turn_id == "turn_abc"
    assert turn.has_speech is False
    assert turn.is_vad_speaking is False
    assert turn.last_speech_time == 0.0

    # Buffer audio
    turn.buffer_audio(b"\x00\x01" * 10)
    assert len(turn.get_audio()) == 20
    assert turn.turn_audio_bytes == 20

    turn.reset_audio()
    assert turn.get_audio() == b""
    assert turn.turn_audio_bytes == 0


@pytest.mark.anyio
async def test_turn_state_close_releases_stt():
    class DummySTT:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    dummy = DummySTT()
    turn = TurnState(turn_id="turn_xyz", stt_client=dummy)  # type: ignore
    await turn.close()
    assert dummy.closed is True
    assert turn.stt_client is None


def test_new_config_fields():
    assert isinstance(settings.app_reload, bool)
    assert settings.stt_finish_timeout == 5.0
    assert settings.smart_turn_threads >= 1
    assert settings.smart_turn_max_workers >= 1











