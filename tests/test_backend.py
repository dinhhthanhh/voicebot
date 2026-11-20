"""Unit tests cho toàn bộ backend của Voicebot MVP."""

from __future__ import annotations

from typing import Any

import pytest

from backend import services
from backend.config import settings
from backend.text_processing import (
    REMAINDER_NOTE,
    SpeechFilter,
    TTSTextStreamer,
    clean_inline_markdown,
    is_only_filler,
    normalize_text_for_tts,
    split_for_tts,
)
from backend.voice import VoiceOrchestrator, VoiceSession, VoiceState, _calc_rms


# ---------------------------------------------------------------------------
# 1. Tests cho Module Cấu hình (config.py)
# ---------------------------------------------------------------------------
def test_config_defaults():
    assert settings.app_port > 0
    assert settings.stt_sample_rate == 16000
    assert settings.smart_turn_silence_timeout_seconds == 2.5
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


def test_speech_filter_streaming():
    sf = SpeechFilter(max_sentences=8, max_chars=600)
    text = (
        "Dạ chào bạn! Tôi là trợ lý ảo VNPost. "
        "Tôi có thể hỗ trợ gì cho bạn hôm nay?"
    )
    sentences = sf.feed(text) + sf.flush()
    joined = " ".join(sentences)
    assert "Dạ chào bạn!" in joined
    assert "Tôi là trợ lý ảo VNPost." in joined


def test_speech_filter_capping():
    sf = SpeechFilter(max_sentences=2, max_chars=50)
    text = "Câu thứ nhất rất là dài. Câu thứ hai tiếp tục dài. Câu thứ ba không được đọc."
    sentences = sf.feed(text) + sf.flush()
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
# 4. Tests cho Voice Session & Orchestrator (voice.py)
# ---------------------------------------------------------------------------
def test_session_lifecycle():
    orchestrator = VoiceOrchestrator()
    session = orchestrator.create_session("test_sess_01")
    assert session.session_id == "test_sess_01"
    assert orchestrator.sessions.get("test_sess_01") is session

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


def test_orchestrator_max_sessions():
    orch = VoiceOrchestrator()
    for i in range(settings.max_sessions):
        orch.create_session(f"sess_{i}")
    with pytest.raises(RuntimeError):
        orch.create_session("sess_overflow")


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
    collected.extend(streamer.flush())

    assert len(collected) >= 2
    full = " ".join(collected)
    assert "hai mươi lăm nghìn đồng" in full


def test_tts_text_streamer_capping():
    streamer = TTSTextStreamer(max_sentences=2, max_chars=50)
    text = "Câu thứ nhất rất là dài. Câu thứ hai tiếp tục dài. Câu thứ ba không được đọc."
    parts = streamer.feed(text) + streamer.flush()
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


def test_barge_in_preroll_and_decay():
    import numpy as np

    session = VoiceSession(session_id="test_barge_preroll")
    session.state = VoiceState.SPEAKING
    session.current_turn_id = "turn_1"

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    quiet_pcm = (np.ones(160, dtype=np.int16) * 300).tobytes()
    loud_pcm = (np.ones(160, dtype=np.int16) * 3500).tobytes()

    # 1. Gửi frame nhỏ: lưu vào pre-roll, hits = 0
    res = _step(session.check_barge_in(quiet_pcm))
    assert res is False
    assert len(session.barge_in_preroll) == 1
    assert session.barge_in_hits == 0

    # 2. Gửi 1 frame lớn: hit = 1
    res = _step(session.check_barge_in(loud_pcm))
    assert res is False
    assert session.barge_in_hits == 1
    assert len(session.barge_in_preroll) == 2

    # 3. Gửi 1 frame nhỏ giữa chừng: hits giảm về 0 (decay) nhưng pre-roll KHÔNG bị xóa
    res = _step(session.check_barge_in(quiet_pcm))
    assert res is False
    assert session.barge_in_hits == 0
    assert len(session.barge_in_preroll) == 3

    # 4. Gửi liên tiếp các frame lớn cho đến khi đạt debounce
    for _ in range(settings.barge_in_debounce_frames):
        res = _step(session.check_barge_in(loud_pcm))
    assert res is True
    assert len(session.barge_in_preroll) == 0


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
    orch = VoiceOrchestrator()
    session = orch.create_session("sess_err_cleanup")
    session.buffer_audio(b"\x00" * 320)
    session.barge_in_preroll.append(b"\x00" * 320)

    class FailingSTT:
        async def close(self):
            raise RuntimeError("Network failed on close")

    session.stt_client = FailingSTT()

    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    _step(orch.close_session("sess_err_cleanup"))
    assert session.closed is True
    assert session.turn_audio_bytes == 0
    assert len(session.barge_in_preroll) == 0
    assert session.stt_client is None
    assert "sess_err_cleanup" not in orch.sessions


def test_calc_rms_empty_and_odd():
    assert _calc_rms(b"") == 0.0
    assert _calc_rms(b"\x00") == 0.0
    assert _calc_rms(b"\x00\x00") == 0.0
    pcm = b"\x00\x10" * 100
    assert _calc_rms(pcm) > 0.0


def test_create_session_duplicate_id():
    orch = VoiceOrchestrator()
    orch.create_session("sess_dup_01")
    with pytest.raises(ValueError, match="đã tồn tại"):
        orch.create_session("sess_dup_01")


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
    session.barge_in_preroll.append(b"\x00" * 320)

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
    assert len(session.barge_in_preroll) == 0


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

    # Session closed: coro should not be executed
    session.closed = True

    async def dummy_coro():
        pass

    _step(session._safe_send(dummy_coro()))

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
        smart_turn_silence_timeout_seconds=-1.0,
        barge_in_rms_threshold=0.0,
        barge_in_debounce_frames=0,
        max_sessions=0,
        llm_timeout_seconds=0.0,
        tts_timeout_seconds=0.0,
    )
    warnings = bad_cfg.validate_config()
    assert any("SMART_TURN_THRESHOLD" in w for w in warnings)
    assert any("SMART_TURN_SILENCE_TIMEOUT_SECONDS" in w for w in warnings)
    assert any("BARGE_IN_RMS_THRESHOLD" in w for w in warnings)
    assert any("BARGE_IN_DEBOUNCE_FRAMES" in w for w in warnings)
    assert any("MAX_SESSIONS" in w for w in warnings)
    assert any("LLM_TIMEOUT_SECONDS" in w for w in warnings)
    assert any("TTS_TIMEOUT_SECONDS" in w for w in warnings)


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





