"""Unit tests cho toàn bộ backend của Voicebot MVP."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest

from backend import main, services
from backend.config import settings
from backend.text_processing import (
    REMAINDER_NOTE,
    TTSStreamer,
    clean_md,
    is_filler,
    normalize_tts,
    split_tts,
)
from backend.voice import TurnState, VoiceSession, VoiceState, _drain_queue


def _step(coro: Any) -> Any:
    """Thực thi một coroutine trong event loop thật."""
    return asyncio.run(coro)


def _make_wav_header(extra_chunks: bytes = b"") -> bytes:
    """Tạo header RIFF/WAVE chuẩn kèm fmt chunk và data chunk."""
    fmt_chunk = b"fmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00\xee\x02\x00\x02\x00\x10\x00"
    body = b"WAVE" + fmt_chunk + extra_chunks + b"data\x04\x00\x00\x00"
    return b"RIFF" + len(body).to_bytes(4, "little") + body


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
def test_is_filler():
    assert is_filler("ừ") is True
    assert is_filler("ừm ờ à") is True
    assert is_filler("à mà") is True
    assert is_filler("") is True
    assert is_filler("chào bạn tôi cần gửi hàng") is False


def test_clean_md():
    assert clean_md("**chữ đậm** và *chữ nghiêng*") == "chữ đậm và chữ nghiêng"
    assert clean_md("# Tiêu đề lớn") == "Tiêu đề lớn"
    assert clean_md("- Gạch đầu dòng") == "Gạch đầu dòng"
    assert clean_md("* Gạch đầu dòng dấu sao") == "Gạch đầu dòng dấu sao"
    assert clean_md("Xem tại [VNPost](https://vnpost.vn) nhé") == "Xem tại VNPost nhé"
    assert clean_md("Định dạng `inline code` và ~~gạch ngang~~ xong") == "Định dạng inline code và gạch ngang xong"
    assert clean_md("Mục 1. Gửi hàng tại bưu cục") == "Mục 1. Gửi hàng tại bưu cục"
    assert clean_md("Xem tại https://vnpost.vn.") == "Xem tại"
    assert clean_md("Tra cứu (https://vnpost.vn) ngay") == "Tra cứu ngay"


def test_tts_streamer_streaming():
    streamer = TTSStreamer(max_sentences=8, max_chars=600)
    text = (
        "Dạ chào bạn! Tôi là trợ lý ảo VNPost. "
        "Tôi có thể hỗ trợ gì cho bạn hôm nay?"
    )
    sentences = streamer.feed(text) + streamer.flush()
    joined = " ".join(sentences)
    assert "Dạ chào bạn!" in joined
    assert "Tôi là trợ lý ảo VNPost." in joined


def test_tts_streamer_capping():
    streamer = TTSStreamer(max_sentences=2, max_chars=50)
    text = "Câu thứ nhất rất là dài. Câu thứ hai tiếp tục dài. Câu thứ ba không được đọc."
    sentences = streamer.feed(text) + streamer.flush()
    assert len(sentences) == 3
    assert sentences[-1] == REMAINDER_NOTE


def test_normalize_tts():
    # Tiền tệ và số điện thoại
    res = normalize_tts("Cước phí là 25.000 đồng về SĐT 0912345678")
    assert "hai mươi lăm nghìn đồng" in res
    assert "không chín một hai ba bốn năm sáu bảy tám" in res

    # Tiền hàng tỷ không bị regex số điện thoại cắt cụt / đọc thành thập phân
    bil_res = normalize_tts("Tổng tiền là 1.000.000.000 đồng")
    assert "một tỷ đồng" in bil_res
    assert "phẩy" not in bil_res

    # Số điện thoại quốc tế +84 dạng viết cách và viết liền
    intl_res = normalize_tts("Gọi +84 912 345 678 hoặc +84912345678")
    assert "không chín một hai ba bốn năm sáu bảy tám" in intl_res
    assert intl_res.count("không chín một hai ba bốn năm sáu bảy tám") == 2

    # Ngày tháng
    date_res = normalize_tts("Hẹn giao ngày 19/11/2026")
    assert "ngày mười chín tháng mười một năm hai nghìn lẻ hai mươi sáu" in date_res

    # Số thập phân và đơn vị đo
    dec_res = normalize_tts("Trọng lượng 2.5kg và 24/7")
    assert "hai phẩy năm ki lô gam" in dec_res
    assert "hai tư trên bảy" in dec_res


def test_split_tts():
    long_text = "Một đoạn văn bản rất dài, " * 30
    parts = split_tts(long_text, limit=100)
    assert len(parts) > 1
    for p in parts:
        assert len(p) <= 100


# ---------------------------------------------------------------------------
# 3. Tests cho Service Adapters (services.py)
# ---------------------------------------------------------------------------
def test_tts_chunk_processing():
    fake_wav_header = _make_wav_header()
    pcm_payload = b"\x01\x02\x03\x04"
    raw_chunk = fake_wav_header + pcm_payload

    data, parsed, leftover = services._process_tts_chunk(
        raw_chunk, header_parsed=False, header_buf=bytearray(), leftover=b""
    )
    assert parsed is True
    assert data == pcm_payload
    assert leftover == b""

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
    assert detector._sync_predict(b"\x00" * 3200) == 0.0


# ---------------------------------------------------------------------------
# 4. Tests cho Voice Session (voice.py)
# ---------------------------------------------------------------------------
def test_session_lifecycle():
    session = VoiceSession("test_sess_01")
    assert session.session_id == "test_sess_01"

    for i in range(20):
        session.add_history("user" if i % 2 == 0 else "assistant", f"msg {i}")
    max_msgs = settings.voice_max_history_pairs * 2
    assert len(session.history) == max_msgs

    messages = session.build_messages()
    assert len(messages) == max_msgs + 1
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_silence_timeout_logic():
    session = VoiceSession(session_id="s_test")
    assert session.is_silence_timeout() is False

    session.current_turn = TurnState(turn_id="turn_test", has_speech=True, last_speech_time=0.0)
    assert session.is_silence_timeout() is False

    session.current_turn.last_speech_time = time.monotonic() - 6.0
    assert session.is_silence_timeout() is True

    session.current_turn.is_vad_speaking = True
    session.current_turn.last_speech_time = time.monotonic() - 1.0
    assert session.is_silence_timeout() is False


def test_phone_not_confused_with_date():
    res = normalize_tts("Liên hệ số 09.12.3456 để được hỗ trợ")
    assert "ngày" not in res


def test_hotline_and_date_spacing():
    res = normalize_tts("gọi tổng đài 1900 54 54 81 hoặc 1900 545481")
    assert "một chín không không năm bốn năm bốn tám một" in res
    assert res.count("một chín không không năm bốn năm bốn tám một") == 2

    date_res = normalize_tts("hẹn giao ngày 12 / 5")
    assert "ngày ngày" not in date_res
    assert "ngày mười hai tháng năm" in date_res


def test_tech_terms_tts():
    res = normalize_tts("mạng 5G hoạt động 24h và xem phim 3D 4K")
    assert "năm gờ" in res
    assert "hai mươi bốn giờ" in res
    assert "ba đê" in res
    assert "bốn ca" in res

    # Phân biệt đơn vị viết thường (5g gam, 4k nghìn) với thuật ngữ viết hoa (5G, 4K)
    unit_res = normalize_tts("Gói hàng nặng 5g giá 4k")
    assert "năm gam" in unit_res
    assert "bốn nghìn" in unit_res


def test_wav_chunk_advanced():
    buf = bytearray()
    d1, p1, l1 = services._process_tts_chunk(b"RI", False, buf, b"")
    assert d1 == b""
    assert p1 is False

    rest_hdr = _make_wav_header(extra_chunks=b"JUNK\x04\x00\x00\x001234")[2:]
    payload = b"\xaa\xbb\xcc\xdd"
    d2, p2, l2 = services._process_tts_chunk(rest_hdr + payload, p1, buf, l1)
    assert p2 is True
    assert d2 == payload
    assert l2 == b""


@pytest.mark.anyio
async def test_thinking_to_idle_when_empty_or_filtered():
    session = VoiceSession(session_id="test_thinking_idle")
    session.current_turn = TurnState(turn_id="turn_123")
    session.state = VoiceState.THINKING

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    await queue.put(None)
    await session._stream_tts("turn_123", queue)
    assert session.state == VoiceState.IDLE


def test_prompt_file_loaded():
    assert "VNPost" in settings.llm_system_prompt
    assert len(settings.llm_system_prompt) > 100


def test_tts_streamer():
    streamer = TTSStreamer(max_sentences=5, max_chars=300)
    deltas = ["Chào ", "bạn! ", "Cước ", "phí ", "là ", "25.000 đồng. ", "Xin ", "cảm ", "ơn!"]
    collected = []
    for d in deltas:
        collected.extend(streamer.feed(d))
    collected.extend(streamer.feed())

    assert len(collected) >= 2
    full = " ".join(collected)
    assert "hai mươi lăm nghìn đồng" in full


def test_error_state_protection():
    s = VoiceSession(session_id="test_error_prot")
    _step(s.set_state(VoiceState.ERROR))
    assert s.state == VoiceState.ERROR

    _step(s.set_state(VoiceState.IDLE))
    assert s.state == VoiceState.ERROR
    _step(s.set_state(VoiceState.SPEAKING))
    assert s.state == VoiceState.ERROR

    _step(s.set_state(VoiceState.IDLE, force=True))
    assert s.state == VoiceState.IDLE


def test_drain_queue():
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
    session.current_turn = TurnState(turn_id="turn_vad")

    _step(session._handle_vad("turn_vad", "speech_started"))
    assert session.current_turn.is_vad_speaking is True
    assert session.current_turn.has_speech is True
    assert session.current_turn.smart_turn_task is None

    session.current_turn.has_speech = False
    _step(session._handle_vad("turn_vad", "speech_ended"))
    assert session.current_turn.is_vad_speaking is False
    assert session.current_turn.smart_turn_task is None


def test_close_session_safe_on_error():
    session = VoiceSession(session_id="sess_err_cleanup")
    session.current_turn = TurnState(turn_id="turn_err")
    session.current_turn.buffer_audio(b"\x00" * 320)

    class FailingSTT:
        async def close(self):
            raise RuntimeError("Network failed on close")

    session.current_turn.stt_client = FailingSTT()  # type: ignore
    _step(session.close())
    assert session.closed is True
    assert session.current_turn is None


def test_add_history_zero_limit():
    original_val = settings.voice_max_history_pairs
    object.__setattr__(settings, "voice_max_history_pairs", 0)
    try:
        session = VoiceSession(session_id="sess_hist_zero")
        session.add_history("user", "xin chào")
        assert len(session.history) == 0
    finally:
        object.__setattr__(settings, "voice_max_history_pairs", original_val)


def test_cancel_turn_safe_when_stt_close_fails():
    session = VoiceSession(session_id="sess_cancel_test")
    session.current_turn = TurnState(turn_id="turn_cancel_01")
    session.state = VoiceState.SPEAKING
    session.current_turn.buffer_audio(b"\x00" * 320)

    class FailingSTT:
        async def close(self):
            raise RuntimeError("STT close failed")

    session.current_turn.stt_client = FailingSTT()  # type: ignore
    _step(session.cancel_turn("turn_cancel_01"))
    assert session.current_turn is None
    assert session.state == VoiceState.IDLE


def test_pipeline_error_isolated_to_turn():
    session = VoiceSession(session_id="sess_turn_isolation")
    session.current_turn = TurnState(turn_id="turn_B")
    session.state = VoiceState.LISTENING

    q: asyncio.Queue[Any] = asyncio.Queue()
    q.put_nowait("dummy")

    _step(session._fail_turn("ERR_CODE", "Error from turn A", turn_id="turn_A", stop_playback=True, queue=q))
    assert session.state == VoiceState.LISTENING
    assert not q.empty()


def test_safe_send_closed_and_disconnect():
    session = VoiceSession(session_id="sess_safe_send")

    session.closed = True
    called_on_closed = []

    async def dummy_send():
        called_on_closed.append(True)

    _step(session._safe_send(dummy_send))
    assert called_on_closed == []

    session.closed = False
    called = []

    async def working_send():
        called.append(True)

    _step(session._safe_send(working_send))
    assert called == [True]

    async def failing_send():
        raise ConnectionResetError("Connection lost")

    _step(session._safe_send(failing_send))
    assert session.closed is True


def test_config_validation_ranges():
    bad_cfg = settings.__class__(
        app_port=0,
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
    assert any("APP_PORT" in w for w in warnings)
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
    stt.websocket = MockWS()  # type: ignore
    stt.is_active = True

    with pytest.raises(services.STTError, match="Lỗi gửi audio tới STT"):
        _step(stt.send_audio(b"\x00\x00" * 160))


# ---------------------------------------------------------------------------
# 5. Tests bổ sung cho các cải tiến Logic (Review fixes)
# ---------------------------------------------------------------------------
def test_match_turn_strict_and_empty_handling():
    session = VoiceSession(session_id="test_match_turn")
    assert session.is_active_turn("turn_1") is False
    assert session.is_active_turn(None, allow_empty=True) is False
    assert session.is_active_turn(None, allow_empty=False) is False

    session.current_turn = TurnState(turn_id="turn_active")
    assert session.is_active_turn("turn_active") is True
    assert session.is_active_turn("turn_old") is False
    assert session.is_active_turn(None, allow_empty=False) is False
    assert session.is_active_turn(None, allow_empty=True) is True
    assert session.is_active_turn("", allow_empty=False) is False
    assert session.is_active_turn("", allow_empty=True) is True


def test_cancel_turn_clears_current_turn_id():
    session = VoiceSession(session_id="test_cancel_id")
    session.current_turn = TurnState(turn_id="turn_to_cancel")
    session.state = VoiceState.SPEAKING

    _step(session.cancel_turn("turn_to_cancel"))
    assert session.current_turn_id is None
    assert session.state == VoiceState.IDLE


def test_playback_done_and_cancel_isolation():
    session = VoiceSession(session_id="test_isolation")
    session.current_turn = TurnState(turn_id="turn_current")
    session.state = VoiceState.SPEAKING

    _step(session.handle_event({"type": "playback.done", "turn_id": "turn_old"}))
    assert session.state == VoiceState.SPEAKING
    assert session.current_turn_id == "turn_current"

    _step(session.handle_event({"type": "playback.done"}))
    assert session.state == VoiceState.SPEAKING

    _step(session.handle_event({"type": "playback.done", "turn_id": "turn_current"}))
    assert session.state == VoiceState.IDLE


def test_wav_chunk_extended_header_over_4096():
    extended_junk = b"JUNK" + (5000).to_bytes(4, "little") + (b"\x00" * 5000)
    fake_header = _make_wav_header(extra_chunks=extended_junk)
    payload = b"\xde\xad\xbe\xef"

    buf = bytearray()
    data, parsed, leftover = services._process_tts_chunk(
        fake_header + payload, header_parsed=False, header_buf=buf, leftover=b""
    )
    assert parsed is True
    assert data == payload
    assert leftover == b""


def test_smart_turn_logs_warning_on_exception(caplog):
    detector = services.SmartTurnDetector(model_path="dummy.onnx")

    def failing_predict(*args, **kwargs):
        raise RuntimeError("Inference crash")

    detector.session = object()  # type: ignore
    detector.feature_extractor = object()  # type: ignore
    detector._attempted = True
    detector._sync_predict = failing_predict  # type: ignore

    with caplog.at_level(logging.WARNING):
        result = _step(detector.predict(b"\x00\x00" * 2000))

    assert result == 0.0
    assert any("Lỗi suy luận Smart Turn" in record.message for record in caplog.records)


def test_stt_error_event_transitions_session_to_error():
    session = VoiceSession(session_id="test_stt_err")
    session.state = VoiceState.LISTENING

    class MockSTTWithError:
        def __init__(self):
            self.closed = False

        async def events(self):
            yield {"type": "error", "error_code": "AUDIO_TOO_LONG", "message": "Quá dài"}

        async def close(self):
            self.closed = True

    mock_stt = MockSTTWithError()
    session.current_turn = TurnState(turn_id="turn_err_1", stt_client=mock_stt)  # type: ignore

    events_emitted = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send

    _step(session._forward_stt(mock_stt, "turn_err_1"))  # type: ignore
    assert session.state == VoiceState.ERROR
    assert session.current_turn.stt_client is None
    assert mock_stt.closed is True
    assert any(ev.get("type") == "session.error" and ev.get("code") == "AUDIO_TOO_LONG" for ev in events_emitted)


def test_handle_audio_frame_stt_error():
    session = VoiceSession(session_id="test_stt_send_err")
    session.state = VoiceState.LISTENING

    class MockFailingSTT:
        is_active = True

        async def send_audio(self, pcm):
            raise services.STTError("STT send failed", code="STT_SEND_ERROR")

        async def close(self):
            self.is_active = False

    session.current_turn = TurnState(turn_id="turn_send_err", stt_client=MockFailingSTT())  # type: ignore
    _step(session.handle_audio_frame(b"\x00\x00" * 160))
    assert session.state == VoiceState.ERROR
    assert session.current_turn.stt_client is None


def test_producer_preserves_queue_on_normal_completion():
    session = VoiceSession(session_id="test_prod_queue")
    session.current_turn = TurnState(turn_id="turn_prod")
    session.state = VoiceState.THINKING

    async def fake_llm(messages):
        yield "Câu một. "
        yield "Câu hai. "

    original_llm = services.llm_stream
    services.llm_stream = fake_llm
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)

    try:
        _step(session._produce_llm("turn_prod", queue, "User prompt"))
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())

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
    session.state = VoiceState.THINKING

    class TimeoutSTT:
        async def finish(self):
            raise services.STTError("Timeout STT", code="STT_TIMEOUT")

    stt = TimeoutSTT()
    session.current_turn = TurnState(turn_id="turn_fb", last_partial_text="Tôi cần gửi bưu phẩm")

    events = []

    async def mock_send(ev):
        events.append(ev)

    session.send_json_cb = mock_send

    async def mock_tg(turn_id, text):
        pass

    session._run_pipeline = mock_tg  # type: ignore
    _step(session._process_turn("turn_fb", stt))  # type: ignore

    final_ev = next((ev for ev in events if ev.get("type") == "transcript.final"), None)
    assert final_ev is not None
    assert final_ev.get("text") == "Tôi cần gửi bưu phẩm"


def test_process_turn_empty_transcript_responds_with_prompt():
    session = VoiceSession(session_id="sess_empty_text")
    session.state = VoiceState.THINKING

    class EmptySTT:
        async def finish(self):
            return ""

    stt = EmptySTT()
    session.current_turn = TurnState(turn_id="turn_empty", last_partial_text="")

    events = []

    async def mock_send(ev):
        events.append(ev)

    session.send_json_cb = mock_send

    filler_prompts = []

    async def mock_filler(turn_id, prompt):
        filler_prompts.append((turn_id, prompt))

    session._speak_prompt = mock_filler  # type: ignore
    _step(session._process_turn("turn_empty", stt))  # type: ignore

    final_ev = next((ev for ev in events if ev.get("type") == "transcript.final"), None)
    assert final_ev is not None
    assert final_ev.get("text") == ""
    assert len(filler_prompts) == 1
    assert filler_prompts[0] == ("turn_empty", "Tôi chưa nghe rõ, bạn vui lòng nói lại nhé.")


def test_process_turn_stt_timeout_no_partial_transitions_to_error():
    session = VoiceSession(session_id="sess_timeout_err")
    session.state = VoiceState.THINKING

    class TimeoutSTT:
        async def finish(self):
            raise services.STTError("Timeout STT", code="STT_TIMEOUT")

    stt = TimeoutSTT()
    session.current_turn = TurnState(turn_id="turn_to_err", last_partial_text="")
    _step(session._process_turn("turn_to_err", stt))  # type: ignore
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

    _step(stt.close())
    _step(stt.close())

    assert stt._closed is True
    assert stt.is_active is False
    assert ws.close_count == 1


def test_stt_recv_loop_propagates_cancelled_error():
    class ClosingWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

    stt = services.STTClient("sess_cancel", "turn_c", "req_c")
    stt.websocket = ClosingWS()  # type: ignore

    with pytest.raises(asyncio.CancelledError):
        _step(stt._recv_loop())

    assert stt.is_active is False
    assert stt.is_done.is_set() is True


def test_parse_sse_delta_strict_validation():
    assert services._parse_sse_delta('{"choices": [null]}') is None
    assert services._parse_sse_delta('{"choices": [{"delta": null}]}') is None
    assert services._parse_sse_delta('{"choices": [{"delta": {"content": 123}}]}') is None
    assert services._parse_sse_delta('{"choices": []}') is None
    assert services._parse_sse_delta('{"choices": "not_a_list"}') is None
    assert services._parse_sse_delta('"just_string"') is None
    assert services._parse_sse_delta("12345") is None
    assert services._parse_sse_delta('{"choices": [{"delta": {"content": "Xin chào"}}]}') == "Xin chào"


def test_wav_data_offset_standard_and_extended():
    std_wav = (
        b"RIFF\x24\x00\x00\x00WAVE"
        b"fmt \x10\x00\x00\x00" + b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
        b"data\x04\x00\x00\x00"
    )
    assert services._wav_data_offset(bytearray(std_wav)) == 44

    list_chunk = b"LIST\x08\x00\x00\x0012345678"
    ext_wav = (
        b"RIFF\x30\x00\x00\x00WAVE"
        b"fmt \x10\x00\x00\x00" + b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
        + list_chunk
        + b"data\x04\x00\x00\x00"
    )
    assert services._wav_data_offset(bytearray(ext_wav)) == 44 + len(list_chunk)
    assert services._wav_data_offset(bytearray(b"\x01\x02\x03\x04" * 10)) == 0
    assert services._wav_data_offset(bytearray(b"RIFF\x00\x00\x00\x00AVI ")) == 0


def test_tts_stream_odd_byte_leftover_warning_and_pad(caplog):
    class MockResp:
        status_code = 200

        async def aiter_bytes(self):
            yield b"\x01\x02\x03\x04"
            yield b"\x05\x06\x07"

    class MockClient:
        is_closed = False

        @asynccontextmanager
        async def stream(self, *args, **kwargs):
            yield MockResp()

    orig_client_fn = services.get_http_client
    services.get_http_client = lambda: MockClient()  # type: ignore

    async def _collect():
        chunks = []
        with caplog.at_level(logging.WARNING):
            async for chunk in services.tts_stream("Test leftover"):
                chunks.append(chunk)
        return chunks

    try:
        chunks = _step(_collect())
        total_bytes = b"".join(chunks)
        assert len(total_bytes) % 2 == 0
        assert len(total_bytes) == 8
        assert any("Dữ liệu PCM từ TTS không hợp lệ" in r.message for r in caplog.records)
    finally:
        services.get_http_client = orig_client_fn


# ---------------------------------------------------------------------------
# 6. Tests cho các luồng quan trọng (Turn order, STT finish, Smart Turn, Main)
# ---------------------------------------------------------------------------
def test_produce_llm_order_text_done_before_sentinel():
    session = VoiceSession(session_id="test_order")
    session.current_turn = TurnState(turn_id="turn_order")
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

    try:
        _step(session._produce_llm("turn_order", queue, "xin chào"))
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

    stt.websocket = MockEmptyWs()  # type: ignore
    _step(stt._recv_loop())
    assert stt.error_payload is not None
    assert stt.error_payload.get("error_code") == "CONNECTION_CLOSED"
    assert stt.is_done.is_set() is True


def test_smart_turn_detector_predict_lock_and_finish_turn_cancels_task():
    detector = services.SmartTurnDetector()
    assert hasattr(detector, "_executor")
    assert hasattr(detector, "shutdown")

    session = VoiceSession(session_id="test_st_cancel")
    session.current_turn = TurnState(turn_id="turn_st")
    session.state = VoiceState.LISTENING

    cancelled = False

    class DummyTask:
        def done(self):
            return False

        def cancel(self):
            nonlocal cancelled
            cancelled = True

    session.current_turn.smart_turn_task = DummyTask()  # type: ignore
    session._spawn_task = lambda coro: coro.close()  # type: ignore

    _step(session.finish_turn())
    assert cancelled is True
    assert session.current_turn.smart_turn_task is None
    assert session.state == VoiceState.THINKING


def test_close_session_cleans_up_even_if_closed_flag_already_set():
    session = VoiceSession(session_id="test_cleanup_after_send_fail")
    session.closed = True
    session.current_turn = TurnState(turn_id="turn_leak_check")
    session.current_turn.buffer_audio(b"\x00" * 320)
    session.history.append({"role": "user", "content": "hello"})

    stt_closed = False

    class MockSTT:
        async def close(self):
            nonlocal stt_closed
            stt_closed = True

    session.current_turn.stt_client = MockSTT()  # type: ignore
    _step(session.close())
    assert session._cleanup_done is True
    assert session.current_turn is None
    assert len(session.history) == 0
    assert stt_closed is True


def test_set_state_error_resets_audio_and_smart_turn_flags():
    session = VoiceSession(session_id="test_error_state_reset")
    session.current_turn = TurnState(turn_id="turn_err", has_speech=True, is_vad_speaking=True)
    session.current_turn.buffer_audio(b"\x00" * 320)
    session.state = VoiceState.LISTENING

    cancelled = False

    class DummyTask:
        def done(self):
            return False

        def cancel(self):
            nonlocal cancelled
            cancelled = True

    session.current_turn.smart_turn_task = DummyTask()  # type: ignore
    _step(session.set_state(VoiceState.ERROR))
    assert session.state == VoiceState.ERROR
    assert session.current_turn.has_speech is False
    assert session.current_turn.is_vad_speaking is False
    assert session.current_turn.turn_audio_bytes == 0
    assert cancelled is True
    assert session.current_turn.smart_turn_task is None
    assert session.current_turn_id == "turn_err"


@pytest.mark.anyio
async def test_smart_turn_self_cancellation_prevention():
    session = VoiceSession(session_id="test_self_cancel")
    session.current_turn = TurnState(turn_id="turn_sc")
    session.state = VoiceState.LISTENING

    spawned_turns = []
    session._spawn_task = lambda coro: spawned_turns.append(coro) or coro.close()  # type: ignore

    async def _mock_smart_turn():
        assert session.current_turn is not None
        session.current_turn.smart_turn_task = asyncio.current_task()
        await session.finish_turn("turn_sc")

    task = asyncio.create_task(_mock_smart_turn())
    await task

    assert task.done() and not task.cancelled()
    assert session.state == VoiceState.THINKING
    assert len(spawned_turns) == 1


@pytest.mark.anyio
async def test_finish_turn_stt_client_decoupling_prevents_stealing():
    session = VoiceSession(session_id="test_stt_decouple")
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

    session.current_turn = TurnState(turn_id="turn_old", stt_client=old_stt)  # type: ignore
    await session.finish_turn("turn_old")
    assert session.current_turn.stt_client is None

    session.current_turn = TurnState(turn_id="turn_new", stt_client=new_stt)  # type: ignore
    await session._process_turn("turn_old", old_stt)  # type: ignore

    assert old_stt.closed is True
    assert session.current_turn.stt_client is new_stt
    assert new_stt.closed is False
    assert new_stt.finished is False


@pytest.mark.anyio
async def test_finish_turn_ignores_stale_turn_id():
    session = VoiceSession(session_id="test_stale_turn")
    session.current_turn = TurnState(turn_id="turn_2")
    session.state = VoiceState.LISTENING

    await session.finish_turn("turn_1")
    assert session.state == VoiceState.LISTENING
    assert session.current_turn_id == "turn_2"


def test_turn_state_dataclass():
    turn = TurnState(turn_id="turn_abc")
    assert turn.turn_id == "turn_abc"
    assert turn.has_speech is False
    assert turn.is_vad_speaking is False
    assert turn.last_speech_time == 0.0

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
    assert settings.smart_turn_max_workers >= 1


def test_sentence_split_numbered_list():
    streamer = TTSStreamer()
    res = streamer.feed("1. Chuẩn bị hàng. 2. Đem ra bưu cục.") + streamer.flush()
    assert len(res) == 2
    assert "Chuẩn bị hàng." in res[0]
    assert "Đem ra bưu cục." in res[1]


def test_large_number_with_unit_tts():
    res = normalize_tts("Tổng cộng 10000000 đồng")
    assert "mười triệu đồng" in res


def test_silence_timeout_stuck_vad():
    session = VoiceSession(session_id="s_stuck_vad")
    session.current_turn = TurnState(turn_id="turn_stuck", has_speech=True, is_vad_speaking=True)
    session.current_turn.last_speech_time = time.monotonic() - 5.0
    assert session.is_silence_timeout() is False

    session.current_turn.last_speech_time = time.monotonic() - 11.0
    assert session.is_silence_timeout() is True


def test_process_turn_filler_tts_error_code():
    session = VoiceSession(session_id="s_filler_err")
    session.state = VoiceState.THINKING
    session.current_turn = TurnState(turn_id="turn_fill")

    class EmptySTT:
        async def finish(self):
            return ""

    stt = EmptySTT()

    async def mock_tts_stream(sentence):
        raise services.TTSError("ZeroTTS sập", code="TTS_STREAM_ERROR")
        yield b""  # pragma: no cover

    original_tts = services.tts_stream
    services.tts_stream = mock_tts_stream

    events = []

    async def mock_send(ev):
        events.append(ev)

    session.send_json_cb = mock_send

    try:
        _step(session._process_turn("turn_fill", stt))  # type: ignore
        assert session.state == VoiceState.ERROR
        err_ev = next((ev for ev in events if ev.get("type") == "session.error"), None)
        assert err_ev is not None
        assert err_ev.get("code") == "TTS_STREAM_ERROR"
    finally:
        services.tts_stream = original_tts


def test_validate_config_new_rules():
    assert settings.smart_turn_max_workers == 2
    orig_finish = settings.stt_finish_timeout
    try:
        object.__setattr__(settings, "stt_finish_timeout", -1.0)
        warnings = settings.validate_config()
        assert any("STT_FINISH_TIMEOUT" in w for w in warnings)
    finally:
        object.__setattr__(settings, "stt_finish_timeout", orig_finish)


@pytest.mark.anyio
async def test_start_turn_cancel_during_connect_closes_orphan_stt():
    session = VoiceSession(session_id="sess_cancel_during_connect")
    connected_event = asyncio.Event()
    proceed_event = asyncio.Event()

    class SlowSTT:
        def __init__(self):
            self.closed = False

        async def events(self):
            if False:
                yield {}

        async def close(self):
            self.closed = True

    slow_stt = SlowSTT()

    async def fake_connect(session_id, turn_id):
        connected_event.set()
        await proceed_event.wait()
        return slow_stt

    orig_connect = services.STTClient.connect
    services.STTClient.connect = fake_connect  # type: ignore

    try:
        start_task = asyncio.create_task(session.start_turn())
        await connected_event.wait()

        # cancel_turn không bị block bởi turn_lock khi STT đang connect
        await asyncio.wait_for(session.cancel_turn(None), timeout=0.2)
        assert session.state == VoiceState.IDLE
        assert session.current_turn is None

        proceed_event.set()
        await start_task
        assert slow_stt.closed is True
    finally:
        services.STTClient.connect = orig_connect  # type: ignore


@pytest.mark.anyio
async def test_forward_stt_ignores_after_thinking():
    session = VoiceSession(session_id="sess_forward_race")
    session.state = VoiceState.THINKING
    session.current_turn = TurnState(turn_id="turn_race")
    events_emitted = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send

    class LateErrorSTT:
        async def events(self):
            yield {"type": "partial", "text": "trễ"}
            yield {"type": "error", "error_code": "STT_ERROR", "message": "lỗi muộn"}

        async def close(self):
            pass

    await session._forward_stt(LateErrorSTT(), "turn_race")  # type: ignore
    assert events_emitted == []
    assert session.state == VoiceState.THINKING


@pytest.mark.anyio
async def test_handle_audio_frame_ignores_odd_bytes_outside_listening():
    session = VoiceSession(session_id="sess_idle_audio")
    session.state = VoiceState.IDLE
    events_emitted = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send
    await session.handle_audio_frame(b"\x01\x02\x03")
    assert events_emitted == []


@pytest.mark.anyio
async def test_gateway_health_and_ws_message_handling():
    health = await main.health_check()
    assert health["status"] == "ok"
    assert isinstance(health["smart_turn_ready"], bool)

    session = VoiceSession(session_id="sess_gw")
    events_emitted = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send
    await main._handle_ws_message(session, {"text": "{bad json"})
    await main._handle_ws_message(session, {"text": "[1, 2]"})
    session.state = VoiceState.SPEAKING
    session.current_turn = TurnState(turn_id="turn_gw")
    await main._handle_ws_message(session, {"text": '{"type": "playback.done", "turn_id": "turn_gw"}'})

    codes = [e.get("code") for e in events_emitted if e.get("type") == "session.error"]
    assert codes == ["INVALID_JSON", "INVALID_PAYLOAD"]
    assert any(e.get("type") == "voice.state" and e.get("state") == "idle" for e in events_emitted)


def test_tts_streamer_flush_does_not_duplicate_short_hold():
    streamer = TTSStreamer(min_chars=8)
    assert streamer.feed("Vâng. ") == []
    assert streamer.flush() == ["Vâng."]


@pytest.mark.anyio
async def test_stream_tts_empty_bytes_stays_thinking_then_idle():
    session = VoiceSession(session_id="sess_empty_tts")
    session.state = VoiceState.THINKING
    session.current_turn = TurnState(turn_id="turn_empty_tts")
    events_emitted = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send

    async def empty_tts(sentence):
        if False:
            yield b""

    orig_tts = services.tts_stream
    services.tts_stream = empty_tts
    try:
        q: asyncio.Queue[str | None] = asyncio.Queue()
        await q.put("Xin chào")
        await q.put(None)
        await session._stream_tts("turn_empty_tts", q)
        assert session.state == VoiceState.IDLE
        assert not any(ev.get("type") == "assistant.audio.done" for ev in events_emitted)
        assert not any(ev.get("state") == "speaking" for ev in events_emitted)
    finally:
        services.tts_stream = orig_tts


@pytest.mark.anyio
async def test_run_pipeline_unwraps_nested_exception_group():
    session = VoiceSession(session_id="sess_eg")
    session.state = VoiceState.THINKING
    session.current_turn = TurnState(turn_id="turn_eg")
    events_emitted = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send

    async def failing_producer(turn_id, queue, final_text):
        raise ExceptionGroup("nested", [services.LLMError("Upstream fail", code="LLM_502")])

    orig_producer = session._produce_llm
    session._produce_llm = failing_producer  # type: ignore
    try:
        await session._run_pipeline("turn_eg", "xin chào")
        assert session.state == VoiceState.ERROR
        err_ev = next((e for e in events_emitted if e.get("type") == "session.error"), None)
        assert err_ev is not None
        assert err_ev.get("code") == "LLM_502"
        assert "Upstream fail" in err_ev.get("message", "")
    finally:
        session._produce_llm = orig_producer


def test_text_processing_superscript_and_unbroken_split():
    assert normalize_tts("Diện tích 10² và 0²") == "Diện tích 10² và 0²"
    parts = split_tts("a" * 25, limit=10)
    assert all(len(p) <= 10 for p in parts)


def test_is_active_turn_handles_non_string_turn_id():
    session = VoiceSession(session_id="sess_non_str")
    session.current_turn = TurnState(turn_id="turn_1")
    assert session.is_active_turn(123) is False  # type: ignore
    assert session.is_active_turn(123, allow_empty=True) is False  # type: ignore


def test_date_vs_time_and_delivery_value_ranges():
    # Khoảng thời gian không bị hiểu nhầm là ngày/tháng
    assert normalize_tts("giao từ 2-3 ngày") == "giao từ hai đến ba ngày"
    assert normalize_tts("dự kiến 3-5 ngày làm việc") == "dự kiến ba đến năm ngày làm việc"
    assert normalize_tts("từ 8-10 giờ sáng") == "từ tám đến mười giờ sáng"
    assert normalize_tts("sau 1-2 tuần") == "sau một đến hai tuần"
    assert normalize_tts("từ 8-12h") == "từ tám đến mười hai giờ"
    assert normalize_tts("giao 2/3 giờ") == "giao hai trên ba giờ"

    # Khoảng giá trị giao hàng (trọng lượng, cước, số đơn) không bị hiểu nhầm là ngày/tháng
    assert normalize_tts("nặng từ 2.5 kg") == "nặng từ hai phẩy năm ki lô gam"
    assert normalize_tts("từ 1.5 - 3.5 kg") == "từ một phẩy năm đến ba phẩy năm ki lô gam"
    assert normalize_tts("từ 5-10 kg") == "từ năm đến mười ki lô gam"
    assert normalize_tts("từ 5-10kg") == "từ năm đến mười ki lô gam"
    assert normalize_tts("cước từ 25.000 - 35.000 đồng") == "cước từ hai mươi lăm nghìn đến ba mươi lăm nghìn đồng"
    assert normalize_tts("cước 15.000đ - 20.000đ") == "cước mười lăm nghìn đồng đến hai mươi nghìn đồng"
    assert normalize_tts("từ 2kg - 5kg") == "từ hai ki lô gam đến năm ki lô gam"
    assert normalize_tts("giao 1-2 đơn") == "giao một đến hai đơn"
    assert normalize_tts("từ 15/10 kg") == "từ mười lăm trên mười ki lô gam"

    # Các dạng ngày tháng và khoảng ngày hợp lệ vẫn hoạt động chính xác
    assert normalize_tts("từ 1/5 đến 3/5") == "từ ngày một tháng năm đến ngày ba tháng năm"
    assert normalize_tts("từ 5-10/12") == "từ ngày năm đến ngày mười tháng mười hai"
    assert normalize_tts("từ 1/5 - 3/5") == "từ ngày một tháng năm đến ngày ba tháng năm"
    assert normalize_tts("ngày 12-5") == "ngày mười hai tháng năm"
    assert normalize_tts("giao 12/5, ngày mai nhận") == "giao ngày mười hai tháng năm, ngày mai nhận"


@pytest.mark.anyio
async def test_stt_connect_cancel_and_empty_error_msg():
    stt = services.STTClient("s", "t", "r")
    stt.error_payload = {"type": "error", "error_code": "ERR", "message": ""}
    with pytest.raises(services.STTError, match="Lỗi mặc định"):
        stt._raise_if_error("ERR", "Lỗi mặc định")

    ws_closed = False

    class BlockingWS:
        async def send(self, data):
            pass

        async def close(self, **_kwargs):
            nonlocal ws_closed
            ws_closed = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(10)
            raise StopAsyncIteration

    orig_ws_connect = services.websockets.connect

    async def fake_ws_connect(*_args, **_kwargs):
        return BlockingWS()

    services.websockets.connect = fake_ws_connect  # type: ignore
    try:
        task = asyncio.create_task(services.STTClient.connect("s_cancel", "t_cancel"))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ws_closed is True
    finally:
        services.websockets.connect = orig_ws_connect


def test_smart_turn_partial_init_failure_resets_ready(tmp_path):
    dummy_model = tmp_path / "dummy.onnx"
    dummy_model.write_bytes(b"fake")
    detector = services.SmartTurnDetector(model_path=dummy_model)

    orig_ort = services.ort.InferenceSession
    orig_ext = services.WhisperFeatureExtractor
    services.ort.InferenceSession = lambda *_args, **_kwargs: object()  # type: ignore
    services.WhisperFeatureExtractor = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fail"))  # type: ignore
    try:
        detector.load_model()
        assert detector.is_ready is False
        assert detector.session is None
        assert detector._attempted is True
    finally:
        services.ort.InferenceSession = orig_ort
        services.WhisperFeatureExtractor = orig_ext


@pytest.mark.anyio
async def test_handle_audio_frame_odd_bytes_in_listening_fails_once():
    session = VoiceSession(session_id="sess_odd_listening")
    session.state = VoiceState.LISTENING

    class DummySTT:
        is_active = True

        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    stt = DummySTT()
    session.current_turn = TurnState(turn_id="turn_odd", stt_client=stt)  # type: ignore

    events_emitted: list[dict[str, Any]] = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send

    await session.handle_audio_frame(b"\x01\x02\x03")
    await session.handle_audio_frame(b"\x04\x05\x06")

    err_events = [e for e in events_emitted if e.get("type") == "session.error"]
    assert len(err_events) == 1
    assert err_events[0].get("code") == "INVALID_AUDIO_FORMAT"
    assert session.state == VoiceState.ERROR
    assert stt.closed is True


@pytest.mark.anyio
async def test_cancel_tasks_preserves_active_tasks_set_identity():
    session = VoiceSession(session_id="sess_set_id")
    orig_set = session.active_tasks

    done_gate = asyncio.Event()

    async def self_canceller():
        await session.cancel_tasks()
        await done_gate.wait()

    async def sleeper():
        await asyncio.sleep(10)

    session._spawn_task(sleeper())
    runner = session._spawn_task(self_canceller())
    await asyncio.sleep(0.01)

    assert session.active_tasks is orig_set
    assert runner in session.active_tasks
    done_gate.set()
    await runner
    await asyncio.sleep(0)
    assert runner not in session.active_tasks


@pytest.mark.anyio
async def test_start_turn_preserves_stt_error_code_on_connect_fail():
    session = VoiceSession(session_id="sess_stt_overload")
    events_emitted: list[dict[str, Any]] = []

    async def mock_send(ev):
        events_emitted.append(ev)

    session.send_json_cb = mock_send
    orig_connect = services.STTClient.connect

    async def fail_connect(session_id, turn_id):
        raise services.STTError("Quá tải", code="OVERLOADED")

    services.STTClient.connect = fail_connect  # type: ignore
    try:
        await session.start_turn()
        err_ev = next((e for e in events_emitted if e.get("type") == "session.error"), None)
        assert err_ev is not None
        assert err_ev.get("code") == "OVERLOADED"
        assert session.state == VoiceState.ERROR
    finally:
        services.STTClient.connect = orig_connect  # type: ignore


@pytest.mark.anyio
async def test_safe_send_handles_disconnect_and_close_message_errors():
    from starlette.websockets import WebSocketDisconnect
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    for exc in (
        WebSocketDisconnect(code=1000),
        ConnectionClosed(Close(1000, "bye"), None),
        RuntimeError('Cannot call "send" once a close message has been sent.'),
    ):
        session = VoiceSession(session_id="sess_send_err")

        async def broken_send(err=exc):
            raise err

        await session._safe_send(broken_send)
        assert session.closed is True


def test_smart_turn_sync_predict_handles_odd_and_oversized_buffer():
    class FakeExt:
        def __call__(self, audio, **_kwargs):
            assert len(audio) == 8 * 16000
            return {"input_features": services.np.zeros((1, 80, 800), dtype=services.np.float32)}

    class FakeSession:
        def run(self, *_args, **_kwargs):
            return [services.np.array([[0.85]], dtype=services.np.float32)]

    detector = services.SmartTurnDetector(model_path="dummy.onnx")
    detector.feature_extractor = FakeExt()  # type: ignore
    detector.session = FakeSession()  # type: ignore

    # > 8s + odd byte count (256000 + 3201)
    prob = detector._sync_predict(b"\x01\x02" * (128000 + 1600) + b"\x03", sample_rate=16000)
    assert abs(prob - 0.85) < 1e-5
    assert detector._sync_predict(b"\x01", sample_rate=16000) == 0.0


@pytest.mark.anyio
async def test_close_http_client_detaches_before_aclose():
    closed = False

    class DummyClient:
        is_closed = False

        async def aclose(self):
            nonlocal closed
            assert services._http_client is None
            closed = True

    orig = services._http_client
    services._http_client = DummyClient()  # type: ignore
    try:
        await services.close_http_client()
        assert closed is True
        assert services._http_client is None
    finally:
        services._http_client = orig


@pytest.mark.anyio
async def test_no_speech_timeout_and_watch_task():
    session = VoiceSession(session_id="sess_no_speech")
    session.state = VoiceState.LISTENING
    session.current_turn = TurnState(
        turn_id="turn_ns",
        has_speech=False,
        started_at=time.monotonic() - (settings.turn_no_speech_timeout_seconds + 1.0),
    )
    assert session.is_silence_timeout() is True

    events = []

    async def mock_send(ev):
        events.append(ev)

    session.send_json_cb = mock_send

    orig_timeout = settings.turn_no_speech_timeout_seconds
    object.__setattr__(settings, "turn_no_speech_timeout_seconds", 0.05)
    try:
        session.current_turn = TurnState(turn_id="turn_ns_watch", has_speech=False)
        session.state = VoiceState.LISTENING
        await session._watch_no_speech_timeout("turn_ns_watch")
        assert session.state == VoiceState.IDLE
        assert session.current_turn is None
        assert any(e.get("type") == "playback.stop" and e.get("turn_id") == "turn_ns_watch" for e in events)
    finally:
        object.__setattr__(settings, "turn_no_speech_timeout_seconds", orig_timeout)


@pytest.mark.anyio
async def test_smart_turn_retries_within_same_pause():
    session = VoiceSession(session_id="sess_st_retry")
    session.state = VoiceState.LISTENING
    session.current_turn = TurnState(turn_id="turn_st_retry", has_speech=True)
    session.current_turn.buffer_audio(b"\x01\x02" * 2000)

    probs = iter([0.35, 0.85])
    calls = 0
    finished = []

    async def fake_predict(_audio):
        nonlocal calls
        calls += 1
        return next(probs)

    async def fake_finish(turn_id=None, **_kwargs):
        finished.append(turn_id)

    orig_predict = services.smart_turn_detector.predict
    orig_finish = session.finish_turn
    orig_sleep = asyncio.sleep

    async def fast_sleep(_sec):
        await orig_sleep(0)

    services.smart_turn_detector.predict = fake_predict  # type: ignore
    session.finish_turn = fake_finish  # type: ignore
    asyncio.sleep = fast_sleep  # type: ignore
    try:
        await session._run_smart_turn("turn_st_retry")
        assert calls == 2
        assert finished == ["turn_st_retry"]
    finally:
        services.smart_turn_detector.predict = orig_predict  # type: ignore
        session.finish_turn = orig_finish  # type: ignore
        asyncio.sleep = orig_sleep


@pytest.mark.anyio
async def test_tts_stream_distinguishes_empty_audio_and_truncated_header():
    class MockResp:
        status_code = 200

        def __init__(self, payloads):
            self._payloads = payloads

        async def aiter_bytes(self):
            for chunk in self._payloads:
                yield chunk

    class MockClient:
        is_closed = False

        def __init__(self, payloads):
            self._payloads = payloads

        @asynccontextmanager
        async def stream(self, *_args, **_kwargs):
            yield MockResp(self._payloads)

    orig_client_fn = services.get_http_client
    try:
        # 1. HTTP 200 nhưng 0 byte
        services.get_http_client = lambda: MockClient([])  # type: ignore
        with pytest.raises(services.TTSError) as exc_empty:
            async for _ in services.tts_stream("Xin chào"):
                pass
        assert exc_empty.value.code == "TTS_EMPTY_AUDIO"

        # 2. HTTP 200 nhưng WAV header bị cắt cụt (< 12 bytes)
        services.get_http_client = lambda: MockClient([b"RIFF\x24\x00"])  # type: ignore
        with pytest.raises(services.TTSError) as exc_hdr:
            async for _ in services.tts_stream("Xin chào"):
                pass
        assert exc_hdr.value.code == "TTS_INVALID_AUDIO"

        # 3. Chuỗi rỗng hợp lệ vẫn return êm không lỗi
        chunks = [c async for c in services.tts_stream("   ")]
        assert chunks == []
    finally:
        services.get_http_client = orig_client_fn


@pytest.mark.anyio
async def test_safe_send_handles_not_connected_and_oserror():
    for exc in (
        OSError("Socket write failed"),
        RuntimeError('WebSocket is not connected. Need to call "accept" first.'),
    ):
        session = VoiceSession(session_id="sess_send_extra")

        async def broken_send(err=exc):
            raise err

        await session._safe_send(broken_send)
        assert session.closed is True
