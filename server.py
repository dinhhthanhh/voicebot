from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from stt import (
    MAX_AUDIO_BYTES,
    STT_SAMPLE_RATE,
    SileroVadModel,
    SmartTurnDetector,
    SttError,
    SttSession,
    VadStreamState,
    get_detector,
    get_vad_model,
    is_hesitating,
    is_only_filler,
    is_reading_sequence,
    normalize_speech_input,
)
from tts import TTS_SAMPLE_RATE, run_turn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Cấu hình dịch vụ kết nối ngoài (STT, LLM, TTS)
# ---------------------------------------------------------------------------
STT_URL = os.environ.get("STT_WS_URL", "wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream")
LLM_URL = os.environ.get("LLM_URL", "https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions")
TTS_URL = os.environ.get("TTS_BASE_URL", "https://ai.vnpost.vn/zero-tts")
VOICE = os.environ.get("TTS_VOICE", "maichi")

INDEX_HTML = Path(__file__).resolve().parent / "index.html"
MAX_HISTORY = 8


class SessionState(StrEnum):
    """Trạng thái phiên thoại giữa bot và client."""

    IDLE = "idle"
    STARTING = "starting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


# ---------------------------------------------------------------------------
# Cấu hình âm thanh & Ngưỡng VAD / Smart Turn
# ---------------------------------------------------------------------------
FRAME_DURATION_MS = 100
# Mỗi frame 100ms = 1600 mẫu * 2 bytes = 3200 bytes
FRAME_BYTES = int(STT_SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0)) * 2
MIN_AUDIO_BYTES = 12_000  # Tối thiểu ~375ms âm thanh trước khi chạy Smart Turn
MAX_PENDING_FRAMES = 150  # Tối đa 15 giây âm thanh đệm chờ khi STT kết nối hoặc thử lại (retry)

# Ngưỡng VAD (Hysteresis: START để bắt đầu, KEEP để duy trì không đứt đoạn)
VAD_PROB_START = 0.60
VAD_PROB_KEEP = 0.40
VAD_RMS_START = 0.0025
VAD_RMS_KEEP = 0.0015

# Ngưỡng xác suất Smart Turn
PROB_VERY_HIGH = 0.85  # Rất tự tin dứt câu (ngắt siêu nhanh ~400ms)
PROB_HIGH = 0.70  # Tự tin dứt câu (~600ms)
PROB_MID = 0.55  # Tự tin trung bình (~1.2s)
WALL_TIMEOUT_S = 2.5  # Hết thời gian chờ im lặng theo đồng hồ thực tế (câu thường)
WALL_EXTENDED_TIMEOUT_S = 3.5  # Hết thời gian chờ tối đa khi ngập ngừng hoặc đọc số


def ms_to_frames(ms: int) -> int:
    """Quy đổi thời gian milliseconds sang số frame (mỗi frame 100ms)."""
    return max(1, (ms + FRAME_DURATION_MS - 1) // FRAME_DURATION_MS)


# Các mốc frame kiểm tra (mỗi frame 100ms)
MIN_SPEECH_FRAMES = ms_to_frames(300)
MIN_SILENCE_FRAMES = ms_to_frames(300)
CHECK_INTERVAL = ms_to_frames(100)  # Kiểm tra mỗi 100ms để bắt tức thì khoảnh khắc dứt câu
FRAMES_ULTRA_FAST = ms_to_frames(500)  # 500ms khi prob >= 0.80
FRAMES_FAST = ms_to_frames(700)  # 700ms khi prob >= 0.65
FRAMES_SLOW = ms_to_frames(1200)  # 1.2s khi prob >= 0.50
FRAMES_SEQUENCE = ms_to_frames(1500)  # 1.5s khi đang đọc số
FILLER_TIMEOUT = ms_to_frames(2000)
SILENCE_TIMEOUT = ms_to_frames(2000)
TIMEOUT_EXTENDED = ms_to_frames(3500)


def decide_turn_end(
    prob: float,
    silence_frames: int,
    wall_silence_s: float,
    *,
    reading: bool,
    hesitating: bool,
) -> str | None:
    """Hàm thuần túy quyết định thời điểm dứt câu dựa trên xác suất và thời gian im lặng.

    Trả về lý do dừng ('smart_turn_done', 'smart_turn_mid', 'sequence_done', 'silence_timeout') hoặc None.
    """
    is_pending = reading or hesitating
    max_frames = TIMEOUT_EXTENDED if is_pending else SILENCE_TIMEOUT
    max_wall_s = WALL_EXTENDED_TIMEOUT_S if is_pending else WALL_TIMEOUT_S

    # 1. Timeout luôn được ưu tiên, không phụ thuộc Smart Turn, chống treo vĩnh viễn
    if silence_frames >= max_frames or wall_silence_s >= max_wall_s:
        return "silence_timeout"

    # 2. Đang đọc số, mã vận đơn hoặc địa chỉ (cần khoảng lặng tự nhiên dài hơn)
    if reading:
        if prob >= PROB_VERY_HIGH and silence_frames >= ms_to_frames(1000):
            return "sequence_done"
        if prob >= PROB_HIGH and silence_frames >= FRAMES_SEQUENCE:
            return "sequence_done"
        return None

    # 3. Đang ngập ngừng hoặc nói câu chưa hoàn chỉnh
    if hesitating:
        if prob >= PROB_HIGH and silence_frames >= FRAMES_SLOW:
            return "smart_turn_done"
        return None

    # 4. Câu nói bình thường: phân tầng theo độ tự tin
    if prob >= PROB_VERY_HIGH and silence_frames >= FRAMES_ULTRA_FAST:
        return "smart_turn_done"
    if prob >= PROB_HIGH and silence_frames >= FRAMES_FAST:
        return "smart_turn_done"
    if prob >= PROB_MID and silence_frames >= FRAMES_SLOW:
        return "smart_turn_mid"

    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo tài nguyên toàn cục (HTTP client pool, model Silero VAD và Smart-Turn)."""
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=60.0,
            write=30.0,
            pool=10.0,
        ),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )
    app.state.detector = await asyncio.to_thread(get_detector)
    app.state.vad_model = await asyncio.to_thread(get_vad_model)
    yield
    await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


class VoiceSession:
    """Quản lý trạng thái và luồng dữ liệu hai chiều của một phiên thoại WebSocket."""

    def __init__(
        self,
        ws: WebSocket,
        http: httpx.AsyncClient,
        detector: SmartTurnDetector | None = None,
        vad_model: SileroVadModel | None = None,
    ) -> None:
        self.ws = ws
        self.http = http
        self.state = SessionState.IDLE
        self.auto_turn = True
        self.history: list[dict[str, str]] = []
        self.stt: SttSession | None = None
        self.turn_task: asyncio.Task | None = None
        self._start_task: asyncio.Task | None = None
        self.metrics: dict[str, float] = {}
        self._send_lock = asyncio.Lock()
        self._vad_lock = asyncio.Lock()

        self._pending_audio: deque[bytes] = deque(maxlen=MAX_PENDING_FRAMES)
        self._is_processing_audio = False

        self.detector = detector or get_detector()
        self.vad = VadStreamState(model=vad_model)
        self.turn_id = 0
        self._detect_task: asyncio.Task | None = None
        self.audio_buf = bytearray()
        self._reset_turn_audio(increment_turn=False)

    def _reset_turn_audio(self, *, increment_turn: bool = True) -> None:
        """Đặt lại bộ đệm âm thanh và trạng thái đếm frame khi kết thúc hoặc hủy lượt thoại."""
        if increment_turn:
            self.turn_id += 1
        current = None
        with suppress(RuntimeError):
            current = asyncio.current_task()
        if self._detect_task and self._detect_task is not current and not self._detect_task.done():
            self._detect_task.cancel()
        self._detect_task = None
        self.vad.reset()
        self.audio_buf.clear()
        self._pending_audio.clear()
        self._is_processing_audio = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.has_speech = False
        self._detecting = False
        self._last_detect_frame = 0
        self.last_partial_text = ""
        self.last_speech_time = time.monotonic()

    async def _cancel_tasks(self) -> None:
        """Hủy và đợi hoàn tất các tác vụ nền đang chạy (_start_task, turn_task, _detect_task)."""
        current = None
        with suppress(RuntimeError):
            current = asyncio.current_task()
        tasks = [
            t
            for t in (self._start_task, self.turn_task, self._detect_task)
            if t and t is not current and not t.done()
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._start_task = None
        self.turn_task = None
        self._detect_task = None

    async def _handle_error(self, message: str) -> None:
        """Dọn dẹp tài nguyên phiên và thông báo lỗi về client."""
        await self.close()
        await self.set_state(SessionState.IDLE)
        await self._send_ws({"type": "error", "message": message})

    def _is_turn_active(self, turn_id: int | None = None) -> bool:
        """Kiểm tra lượt thoại hiện tại còn hợp lệ để tiếp tục xử lý hay không."""
        match_turn = turn_id is None or turn_id == self.turn_id
        return match_turn and self.state == SessionState.LISTENING and self.stt is not None

    async def set_state(
        self, state: SessionState | str, ui_state: SessionState | str | None = None
    ) -> None:
        """Cập nhật trạng thái máy trạng thái nội bộ và gửi gói tin đồng bộ về client UI."""
        self.state = SessionState(state)
        await self._send_ws({"type": "state", "state": ui_state if ui_state is not None else state})

    async def close(self) -> None:
        """Dọn dẹp và giải phóng toàn bộ tác vụ nền khi đóng kết nối WebSocket."""
        async with self._vad_lock:
            self._reset_turn_audio()
        await self._cancel_tasks()
        await self._abort_stt()

    async def cancel_turn(self) -> None:
        """Hủy bỏ lượt thoại đang chạy (ngắt bot khi người dùng chen ngang)."""
        await self.close()
        await self._send_ws({"type": "stop_audio"})
        await self.set_state(SessionState.IDLE)

    async def on_audio(self, data: bytes) -> None:
        """Tiếp nhận dữ liệu PCM thô từ trình duyệt và chuyển tiếp đến STT/VAD."""
        if not data:
            return

        if len(data) > FRAME_BYTES * 2:
            await self._send_ws(
                {
                    "type": "error",
                    "message": "Khung âm thanh không hợp lệ (vượt quá kích thước tối đa).",
                }
            )
            return

        frame = data[: len(data) // 2 * 2]
        if not frame:
            return

        if self.state == SessionState.STARTING:
            if len(self._pending_audio) >= MAX_PENDING_FRAMES:
                logger.warning(
                    "[Audio Queue] Hàng đợi pending_audio đầy (%d frames), frame cũ nhất bị loại bỏ.",
                    MAX_PENDING_FRAMES,
                )
            self._pending_audio.append(frame)
            return

        if not self._is_turn_active() or self.stt is None:
            return

        self._pending_audio.append(frame)
        await self._drain_pending_audio()

    async def _drain_pending_audio(self) -> None:
        """Xả tuần tự các khung âm thanh trong _pending_audio tới STT/VAD theo thứ tự FIFO."""
        if self._is_processing_audio:
            return
        self._is_processing_audio = True
        try:
            while self._pending_audio and self._is_turn_active():
                frame = self._pending_audio.popleft()
                if not await self._process_frame(frame):
                    break
        finally:
            self._is_processing_audio = False

    async def _process_frame(self, frame: bytes) -> bool:
        """Chuyển tiếp khung âm thanh tới STT và cập nhật VAD."""
        if not self._is_turn_active() or self.stt is None:
            return False
        try:
            await self.stt.send_frame(frame)
        except Exception:
            await self._handle_error("Mất kết nối dịch vụ nhận dạng giọng nói.")
            return False

        async with self._vad_lock:
            await self._process_vad(frame)
        return True

    async def _is_speaking(self, frame: bytes, current_turn: int) -> bool | None:
        """Đo năng lượng RMS và tính xác suất giọng người qua Silero VAD."""
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2)))

        if not self.vad.model.is_available:
            return rms >= VAD_RMS_START

        speech_prob = await asyncio.to_thread(self.vad.speech_prob, samples)
        if not self._is_turn_active(current_turn):
            return None

        # Cơ chế hysteresis: khi đã có tiếng nói thì hạ thấp ngưỡng để không bị đứt đoạn âm tiết nhẹ
        prob_threshold = VAD_PROB_KEEP if self.has_speech else VAD_PROB_START
        rms_threshold = VAD_RMS_KEEP if self.has_speech else VAD_RMS_START

        # Khi Silero VAD tự tin là tiếng người, ưu tiên nhận diện kể cả khi nói thì thầm/nói nhỏ
        if speech_prob >= 0.70:
            return rms >= VAD_RMS_KEEP

        return speech_prob >= prob_threshold and rms >= rms_threshold

    def _update_vad(self, is_speaking: bool, now: float) -> None:
        """Cập nhật bộ đếm frame tiếng nói / im lặng dựa trên kết quả phát hiện."""
        if is_speaking:
            self.has_speech = True
            self.speech_frames += 1
            self.silence_frames = 0
            self.last_speech_time = now
            self._last_detect_frame = 0
        elif self.has_speech:
            self.silence_frames += 1

    def _schedule_detect(self) -> None:
        """Kiểm tra điều kiện kích hoạt tác vụ Smart-Turn ngắt câu tự động."""
        silence_passed = self.silence_frames - self._last_detect_frame
        can_trigger = (
            self.has_speech
            and self.speech_frames >= MIN_SPEECH_FRAMES
            and self.silence_frames >= MIN_SILENCE_FRAMES
            and silence_passed >= CHECK_INTERVAL
            and not self._detecting
        )
        if can_trigger:
            self._detecting = True
            self._last_detect_frame = self.silence_frames
            self._detect_task = asyncio.create_task(self._check_turn_end(self.turn_id))

    async def _process_vad(self, frame: bytes) -> None:
        """Xử lý từng khung âm thanh qua bộ phân loại VAD và bộ đệm trượt."""
        if not self.auto_turn:
            return

        self.audio_buf.extend(frame)
        if len(self.audio_buf) > MAX_AUDIO_BYTES:
            del self.audio_buf[:-MAX_AUDIO_BYTES]

        is_speaking = await self._is_speaking(frame, self.turn_id)
        if is_speaking is None:
            return

        now = time.monotonic()
        self._update_vad(is_speaking, now)
        self._schedule_detect()

    async def _on_filler_timeout(self, turn_id: int) -> bool:
        """Xử lý trường hợp người dùng chỉ nói từ đệm (ừ, à, ờ...) rồi im lặng quá 2 giây."""
        if self.silence_frames >= FILLER_TIMEOUT and self._is_turn_active(turn_id):
            await self._send_ws(
                {
                    "type": "auto_stop",
                    "probability": 0.0,
                    "reason": "filler_timeout",
                }
            )
            await self.stop_turn()
            return True
        return False

    async def _predict_turn_prob(self) -> float:
        """Chạy dự đoán mô hình Smart-Turn trên luồng worker riêng."""
        if not self.detector.is_available:
            return 0.0
        audio_f32 = np.frombuffer(self.audio_buf, dtype=np.int16).astype(np.float32) / 32768.0
        res = await asyncio.to_thread(self.detector.predict, audio_f32)
        return float(res.get("probability", 0.0))

    async def _check_turn_end(self, turn_id: int) -> None:
        """Định kỳ đánh giá xem người dùng đã hoàn thành câu nói hay chưa để chuyển lượt bot."""
        try:
            can_check = (
                self._is_turn_active(turn_id)
                and len(self.audio_buf) >= MIN_AUDIO_BYTES
                and self.speech_frames >= MIN_SPEECH_FRAMES
            )
            if not can_check:
                return

            is_only_filler_text = (
                is_only_filler(self.last_partial_text) if self.last_partial_text else False
            )
            if is_only_filler_text:
                await self._on_filler_timeout(turn_id)
                return

            prob = await self._predict_turn_prob()
            if not self._is_turn_active(turn_id):
                return

            wall_silence_s = time.monotonic() - self.last_speech_time
            raw_partial = self.last_partial_text
            reading = is_reading_sequence(raw_partial) if raw_partial else False
            hesitating = is_hesitating(raw_partial) if raw_partial else False

            reason = decide_turn_end(
                prob,
                self.silence_frames,
                wall_silence_s,
                reading=reading,
                hesitating=hesitating,
            )

            if reason and self._is_turn_active(turn_id):
                await self._send_ws(
                    {
                        "type": "auto_stop",
                        "probability": round(prob, 3),
                        "reason": reason,
                    }
                )
                await self.stop_turn()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("[Smart Turn Error] %s", exc)
        finally:
            if turn_id == self.turn_id:
                self._detecting = False

    async def start_turn(self) -> None:
        """Bắt đầu một lượt nhận dạng giọng nói mới từ người dùng."""
        if self.state in (SessionState.THINKING, SessionState.SPEAKING):
            await self._send_ws({"type": "busy"})
            return

        await self._abort_stt()
        async with self._vad_lock:
            self._reset_turn_audio()
        await self.set_state(SessionState.STARTING, ui_state=SessionState.LISTENING)

        stt = SttSession(STT_URL, self._on_stt_event)
        self.stt = stt
        try:
            await stt.open()
        except SttError as exc:
            self.stt = None
            await self._handle_error(exc.message)
            return
        except Exception:
            self.stt = None
            await self._handle_error("Không kết nối được dịch vụ nhận dạng giọng nói.")
            return
        except BaseException:
            self.stt = None
            with suppress(Exception):
                await stt.abort()
            raise

        self.state = SessionState.LISTENING

        await self._drain_pending_audio()

    async def stop_turn(self) -> None:
        """Dừng nhận dạng và chuyển trạng thái sang suy nghĩ / tổng hợp câu trả lời."""
        if self._start_task and not self._start_task.done():
            await asyncio.gather(self._start_task, return_exceptions=True)
        if not self._is_turn_active() or self.stt is None:
            return

        stt = self.stt
        self.stt = None
        fallback_text = self.last_partial_text.strip()
        async with self._vad_lock:
            self._reset_turn_audio()
        await self.set_state(SessionState.THINKING)
        self.turn_task = asyncio.create_task(self._execute_turn(stt, fallback_text))

    async def _finalize_stt(self, stt: SttSession, fallback_text: str) -> str:
        """Đợi kết quả STT cuối cùng và chuẩn hóa văn bản trước khi đưa vào LLM."""
        try:
            raw_text = (await stt.finish()).strip()
            if not raw_text and fallback_text:
                raw_text = fallback_text
            return normalize_speech_input(raw_text)
        except (SttError, Exception) as exc:
            if fallback_text:
                return normalize_speech_input(fallback_text)
            err_msg = (
                exc.message
                if isinstance(exc, SttError)
                else "Nhận dạng giọng nói thất bại, hãy thử lại."
            )
            await self._send_ws({"type": "error", "message": err_msg})
            return ""

    async def _run_dialogue(self, text: str) -> None:
        """Chạy luồng sinh câu trả lời LLM và đồng thời tổng hợp giọng nói phát về client."""
        reply = await run_turn(
            text,
            history=self.history,
            llm_url=LLM_URL,
            tts_url=TTS_URL,
            voice=VOICE,
            client=self.http,
            send_raw=self._send_raw,
            send_audio=self._send_audio,
            metrics=self.metrics,
        )
        self.history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply},
        ]
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        await self._send_ws(
            {
                "type": "metrics",
                **{k: round(v, 2) for k, v in self.metrics.items()},
            }
        )

    async def _execute_turn(self, stt: SttSession, fallback_text: str = "") -> None:
        """Điều phối một lượt thoại hoàn chỉnh: kết thúc STT -> xử lý LLM -> phát TTS."""
        t_stop = time.perf_counter()
        try:
            text = await self._finalize_stt(stt, fallback_text)
            await self._send_ws(
                {
                    "type": "final",
                    "text": text,
                    "stt_final_s": round(time.perf_counter() - t_stop, 2),
                }
            )
            if not text or is_only_filler(text):
                msg = (
                    "Tôi đang lắng nghe, bạn hãy nói rõ câu hỏi hoặc yêu cầu nhé."
                    if text
                    else "Không nghe rõ, hãy nói lại."
                )
                await self._send_ws({"type": "info", "message": msg})
                await self._send_ws({"type": "bot_done"})
                return

            await self._run_dialogue(text)
            await self._send_ws({"type": "bot_done"})
        except asyncio.CancelledError:
            await stt.abort()
            raise
        except Exception as exc:
            await self._send_ws(
                {"type": "error", "message": f"Lỗi khi tạo câu trả lời: {type(exc).__name__}"}
            )
            await self._send_ws({"type": "bot_done"})
        finally:
            if self.turn_task is asyncio.current_task():
                self.turn_task = None
            await self.set_state(SessionState.IDLE)

    async def _send_ws(self, data: dict[str, Any] | bytes) -> None:
        """Gửi dữ liệu (JSON hoặc PCM audio) an toàn qua WebSocket có khóa phân luồng."""
        async with self._send_lock:
            with suppress(WebSocketDisconnect, RuntimeError):
                if isinstance(data, bytes):
                    await self.ws.send_bytes(data)
                else:
                    await self.ws.send_json(data)

    async def on_control(self, msg: dict[str, Any]) -> None:
        """Xử lý các thông điệp điều khiển JSON từ client."""
        msg_type = msg.get("type")
        if msg_type == "config":
            if "auto_turn" in msg:
                self.auto_turn = bool(msg["auto_turn"])
        elif msg_type == "start_turn":
            if self._start_task and not self._start_task.done():
                return
            self._start_task = asyncio.create_task(self.start_turn())
        elif msg_type == "stop_turn":
            await self.stop_turn()
        elif msg_type == "interrupt":
            await self.cancel_turn()
        elif msg_type == "reset":
            await self.cancel_turn()
            self.history.clear()

    async def _send_raw(self, text: str) -> None:
        await self._send_ws({"type": "bot_raw", "text": text})

    async def _send_audio(self, pcm: bytes) -> None:
        """Gửi âm thanh TTS PCM16 về client kèm gói tin định dạng đầu tiên."""
        if self.state == SessionState.THINKING:
            await self._send_ws(
                {"type": "tts_format", "sample_rate": TTS_SAMPLE_RATE, "channels": 1}
            )
            await self.set_state(SessionState.SPEAKING)
        await self._send_ws(pcm)

    async def _on_stt_event(self, kind: str, payload: Any) -> None:
        """Callback nhận kết quả tạm thời (partial) hoặc lỗi từ STT session."""
        if kind == "partial":
            if self.state != SessionState.LISTENING:
                return
            new_text = str(payload or "").strip()
            if new_text:
                self.last_partial_text = new_text
                self.has_speech = True
                self.speech_frames = max(self.speech_frames, MIN_SPEECH_FRAMES)
            await self._send_ws({"type": "partial", "text": payload})
        elif kind == "error" and self.state in (SessionState.STARTING, SessionState.LISTENING):
            # Nếu đã chuyển sang THINKING (đang chạy _finalize_stt), để stt.finish() xử lý cùng fallback_text
            err_msg = (
                payload.message
                if isinstance(payload, SttError)
                else SttError(payload).message
            )
            await self._handle_error(err_msg)

    async def _abort_stt(self) -> None:
        """Ngắt kết nối phiên STT ngay lập tức nếu đang hoạt động."""
        if self.stt:
            stt, self.stt = self.stt, None
            await stt.abort()


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket) -> None:
    """Endpoint WebSocket chính kết nối với trình duyệt người dùng."""
    await ws.accept()
    session = VoiceSession(
        ws,
        app.state.http,
        detector=getattr(app.state, "detector", None),
        vad_model=getattr(app.state, "vad_model", None),
    )
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await session.on_audio(msg["bytes"])
            elif msg.get("text") is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError as exc:
                    logger.warning("[WebSocket] Lỗi JSON: %s", exc)
                else:
                    if isinstance(payload, dict):
                        await session.on_control(payload)
                    else:
                        logger.warning(
                            "[WebSocket] Gói tin không phải JSON object: %s",
                            type(payload).__name__,
                        )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("[WebSocket Error] %s", exc)
    finally:
        await session.close()
