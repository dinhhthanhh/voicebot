"""Module cấu hình ứng dụng Voicebot MVP.

Đọc biến môi trường và cung cấp settings cho toàn bộ hệ thống.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent

for env_path in (_BACKEND_DIR / ".env", _ROOT_DIR / ".env"):
    if env_path.is_file():
        load_dotenv(env_path)
        break

_PROMPT_FILE = _BACKEND_DIR / "prompt.txt"
DEFAULT_SYSTEM_PROMPT = (
    _PROMPT_FILE.read_text(encoding="utf-8")
    if _PROMPT_FILE.is_file()
    else "Bạn là trợ lý giọng nói tiếng Việt của VNPost. Trả lời ngắn gọn dưới 3 câu."
)


@dataclass(frozen=True)
class Settings:
    """Cấu hình tập trung cho Voicebot MVP."""

    # Server Gateway
    app_host: str = field(default_factory=lambda: os.getenv("APP_HOST", "127.0.0.1"))
    app_port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    app_log_level: str = field(default_factory=lambda: os.getenv("APP_LOG_LEVEL", "INFO"))
    cors_allow_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()
        ]
    )
    max_sessions: int = field(default_factory=lambda: int(os.getenv("MAX_SESSIONS", "10")))

    # LLM Settings
    llm_chat_completions_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_CHAT_COMPLETIONS_URL",
            "https://ai.vnpost.vn/vllm-qwen38-27b/v1/chat/completions",
        )
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))
    llm_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("LLM_REASONING_EFFORT", "low")
    )
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    )
    llm_system_prompt: str = field(
        default_factory=lambda: os.getenv("LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    )

    # STT Settings
    stt_ws_url: str = field(
        default_factory=lambda: os.getenv(
            "STT_WS_URL", "wss://ai.vnpost.vn/qwen3-asr-1p7b/v1/stream"
        )
    )
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "qwen3-asr-1p7b-ft"))
    stt_language: str = field(default_factory=lambda: os.getenv("STT_LANGUAGE", "Vietnamese"))
    stt_sample_rate: int = field(default_factory=lambda: int(os.getenv("STT_SAMPLE_RATE", "16000")))
    stt_vad: str = field(default_factory=lambda: os.getenv("STT_VAD", "silero"))

    # TTS Settings
    tts_base_url: str = field(
        default_factory=lambda: os.getenv("TTS_BASE_URL", "https://ai.vnpost.vn/zero-tts")
    )
    tts_stream_path: str = field(
        default_factory=lambda: os.getenv("TTS_STREAM_PATH", "/tts/stream")
    )
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "maichi"))
    tts_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("TTS_TIMEOUT_SECONDS", "90"))
    )

    # Orchestrator & Session Settings
    voice_max_history_pairs: int = field(
        default_factory=lambda: int(os.getenv("VOICE_MAX_HISTORY_PAIRS", "8"))
    )

    # Thư mục frontend và models
    frontend_dir: Path = field(default_factory=lambda: _ROOT_DIR / "frontend")
    smart_turn_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("SMART_TURN_MODEL_PATH", str(_ROOT_DIR / "models" / "smart-turn-v3.2-cpu.onnx"))
        )
    )
    smart_turn_threshold: float = field(
        default_factory=lambda: float(os.getenv("SMART_TURN_THRESHOLD", "0.70"))
    )
    smart_turn_silence_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("SMART_TURN_SILENCE_TIMEOUT_SECONDS", "2.5"))
    )
    barge_in_rms_threshold: float = field(
        default_factory=lambda: float(os.getenv("BARGE_IN_RMS_THRESHOLD", "0.045"))
    )
    barge_in_debounce_frames: int = field(
        default_factory=lambda: int(os.getenv("BARGE_IN_DEBOUNCE_FRAMES", "4"))
    )
    barge_in_preroll_frames: int = field(
        default_factory=lambda: int(os.getenv("BARGE_IN_PREROLL_FRAMES", "8"))
    )

    def validate_config(self) -> list[str]:
        """Kiểm tra tính hợp lệ của cấu hình, trả về danh sách cảnh báo."""
        warnings: list[str] = []
        required_urls = [
            ("STT_WS_URL", self.stt_ws_url),
            ("LLM_CHAT_COMPLETIONS_URL", self.llm_chat_completions_url),
            ("TTS_BASE_URL", self.tts_base_url),
        ]
        for name, val in required_urls:
            if not val:
                warnings.append(f"{name} chưa được thiết lập.")

        rules: list[tuple[bool, str]] = [
            (not (0 < self.smart_turn_threshold <= 1.0), f"SMART_TURN_THRESHOLD ({self.smart_turn_threshold}) nên nằm trong khoảng (0, 1.0]."),
            (self.smart_turn_silence_timeout_seconds <= 0, f"SMART_TURN_SILENCE_TIMEOUT_SECONDS ({self.smart_turn_silence_timeout_seconds}) phải lớn hơn 0."),
            (self.barge_in_rms_threshold <= 0, f"BARGE_IN_RMS_THRESHOLD ({self.barge_in_rms_threshold}) phải lớn hơn 0."),
            (self.barge_in_debounce_frames < 1, f"BARGE_IN_DEBOUNCE_FRAMES ({self.barge_in_debounce_frames}) phải >= 1."),
            (self.max_sessions < 1, f"MAX_SESSIONS ({self.max_sessions}) phải >= 1."),
            (self.llm_timeout_seconds <= 0, f"LLM_TIMEOUT_SECONDS ({self.llm_timeout_seconds}) phải lớn hơn 0."),
            (self.tts_timeout_seconds <= 0, f"TTS_TIMEOUT_SECONDS ({self.tts_timeout_seconds}) phải lớn hơn 0."),
        ]
        for condition, msg in rules:
            if condition:
                warnings.append(msg)
        return warnings


settings = Settings()
