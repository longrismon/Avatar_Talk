"""
Pydantic v2 configuration system for Avatar Talk engine.

Loads engine/config.yaml, performs ${ENV_VAR} interpolation, and validates
all settings into typed models.
"""

from __future__ import annotations

import os
import re
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Env-var interpolation
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def _interpolate_env_vars(value):
    """Recursively replace ${VAR} references with env var values."""
    if isinstance(value, str):
        def replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")
        return _ENV_VAR_PATTERN.sub(replace, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# STT models
# ---------------------------------------------------------------------------

class WhisperConfig(BaseModel):
    model: str = "large-v3"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    compute_type: Literal["float16", "int8", "int8_float16"] = "float16"
    language: str = "en"


class GoogleSTTConfig(BaseModel):
    credentials_path: str = "./credentials/google-stt.json"
    language_code: str = "en-US"


class STTConfig(BaseModel):
    primary: Literal["faster-whisper", "google-stt"] = "faster-whisper"
    fallback: Literal["faster-whisper", "google-stt"] = "google-stt"
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    google: GoogleSTTConfig = Field(default_factory=GoogleSTTConfig)


# ---------------------------------------------------------------------------
# TTS models
# ---------------------------------------------------------------------------

class ElevenLabsConfig(BaseModel):
    api_key: str = ""
    voice_id: str = ""
    model_id: str = "eleven_turbo_v2_5"
    latency_optimization: int = Field(default=3, ge=1, le=4)


class CoquiConfig(BaseModel):
    model_path: str = "./models/coqui-xtts"
    voice_sample: str = "./voice/my-voice-sample.wav"
    device: Literal["cuda", "cpu"] = "cuda"


class TTSConfig(BaseModel):
    primary: Literal["elevenlabs", "coqui-xtts"] = "elevenlabs"
    fallback: Literal["elevenlabs", "coqui-xtts"] = "coqui-xtts"
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)
    coqui: CoquiConfig = Field(default_factory=CoquiConfig)


# ---------------------------------------------------------------------------
# VAD model
# ---------------------------------------------------------------------------

class VADConfig(BaseModel):
    engine: Literal["silero", "webrtc"] = "silero"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_silence_duration_ms: int = Field(default=1500, ge=100)
    speech_pad_ms: int = Field(default=300, ge=0)


# ---------------------------------------------------------------------------
# Audio aggregate
# ---------------------------------------------------------------------------

class AudioConfig(BaseModel):
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)


# ---------------------------------------------------------------------------
# LLM models
# ---------------------------------------------------------------------------

class AnthropicConfig(BaseModel):
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = Field(default=1500, ge=100, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=8.0, ge=1.0)


class OpenAIConfig(BaseModel):
    api_key: str = ""
    model: str = "gpt-4o"
    max_tokens: int = Field(default=1500, ge=100, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class OllamaConfig(BaseModel):
    model: str = "llama3:8b-instruct"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=15.0, ge=1.0)


class CustomLLMConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = Field(default=1500, ge=100, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=15.0, ge=1.0)


class LLMConfig(BaseModel):
    primary: Literal["anthropic", "openai", "ollama", "custom"] = "anthropic"
    fallback: Literal["anthropic", "openai", "ollama", "custom"] = "ollama"
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    custom: CustomLLMConfig = Field(default_factory=CustomLLMConfig)


# ---------------------------------------------------------------------------
# LipSync models
# ---------------------------------------------------------------------------

class Wav2LipConfig(BaseModel):
    model_path: str = "./models/wav2lip/wav2lip_gan.pth"
    face_detect_path: str = "./models/wav2lip/s3fd.pth"


class SadTalkerConfig(BaseModel):
    model_path: str = "./models/sadtalker/"


class LipSyncConfig(BaseModel):
    enabled: bool = False
    engine: Literal["wav2lip", "sadtalker"] = "wav2lip"
    reference_image: str = "./face/reference.png"
    fps: int = Field(default=25, ge=1, le=60)
    device: Literal["cuda", "cpu"] = "cuda"
    wav2lip: Wav2LipConfig = Field(default_factory=Wav2LipConfig)
    sadtalker: SadTalkerConfig = Field(default_factory=SadTalkerConfig)


# ---------------------------------------------------------------------------
# Browser models
# ---------------------------------------------------------------------------

class AppURLsConfig(BaseModel):
    teams: str = "https://teams.microsoft.com"
    slack: str = "https://app.slack.com"
    discord: str = "https://discord.com/app"


class BrowserConfig(BaseModel):
    engine: Literal["playwright"] = "playwright"
    browser_type: Literal["chromium", "firefox", "webkit"] = "chromium"
    headless: bool = False
    user_data_dir: str = "./browser_profile"
    default_step_timeout_ms: int = Field(default=15000, ge=1000)
    max_retries_per_step: int = Field(default=1, ge=0, le=5)
    default_app: str = "teams"
    app_urls: AppURLsConfig = Field(default_factory=AppURLsConfig)


# ---------------------------------------------------------------------------
# Virtual device models
# ---------------------------------------------------------------------------

class LinuxAudioConfig(BaseModel):
    sink_name: str = "avatar_agent_mic"
    module: str = "module-null-sink"


class MacAudioConfig(BaseModel):
    driver: str = "blackhole"


class WindowsAudioConfig(BaseModel):
    driver: str = "vb-cable"


class VirtualAudioConfig(BaseModel):
    linux: LinuxAudioConfig = Field(default_factory=LinuxAudioConfig)
    macos: MacAudioConfig = Field(default_factory=MacAudioConfig)
    windows: WindowsAudioConfig = Field(default_factory=WindowsAudioConfig)


class LinuxCameraConfig(BaseModel):
    device: str = "/dev/video10"
    module: str = "v4l2loopback"


class MacCameraConfig(BaseModel):
    driver: str = "obs"


class WindowsCameraConfig(BaseModel):
    driver: str = "obs"


class VirtualCameraConfig(BaseModel):
    linux: LinuxCameraConfig = Field(default_factory=LinuxCameraConfig)
    macos: MacCameraConfig = Field(default_factory=MacCameraConfig)
    windows: WindowsCameraConfig = Field(default_factory=WindowsCameraConfig)


class VirtualDevicesConfig(BaseModel):
    audio: VirtualAudioConfig = Field(default_factory=VirtualAudioConfig)
    camera: VirtualCameraConfig = Field(default_factory=VirtualCameraConfig)


# ---------------------------------------------------------------------------
# Notifications models
# ---------------------------------------------------------------------------

class FirebaseConfig(BaseModel):
    credentials_path: str = "./credentials/firebase-admin.json"


class NotificationsConfig(BaseModel):
    push_enabled: bool = True
    firebase: FirebaseConfig = Field(default_factory=FirebaseConfig)
    events: list[str] = Field(default_factory=lambda: [
        "review_started", "intervention_needed", "call_connected", "call_ended"
    ])


# ---------------------------------------------------------------------------
# Logging model
# ---------------------------------------------------------------------------

class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    file: Optional[str] = "./logs/agent.log"
    max_size_mb: int = Field(default=50, ge=1)
    backup_count: int = Field(default=5, ge=0)


# ---------------------------------------------------------------------------
# Top-level section models
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    port: int = Field(default=9600, ge=1024, le=65535)
    host: str = "0.0.0.0"
    auth_token: str = "change-me-in-production"
    cors_origins: list[str] = Field(default_factory=list)
    mdns_enabled: bool = True
    mdns_name: str = "avatar-agent-desktop"


class PrincipalConfig(BaseModel):
    profile_path: str = "./profiles/default.json"


class ReviewConfig(BaseModel):
    timeout_seconds: float = Field(default=5.0, ge=2.0, le=15.0)
    auto_select_recommended: bool = True
    keyboard_shortcuts_enabled: bool = True


# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    principal: PrincipalConfig = Field(default_factory=PrincipalConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    lipsync: LipSyncConfig = Field(default_factory=LipSyncConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    virtual_devices: VirtualDevicesConfig = Field(default_factory=VirtualDevicesConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Load / singleton
# ---------------------------------------------------------------------------

_config: Settings | None = None


def load_config(path: str = "config.yaml") -> Settings:
    """Read YAML file, interpolate env vars, validate into Settings."""
    global _config
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _interpolate_env_vars(raw)
    _config = Settings(**raw)
    return _config


def get_config() -> Settings:
    """Return the cached Settings instance; raises if not yet loaded."""
    global _config
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config
