"""Audio module — STT, TTS, VAD, and virtual device helpers."""
from .stt_interface import STTClient, TranscriptionResult
from .tts_interface import TTSClient
from .vad import SileroVAD, VADResult, create_vad


def create_stt(config) -> STTClient:
    """Factory: create the primary STT client with automatic GPU detection.

    If ``faster-whisper`` is requested but CUDA is unavailable or torch is not
    installed, falls back to Google STT and logs a warning.

    Args:
        config: ``AudioConfig`` (or duck-typed equivalent) from engine/config.py.
    """
    primary = config.stt.primary

    has_cuda = False
    if primary == "faster-whisper":
        try:
            import torch
            has_cuda = (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).total_memory > 8e9
            )
        except (ImportError, RuntimeError):
            pass

    if primary == "faster-whisper" and has_cuda:
        from .stt_whisper import WhisperSTT
        c = config.stt.whisper
        return WhisperSTT(
            model_size=c.model,
            device=c.device,
            compute_type=c.compute_type,
            language=c.language,
        )

    if primary == "faster-whisper" and not has_cuda:
        from engine.logging_config import get_logger
        get_logger("audio").warning(
            "stt_gpu_unavailable",
            requested="faster-whisper",
            falling_back_to=config.stt.fallback,
        )

    from .stt_google import GoogleSTT
    c = config.stt.google
    return GoogleSTT(credentials_path=c.credentials_path, language_code=c.language_code)


def create_tts(config) -> TTSClient:
    """Factory: create the primary TTS client from config.

    Args:
        config: ``AudioConfig`` (or duck-typed equivalent) from engine/config.py.
    """
    primary = config.tts.primary
    if primary == "elevenlabs":
        from .tts_elevenlabs import ElevenLabsTTS
        c = config.tts.elevenlabs
        return ElevenLabsTTS(
            api_key=c.api_key,
            voice_id=c.voice_id,
            model_id=c.model_id,
            latency_optimization=c.latency_optimization,
        )
    elif primary == "coqui-xtts":
        from .tts_coqui import CoquiTTS
        c = config.tts.coqui
        return CoquiTTS(
            model_path=c.model_path,
            voice_sample=c.voice_sample,
            device=c.device,
        )
    else:
        raise ValueError(f"Unsupported TTS primary: {primary!r}")


__all__ = [
    "STTClient", "TranscriptionResult",
    "TTSClient",
    "SileroVAD", "VADResult", "create_vad",
    "create_stt", "create_tts",
]
