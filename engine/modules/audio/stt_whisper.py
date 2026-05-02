"""faster-whisper STT with sliding-window partial transcription."""
import asyncio
from typing import Optional

from .stt_interface import STTClient, TranscriptionResult
from engine.logging_config import get_logger

log = get_logger("stt.whisper")

_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # int16

# 500 ms sliding window, 100 ms step for quasi-streaming partials (ADR-002 §4)
_WINDOW_BYTES = int(0.5 * _SAMPLE_RATE) * _BYTES_PER_SAMPLE   # 16 000 bytes
_STEP_BYTES = int(0.1 * _SAMPLE_RATE) * _BYTES_PER_SAMPLE     # 3 200 bytes

# Re-export for tests
WINDOW_BYTES = _WINDOW_BYTES


class WhisperSTT(STTClient):
    """faster-whisper STT client.

    - ``transcribe_chunk``: maintains a 500 ms rolling window; runs Whisper on
      each new 100 ms of audio for a quasi-streaming partial transcript.
    - ``transcribe_utterance``: runs a fresh full-utterance transcription on the
      complete buffered audio for maximum accuracy; resets the partial buffer.

    Auto-detects CUDA at load time; falls back to CPU/int8 if unavailable.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None
        self._chunk_buffer = bytearray()

    async def load(self) -> None:
        """Load the Whisper model (blocks in a thread executor)."""
        if self._model is not None:
            return
        log.info("whisper_loading", model=self._model_size, device=self._device)
        self._model = await asyncio.get_running_loop().run_in_executor(
            None, self._load_sync
        )
        log.info("whisper_loaded", model=self._model_size)

    def _load_sync(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is required. Install with: pip install faster-whisper"
            ) from exc

        device = self._device
        compute_type = self._compute_type
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                log.warning("whisper_cuda_unavailable", falling_back_to="cpu")
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            device = "cpu"
            compute_type = "int8"

        return WhisperModel(self._model_size, device=device, compute_type=compute_type)

    async def transcribe_chunk(self, audio_bytes: bytes) -> TranscriptionResult:
        """Add bytes to the rolling buffer and transcribe the current 500 ms window."""
        if self._model is None:
            raise RuntimeError("WhisperSTT.load() must be awaited before transcribing")

        self._chunk_buffer.extend(audio_bytes)
        # Keep only the latest window worth of audio
        if len(self._chunk_buffer) > _WINDOW_BYTES:
            del self._chunk_buffer[: len(self._chunk_buffer) - _WINDOW_BYTES]

        if len(self._chunk_buffer) < _STEP_BYTES:
            return TranscriptionResult(text="", is_final=False)

        text = await self._run_transcribe(bytes(self._chunk_buffer))
        return TranscriptionResult(text=text, is_final=False, language=self._language)

    async def transcribe_utterance(self, audio_bytes: bytes) -> TranscriptionResult:
        """Transcribe a complete utterance; resets the partial buffer."""
        if self._model is None:
            raise RuntimeError("WhisperSTT.load() must be awaited before transcribing")

        if not audio_bytes:
            return TranscriptionResult(text="", is_final=True)

        text = await self._run_transcribe(audio_bytes)
        self._chunk_buffer = bytearray()
        log.info("utterance_transcribed", preview=text[:80] if text else "")
        return TranscriptionResult(text=text, is_final=True, language=self._language)

    async def _run_transcribe(self, audio_bytes: bytes) -> str:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._transcribe_sync, audio_bytes
        )

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        import numpy as np
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            samples,
            language=self._language,
            beam_size=5,
            vad_filter=False,  # VAD is handled externally
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def aclose(self) -> None:
        self._model = None
        log.info("whisper_closed")
