"""Coqui XTTS v2 — local TTS fallback (no API key required)."""
import asyncio
from typing import AsyncIterator

from .tts_interface import TTSClient
from engine.logging_config import get_logger

log = get_logger("tts.coqui")

_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = int(0.1 * _SAMPLE_RATE)  # 100 ms chunks
_CHUNK_BYTES = _CHUNK_SAMPLES * 2          # int16


class CoquiTTS(TTSClient):
    """Coqui XTTS v2 local TTS.

    Fully offline fallback activated when ElevenLabs is unavailable.
    First-chunk latency ~800–1500 ms depending on GPU.  Requires the
    ``TTS`` package and a voice sample WAV of at least 10 s.

    Call ``await load()`` once at startup before synthesizing.
    """

    def __init__(
        self,
        model_path: str,
        voice_sample: str,
        device: str = "cuda",
    ) -> None:
        self._model_path = model_path
        self._voice_sample = voice_sample
        self._device = device
        self._tts = None

    async def load(self) -> None:
        """Load the XTTS model in a thread executor (slow — do at startup)."""
        if self._tts is not None:
            return
        log.info("coqui_loading", model=self._model_path, device=self._device)
        self._tts = await asyncio.get_running_loop().run_in_executor(
            None, self._load_sync
        )
        log.info("coqui_loaded")

    def _load_sync(self):
        try:
            from TTS.api import TTS as CoquiLib
        except ImportError as exc:
            raise RuntimeError(
                "TTS library is required for Coqui. Install with: pip install TTS"
            ) from exc
        return CoquiLib(
            model_path=self._model_path,
            config_path=f"{self._model_path}/config.json",
            progress_bar=False,
            gpu=(self._device == "cuda"),
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text with XTTS and yield 100 ms PCM chunks."""
        if self._tts is None:
            raise RuntimeError("CoquiTTS.load() must be awaited before synthesize()")

        log.info("coqui_start", preview=text[:60])
        loop = asyncio.get_running_loop()
        wav = await loop.run_in_executor(None, self._synthesize_sync, text)

        import numpy as np
        raw = (np.array(wav, dtype=np.float32) * 32767).astype(np.int16).tobytes()
        for i in range(0, len(raw), _CHUNK_BYTES):
            yield raw[i : i + _CHUNK_BYTES]

        log.info("coqui_complete")

    def _synthesize_sync(self, text: str):
        return self._tts.tts(
            text=text,
            speaker_wav=self._voice_sample,
            language="en",
        )

    async def aclose(self) -> None:
        self._tts = None
        log.info("coqui_closed")
