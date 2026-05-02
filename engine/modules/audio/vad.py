"""Voice Activity Detection — Silero VAD primary, WebRTC energy fallback."""
import struct
import time
from dataclasses import dataclass
from typing import Optional

from engine.logging_config import get_logger

log = get_logger("vad")

_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # int16


@dataclass
class VADResult:
    is_speech: bool
    probability: float
    utterance_complete: bool  # True when silence after speech passes the threshold


class SileroVAD:
    """Silero VAD wrapper for utterance-boundary detection.

    Processes audio in 512-sample (32 ms) chunks at 16 kHz — the minimum chunk
    size required by the Silero model.  Signals ``utterance_complete`` when
    continuous silence after a speech segment exceeds ``min_silence_ms``.

    Call ``await load()`` once before the first ``process_chunk()`` call.
    """

    CHUNK_SAMPLES = 512          # 32 ms at 16 kHz — Silero requirement
    CHUNK_BYTES = CHUNK_SAMPLES * _BYTES_PER_SAMPLE

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 1500,
        speech_pad_ms: int = 300,
    ) -> None:
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._speech_pad_ms = speech_pad_ms
        self._model = None
        self._buffer = bytearray()
        self._speech_started = False
        self._silence_start: Optional[float] = None
        self._loaded = False

    async def load(self) -> None:
        """Load the Silero VAD model via torch.hub (downloads on first use)."""
        if self._loaded:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is required for SileroVAD. Install with: pip install torch"
            ) from exc
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._loaded = True
            log.info("silero_vad_loaded", threshold=self._threshold)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Silero VAD: {exc}") from exc

    def process_chunk(self, audio_bytes: bytes) -> list[VADResult]:
        """Add raw PCM bytes to the internal buffer and process 512-sample chunks.

        Args:
            audio_bytes: Raw int16 PCM at 16 kHz, any length.

        Returns:
            One VADResult per completed 512-sample chunk; empty list if not
            enough data has accumulated yet.
        """
        if not self._loaded:
            raise RuntimeError("SileroVAD.load() must be awaited before process_chunk()")

        import torch

        self._buffer.extend(audio_bytes)
        results: list[VADResult] = []

        while len(self._buffer) >= self.CHUNK_BYTES:
            chunk = bytes(self._buffer[: self.CHUNK_BYTES])
            del self._buffer[: self.CHUNK_BYTES]

            samples = struct.unpack(f"<{self.CHUNK_SAMPLES}h", chunk)
            tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0
            prob = float(self._model(tensor, _SAMPLE_RATE).item())
            is_speech = prob >= self._threshold
            utterance_complete = False

            if is_speech:
                self._speech_started = True
                self._silence_start = None
            elif self._speech_started:
                now = time.monotonic()
                if self._silence_start is None:
                    self._silence_start = now
                silence_ms = (now - self._silence_start) * 1000
                if silence_ms >= self._min_silence_ms:
                    utterance_complete = True
                    self._speech_started = False
                    self._silence_start = None
                    log.debug("utterance_complete", silence_ms=int(silence_ms))

            results.append(VADResult(
                is_speech=is_speech,
                probability=prob,
                utterance_complete=utterance_complete,
            ))

        return results

    def reset(self) -> None:
        """Reset state for a new call or after an utterance has been dispatched."""
        self._buffer = bytearray()
        self._speech_started = False
        self._silence_start = None
        log.debug("vad_reset")


def create_vad(config) -> SileroVAD:
    """Factory: create a SileroVAD from a VADConfig instance."""
    if config.engine != "silero":
        log.warning("vad_engine_unsupported", requested=config.engine, using="silero")
    return SileroVAD(
        threshold=config.threshold,
        min_silence_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms,
    )
