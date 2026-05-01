"""Abstract base class for Speech-to-Text providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    text: str
    is_final: bool
    confidence: float = 1.0
    language: str = "en"


class STTClient(ABC):
    """Abstract interface for STT providers.

    All implementations receive raw int16 PCM mono audio at 16 kHz.
    """

    @abstractmethod
    async def transcribe_chunk(self, audio_bytes: bytes) -> TranscriptionResult:
        """Process a partial audio chunk; returns a non-final partial transcript.

        Args:
            audio_bytes: Raw int16 PCM bytes at 16 kHz mono.
        """

    @abstractmethod
    async def transcribe_utterance(self, audio_bytes: bytes) -> TranscriptionResult:
        """Transcribe a complete utterance for maximum accuracy.

        Called when VAD signals end-of-utterance. Implementations should run
        a fresh, full-context transcription on the entire buffered audio.

        Args:
            audio_bytes: Complete utterance — raw int16 PCM bytes at 16 kHz mono.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Release model resources."""
