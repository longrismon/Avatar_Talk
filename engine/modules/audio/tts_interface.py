"""Abstract base class for Text-to-Speech providers."""
from abc import ABC, abstractmethod
from typing import AsyncIterator


class TTSClient(ABC):
    """Abstract interface for TTS providers.

    Implementations yield raw int16 PCM mono audio chunks at 24 kHz unless
    otherwise documented on the concrete class.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesized audio for the given text.

        Args:
            text: Text to synthesize.

        Yields:
            Raw int16 PCM bytes at 24 kHz mono.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Release resources."""
