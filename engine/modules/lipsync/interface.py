"""LipSync interface — abstract base class for lip-sync providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LipSyncFrame:
    """A single decoded video frame from a lip-sync result."""
    bgr: bytes    # raw BGR bytes (height × width × 3)
    width: int
    height: int
    pts_ms: int   # presentation timestamp in milliseconds


class LipSyncClient(ABC):
    """Abstract interface for lip-sync model providers."""

    @abstractmethod
    async def load(self) -> None:
        """Load model weights into memory (idempotent)."""

    @abstractmethod
    async def generate_video(
        self,
        audio_pcm: bytes,
        reference_image: str,
        sample_rate: int = 16000,
    ) -> bytes:
        """Generate a lip-synced MP4 video from raw int16 PCM audio.

        Args:
            audio_pcm: Raw int16 PCM bytes at sample_rate Hz, mono.
            reference_image: Path to the reference face image (JPEG/PNG).
            sample_rate: Audio sample rate in Hz (default 16 000).

        Returns:
            Raw MP4 bytes containing the animated face video.

        Raises:
            RuntimeError: If load() has not been called.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Release model weights and free resources."""
