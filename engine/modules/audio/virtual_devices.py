"""Virtual audio device management and real-time audio injection/capture."""
import asyncio
import platform
import subprocess
from typing import AsyncIterator, Optional

from engine.logging_config import get_logger

log = get_logger("virtual_devices")

_INJECT_SAMPLE_RATE = 24000  # Match ElevenLabs TTS output
_CAPTURE_SAMPLE_RATE = 16000  # Match STT input requirement
_CHANNELS = 1
_DTYPE = "int16"
_CAPTURE_CHUNK_MS = 32        # 512 samples at 16 kHz — Silero VAD chunk size


async def inject_audio(
    audio_chunks: AsyncIterator[bytes],
    device: str,
    sample_rate: int = _INJECT_SAMPLE_RATE,
) -> None:
    """Inject audio chunks into a virtual mic device via sounddevice.

    Args:
        audio_chunks: Async iterator of raw int16 PCM bytes at ``sample_rate``.
        device: sounddevice device name or index string.
        sample_rate: Must match the audio data (default 24 kHz for ElevenLabs).
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required. Install with: pip install sounddevice"
        ) from exc

    loop = asyncio.get_running_loop()
    log.info("inject_start", device=device, sample_rate=sample_rate)
    chunk_count = 0

    with sd.RawOutputStream(
        device=device,
        samplerate=sample_rate,
        channels=_CHANNELS,
        dtype=_DTYPE,
        blocksize=0,
    ) as stream:
        async for chunk in audio_chunks:
            if chunk:
                await loop.run_in_executor(None, stream.write, chunk)
                chunk_count += 1

    log.info("inject_complete", chunks=chunk_count)


async def capture_audio(
    device: str,
    chunk_ms: int = _CAPTURE_CHUNK_MS,
    sample_rate: int = _CAPTURE_SAMPLE_RATE,
) -> AsyncIterator[bytes]:
    """Capture audio from a device as an async stream of raw PCM bytes.

    Yields 32 ms (512-sample) chunks by default — sized for Silero VAD
    compatibility.

    Args:
        device: sounddevice device name or index string.
        chunk_ms: Duration per chunk in milliseconds.
        sample_rate: Capture sample rate (default 16 kHz for STT).
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required. Install with: pip install sounddevice"
        ) from exc

    chunk_samples = int(chunk_ms / 1000 * sample_rate)
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=50)
    loop = asyncio.get_running_loop()

    def _callback(indata, frames, time_info, status):
        if status:
            log.warning("capture_status", status=str(status))
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    log.info("capture_start", device=device, sample_rate=sample_rate, chunk_ms=chunk_ms)
    with sd.RawInputStream(
        device=device,
        samplerate=sample_rate,
        channels=_CHANNELS,
        dtype=_DTYPE,
        blocksize=chunk_samples,
        callback=_callback,
    ):
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


# ---------------------------------------------------------------------------
# Platform-specific setup helpers
# ---------------------------------------------------------------------------

def setup_linux_virtual_mic(sink_name: str = "avatar_agent_mic") -> bool:
    """Create a PulseAudio null sink for Linux virtual mic injection.

    Idempotent — returns True if the sink already exists or was created.
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True, text=True, timeout=5,
        )
        if sink_name in result.stdout:
            log.info("virtual_mic_exists", sink_name=sink_name)
            return True
        subprocess.run(
            ["pactl", "load-module", "module-null-sink",
             f"sink_name={sink_name}",
             f"sink_properties=device.description={sink_name}"],
            check=True, timeout=5,
        )
        log.info("virtual_mic_created", sink_name=sink_name)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as exc:
        log.error("virtual_mic_failed", error=str(exc))
        return False


def get_virtual_mic_device(config) -> str:
    """Return the virtual mic device name for the current platform from config."""
    os_name = platform.system()
    if os_name == "Linux":
        return config.audio.linux.sink_name
    elif os_name == "Darwin":
        return config.audio.macos.driver
    elif os_name == "Windows":
        return config.audio.windows.driver
    return "default"
