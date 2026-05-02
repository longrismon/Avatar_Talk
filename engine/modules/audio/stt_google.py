"""Google Cloud Speech-to-Text — fallback when no CUDA GPU is available."""
import os

from .stt_interface import STTClient, TranscriptionResult
from engine.logging_config import get_logger

log = get_logger("stt.google")

_SAMPLE_RATE = 16000


class GoogleSTT(STTClient):
    """Google Cloud STT (synchronous recognize API).

    Used as the ``faster-whisper`` fallback on CPU-only machines.
    Requires the ``google-cloud-speech`` package and either
    ``GOOGLE_APPLICATION_CREDENTIALS`` env var or a ``credentials_path``.

    Partial transcription is not optimized — ``transcribe_chunk`` always
    returns an empty result; only ``transcribe_utterance`` hits the API.
    """

    def __init__(self, credentials_path: str = "", language_code: str = "en-US") -> None:
        self._credentials_path = credentials_path
        self._language_code = language_code
        self._client = None
        self._speech = None

    async def load(self) -> None:
        """Initialize the Google STT async client."""
        if self._client is not None:
            return
        try:
            from google.cloud import speech as google_speech
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-speech is required. Install with: "
                "pip install google-cloud-speech"
            ) from exc

        if self._credentials_path:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", self._credentials_path)

        self._client = google_speech.SpeechAsyncClient()
        self._speech = google_speech
        log.info("google_stt_loaded", language=self._language_code)

    async def transcribe_chunk(self, audio_bytes: bytes) -> TranscriptionResult:
        return TranscriptionResult(text="", is_final=False)

    async def transcribe_utterance(self, audio_bytes: bytes) -> TranscriptionResult:
        if self._client is None:
            raise RuntimeError("GoogleSTT.load() must be awaited before transcribing")
        if not audio_bytes:
            return TranscriptionResult(text="", is_final=True)

        config = self._speech.RecognitionConfig(
            encoding=self._speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=_SAMPLE_RATE,
            language_code=self._language_code,
        )
        audio = self._speech.RecognitionAudio(content=audio_bytes)
        response = await self._client.recognize(config=config, audio=audio)
        text = " ".join(
            result.alternatives[0].transcript
            for result in response.results
            if result.alternatives
        ).strip()
        log.info("google_utterance_transcribed", preview=text[:80] if text else "")
        return TranscriptionResult(
            text=text,
            is_final=True,
            language=self._language_code[:2],
        )

    async def aclose(self) -> None:
        self._client = None
        log.info("google_stt_closed")
