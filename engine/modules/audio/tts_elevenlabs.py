"""ElevenLabs TTS via WebSocket streaming API (PCM 24 kHz mono)."""
import asyncio
import base64
import json
from typing import AsyncIterator

import websockets

from .tts_interface import TTSClient
from engine.logging_config import get_logger

log = get_logger("tts.elevenlabs")

_WS_URL = (
    "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
    "?model_id={model_id}&output_format=pcm_24000"
    "&optimize_streaming_latency={latency}"
)


class ElevenLabsTTS(TTSClient):
    """ElevenLabs streaming TTS over WebSocket.

    Yields raw int16 PCM chunks at 24 kHz mono.  First-chunk latency is
    typically 250–350 ms at latency_optimization=3.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_turbo_v2_5",
        latency_optimization: int = 3,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._latency = latency_optimization

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks for ``text`` via ElevenLabs WebSocket."""
        import websockets

        url = _WS_URL.format(
            voice_id=self._voice_id,
            model_id=self._model_id,
            latency=self._latency,
        )
        log.info("elevenlabs_start", preview=text[:60], model=self._model_id)
        chunk_count = 0

        async with websockets.connect(
            url,
            additional_headers={"xi-api-key": self._api_key},
        ) as ws:
            # BOS sentinel with voice settings
            await ws.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "generation_config": {"chunk_length_schedule": [120, 160, 250, 290]},
            }))
            # The text to speak
            await ws.send(json.dumps({"text": text}))
            # EOS sentinel
            await ws.send(json.dumps({"text": ""}))

            async for raw_msg in ws:
                msg = json.loads(raw_msg)
                if msg.get("audio"):
                    audio_bytes = base64.b64decode(msg["audio"])
                    chunk_count += 1
                    if chunk_count == 1:
                        log.debug("elevenlabs_first_chunk")
                    yield audio_bytes
                if msg.get("isFinal"):
                    break

        log.info("elevenlabs_complete", chunks=chunk_count)

    async def aclose(self) -> None:
        log.info("elevenlabs_closed")
