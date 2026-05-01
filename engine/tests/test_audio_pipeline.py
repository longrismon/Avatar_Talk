"""Tests for Phase 2 audio pipeline — VAD, STT, TTS, virtual devices."""
import asyncio
import base64
import json
import struct
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_torch(speech_prob: float = 0.9) -> MagicMock:
    """Return a minimal torch mock that satisfies vad.process_chunk."""
    mock_torch = MagicMock()
    # torch.tensor(...).unsqueeze(0) / 32768.0 → just needs to be passable to model
    mock_tensor = MagicMock()
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_torch.tensor.return_value = mock_tensor
    mock_torch.float32 = "float32"
    return mock_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pcm_chunk(n_samples: int = 512, value: int = 0) -> bytes:
    """Create a raw int16 PCM chunk of n_samples all set to value."""
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


# ---------------------------------------------------------------------------
# SileroVAD
# ---------------------------------------------------------------------------

class TestSileroVAD:
    def test_init_defaults(self):
        from engine.modules.audio.vad import SileroVAD
        vad = SileroVAD()
        assert vad._threshold == 0.5
        assert vad._min_silence_ms == 1500

    def test_process_requires_load(self):
        from engine.modules.audio.vad import SileroVAD
        vad = SileroVAD()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            vad.process_chunk(make_pcm_chunk())

    def test_too_few_bytes_returns_empty(self):
        from engine.modules.audio.vad import SileroVAD
        vad = SileroVAD()
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.9
        vad._model = mock_model
        vad._loaded = True
        # 256 samples = 512 bytes < CHUNK_BYTES (1024)
        with patch.dict(sys.modules, {"torch": _make_mock_torch()}):
            results = vad.process_chunk(make_pcm_chunk(256))
        assert results == []

    def test_speech_chunk_detected(self):
        from engine.modules.audio.vad import SileroVAD
        vad = SileroVAD(threshold=0.5)
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.9
        vad._model = mock_model
        vad._loaded = True

        with patch.dict(sys.modules, {"torch": _make_mock_torch()}):
            results = vad.process_chunk(make_pcm_chunk())
        assert len(results) == 1
        assert results[0].is_speech is True
        assert results[0].utterance_complete is False

    def test_utterance_complete_after_silence(self):
        from engine.modules.audio.vad import SileroVAD
        vad = SileroVAD(threshold=0.5, min_silence_ms=0)
        mock_model = MagicMock()
        vad._model = mock_model
        vad._loaded = True

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            mock_model.return_value.item.return_value = 0.9
            vad.process_chunk(make_pcm_chunk())  # speech
            mock_model.return_value.item.return_value = 0.1
            results = vad.process_chunk(make_pcm_chunk())  # silence
        assert any(r.utterance_complete for r in results)

    def test_reset_clears_state(self):
        from engine.modules.audio.vad import SileroVAD
        vad = SileroVAD()
        vad._speech_started = True
        vad._buffer.extend(b"\x00" * 64)
        vad.reset()
        assert not vad._speech_started
        assert len(vad._buffer) == 0

    def test_silence_before_speech_never_completes(self):
        from engine.modules.audio.vad import SileroVAD
        vad = SileroVAD(threshold=0.5, min_silence_ms=0)
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.1
        vad._model = mock_model
        vad._loaded = True

        with patch.dict(sys.modules, {"torch": _make_mock_torch()}):
            results = vad.process_chunk(make_pcm_chunk())
        assert not any(r.utterance_complete for r in results)


# ---------------------------------------------------------------------------
# WhisperSTT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestWhisperSTT:
    async def test_requires_load(self):
        from engine.modules.audio.stt_whisper import WhisperSTT
        stt = WhisperSTT()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            await stt.transcribe_chunk(b"\x00" * 100)

    async def test_transcribe_utterance_empty_bytes(self):
        from engine.modules.audio.stt_whisper import WhisperSTT
        stt = WhisperSTT()
        stt._model = MagicMock()  # skip load
        result = await stt.transcribe_utterance(b"")
        assert result.is_final is True
        assert result.text == ""

    async def test_transcribe_utterance_returns_text(self):
        from engine.modules.audio.stt_whisper import WhisperSTT
        import numpy as np

        stt = WhisperSTT(language="en")
        seg = MagicMock()
        seg.text = "  Hello world  "
        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([seg], MagicMock())

        audio = np.zeros(16000, dtype=np.int16).tobytes()
        result = await stt.transcribe_utterance(audio)
        assert result.is_final is True
        assert result.text == "Hello world"
        assert result.language == "en"

    async def test_partial_chunk_buffer_capped_at_window(self):
        from engine.modules.audio.stt_whisper import WhisperSTT, WINDOW_BYTES
        import numpy as np

        stt = WhisperSTT()
        seg = MagicMock()
        seg.text = "partial"
        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([seg], MagicMock())

        # Feed 600 ms of audio — more than the 500 ms window
        audio = np.zeros(int(0.6 * 16000), dtype=np.int16).tobytes()
        await stt.transcribe_chunk(audio)
        assert len(stt._chunk_buffer) <= WINDOW_BYTES

    async def test_transcribe_utterance_resets_partial_buffer(self):
        from engine.modules.audio.stt_whisper import WhisperSTT
        import numpy as np

        stt = WhisperSTT()
        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([], MagicMock())
        stt._chunk_buffer.extend(b"\x00" * 1000)

        audio = np.zeros(16000, dtype=np.int16).tobytes()
        await stt.transcribe_utterance(audio)
        assert len(stt._chunk_buffer) == 0

    async def test_aclose(self):
        from engine.modules.audio.stt_whisper import WhisperSTT
        stt = WhisperSTT()
        stt._model = MagicMock()
        await stt.aclose()
        assert stt._model is None


# ---------------------------------------------------------------------------
# ElevenLabsTTS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestElevenLabsTTS:
    async def test_synthesize_yields_audio_chunks(self):
        from engine.modules.audio.tts_elevenlabs import ElevenLabsTTS

        fake_audio = b"\x01\x02" * 100
        messages = [
            json.dumps({"audio": base64.b64encode(fake_audio).decode(), "isFinal": False}),
            json.dumps({"audio": base64.b64encode(fake_audio).decode(), "isFinal": True}),
        ]

        class FakeWS:
            async def send(self, _): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            def __aiter__(self): return self
            _msgs = list(messages)
            async def __anext__(self):
                if self._msgs:
                    return self._msgs.pop(0)
                raise StopAsyncIteration

        tts = ElevenLabsTTS(api_key="k", voice_id="v")
        with patch("engine.modules.audio.tts_elevenlabs.websockets.connect", return_value=FakeWS()):
            chunks = [c async for c in tts.synthesize("Hello")]

        assert len(chunks) == 2
        assert chunks[0] == fake_audio

    async def test_aclose_no_error(self):
        from engine.modules.audio.tts_elevenlabs import ElevenLabsTTS
        await ElevenLabsTTS(api_key="k", voice_id="v").aclose()

    async def test_synthesize_stops_at_is_final(self):
        from engine.modules.audio.tts_elevenlabs import ElevenLabsTTS

        fake_audio = b"\x00" * 48
        # 3 messages: 2 audio + final
        msgs = [
            json.dumps({"audio": base64.b64encode(fake_audio).decode(), "isFinal": False}),
            json.dumps({"audio": base64.b64encode(fake_audio).decode(), "isFinal": False}),
            json.dumps({"audio": base64.b64encode(fake_audio).decode(), "isFinal": True}),
            json.dumps({"audio": base64.b64encode(fake_audio).decode(), "isFinal": False}),  # after final — should not be yielded
        ]

        class FakeWS:
            async def send(self, _): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            def __aiter__(self): return self
            _msgs = list(msgs)
            async def __anext__(self):
                if self._msgs:
                    return self._msgs.pop(0)
                raise StopAsyncIteration

        tts = ElevenLabsTTS(api_key="k", voice_id="v")
        with patch("engine.modules.audio.tts_elevenlabs.websockets.connect", return_value=FakeWS()):
            chunks = [c async for c in tts.synthesize("Hi")]

        assert len(chunks) == 3  # stops at isFinal=True, inclusive


# ---------------------------------------------------------------------------
# CoquiTTS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCoquiTTS:
    async def test_requires_load(self):
        from engine.modules.audio.tts_coqui import CoquiTTS
        tts = CoquiTTS(model_path="./m", voice_sample="./v.wav")
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            async for _ in tts.synthesize("hi"):
                pass

    async def test_synthesize_yields_chunks(self):
        from engine.modules.audio.tts_coqui import CoquiTTS, _CHUNK_BYTES
        import numpy as np

        tts = CoquiTTS(model_path="./m", voice_sample="./v.wav")
        # 2400 samples = 100 ms @ 24 kHz → exactly 1 chunk
        tts._tts = MagicMock()
        tts._tts.tts.return_value = [0.0] * 2400

        chunks = [c async for c in tts.synthesize("Hello")]
        assert len(chunks) >= 1
        assert all(len(c) <= _CHUNK_BYTES for c in chunks)

    async def test_aclose(self):
        from engine.modules.audio.tts_coqui import CoquiTTS
        tts = CoquiTTS(model_path="./m", voice_sample="./v.wav")
        tts._tts = MagicMock()
        await tts.aclose()
        assert tts._tts is None


# ---------------------------------------------------------------------------
# virtual_devices
# ---------------------------------------------------------------------------

class TestVirtualDevices:
    @pytest.mark.asyncio
    async def test_inject_audio_calls_stream_write(self):
        from engine.modules.audio.virtual_devices import inject_audio

        async def fake_chunks():
            yield b"\x00\x01" * 512
            yield b"\x00\x02" * 512

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.write = MagicMock()

        mock_sd = MagicMock()
        mock_sd.RawOutputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            await inject_audio(fake_chunks(), device="test-mic")

        assert mock_stream.write.call_count == 2

    @pytest.mark.asyncio
    async def test_inject_audio_skips_empty_chunks(self):
        from engine.modules.audio.virtual_devices import inject_audio

        async def fake_chunks():
            yield b""
            yield b"\x00" * 64

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.write = MagicMock()

        mock_sd = MagicMock()
        mock_sd.RawOutputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            await inject_audio(fake_chunks(), device="test-mic")

        assert mock_stream.write.call_count == 1  # empty chunk skipped

    def test_setup_linux_virtual_mic_already_exists(self):
        from engine.modules.audio.virtual_devices import setup_linux_virtual_mic
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="avatar_agent_mic something")
            result = setup_linux_virtual_mic("avatar_agent_mic")
        assert result is True
        assert mock_run.call_count == 1  # only the list call, no create

    def test_setup_linux_virtual_mic_creates(self):
        from engine.modules.audio.virtual_devices import setup_linux_virtual_mic
        call_results = [
            MagicMock(stdout="other_sink"),  # list call
            MagicMock(),                     # load-module call
        ]
        with patch("subprocess.run", side_effect=call_results):
            result = setup_linux_virtual_mic("avatar_agent_mic")
        assert result is True


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

class TestFactories:
    def test_create_tts_elevenlabs(self):
        from engine.modules.audio import create_tts
        from engine.modules.audio.tts_elevenlabs import ElevenLabsTTS

        cfg = MagicMock()
        cfg.tts.primary = "elevenlabs"
        cfg.tts.elevenlabs.api_key = "key"
        cfg.tts.elevenlabs.voice_id = "voice"
        cfg.tts.elevenlabs.model_id = "eleven_turbo_v2_5"
        cfg.tts.elevenlabs.latency_optimization = 3

        tts = create_tts(cfg)
        assert isinstance(tts, ElevenLabsTTS)

    def test_create_tts_coqui(self):
        from engine.modules.audio import create_tts
        from engine.modules.audio.tts_coqui import CoquiTTS

        cfg = MagicMock()
        cfg.tts.primary = "coqui-xtts"
        cfg.tts.coqui.model_path = "./m"
        cfg.tts.coqui.voice_sample = "./v.wav"
        cfg.tts.coqui.device = "cpu"

        tts = create_tts(cfg)
        assert isinstance(tts, CoquiTTS)

    def test_create_tts_invalid_raises(self):
        from engine.modules.audio import create_tts
        cfg = MagicMock()
        cfg.tts.primary = "unknown-provider"
        with pytest.raises(ValueError, match="Unsupported TTS"):
            create_tts(cfg)

    def test_create_stt_falls_back_to_google_without_cuda(self):
        from engine.modules.audio import create_stt
        from engine.modules.audio.stt_google import GoogleSTT

        cfg = MagicMock()
        cfg.stt.primary = "faster-whisper"
        cfg.stt.fallback = "google-stt"
        cfg.stt.google.credentials_path = ""
        cfg.stt.google.language_code = "en-US"

        with patch.dict("sys.modules", {"torch": None}):
            stt = create_stt(cfg)
        assert isinstance(stt, GoogleSTT)

    def test_create_vad(self):
        from engine.modules.audio.vad import create_vad, SileroVAD

        cfg = MagicMock()
        cfg.engine = "silero"
        cfg.threshold = 0.6
        cfg.min_silence_duration_ms = 2000
        cfg.speech_pad_ms = 200

        vad = create_vad(cfg)
        assert isinstance(vad, SileroVAD)
        assert vad._threshold == 0.6
        assert vad._min_silence_ms == 2000
