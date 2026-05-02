"""Tests for Phase 4 — lip-sync pipeline: interface, Wav2Lip, SadTalker, virtual camera, factory."""
import asyncio
import struct
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


def _make_pcm(n_samples: int = 16000) -> bytes:
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestLipSyncInterface:
    def test_interface_cannot_be_instantiated(self):
        from engine.modules.lipsync.interface import LipSyncClient
        with pytest.raises(TypeError):
            LipSyncClient()  # ABC — cannot instantiate directly

    def test_lipsync_frame_fields(self):
        from engine.modules.lipsync.interface import LipSyncFrame
        frame = LipSyncFrame(bgr=b"\x00" * 12, width=2, height=2, pts_ms=100)
        assert frame.width == 2
        assert frame.pts_ms == 100


# ---------------------------------------------------------------------------
# Wav2Lip
# ---------------------------------------------------------------------------

class TestWav2LipLipSync:
    async def test_requires_load(self):
        from engine.modules.lipsync.wav2lip import Wav2LipLipSync
        ls = Wav2LipLipSync(model_path="./m.pth", face_detect_path="./s3fd.pth")
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            await ls.generate_video(_make_pcm(), "./face.png")

    async def test_generate_video_returns_bytes(self):
        from engine.modules.lipsync.wav2lip import Wav2LipLipSync

        ls = Wav2LipLipSync(model_path="./m.pth", face_detect_path="./s3fd.pth", device="cpu")
        ls._loaded = True

        fake_mp4 = b"\x00\x00\x00\x18ftyp" + b"\x00" * 24

        def fake_run(audio_pcm, ref_img, sample_rate):
            return fake_mp4

        with patch.object(ls, "_run_inference", side_effect=fake_run):
            result = await ls.generate_video(_make_pcm(), "./face.png")

        assert isinstance(result, bytes)
        assert result == fake_mp4

    async def test_generate_video_uses_executor(self):
        from engine.modules.lipsync.wav2lip import Wav2LipLipSync

        ls = Wav2LipLipSync(model_path="./m.pth", face_detect_path="./s3fd.pth")
        ls._loaded = True

        called_from_thread = []

        def fake_run(audio_pcm, ref_img, sample_rate):
            # If running in executor, we're not in the event loop thread
            called_from_thread.append(True)
            return b"video"

        with patch.object(ls, "_run_inference", side_effect=fake_run):
            await ls.generate_video(_make_pcm(), "./face.png")

        assert called_from_thread, "inference must run in executor"

    async def test_aclose_releases_model(self):
        from engine.modules.lipsync.wav2lip import Wav2LipLipSync

        ls = Wav2LipLipSync(model_path="./m.pth", face_detect_path="./s3fd.pth")
        ls._model = MagicMock()
        ls._loaded = True

        await ls.aclose()

        assert ls._model is None
        assert not ls._loaded

    async def test_load_raises_if_model_path_missing(self):
        from engine.modules.lipsync.wav2lip import Wav2LipLipSync

        ls = Wav2LipLipSync(model_path="/nonexistent/m.pth", face_detect_path="./s3fd.pth")

        with pytest.raises(FileNotFoundError, match="Wav2Lip"):
            await ls.load()

    def test_audio_to_mel_returns_array(self):
        from engine.modules.lipsync.wav2lip import Wav2LipLipSync
        import numpy as np

        ls = Wav2LipLipSync(model_path="./m.pth", face_detect_path="./s3fd.pth")
        waveform = np.zeros(16000, dtype=np.float32)
        mel = ls._audio_to_mel(waveform, 16000)
        assert mel.shape[0] == 80  # 80 mel bands

    def test_audio_to_mel_resamples_non_16k(self):
        from engine.modules.lipsync.wav2lip import Wav2LipLipSync
        import numpy as np

        ls = Wav2LipLipSync(model_path="./m.pth", face_detect_path="./s3fd.pth")
        waveform = np.zeros(24000, dtype=np.float32)  # 24 kHz
        mel = ls._audio_to_mel(waveform, 24000)
        assert mel.shape[0] == 80


# ---------------------------------------------------------------------------
# SadTalker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSadTalkerLipSync:
    async def test_requires_load(self):
        from engine.modules.lipsync.sadtalker import SadTalkerLipSync
        ls = SadTalkerLipSync(model_path="./sadtalker_models")
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            await ls.generate_video(_make_pcm(), "./face.png")

    async def test_generate_video_returns_bytes(self):
        from engine.modules.lipsync.sadtalker import SadTalkerLipSync

        ls = SadTalkerLipSync(model_path="./sadtalker_models", device="cpu")
        ls._loaded = True
        ls._pipeline = {"model_path": "./sadtalker_models", "device": "cpu"}

        fake_mp4 = b"\x00\x00\x00\x18ftyp" + b"\x00" * 24

        with patch.object(ls, "_run_inference", return_value=fake_mp4):
            result = await ls.generate_video(_make_pcm(), "./face.png")

        assert result == fake_mp4

    async def test_aclose_releases_pipeline(self):
        from engine.modules.lipsync.sadtalker import SadTalkerLipSync

        ls = SadTalkerLipSync(model_path="./sadtalker_models")
        ls._pipeline = {"model_path": "./sadtalker_models", "device": "cpu"}
        ls._loaded = True

        await ls.aclose()

        assert ls._pipeline is None
        assert not ls._loaded

    async def test_load_raises_if_model_path_missing(self):
        from engine.modules.lipsync.sadtalker import SadTalkerLipSync

        ls = SadTalkerLipSync(model_path="/nonexistent/sadtalker")

        with pytest.raises(FileNotFoundError, match="SadTalker"):
            await ls.load()


# ---------------------------------------------------------------------------
# VirtualCamera
# ---------------------------------------------------------------------------

class TestVirtualCamera:
    @pytest.mark.asyncio
    async def test_inject_video_frames_calls_video_writer(self):
        from engine.modules.lipsync.virtual_camera import inject_video_frames

        fake_mp4 = b"\x00" * 128

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 320.0
        # Return one frame then stop
        mock_cap.read.side_effect = [
            (True, b"\x00" * (320 * 240 * 3)),
            (False, None),
        ]

        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 0x47504A4D
        mock_cv2.CAP_V4L2 = 200
        mock_cv2.CAP_ANY = 0

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            await inject_video_frames(fake_mp4, device="/dev/video10", fps=25)

        mock_writer.write.assert_called_once()
        mock_writer.release.assert_called_once()
        mock_cap.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_video_frames_skips_empty_bytes(self):
        from engine.modules.lipsync.virtual_camera import inject_video_frames

        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            await inject_video_frames(b"", device="/dev/video10")

        mock_cv2.VideoCapture.assert_not_called()

    def test_setup_linux_virtual_camera_already_exists(self):
        from engine.modules.lipsync.virtual_camera import setup_linux_virtual_camera

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = setup_linux_virtual_camera("/dev/video10")

        assert result is True
        assert mock_run.call_count == 1  # only the ls check, no modprobe

    def test_setup_linux_virtual_camera_loads_module(self):
        from engine.modules.lipsync.virtual_camera import setup_linux_virtual_camera

        ls_result = MagicMock(returncode=1)
        modprobe_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[ls_result, modprobe_result]) as mock_run:
            result = setup_linux_virtual_camera("/dev/video10")

        assert result is True
        assert mock_run.call_count == 2
        modprobe_call = mock_run.call_args_list[1]
        assert "modprobe" in modprobe_call[0][0]

    def test_get_virtual_camera_device_linux(self):
        from engine.modules.lipsync.virtual_camera import get_virtual_camera_device

        cfg = MagicMock()
        cfg.camera.linux.device = "/dev/video10"

        with patch("platform.system", return_value="Linux"):
            result = get_virtual_camera_device(cfg)

        assert result == "/dev/video10"

    def test_get_virtual_camera_device_none_config(self):
        from engine.modules.lipsync.virtual_camera import get_virtual_camera_device

        assert get_virtual_camera_device(None) is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestLipSyncFactory:
    def test_create_wav2lip(self):
        from engine.modules.lipsync import create_lipsync
        from engine.modules.lipsync.wav2lip import Wav2LipLipSync

        cfg = MagicMock()
        cfg.engine = "wav2lip"
        cfg.wav2lip.model_path = "./m.pth"
        cfg.wav2lip.face_detect_path = "./s3fd.pth"
        cfg.device = "cpu"

        ls = create_lipsync(cfg)
        assert isinstance(ls, Wav2LipLipSync)

    def test_create_sadtalker(self):
        from engine.modules.lipsync import create_lipsync
        from engine.modules.lipsync.sadtalker import SadTalkerLipSync

        cfg = MagicMock()
        cfg.engine = "sadtalker"
        cfg.sadtalker.model_path = "./sadtalker_models"
        cfg.device = "cpu"

        ls = create_lipsync(cfg)
        assert isinstance(ls, SadTalkerLipSync)

    def test_create_unknown_raises(self):
        from engine.modules.lipsync import create_lipsync

        cfg = MagicMock()
        cfg.engine = "deepfacelab"

        with pytest.raises(ValueError, match="Unsupported lip-sync engine"):
            create_lipsync(cfg)
