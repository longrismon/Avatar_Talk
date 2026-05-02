"""Wav2Lip lip-sync client — animates a reference face image to match audio."""
import asyncio
import os
import tempfile

from .interface import LipSyncClient
from engine.logging_config import get_logger

log = get_logger("lipsync.wav2lip")

_WAV2LIP_FPS = 25
_MEL_STEP_SIZE = 16
_FACE_CROP_SIZE = (96, 96)


class Wav2LipLipSync(LipSyncClient):
    """Wav2Lip-based lip-sync: animates a cropped face region to match audio.

    Model weights must be downloaded separately:
        - wav2lip_gan.pth  (Wav2Lip GAN, better visual quality)
        - s3fd.pth         (S3FD face detector)

    Inference runs in a thread executor so the event loop is never blocked.
    """

    def __init__(self, model_path: str, face_detect_path: str, device: str = "cuda") -> None:
        self._model_path = model_path
        self._face_detect_path = face_detect_path
        self._device = device
        self._model = None
        self._loaded = False

    async def load(self) -> None:
        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_models)

    def _load_models(self) -> None:
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Wav2Lip weights not found: {self._model_path}")
        if not os.path.exists(self._face_detect_path):
            raise FileNotFoundError(f"Face detector weights not found: {self._face_detect_path}")

        import torch
        import cv2  # noqa: F401

        try:
            from wav2lip.models import Wav2Lip as _Wav2LipModel
        except ImportError:
            raise ImportError(
                "wav2lip package not installed. "
                "Clone https://github.com/Rudrabha/Wav2Lip and install dependencies."
            )

        checkpoint = torch.load(self._model_path, map_location=self._device)
        model = _Wav2LipModel()
        model.load_state_dict(checkpoint["state_dict"])
        self._model = model.to(self._device).eval()
        self._loaded = True
        log.info("wav2lip_loaded", model_path=self._model_path, device=self._device)

    async def generate_video(
        self,
        audio_pcm: bytes,
        reference_image: str,
        sample_rate: int = 16000,
    ) -> bytes:
        if not self._loaded:
            raise RuntimeError("Wav2LipLipSync.load() must be called before generate_video()")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._run_inference, audio_pcm, reference_image, sample_rate
        )

    def _run_inference(self, audio_pcm: bytes, reference_image: str, sample_rate: int) -> bytes:
        import cv2
        import numpy as np
        import torch

        img = cv2.imread(reference_image)
        if img is None:
            raise FileNotFoundError(f"Reference image not found: {reference_image}")
        h, w = img.shape[:2]

        audio_np = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        mel = self._audio_to_mel(audio_np, sample_rate)

        fps = _WAV2LIP_FPS
        n_frames = max(1, int(len(audio_pcm) / (2 * sample_rate) * fps))
        frames = [img.copy() for _ in range(n_frames)]

        mel_chunks = []
        mel_step = _MEL_STEP_SIZE
        for i in range(n_frames):
            start = int(i * (mel.shape[1] / n_frames))
            chunk = mel[:, max(0, start - mel_step // 2): start + mel_step // 2]
            if chunk.shape[1] < mel_step:
                chunk = np.pad(chunk, ((0, 0), (0, mel_step - chunk.shape[1])), mode="edge")
            mel_chunks.append(chunk)

        batch_size = 128
        predictions = []
        for i in range(0, n_frames, batch_size):
            mel_batch = np.array(mel_chunks[i: i + batch_size])[:, np.newaxis]
            img_batch = np.array(
                [cv2.resize(f, _FACE_CROP_SIZE) for f in frames[i: i + batch_size]]
            ).astype(np.float32) / 255.0
            img_batch = img_batch.transpose(0, 3, 1, 2)

            with torch.no_grad():
                pred = self._model(
                    torch.FloatTensor(mel_batch).to(self._device),
                    torch.FloatTensor(img_batch).to(self._device),
                )
            pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            predictions.extend(pred_np.astype(np.uint8))

        output_frames = [cv2.resize(pred, (w, h)) for pred in predictions]
        return self._encode_mp4(output_frames, fps, w, h)

    def _audio_to_mel(self, waveform, sample_rate: int):
        import numpy as np

        if sample_rate != 16000:
            new_len = int(len(waveform) * 16000 / sample_rate)
            waveform = np.interp(
                np.linspace(0, len(waveform) - 1, new_len),
                np.arange(len(waveform)),
                waveform,
            )

        n_fft, hop_length, n_mels = 800, 200, 80
        n_frames_stft = max(1, 1 + (len(waveform) - n_fft) // hop_length)
        window = np.hanning(n_fft)
        stft = np.zeros((n_fft // 2 + 1, n_frames_stft), dtype=np.float32)
        for i in range(n_frames_stft):
            frame = waveform[i * hop_length: i * hop_length + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            stft[:, i] = np.abs(np.fft.rfft(frame * window))

        mel_fb = np.linspace(0, n_fft // 2, n_mels + 2).astype(int)
        mel = np.zeros((n_mels, stft.shape[1]), dtype=np.float32)
        for m in range(n_mels):
            mel[m] = stft[mel_fb[m]: mel_fb[m + 2]].mean(axis=0)
        return np.log(np.clip(mel, 1e-5, None))

    def _encode_mp4(self, frames, fps: int, width: int, height: int) -> bytes:
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)

    async def aclose(self) -> None:
        self._model = None
        self._loaded = False
        log.info("wav2lip_closed")
