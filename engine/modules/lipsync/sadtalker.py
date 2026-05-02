"""SadTalker lip-sync client — expressive head-pose driven face animation."""
import asyncio
import os
import tempfile

from .interface import LipSyncClient
from engine.logging_config import get_logger

log = get_logger("lipsync.sadtalker")


class SadTalkerLipSync(LipSyncClient):
    """SadTalker-based lip-sync: full head-pose + expression animation.

    Slower than Wav2Lip (~5–10× per second of audio) but produces more
    expressive and natural head motion.

    Model weights must be downloaded from:
        https://github.com/OpenTalker/SadTalker

    Inference runs in a thread executor so the event loop is never blocked.
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self._model_path = model_path
        self._device = device
        self._pipeline = None
        self._loaded = False

    async def load(self) -> None:
        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_pipeline)

    def _load_pipeline(self) -> None:
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"SadTalker model directory not found: {self._model_path}")

        try:
            from sadtalker.inference import SadTalker as _SadTalkerAPI  # noqa: F401
        except ImportError:
            raise ImportError(
                "SadTalker package not installed. "
                "Clone https://github.com/OpenTalker/SadTalker and add it to sys.path."
            )

        self._pipeline = {
            "model_path": self._model_path,
            "device": self._device,
        }
        self._loaded = True
        log.info("sadtalker_loaded", model_path=self._model_path, device=self._device)

    async def generate_video(
        self,
        audio_pcm: bytes,
        reference_image: str,
        sample_rate: int = 16000,
    ) -> bytes:
        if not self._loaded:
            raise RuntimeError("SadTalkerLipSync.load() must be called before generate_video()")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._run_inference, audio_pcm, reference_image, sample_rate
        )

    def _run_inference(self, audio_pcm: bytes, reference_image: str, sample_rate: int) -> bytes:
        import numpy as np

        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile not installed; required by SadTalker (pip install soundfile)")

        audio_np = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "input.wav")
            sf.write(wav_path, audio_np, sample_rate)

            from sadtalker.inference import SadTalker as _SadTalkerAPI
            api = _SadTalkerAPI(
                checkpoint_path=self._pipeline["model_path"],
                config_path=os.path.join(self._pipeline["model_path"], "config"),
                device=self._pipeline["device"],
            )
            output_path = api.test(
                source_image=reference_image,
                driven_audio=wav_path,
                result_dir=tmp_dir,
            )
            with open(output_path, "rb") as f:
                return f.read()

    async def aclose(self) -> None:
        self._pipeline = None
        self._loaded = False
        log.info("sadtalker_closed")
