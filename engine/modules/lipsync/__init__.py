"""Avatar Agent lip-sync module — factory and public API."""
from .interface import LipSyncClient, LipSyncFrame


def create_lipsync(config) -> LipSyncClient:
    """Create a LipSyncClient from a LipSyncConfig instance.

    Args:
        config: LipSyncConfig from engine/config.py.

    Returns:
        A LipSyncClient subclass instance (not yet loaded).

    Raises:
        ValueError: If config.engine is not a recognised provider.
    """
    engine = config.engine

    if engine == "wav2lip":
        from .wav2lip import Wav2LipLipSync
        return Wav2LipLipSync(
            model_path=config.wav2lip.model_path,
            face_detect_path=config.wav2lip.face_detect_path,
            device=config.device,
        )

    if engine == "sadtalker":
        from .sadtalker import SadTalkerLipSync
        return SadTalkerLipSync(
            model_path=config.sadtalker.model_path,
            device=config.device,
        )

    raise ValueError(f"Unsupported lip-sync engine: {engine!r}. Expected 'wav2lip' or 'sadtalker'.")


__all__ = ["LipSyncClient", "LipSyncFrame", "create_lipsync"]
