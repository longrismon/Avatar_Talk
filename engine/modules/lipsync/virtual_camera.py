"""Virtual camera — inject video frames into a v4l2loopback device."""
import asyncio
import os
import platform
import subprocess
import tempfile
from typing import Optional

from engine.logging_config import get_logger

log = get_logger("lipsync.virtual_camera")


async def inject_video_frames(
    mp4_bytes: bytes,
    device: str,
    fps: int = 25,
) -> None:
    """Decode MP4 bytes and push BGR frames to a virtual camera device.

    Runs the blocking OpenCV decode/write in a thread executor.

    Args:
        mp4_bytes: Raw MP4 video bytes from a LipSyncClient.
        device: v4l2loopback device path (e.g. '/dev/video10').
        fps: Output frame rate.
    """
    if not mp4_bytes:
        log.warning("inject_video_frames_empty")
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _write_frames_sync, mp4_bytes, device, fps)


def _write_frames_sync(mp4_bytes: bytes, device: str, fps: int) -> None:
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(mp4_bytes)
        tmp_path = f.name

    frame_count = 0
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            log.error("virtual_camera_open_failed", tmp=tmp_path)
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        out = cv2.VideoWriter(
            device, backend, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )
        if not out.isOpened():
            log.error("virtual_camera_writer_failed", device=device)
            cap.release()
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        log.info("virtual_camera_inject_complete", frames=frame_count, device=device)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def setup_linux_virtual_camera(device: str = "/dev/video10") -> bool:
    """Ensure a v4l2loopback virtual camera device exists.

    Returns True if the device already exists or was successfully created.

    Raises:
        subprocess.CalledProcessError: If modprobe fails.
    """
    result = subprocess.run(["ls", device], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("virtual_camera_exists", device=device)
        return True

    video_nr = device.replace("/dev/video", "")
    subprocess.run(
        [
            "modprobe", "v4l2loopback",
            f"video_nr={video_nr}",
            "card_label=avatar_agent_cam",
            "exclusive_caps=1",
        ],
        check=True,
    )
    log.info("virtual_camera_created", device=device)
    return True


def get_virtual_camera_device(config) -> Optional[str]:
    """Return the platform-appropriate virtual camera device name from config.

    Args:
        config: VirtualDevicesConfig (engine/config.py).
    """
    if config is None:
        return None
    try:
        sys = platform.system()
        if sys == "Linux":
            return config.camera.linux.device
        elif sys == "Darwin":
            return config.camera.macos.driver
        else:
            return config.camera.windows.driver
    except AttributeError:
        return None
