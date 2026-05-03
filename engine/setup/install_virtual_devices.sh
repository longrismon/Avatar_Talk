#!/usr/bin/env bash
# install_virtual_devices.sh — one-command setup for Avatar Agent virtual devices
#
# Linux:   creates a PulseAudio null-sink virtual mic + loads v4l2loopback virtual camera
# macOS:   prints instructions for BlackHole (audio) and OBS Virtual Camera (video)
# Windows: prints instructions for VB-Cable (audio) and OBS Virtual Camera (video)
#
# Usage:
#   chmod +x engine/setup/install_virtual_devices.sh
#   ./engine/setup/install_virtual_devices.sh
#   ./engine/setup/install_virtual_devices.sh --uninstall

set -euo pipefail

SINK_NAME="avatar_agent_mic"
V4L2_DEVICE=10
UNINSTALL=false

for arg in "$@"; do
  case "$arg" in
    --uninstall) UNINSTALL=true ;;
  esac
done

OS="$(uname -s)"

# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------

linux_install_audio() {
  echo ">>> [Audio] Loading PulseAudio null-sink: ${SINK_NAME}"
  if pactl list sinks short 2>/dev/null | grep -q "${SINK_NAME}"; then
    echo "    Already loaded — skipping."
  else
    pactl load-module module-null-sink sink_name="${SINK_NAME}" \
      sink_properties="device.description='Avatar Agent Virtual Mic'"
    echo "    Loaded. Sink: ${SINK_NAME}"
  fi

  echo ">>> [Audio] Persisting via /etc/pulse/default.pa"
  ENTRY="load-module module-null-sink sink_name=${SINK_NAME} sink_properties=device.description=Avatar_Agent_Virtual_Mic"
  if ! grep -q "${SINK_NAME}" /etc/pulse/default.pa 2>/dev/null; then
    echo "${ENTRY}" | sudo tee -a /etc/pulse/default.pa > /dev/null
    echo "    Added to default.pa"
  else
    echo "    Already in default.pa — skipping."
  fi
}

linux_uninstall_audio() {
  echo ">>> [Audio] Unloading PulseAudio null-sink: ${SINK_NAME}"
  MODULE_ID=$(pactl list modules short 2>/dev/null | grep "${SINK_NAME}" | awk '{print $1}' | head -1)
  if [ -n "${MODULE_ID}" ]; then
    pactl unload-module "${MODULE_ID}"
    echo "    Unloaded module ${MODULE_ID}"
  else
    echo "    Not loaded — skipping."
  fi
  if [ -f /etc/pulse/default.pa ]; then
    sudo sed -i "/${SINK_NAME}/d" /etc/pulse/default.pa
    echo "    Removed from default.pa"
  fi
}

linux_install_camera() {
  echo ">>> [Camera] Installing v4l2loopback kernel module"
  if ! dpkg -l v4l2loopback-dkms &>/dev/null && ! modinfo v4l2loopback &>/dev/null; then
    echo "    v4l2loopback not found — installing via apt..."
    sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils
  else
    echo "    v4l2loopback already installed."
  fi

  echo ">>> [Camera] Loading v4l2loopback at /dev/video${V4L2_DEVICE}"
  if ls /dev/video"${V4L2_DEVICE}" &>/dev/null; then
    echo "    /dev/video${V4L2_DEVICE} already exists — skipping modprobe."
  else
    sudo modprobe v4l2loopback video_nr="${V4L2_DEVICE}" card_label="Avatar Agent Camera" exclusive_caps=1
    echo "    Loaded /dev/video${V4L2_DEVICE}"
  fi

  echo ">>> [Camera] Persisting via /etc/modules-load.d"
  if [ ! -f /etc/modules-load.d/v4l2loopback.conf ]; then
    echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf > /dev/null
  fi
  OPTS="options v4l2loopback video_nr=${V4L2_DEVICE} card_label=Avatar_Agent_Camera exclusive_caps=1"
  if [ ! -f /etc/modprobe.d/v4l2loopback.conf ]; then
    echo "${OPTS}" | sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null
    echo "    Persisted to /etc/modprobe.d/v4l2loopback.conf"
  else
    echo "    /etc/modprobe.d/v4l2loopback.conf already exists — skipping."
  fi
}

linux_uninstall_camera() {
  echo ">>> [Camera] Unloading v4l2loopback"
  if lsmod | grep -q v4l2loopback; then
    sudo modprobe -r v4l2loopback || echo "    Cannot remove — device may be in use."
  else
    echo "    Not loaded — skipping."
  fi
  sudo rm -f /etc/modules-load.d/v4l2loopback.conf /etc/modprobe.d/v4l2loopback.conf
  echo "    Removed persistence files."
}

# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------

macos_instructions() {
  cat <<'EOF'
>>> macOS virtual device setup

[Audio — BlackHole]
1. Install BlackHole (2-channel is sufficient):
     brew install blackhole-2ch
   Or download from https://existential.audio/blackhole/
2. Open Audio MIDI Setup, create a Multi-Output Device that includes
   your speakers + BlackHole 2ch as the secondary output.
3. In avatar_talk config.yaml, set:
     virtual_devices.audio.macos.driver: blackhole

[Camera — OBS Virtual Camera]
1. Install OBS Studio:  brew install --cask obs
2. In OBS → Tools → Virtual Camera → Start Virtual Camera.
3. In avatar_talk config.yaml, set:
     virtual_devices.camera.macos.driver: obs

Verify:
  python -c "import sounddevice; print(sounddevice.query_devices())"
  # Look for "BlackHole 2ch" in the output
EOF
}

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

windows_instructions() {
  cat <<'EOF'
>>> Windows virtual device setup

[Audio — VB-Cable]
1. Download VB-CABLE from https://vb-audio.com/Cable/
2. Run the installer as Administrator, reboot.
3. In avatar_talk config.yaml, set:
     virtual_devices.audio.windows.driver: vb-cable

[Camera — OBS Virtual Camera]
1. Install OBS Studio from https://obsproject.com/
2. In OBS → Tools → Virtual Camera → Start Virtual Camera.
3. In avatar_talk config.yaml, set:
     virtual_devices.camera.windows.driver: obs

Verify:
  python -c "import sounddevice; print(sounddevice.query_devices())"
  # Look for "CABLE Output (VB-Audio Virtual Cable)" in the output
EOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "======================================================"
echo " Avatar Agent — Virtual Device Installer"
echo " OS: ${OS}  |  uninstall=${UNINSTALL}"
echo "======================================================"
echo ""

case "${OS}" in
  Linux)
    if ${UNINSTALL}; then
      linux_uninstall_audio
      linux_uninstall_camera
    else
      linux_install_audio
      linux_install_camera
      echo ""
      echo "======================================================"
      echo " Done! Update engine/config.yaml if needed:"
      echo "   virtual_devices.audio.linux.sink_name: ${SINK_NAME}"
      echo "   virtual_devices.camera.linux.device: /dev/video${V4L2_DEVICE}"
      echo "======================================================"
    fi
    ;;
  Darwin)
    macos_instructions
    ;;
  MINGW*|CYGWIN*|MSYS*|Windows_NT)
    windows_instructions
    ;;
  *)
    echo "Unsupported OS: ${OS}" >&2
    exit 1
    ;;
esac
