#!/usr/bin/env bash
# =============================================================================
# EC2 Isaac Sim + IsaacLab Setup Script (Ubuntu 24.04, NVIDIA A10G)
# =============================================================================
# Idempotent — safe to re-run after reboot or partial failure.
#
# Usage: bash setup_ec2_isaac.sh
# =============================================================================
set -euo pipefail

LOG_FILE="$HOME/setup_isaac.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Setup started at $(date) ==="

# ── Detect Ubuntu version ────────────────────────────────────────────────────
source /etc/os-release 2>/dev/null || true
UBUNTU_VER="${VERSION_ID:-unknown}"
echo ">>> Detected Ubuntu $UBUNTU_VER ($VERSION_CODENAME)"

# ── Step 0: NVIDIA Drivers ──────────────────────────────────────────────────
echo ">>> Step 0: Checking/Installing NVIDIA drivers..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "NVIDIA drivers already installed:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "Installing NVIDIA drivers via CUDA repo..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y cuda-drivers
    echo ">>> NVIDIA drivers installed. REBOOT REQUIRED."
    echo ">>> Run: sudo reboot"
    echo ">>> Then re-run this script."
    exit 0
fi

# ── Step 1: System Dependencies ─────────────────────────────────────────────
echo ">>> Step 1: Installing system dependencies..."
sudo apt-get update -qq

# Ubuntu 24.04 replaced libgl1-mesa-glx with libgl1
if [[ "$UBUNTU_VER" == "24.04" ]]; then
    GL_PKG="libgl1"
else
    GL_PKG="libgl1-mesa-glx"
fi

sudo apt-get install -y \
    build-essential git curl wget unzip \
    "$GL_PKG" libglib2.0-0 libsm6 libxrender1 libxext6 \
    libxkbcommon0 libvulkan1 vulkan-tools \
    mesa-utils xdg-utils

echo ">>> System dependencies installed."

# ── Step 2: Miniconda ───────────────────────────────────────────────────────
echo ">>> Step 2: Installing Miniconda..."
CONDA_DIR="$HOME/miniconda3"
if [ -d "$CONDA_DIR" ]; then
    echo "Miniconda already installed at $CONDA_DIR"
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "Miniconda installed."
fi

# Initialize conda for this shell session
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
$CONDA_DIR/bin/conda init bash 2>/dev/null || true

# Accept Conda ToS (required for non-interactive use)
echo ">>> Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo ">>> Miniconda ready."

# ── Step 3: Create Conda Environment ────────────────────────────────────────
echo ">>> Step 3: Creating isaaclab conda environment (Python 3.11)..."
ENV_NAME="isaaclab"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists."
else
    conda create -n "$ENV_NAME" python=3.11 -y
    echo "Conda env '$ENV_NAME' created."
fi

conda activate "$ENV_NAME"
echo ">>> Python: $(python --version)"

# ── Step 4: Install Isaac Sim (pip) ─────────────────────────────────────────
echo ">>> Step 4: Installing Isaac Sim 5.x via pip..."
if python -c "import isaacsim" 2>/dev/null; then
    echo "Isaac Sim already installed."
else
    pip install --quiet \
        isaacsim-rl \
        isaacsim-replicator \
        isaacsim-extscache-physics \
        isaacsim-extscache-kit-sdk
    echo "Isaac Sim installed."
fi

# ── Step 5: Install IsaacLab ────────────────────────────────────────────────
echo ">>> Step 5: Installing IsaacLab..."
ISAACLAB_DIR="$HOME/IsaacLab"

if [ ! -d "$ISAACLAB_DIR" ]; then
    echo "Cloning IsaacLab..."
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

cd "$ISAACLAB_DIR"

# Pin setuptools to avoid flat-layout error
pip install --quiet "setuptools<75.0.0"

# Pre-install ray to avoid resolution-too-deep
if ! python -c "import ray" 2>/dev/null; then
    echo "Installing ray..."
    pip install --quiet "ray[default]==2.45.0"
fi

# Install rl_games with legacy resolver
if ! python -c "import rl_games" 2>/dev/null; then
    echo "Installing rl_games..."
    pip install --quiet --use-deprecated=legacy-resolver \
        "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11"
fi

# Install IsaacLab core (--no-deps to skip dex-retargeting)
if ! python -c "import isaaclab" 2>/dev/null; then
    echo "Installing isaaclab core..."
    pip install --quiet --no-deps -e "source/isaaclab"
    pip install --quiet toml "gymnasium==1.2.1" trimesh einops warp-lang \
        "prettytable==3.3.0" flatdict
fi

# Install IsaacLab extensions
for ext in isaaclab_assets isaaclab_tasks isaaclab_rl; do
    if [ -d "source/$ext" ]; then
        echo "Installing $ext..."
        pip install --quiet --use-deprecated=legacy-resolver -e "source/$ext" 2>/dev/null || \
        pip install --quiet --no-deps -e "source/$ext"
    fi
done

# Extra packages for training
pip install --quiet tensorboard "imageio[ffmpeg]" 2>/dev/null || true

cd "$HOME"
echo ">>> IsaacLab installed."

# ── Step 6: Clone/Update Project Repo ───────────────────────────────────────
echo ">>> Step 6: Setting up project repo..."
REPO_DIR="$HOME/claude_code"

if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Raj9408612613/claude_code.git "$REPO_DIR"
else
    echo "Project repo already at $REPO_DIR"
    cd "$REPO_DIR" && git pull origin main 2>/dev/null || true
fi

cd "$HOME"
echo ">>> Project repo ready."

# ── Step 7: NICE DCV (Remote Desktop) ──────────────────────────────────────
echo ">>> Step 7: Checking NICE DCV..."
if command -v dcv &>/dev/null; then
    echo "DCV already installed."
    sudo systemctl enable dcvserver 2>/dev/null || true
    sudo systemctl start dcvserver 2>/dev/null || true
    sudo dcv create-session --owner ubuntu --type virtual my-session 2>/dev/null || true
else
    echo "DCV not installed. Skipping (optional for headless training)."
    echo "To install DCV for remote desktop, see: https://docs.aws.amazon.com/dcv/"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Setup complete at $(date)"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. source ~/.bashrc"
echo "  2. conda activate isaaclab"
echo "  3. cd ~/claude_code && bash scripts/verify_setup.sh"
echo "  4. bash scripts/smoke_test.sh"
echo ""
echo "Log saved to: $LOG_FILE"
