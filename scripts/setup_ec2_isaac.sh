#!/usr/bin/env bash
# =============================================================================
# EC2 Isaac Sim + IsaacLab Setup — Docker Container Approach
# =============================================================================
# Supports: Ubuntu 22.04 / 24.04, NVIDIA GPUs (RTX 6000, A10G, etc.)
# Idempotent — safe to re-run after reboot or partial failure.
#
# Architecture:
#   Host:      NVIDIA drivers, CUDA, Docker, conda (for mock env testing)
#   Container: Isaac Sim + IsaacLab + Omniverse (for full training & USD conversion)
#
# Usage: bash setup_ec2_isaac.sh
#
# After completion:
#   source ~/.bashrc && conda activate isaaclab
#   cd ~/claude_code
#   bash scripts/verify_setup.sh          # Verify all components
#   bash scripts/smoke_test.sh            # Mock env test (host)
#   bash scripts/smoke_test.sh --full     # Full Isaac Lab test (container)
#   bash scripts/isaac_run.sh             # Interactive container shell
# =============================================================================
set -euo pipefail

LOG_FILE="$HOME/setup_isaac.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Setup started at $(date) ==="

# ── Detect Ubuntu version ────────────────────────────────────────────────────
source /etc/os-release 2>/dev/null || true
UBUNTU_VER="${VERSION_ID:-unknown}"
UBUNTU_CODENAME="${VERSION_CODENAME:-unknown}"
echo ">>> Detected Ubuntu $UBUNTU_VER ($UBUNTU_CODENAME)"

# Determine CUDA repo name
CUDA_REPO="ubuntu2404"
if [[ "$UBUNTU_VER" == "22.04" ]]; then
    CUDA_REPO="ubuntu2204"
fi

# =============================================================================
# STEP 0: NVIDIA Drivers + CUDA Toolkit
# =============================================================================
echo ">>> Step 0: Checking/Installing NVIDIA drivers + CUDA..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "NVIDIA drivers already installed:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "Installing NVIDIA drivers + CUDA toolkit..."
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y cuda-drivers cuda-toolkit
    echo "=============================================="
    echo "  NVIDIA drivers + CUDA installed."
    echo "  REBOOT REQUIRED."
    echo "  Run: sudo reboot"
    echo "  Then re-run this script."
    echo "=============================================="
    exit 0
fi

# Install full CUDA toolkit if not present (needed for GPU containers)
if ! command -v nvcc &>/dev/null; then
    echo "Installing CUDA toolkit..."
    if ! grep -q "cuda" /etc/apt/sources.list.d/*.list 2>/dev/null; then
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update -qq
    fi
    sudo apt-get install -y cuda-toolkit
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> "$HOME/.bashrc"
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> "$HOME/.bashrc"
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
    echo "CUDA toolkit installed."
fi

# =============================================================================
# STEP 1: System Dependencies
# =============================================================================
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
    mesa-utils xdg-utils ca-certificates gnupg lsb-release

echo ">>> System dependencies installed."

# =============================================================================
# STEP 2: Docker + NVIDIA Container Toolkit
# =============================================================================
echo ">>> Step 2: Installing Docker + NVIDIA Container Toolkit..."

# Docker
if command -v docker &>/dev/null; then
    echo "Docker already installed."
else
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    echo "Docker installed."
fi

# Add user to docker group (avoids needing sudo for docker commands)
if ! groups "$USER" | grep -q docker; then
    sudo usermod -aG docker "$USER"
    echo "Added $USER to docker group (will take effect on next login)."
fi

# NVIDIA Container Toolkit
if dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    echo "NVIDIA Container Toolkit already installed."
else
    echo "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed."
fi

# Verify GPU is visible in Docker
echo "Verifying GPU access in Docker..."
if sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "Docker GPU access: OK"
else
    echo "WARNING: Docker GPU access failed. Continuing anyway..."
fi

# =============================================================================
# STEP 3: Clone Project Repo
# =============================================================================
echo ">>> Step 3: Setting up project repo..."
REPO_DIR="$HOME/claude_code"

if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Raj9408612613/claude_code.git "$REPO_DIR"
else
    echo "Project repo already at $REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin nv-omni-spot-tr 2>/dev/null || true
git checkout nv-omni-spot-tr 2>/dev/null || true
cd "$HOME"
echo ">>> Project repo ready at $REPO_DIR (branch: nv-omni-spot-tr)"

# =============================================================================
# STEP 4: Pull Isaac Sim Container (~15GB)
# =============================================================================
echo ">>> Step 4: Pulling Isaac Sim Docker container..."
ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim:4.5.0"

if sudo docker image inspect "$ISAAC_IMAGE" &>/dev/null; then
    echo "Isaac Sim container already pulled."
else
    echo "Pulling Isaac Sim container (~15GB, this will take a while)..."
    sudo docker pull "$ISAAC_IMAGE"
    echo "Isaac Sim container pulled."
fi

# =============================================================================
# STEP 5: Build Custom Image (Isaac Sim + IsaacLab + project deps)
# =============================================================================
echo ">>> Step 5: Building isaac-lab-spot Docker image..."
CUSTOM_IMAGE="isaac-lab-spot:latest"

if sudo docker image inspect "$CUSTOM_IMAGE" &>/dev/null; then
    echo "Custom image '$CUSTOM_IMAGE' already exists."
    echo "  To rebuild: sudo docker rmi $CUSTOM_IMAGE && re-run this script"
else
    echo "Building custom Docker image with IsaacLab..."
    cd "$REPO_DIR"
    sudo docker build -t "$CUSTOM_IMAGE" -f Dockerfile.isaac .
    cd "$HOME"
    echo "Custom image built: $CUSTOM_IMAGE"
fi

# =============================================================================
# STEP 6: MJCF → USD Conversion (inside container)
# =============================================================================
echo ">>> Step 6: MJCF → USD conversion..."
USD_FILE="$REPO_DIR/models/spot_scene.usd"

if [ -f "$USD_FILE" ]; then
    echo "USD file already exists at $USD_FILE"
else
    echo "Converting MJCF → USD inside Isaac Sim container..."
    sudo docker run --rm --gpus all \
        -e "ACCEPT_EULA=Y" \
        -v "$REPO_DIR":/workspace \
        "$CUSTOM_IMAGE" \
        /isaac-sim/python.sh -c "
import sys
try:
    import isaacsim
    from omni.isaac.kit import SimulationApp
    app = SimulationApp({'headless': True})

    # Try the MJCF importer extension
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate('omni.importer.mjcf', True)

    from omni.importer.mjcf import _mjcf
    importer = _mjcf.acquire_mjcf_interface()

    import_config = _mjcf.ImportConfig()
    import_config.fix_base = False
    import_config.import_sites = True
    import_config.self_collision = False

    importer.create_asset_from_mjcf(
        '/workspace/models/spot_scene.xml',
        '/workspace/models',
        'spot_scene',
        import_config
    )
    print('USD saved via Isaac Sim MJCF importer')
    app.close()
except Exception as e:
    print(f'Isaac Sim MJCF importer failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1 || {
        echo "Isaac Sim importer failed. Trying IsaacLab MjcfConverter..."
        sudo docker run --rm --gpus all \
            -e "ACCEPT_EULA=Y" \
            -v "$REPO_DIR":/workspace \
            "$CUSTOM_IMAGE" \
            /isaac-sim/python.sh -c "
import sys
try:
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    cfg = MjcfConverterCfg(
        asset_path='/workspace/models/spot_scene.xml',
        usd_dir='/workspace/models',
        usd_file_name='spot_scene.usd',
        fix_base=False,
        import_sites=True,
        self_collision=False,
    )
    converter = MjcfConverter(cfg)
    print(f'USD saved via IsaacLab: {converter.usd_path}')
    app.close()
except Exception as e:
    print(f'IsaacLab converter also failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1
    }

    if [ -f "$USD_FILE" ]; then
        echo ">>> USD conversion successful: $USD_FILE"
    else
        echo ">>> WARNING: USD conversion failed."
        echo "    The container-based conversion requires a GPU with full RTX support."
        echo "    You can retry manually: bash scripts/isaac_run.sh"
    fi
fi

# =============================================================================
# STEP 7: Conda Environment (for host-side mock testing)
# =============================================================================
echo ">>> Step 7: Installing Miniconda + host Python environment..."
CONDA_DIR="$HOME/miniconda3"
if [ -d "$CONDA_DIR" ]; then
    echo "Miniconda already installed at $CONDA_DIR"
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm -f /tmp/miniconda.sh
    echo "Miniconda installed."
fi

eval "$($CONDA_DIR/bin/conda shell.bash hook)"
$CONDA_DIR/bin/conda init bash 2>/dev/null || true

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

ENV_NAME="isaaclab"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists."
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"

# Install PyTorch + training deps on host (for mock env testing without container)
pip install torch torchvision torchaudio tensorboard "imageio[ffmpeg]" 2>/dev/null || true

echo ">>> Host Python environment ready: $(python --version)"

# =============================================================================
# STEP 8: Quick Validation
# =============================================================================
echo ""
echo ">>> Step 8: Quick validation..."
echo ""
echo "--- Host ---"
echo -n "  NVIDIA driver: "; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "FAIL"
echo -n "  CUDA toolkit:  "; nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' || echo "not installed (OK — using container)"
echo -n "  Python:         "; python --version 2>/dev/null || echo "FAIL"
echo -n "  PyTorch CUDA:   "; python -c "import torch; print(f'{torch.__version__}, CUDA={torch.cuda.is_available()}')" 2>/dev/null || echo "FAIL"
echo -n "  Mock training:  "
PYTHONPATH="$REPO_DIR" python -c "
from omni_spot.mock_env import MockSpotEnv
from omni_spot.ppo import PPOTrainer
env = MockSpotEnv(num_envs=4, device='cuda')
trainer = PPOTrainer(n_envs=4, n_steps=2, lr=3e-4, device='cuda')
obs, _ = env.reset()
obs, batch, stats = trainer.collect_rollout(env, obs)
trainer.update(batch)
print(f'OK (rew={stats[\"rew_mean\"]:.3f})')
" 2>/dev/null || echo "FAIL"

echo ""
echo "--- Container ---"
echo -n "  Docker:          "; docker --version 2>/dev/null | awk '{print $3}' || echo "FAIL"
echo -n "  GPU in Docker:   "; sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "FAIL"
echo -n "  Isaac Sim image: "; sudo docker image inspect "$ISAAC_IMAGE" &>/dev/null && echo "OK" || echo "NOT PULLED"
echo -n "  Custom image:    "; sudo docker image inspect "$CUSTOM_IMAGE" &>/dev/null && echo "OK" || echo "NOT BUILT"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "  Setup complete at $(date)"
echo "=============================================="
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "Docker images:"
sudo docker images --format "  {{.Repository}}:{{.Tag}}  {{.Size}}" | grep -E "isaac|nvidia" || true
echo ""
if [ -f "$USD_FILE" ]; then
    echo "  USD model: $USD_FILE [READY]"
else
    echo "  USD model: NOT YET GENERATED"
    echo "  Retry: bash scripts/isaac_run.sh  (then convert inside container)"
fi
echo ""
echo "Usage:"
echo "  source ~/.bashrc && conda activate isaaclab"
echo ""
echo "  # Host (mock env, no Omniverse):"
echo "  cd ~/claude_code && bash scripts/smoke_test.sh"
echo ""
echo "  # Container (full Isaac Lab + Omniverse):"
echo "  bash scripts/isaac_run.sh                                    # interactive shell"
echo "  bash scripts/isaac_run.sh python -m omni_spot.train --num_envs 64  # training"
echo "  bash scripts/smoke_test.sh --full                            # full smoke test"
echo ""
echo "Log saved to: $LOG_FILE"
