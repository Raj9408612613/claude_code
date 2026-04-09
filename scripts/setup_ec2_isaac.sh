#!/usr/bin/env bash
# =============================================================================
# EC2 Isaac Sim + IsaacLab Setup Script
# =============================================================================
# Supports: Ubuntu 22.04 / 24.04, NVIDIA GPUs (A10G, RTX 6000, etc.)
# Idempotent — safe to re-run after reboot or partial failure.
#
# Usage: bash setup_ec2_isaac.sh
#
# After completion:
#   source ~/.bashrc && conda activate isaaclab
#   cd ~/claude_code && bash scripts/verify_setup.sh
#   bash scripts/smoke_test.sh
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

# ── Step 0: NVIDIA Drivers ──────────────────────────────────────────────────
echo ">>> Step 0: Checking/Installing NVIDIA drivers..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "NVIDIA drivers already installed:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "Installing NVIDIA drivers via CUDA repo..."
    # Use correct repo URL for detected Ubuntu version
    CUDA_REPO="ubuntu2404"
    if [[ "$UBUNTU_VER" == "22.04" ]]; then
        CUDA_REPO="ubuntu2204"
    fi
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y cuda-drivers
    echo "=============================================="
    echo "  NVIDIA drivers installed. REBOOT REQUIRED."
    echo "  Run: sudo reboot"
    echo "  Then re-run this script."
    echo "=============================================="
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
    rm -f /tmp/miniconda.sh
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
    pip install \
        isaacsim-rl \
        isaacsim-replicator \
        isaacsim-extscache-physics \
        isaacsim-extscache-kit-sdk
    echo "Isaac Sim installed."
fi

# Accept Isaac Sim EULA non-interactively
EULA_DIR="$HOME/.nvidia-omniverse/config"
mkdir -p "$EULA_DIR"
if [ ! -f "$EULA_DIR/eula_accepted" ]; then
    cat > "$EULA_DIR/eula_accepted" << 'EULA'
{"eula_accepted": true}
EULA
    echo "Isaac Sim EULA accepted."
fi

# ── Step 5: Install IsaacLab ────────────────────────────────────────────────
echo ">>> Step 5: Installing IsaacLab..."
ISAACLAB_DIR="$HOME/IsaacLab"

if [ ! -d "$ISAACLAB_DIR" ]; then
    echo "Cloning IsaacLab..."
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

cd "$ISAACLAB_DIR"

# Pin setuptools to avoid flat-layout error (setuptools>=75 rejects IsaacLab layout)
pip install "setuptools<75.0.0"

# Pre-install ray to avoid pip resolution-too-deep error
if ! python -c "import ray" 2>/dev/null; then
    echo "Installing ray..."
    pip install "ray[default]==2.45.0"
fi

# Install rl_games with legacy resolver (avoids ray dependency explosion)
if ! python -c "import rl_games" 2>/dev/null; then
    echo "Installing rl_games..."
    pip install --use-deprecated=legacy-resolver \
        "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11"
fi

# Install IsaacLab core with --no-deps (skips dex-retargeting which has no 3.11 wheel)
if ! python -c "import isaaclab" 2>/dev/null; then
    echo "Installing isaaclab core..."
    pip install --no-deps -e "source/isaaclab"
    # Manually install the deps we actually need
    pip install toml "gymnasium==1.2.1" trimesh einops warp-lang \
        "prettytable==3.3.0" flatdict
fi

# Install IsaacLab extensions
for ext in isaaclab_assets isaaclab_tasks isaaclab_rl; do
    if [ -d "source/$ext" ]; then
        echo "Installing $ext..."
        pip install --use-deprecated=legacy-resolver -e "source/$ext" 2>/dev/null || \
        pip install --no-deps -e "source/$ext"
    fi
done

# Extra packages for training & conversion
pip install tensorboard "imageio[ffmpeg]" mujoco usd-core 2>/dev/null || true

cd "$HOME"
echo ">>> IsaacLab installed."

# ── Step 6: Clone/Update Project Repo ───────────────────────────────────────
echo ">>> Step 6: Setting up project repo..."
REPO_DIR="$HOME/claude_code"

if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Raj9408612613/claude_code.git "$REPO_DIR"
else
    echo "Project repo already at $REPO_DIR"
fi

cd "$REPO_DIR"
# Checkout the branch with omni_spot code
git fetch origin nv-omni-spot-tr 2>/dev/null || true
git checkout nv-omni-spot-tr 2>/dev/null || true

cd "$HOME"
echo ">>> Project repo ready at $REPO_DIR (branch: nv-omni-spot-tr)"

# ── Step 7: MJCF → USD Conversion ──────────────────────────────────────────
echo ">>> Step 7: MJCF → USD conversion..."
USD_FILE="$REPO_DIR/models/spot_scene.usd"

if [ -f "$USD_FILE" ]; then
    echo "USD file already exists at $USD_FILE"
else
    echo "Attempting MJCF→USD conversion..."

    # Try Method 1: Isaac Sim SimulationApp (works on RTX GPUs like RTX 6000)
    PYTHONPATH="$REPO_DIR" python -c "
import sys
try:
    import isaacsim
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    cfg = MjcfConverterCfg(
        asset_path='$REPO_DIR/models/spot_scene.xml',
        usd_dir='$REPO_DIR/models',
        usd_file_name='spot_scene.usd',
        fix_base=False,
        import_sites=True,
        self_collision=False,
    )
    converter = MjcfConverter(cfg)
    print(f'USD saved via IsaacLab: {converter.usd_path}')
    app.close()
except Exception as e:
    print(f'IsaacLab converter failed: {e}')
    print('Falling back to mujoco + usd-core converter...')
    sys.exit(1)
" 2>/dev/null || {
        # Method 2: Lightweight mujoco + usd-core (no Isaac Sim runtime needed)
        echo "Using mujoco + usd-core fallback converter..."
        PYTHONPATH="$REPO_DIR" python -c "
import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

model = mujoco.MjModel.from_xml_path('$REPO_DIR/models/spot_scene.xml')
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)

stage = Usd.Stage.CreateNew('$REPO_DIR/models/spot_scene.usd')
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

root = UsdGeom.Xform.Define(stage, '/World')
robot = UsdGeom.Xform.Define(stage, '/World/Spot')

# Track body-to-path mapping for hierarchy
body_paths = {}
body_paths[0] = '/World/Spot'

for i in range(model.nbody):
    name = model.body(i).name or f'body_{i}'
    name = name.replace(' ', '_').replace('-', '_').replace('.', '_')

    if i == 0:
        path = '/World/Spot'
    else:
        parent_id = model.body_parentid[i]
        parent_path = body_paths.get(parent_id, '/World/Spot')
        path = f'{parent_path}/{name}'

    body_paths[i] = path
    xform = UsdGeom.Xform.Define(stage, path)

    pos = model.body_pos[i]
    quat = model.body_quat[i]  # w,x,y,z
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xform.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

# Add geoms
for g in range(model.ngeom):
    geom = model.geom(g)
    body_id = geom.bodyid
    geom_name = geom.name or f'geom_{g}'
    geom_name = geom_name.replace(' ', '_').replace('-', '_').replace('.', '_')
    body_path = body_paths.get(body_id, '/World/Spot')
    geom_path = f'{body_path}/{geom_name}'

    if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
        prim = UsdGeom.Cube.Define(stage, geom_path)
        prim.GetSizeAttr().Set(2.0)
        sz = geom.size
        UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(float(sz[0]), float(sz[1]), float(sz[2])))
    elif geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
        prim = UsdGeom.Sphere.Define(stage, geom_path)
        prim.GetRadiusAttr().Set(float(geom.size[0]))
    elif geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        prim = UsdGeom.Capsule.Define(stage, geom_path)
        prim.GetRadiusAttr().Set(float(geom.size[0]))
        prim.GetHeightAttr().Set(float(geom.size[1]) * 2)
    elif geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        prim = UsdGeom.Cylinder.Define(stage, geom_path)
        prim.GetRadiusAttr().Set(float(geom.size[0]))
        prim.GetHeightAttr().Set(float(geom.size[1]) * 2)
    elif geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = geom.dataid
        prim = UsdGeom.Mesh.Define(stage, geom_path)
        vs = model.mesh_vertadr[mesh_id]
        vn = model.mesh_vertnum[mesh_id]
        verts = model.mesh_vert[vs:vs+vn]
        fs = model.mesh_faceadr[mesh_id]
        fn = model.mesh_facenum[mesh_id]
        faces = model.mesh_face[fs:fs+fn]
        prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in verts.tolist()])
        prim.GetFaceVertexCountsAttr().Set([3] * fn)
        prim.GetFaceVertexIndicesAttr().Set(faces.flatten().tolist())
    elif geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
        prim = UsdGeom.Mesh.Define(stage, geom_path)
        s = 10.0
        prim.GetPointsAttr().Set([Gf.Vec3f(-s,-s,0), Gf.Vec3f(s,-s,0), Gf.Vec3f(s,s,0), Gf.Vec3f(-s,s,0)])
        prim.GetFaceVertexCountsAttr().Set([4])
        prim.GetFaceVertexIndicesAttr().Set([0,1,2,3])
    else:
        continue

    gp = geom.pos
    gq = geom.quat
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(geom_path))
    xf.AddTranslateOp(opSuffix='local').Set(Gf.Vec3d(float(gp[0]), float(gp[1]), float(gp[2])))
    xf.AddOrientOp(opSuffix='local').Set(Gf.Quatf(float(gq[0]), float(gq[1]), float(gq[2]), float(gq[3])))

# Add joints
for j in range(model.njnt):
    jnt = model.jnt(j)
    jnt_name = jnt.name or f'joint_{j}'
    jnt_name = jnt_name.replace(' ', '_').replace('-', '_').replace('.', '_')
    body_id = jnt.bodyid
    body_path = body_paths.get(body_id, '/World/Spot')
    jnt_path = f'{body_path}/{jnt_name}'

    if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
        rev = UsdPhysics.RevoluteJoint.Define(stage, jnt_path)
        axis = jnt.axis
        if abs(axis[0]) > 0.5:
            rev.GetAxisAttr().Set('X')
        elif abs(axis[1]) > 0.5:
            rev.GetAxisAttr().Set('Y')
        else:
            rev.GetAxisAttr().Set('Z')
        if jnt.limited:
            lo = float(np.degrees(jnt.range[0]))
            hi = float(np.degrees(jnt.range[1]))
            rev.GetLowerLimitAttr().Set(lo)
            rev.GetUpperLimitAttr().Set(hi)

stage.GetRootLayer().Save()
print(f'USD saved: $REPO_DIR/models/spot_scene.usd')
print(f'Bodies: {model.nbody}, Geoms: {model.ngeom}, Joints: {model.njnt}, Meshes: {model.nmesh}')
"
    }

    if [ -f "$USD_FILE" ]; then
        echo ">>> USD conversion successful!"
    else
        echo ">>> WARNING: USD conversion failed. You can retry manually later."
    fi
fi

# ── Step 8: Docker + NVIDIA Container Toolkit (optional, for full Isaac Sim) ─
echo ">>> Step 8: Installing Docker + NVIDIA Container Toolkit..."
if command -v docker &>/dev/null; then
    echo "Docker already installed."
else
    sudo apt-get install -y docker.io
    sudo usermod -aG docker "$USER"
    sudo systemctl start docker
    sudo systemctl enable docker
    echo "Docker installed."
fi

if dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    echo "NVIDIA Container Toolkit already installed."
else
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

# ── Step 9: Quick Validation ────────────────────────────────────────────────
echo ""
echo ">>> Step 9: Quick validation..."
echo -n "  Python: "; python --version
echo -n "  PyTorch CUDA: "; python -c "import torch; print(torch.cuda.is_available())"
echo -n "  IsaacLab: "; python -c "import isaaclab; print('OK')" 2>/dev/null || echo "FAIL"
echo -n "  Isaac Sim: "; python -c "import isaacsim; print('OK')" 2>/dev/null || echo "FAIL"
echo -n "  Mock training: "
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

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Setup complete at $(date)"
echo "=============================================="
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
if [ -f "$REPO_DIR/models/spot_scene.usd" ]; then
    echo "  USD model: $REPO_DIR/models/spot_scene.usd [READY]"
else
    echo "  USD model: NOT YET GENERATED (see Step 7 in log)"
fi
echo ""
echo "Next steps:"
echo "  1. source ~/.bashrc"
echo "  2. conda activate isaaclab"
echo "  3. cd ~/claude_code && bash scripts/verify_setup.sh"
echo "  4. bash scripts/smoke_test.sh"
echo ""
echo "Log saved to: $LOG_FILE"
