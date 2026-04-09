Here's the full context file — copy and save it as `AGENT_CONTEXT.md` or paste it into your next Claude Code session:

---

```markdown
# EC2 Isaac Sim + IsaacLab Setup — Full Context

## Project Overview
- **Repo**: https://github.com/Raj9408612613/claude_code.git
- **Branch**: claude/review-isaac-lab-compatibility-Q1vx1
- **Goal**: Train a Spot robot in Isaac Lab (PhysX 5 GPU physics) using PPO
- **Key directory**: `omni_spot/` — training code, env config, MJCF→USD converter

### omni_spot/ structure
```
omni_spot/
├── __init__.py
├── config.py                  # Hyperparameters & constants
├── convert_mjcf_to_usd.py    # MJCF→USD converter (needs Omniverse Kit runtime)
├── train.py                   # Training entry point (Isaac Lab DirectRLEnv)
├── spot_env.py                # DirectRLEnv implementation
├── spot_env_cfg.py            # Isaac Lab environment config
├── spot_actor_critic.py       # PyTorch Actor-Critic network
├── ppo.py                     # PyTorch PPO trainer
├── reward.py                  # Reward computation
├── diagnostics.py             # Training diagnostics
├── video_recorder.py          # Video recording
├── physics_tuning.py          # Physics parameter tuning
└── mock_env.py                # Pure PyTorch mock env (no Omniverse needed)
```

### Model files (in repo root)
```
models/
├── spot_scene.xml             # MJCF model of Spot robot
└── assets/                    # 23 OBJ mesh files for Spot
```
- **No spot_scene.usd exists yet** — must be generated via MJCF conversion

---

## All Known Issues & Fixes (Discovered Through Debugging)

### 1. Python Version Incompatibility
- **Problem**: Conda defaults to Python 3.13. Isaac Sim has NO wheels for 3.13.
- Isaac Sim 4.x → Python 3.10, 5.x → Python 3.11, 6.0 → Python 3.12
- **Fix**: `conda create -n isaaclab python=3.11 -y`

### 2. Conda Terms of Service
- **Problem**: `conda create` fails with `CondaToSNonInteractiveError`
- **Fix**:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 3. pip resolution-too-deep Error
- **Problem**: `./isaaclab.sh --install` fails. Root cause: `rl-games` requires `ray>=2.45.0` which has a massive dep tree causing combinatorial explosion.
- **Fix**: Pre-install ray, use legacy resolver for rl_games
```bash
pip install "ray[default]==2.45.0"
pip install --use-deprecated=legacy-resolver \
    "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11"
```

### 4. setuptools Flat-Layout Error
- **Problem**: setuptools>=75 rejects IsaacLab's multi-package layout
- **Fix**: `pip install "setuptools<75.0.0"`

### 5. dex-retargeting==0.4.6 Not Available
- **Problem**: No wheel for Python 3.11. It's for hand manipulation — NOT needed for Spot.
- **Fix**: Install isaaclab with `--no-deps`, then manually install needed deps
```bash
pip install --no-deps -e "source/isaaclab"
pip install toml gymnasium==1.2.1 trimesh einops warp-lang prettytable==3.3.0 flatdict
```

### 6. Missing toml Module
- **Problem**: `import isaaclab` fails because `--no-deps` skipped toml
- **Fix**: Included in manual dep install above

### 7. IsaacLab Source Layout (`~/IsaacLab/source/`)
Install in this order:
1. `source/isaaclab` — core (use `--no-deps`)
2. `source/isaaclab_assets`
3. `source/isaaclab_tasks`
4. `source/isaaclab_rl`
5. `source/isaaclab_contrib` (optional)
6. `source/isaaclab_mimic` (optional)

### 8. MJCF→USD Conversion (UNRESOLVED)
- `convert_mjcf_to_usd.py` requires full Omniverse Kit runtime (`omni.kit.app`), NOT available in pip-installed Isaac Sim
- **Next step to try**: IsaacLab's built-in converter:
```python
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
cfg = MjcfConverterCfg(
    asset_path='models/spot_scene.xml',
    usd_dir='models',
    usd_file_name='spot_scene.usd',
    fix_base=False, import_sites=True, self_collision=False,
)
converter = MjcfConverter(cfg)
```
- If that also needs Kit runtime, alternatives:
  1. Install Isaac Sim via Omniverse Launcher (not pip)
  2. Pre-generate USD and commit to repo
  3. Modify spot_env_cfg.py to load MJCF directly

### 9. omni_spot Module Not Found
- **Problem**: Not an installed package
- **Fix**: `PYTHONPATH=. python -m omni_spot.train ...` from `~/claude_code`

### 10. pip Dependency Warnings (NOT errors)
- After `--no-deps`, pip warns about missing optional deps (dex-retargeting, transformers, pytest, etc.)
- **These are warnings, not errors.** Not needed for Spot training.

---

## Verified Working Install Order

```bash
# 0. Prerequisites: Ubuntu 22.04 EC2 with NVIDIA GPU drivers installed

# 1. System deps + NICE DCV
sudo apt update -y
sudo apt install -y build-essential git wget curl unzip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ubuntu-desktop gdm3
# DCV install (see DCV section below)

# 2. Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init

# 3. Accept ToS + create env
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n isaaclab python=3.11 -y
conda activate isaaclab

# 4. Isaac Sim (takes a while)
pip install --upgrade pip setuptools wheel
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk

# 5. IsaacLab
cd ~ && git clone https://github.com/isaac-sim/IsaacLab.git && cd ~/IsaacLab
pip install "ray[default]==2.45.0"
pip install "setuptools<75.0.0"
pip install --use-deprecated=legacy-resolver \
    "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11"
pip install --no-deps -e "source/isaaclab"
pip install toml gymnasium==1.2.1 trimesh einops warp-lang prettytable==3.3.0 flatdict
pip install --use-deprecated=legacy-resolver -e "source/isaaclab_assets"
pip install --use-deprecated=legacy-resolver -e "source/isaaclab_tasks"
pip install --use-deprecated=legacy-resolver -e "source/isaaclab_rl"
pip install tensorboard "imageio[ffmpeg]"

# 6. Verify
python -c "import isaaclab; print('IsaacLab OK')"
python -c "import isaacsim; print('Isaac Sim OK')"

# 7. Clone project
cd ~ && git clone https://github.com/Raj9408612613/claude_code.git
cd ~/claude_code && git checkout claude/review-isaac-lab-compatibility-Q1vx1

# 8. Convert MJCF→USD (TBD — see issue #8 above)

# 9. Smoke test
cd ~/claude_code
PYTHONPATH=. python -m omni_spot.train --num_envs 64 --n_steps 128 --total_updates 5
```

## NICE DCV Setup
```bash
wget https://d1uj6qtbmh3dt5.cloudfront.net/nice-dcv-ubuntu2204-x86_64.tgz
tar -xzf nice-dcv-ubuntu2204-x86_64.tgz && cd nice-dcv-*-x86_64/
sudo apt install -y ./nice-dcv-server_*.deb ./nice-dcv-web-viewer_*.deb ./nice-xdcv_*.deb
sudo systemctl enable dcvserver && sudo systemctl start dcvserver
sudo dcv create-session --owner ubuntu --type virtual my-session
sudo passwd ubuntu
# Open port 8443 in EC2 Security Group → https://<public-ip>:8443
```

## What's Left
1. **Resolve MJCF→USD conversion** — try isaaclab.sim.converters.MjcfConverter
2. **Run smoke test** — verify training loop end-to-end
3. **Create scripts/setup_ec2_isaac.sh** — idempotent script with all fixes
```

