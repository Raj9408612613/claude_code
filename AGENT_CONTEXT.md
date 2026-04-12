# Agent Handoff Context
**Date:** 2026-04-12  
**Repo:** `raj9408612613/claude_code`  
**Main branch:** `nv-omni-spot-tr` (always merge Claude branches here, never track `claude/...` branches on EC2)

---

## EC2 Instance
- **IP:** `ec2-54-196-117-72.compute-1.amazonaws.com` (may change on reboot if no Elastic IP)
- **User:** `ubuntu` (NOT root)
- **SSH key:** `g7e-nu-omni-tr.pem`
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition, **96GB VRAM**, CUDA 13.0
- **RAM:** 63GB | **Disk:** 145GB (43% used) | **OS:** Ubuntu 24.04

---

## Project
Spot robot RL training using Isaac Lab + Isaac Sim 4.5 inside Docker.  
Repo lives at `~/claude_code` on EC2.

---

## Current State (as of handoff)
- `isaac-lab-spot:latest` Docker image is built and working
- Shader/compute caches are saved at `~/.isaac_cache/` on host
- About to rerun `smoke_test.sh --full` with fixes applied
- Level 1 (MockSpotEnv + PPO) **PASSED** — 134 SPS, 8 envs
- Level 2 (Isaac Lab full) — **not yet confirmed passing**, still being tested

---

## Key Fixes Applied (all pushed to nv-omni-spot-tr)

### 1. Python output buffering (most recent fix)
`PYTHONUNBUFFERED=1` added to all `docker run` commands in:
- `scripts/smoke_test.sh`
- `scripts/isaac_run.sh`

Without this, training output never appears in terminal (Python buffers stdout in Docker).

### 2. Shader cache persistence
All 4 cache dirs mounted from host into container in both scripts:
```
~/.isaac_cache/kit          → /root/.cache/kit
~/.isaac_cache/ov           → /root/.nvidia-omniverse
~/.isaac_cache/glcache      → /root/.cache/nvidia/GLCache
~/.isaac_cache/computecache → /root/.nv/ComputeCache
```
Cache is owned by root (container runs as root) — permission errors for ubuntu user are expected and harmless.

### 3. Configurable NUM_ENVS
`smoke_test.sh` now accepts env vars:
```bash
NUM_ENVS=8 bash scripts/smoke_test.sh --full   # use 8 envs instead of 64
```

### 4. Fixed monitor.sh
Was accidentally saved as a generator command instead of the actual script.

---

## New Scripts Added
| Script | Purpose |
|---|---|
| `scripts/monitor.sh` | Background hardware monitor — logs GPU/CPU/disk to `~/hw_monitor.log` every 5s |
| `scripts/diagnose.sh` | One-shot diagnostics — shows containers, GPU processes, VRAM, disk I/O, log tails |
| `scripts/download_assets.sh` | Downloads `spot_omniverse.usd` from NVIDIA Omniverse to `models/` |

---

## How to Run Training
```bash
# On EC2 — always pull first
cd ~/claude_code && git pull origin nv-omni-spot-tr

# Use screen to protect from SSH drops
screen -S training

# Window 0 — smoke test (Ctrl+A C to create window 1)
export NUM_ENVS=8
bash scripts/smoke_test.sh --full

# Window 1 — monitor (Ctrl+A C)
bash scripts/monitor.sh

# Switch windows: Ctrl+A N / Ctrl+A P
# Detach: Ctrl+A D
# Reattach: screen -r training
```

---

## Isaac Sim Startup Behaviour (normal, not bugs)
- Takes **~5-6 minutes** to load first time (PSO shader compilation)
- Subsequent runs faster due to cache
- Last log line before training: `Isaac Sim Full Streaming App is loaded.`
- ROS2 Bridge startup failed — **expected**, not needed for RL training
- IRAY GPU unsupported warning — **expected**, Blackwell is newer than iray supports
- `No windowing` warnings — **expected**, running headless

---

## Known Issues / Watch Out For
1. **Two containers running simultaneously** — caused resource exhaustion before. Always check `sudo docker ps` before starting a new run. Kill all with `sudo docker kill $(sudo docker ps -q)`.
2. **Instance freezes under load** — happened once when CPU hit 100% during Isaac Sim init. If SSH times out, reboot from AWS Console.
3. **sudo docker inside nohup** — doesn't work reliably (gRPC disconnect). Use `screen` instead of `nohup` for long-running jobs.
4. **kit/ cache only 4KB** — may indicate PSO cache is stored differently in Isaac Sim 4.5. Monitor startup time on next run to confirm caching is working.

---

## Asset Locations
```
models/spot_scene.xml        # MJCF source
models/spot_scene.usd        # USD (converted from MJCF)
models/spot_omniverse.usd    # Official Spot USD from NVIDIA Omniverse
models/assets/               # Mesh .obj files (23 files)
```
Assets on host are auto-available inside container at `/workspace/models/`.
