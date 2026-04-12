"""
Omniverse Isaac Lab Training Entry Point — Spot Navigation
============================================================
Ported from mjx_train.py. Uses PyTorch + Isaac Lab.

IMPORTANT: Isaac Lab sub-modules (isaaclab.sim, isaaclab.envs, etc.)
require the Omniverse simulation app to be running. We must launch it
via AppLauncher BEFORE importing any Isaac Lab config/env classes.

Usage (requires Isaac Lab / Omniverse):
    /isaac-sim/python.sh -m omni_spot.train --num_envs 4096 --n_steps 2048 --total_updates 500

For quick smoke test:
    /isaac-sim/python.sh -m omni_spot.train --num_envs 64 --n_steps 128 --total_updates 5
"""

import argparse
import os
import sys
import time
import csv
import json
from datetime import datetime

# ── Step 1: Launch Isaac Sim BEFORE importing Isaac Lab sub-modules ──
# This MUST happen before any 'from isaaclab.sim import ...' etc.
print("[INIT] Launching Isaac Sim (first run takes ~5 min for shader compilation)...")
try:
    from isaaclab.app import AppLauncher
except ImportError:
    try:
        from omni.isaac.lab.app import AppLauncher
    except ImportError:
        print("[ERROR] Cannot import AppLauncher from isaaclab or omni.isaac.lab")
        print("        Is Isaac Lab installed? pip list | grep isaaclab")
        sys.exit(1)

_parser = argparse.ArgumentParser(description="Spot Navigation RL — Isaac Lab")
# Environment
_parser.add_argument("--num_envs",      type=int,   default=4096)
_parser.add_argument("--n_steps",       type=int,   default=2048,
                     help="Rollout steps per update")
# Training
_parser.add_argument("--total_updates", type=int,   default=500)
_parser.add_argument("--lr",            type=float, default=3e-4)
_parser.add_argument("--seed",          type=int,   default=42)
# Logging
_parser.add_argument("--log_interval",  type=int,   default=1)
_parser.add_argument("--save_interval", type=int,   default=50)
_parser.add_argument("--log_dir",       type=str,   default="omni_logs")
# Resume
_parser.add_argument("--resume",        type=str,   default=None,
                     help="Path to checkpoint to resume from")
# Profiling
_parser.add_argument("--profile",       type=int,   default=0, metavar="N",
                     help="Profile first N updates with per-component timing")
# AppLauncher adds --headless, --device, --enable_cameras, etc.
AppLauncher.add_app_launcher_args(_parser)
args = _parser.parse_args()

# Launch the simulation app (starts Kit, loads extensions, compiles shaders)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[INIT] Isaac Sim launched successfully.")

# ── Step 2: NOW safe to import Isaac Lab sub-modules & PyTorch ───────
import torch

from .config import LR
from .ppo import PPOTrainer
from .diagnostics import print_diagnostics


class SimpleLogger:
    """CSV + TensorBoard logger. Same as mjx_train.py version."""

    def __init__(self, log_dir: str, run_id: str):
        self.log_dir = os.path.join(log_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "train_log.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = None

        # TensorBoard (optional)
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_path = os.path.join(self.log_dir, "tb")
            self.tb_writer = SummaryWriter(tb_path)
            print(f"[LOG] TensorBoard logging to {tb_path}")
        except ImportError:
            print("[LOG] TensorBoard not available, using CSV only")

    def log(self, update, timesteps, wall_time, rollout_stats, update_info,
            rollout_sec, update_sec):
        row = {
            "update": update,
            "timesteps": timesteps,
            "wall_time": f"{wall_time:.1f}",
            "rew_mean": f"{rollout_stats['rew_mean']:.4f}",
            "rew_min":  f"{rollout_stats['rew_min']:.4f}",
            "rew_max":  f"{rollout_stats['rew_max']:.4f}",
            "done_rate": f"{rollout_stats['done_rate']:.4f}",
            "ep_count": rollout_stats['ep_count'],
            "policy_loss": f"{update_info.get('policy_loss', 0):.6f}",
            "value_loss":  f"{update_info.get('value_loss', 0):.6f}",
            "entropy":     f"{update_info.get('entropy', 0):.6f}",
            "total_loss":  f"{update_info.get('total_loss', 0):.6f}",
            "rollout_sec": f"{rollout_sec:.2f}",
            "update_sec":  f"{update_sec:.2f}",
        }

        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=row.keys())
            self.csv_writer.writeheader()
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        if self.tb_writer:
            self.tb_writer.add_scalar("reward/mean", rollout_stats["rew_mean"], timesteps)
            self.tb_writer.add_scalar("reward/min",  rollout_stats["rew_min"],  timesteps)
            self.tb_writer.add_scalar("reward/max",  rollout_stats["rew_max"],  timesteps)
            self.tb_writer.add_scalar("episode/done_rate", rollout_stats["done_rate"], timesteps)
            self.tb_writer.add_scalar("loss/policy", update_info.get("policy_loss", 0), timesteps)
            self.tb_writer.add_scalar("loss/value",  update_info.get("value_loss", 0), timesteps)
            self.tb_writer.add_scalar("loss/entropy", update_info.get("entropy", 0), timesteps)
            self.tb_writer.add_scalar("loss/total",  update_info.get("total_loss", 0), timesteps)
            self.tb_writer.add_scalar("timing/rollout_sec", rollout_sec, timesteps)
            self.tb_writer.add_scalar("timing/update_sec",  update_sec, timesteps)

    def close(self):
        self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()


def report_gpu_memory(label=""):
    """Print actual GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        max_alloc = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [GPU {label}] {allocated:.2f} GB allocated, "
              f"{reserved:.2f} GB reserved, {max_alloc:.2f} GB peak")


def main():
    # args already parsed at module level (before AppLauncher)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"{'='*60}")
    print(f"  Spot Navigation RL — Isaac Lab (PyTorch)")
    print(f"  Envs: {args.num_envs}  Steps: {args.n_steps}  "
          f"Updates: {args.total_updates}  LR: {args.lr}")
    print(f"{'='*60}")

    # ── Logger ──────────────────────────────────────────────────────
    logger = SimpleLogger(args.log_dir, run_id)

    # ── Environment (Isaac Lab) ─────────────────────────────────────
    # Simulation app is already running (launched at module level).
    # Now Isaac Lab sub-module imports will work.
    print("[INIT] Creating Isaac Lab environment...")
    t0 = time.time()

    from .spot_env_cfg import SpotNavEnvCfg, HAS_ISAAC, _ISAAC_IMPORT_ERROR
    from .spot_env import SpotNavEnv
    from .physics_tuning import apply_tuning

    if not HAS_ISAAC:
        print(f"[ERROR] Isaac Lab import failed: {_ISAAC_IMPORT_ERROR}")
        sys.exit(1)

    env_cfg = SpotNavEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    apply_tuning(env_cfg.sim, env_cfg.scene)
    env = SpotNavEnv(cfg=env_cfg)

    print(f"[INIT] Environment created in {time.time()-t0:.1f}s")
    report_gpu_memory("after env creation")

    # ── Trainer ─────────────────────────────────────────────────────
    print("[INIT] Creating PPO trainer...")
    trainer = PPOTrainer(
        n_envs  = args.num_envs,
        n_steps = args.n_steps,
        lr      = args.lr,
    )

    if args.resume:
        print(f"[INIT] Resuming from {args.resume}")
        trainer.load(args.resume)

    # ── Initial reset ───────────────────────────────────────────────
    print("[INIT] Resetting environments...")
    obs, _ = env.reset()
    report_gpu_memory("after reset")
    print("[INIT] Reset complete. Starting training.\n")

    # ── Training loop ───────────────────────────────────────────────
    total_timesteps = 0
    train_start = time.time()
    best_reward = float("-inf")

    for update in range(1, args.total_updates + 1):
        do_profile = args.profile > 0 and update <= args.profile

        # Collect rollout
        t_roll = time.time()
        obs, batch, rollout_stats = trainer.collect_rollout(
            env, obs, profile=do_profile,
        )
        rollout_sec = time.time() - t_roll

        # PPO update
        t_upd = time.time()
        update_info = trainer.update(batch)
        update_sec = time.time() - t_upd

        # Profiling
        if do_profile:
            timing = rollout_stats.get("_timing", {})
            if timing:
                print(f"  [PROFILE update {update}] "
                      f"inference={timing.get('inference_sec', 0):.2f}s  "
                      f"env_step={timing.get('env_step_sec', 0):.2f}s")
            report_gpu_memory(f"after update {update}")

        total_timesteps += args.num_envs * args.n_steps
        wall_time = time.time() - train_start

        # Log
        if update % args.log_interval == 0:
            mean_rew = rollout_stats["rew_mean"]
            sps = (args.num_envs * args.n_steps) / (rollout_sec + update_sec)
            print(f"[{update:>4d}/{args.total_updates}]  "
                  f"rew={mean_rew:>8.2f}  "
                  f"eps={rollout_stats['ep_count']:>5d}  "
                  f"roll={rollout_sec:.1f}s  upd={update_sec:.1f}s  "
                  f"SPS={sps:,.0f}  total={total_timesteps:,}")

            logger.log(update, total_timesteps, wall_time,
                       rollout_stats, update_info, rollout_sec, update_sec)

            # Diagnostics
            diag = rollout_stats.get("_diag", {})
            if diag:
                print_diagnostics(update, diag, update_info)

        # Save
        if update % args.save_interval == 0:
            path = os.path.join(logger.log_dir, f"ckpt_{update:05d}.pt")
            trainer.save(path)
            print(f"  [SAVE] {path}")

        if rollout_stats["rew_mean"] > best_reward:
            best_reward = rollout_stats["rew_mean"]
            trainer.save(os.path.join(logger.log_dir, "best.pt"))

    # Final save
    trainer.save(os.path.join(logger.log_dir, "final.pt"))
    logger.close()
    print(f"\n[DONE] Training complete. {total_timesteps:,} total timesteps.")

    # Write success marker (smoke_test.sh checks this to avoid false positives
    # from Isaac Sim's shutdown masking the Python exit code)
    marker = os.path.join(args.log_dir, "SUCCESS")
    with open(marker, "w") as f:
        f.write(f"{total_timesteps}\n")

    # Shut down Isaac Sim
    simulation_app.close()


if __name__ == "__main__":
    main()
