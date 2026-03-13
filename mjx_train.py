"""
MJX Training Entry Point
=========================
Main training loop for Spot robot navigation with PPO.

Usage (Colab Pro / L4 / A100):
    python mjx_train.py --n_envs 512 --n_steps 2048 --total_updates 500
    python mjx_train.py --n_envs 512 --n_steps 1024 --total_updates 1000 --log_interval 1

For quick smoke test:
    python mjx_train.py --n_envs 64 --n_steps 128 --total_updates 5 --log_interval 1
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime

import jax
import jax.numpy as jnp

from mjx_nav_env import SpotMJXEnv
from jax_ppo import PPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train Spot navigation via PPO + MJX")
    # Environment
    p.add_argument("--n_envs",        type=int,   default=512,
                   help="Number of parallel environments")
    p.add_argument("--n_steps",       type=int,   default=2048,
                   help="Rollout length per update")
    p.add_argument("--xml_path",      type=str,   default="models/spot_scene.xml",
                   help="Path to MuJoCo XML scene")
    # Training
    p.add_argument("--total_updates", type=int,   default=500,
                   help="Total PPO update iterations")
    p.add_argument("--lr",            type=float, default=3e-4,
                   help="Learning rate")
    p.add_argument("--seed",          type=int,   default=42,
                   help="Random seed")
    # Logging / checkpointing
    p.add_argument("--log_interval",  type=int,   default=1,
                   help="Print stats every N updates")
    p.add_argument("--save_interval", type=int,   default=50,
                   help="Save checkpoint every N updates")
    p.add_argument("--ckpt_dir",      type=str,   default="checkpoints",
                   help="Checkpoint directory")
    p.add_argument("--tb_dir",        type=str,   default="tb_logs",
                   help="TensorBoard log directory")
    # Noise
    p.add_argument("--no_noise",      action="store_true",
                   help="Disable depth noise for debugging")
    # Resume
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


class SimpleLogger:
    """Minimal CSV + stdout logger. TensorBoard optional."""

    def __init__(self, log_dir, tb_dir=None):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "train_log.csv")
        self.tb_writer = None

        # CSV header
        with open(self.csv_path, "w") as f:
            f.write("update,timesteps,wall_time,rew_mean,rew_min,rew_max,"
                    "done_rate,ep_count,policy_loss,value_loss,entropy,"
                    "total_loss,rollout_sec,update_sec\n")

        # Optional TensorBoard
        if tb_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(tb_dir)
                print(f"[LOG] TensorBoard logging to {tb_dir}")
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter
                    self.tb_writer = SummaryWriter(tb_dir)
                    print(f"[LOG] TensorboardX logging to {tb_dir}")
                except ImportError:
                    print("[LOG] No TensorBoard library found. CSV-only logging.")

    def log(self, update, timesteps, wall_time, rollout_stats, update_info,
            rollout_sec, update_sec):
        row = {
            "update":      update,
            "timesteps":   timesteps,
            "wall_time":   f"{wall_time:.1f}",
            "rew_mean":    f"{rollout_stats['rew_mean']:.4f}",
            "rew_min":     f"{rollout_stats['rew_min']:.4f}",
            "rew_max":     f"{rollout_stats['rew_max']:.4f}",
            "done_rate":   f"{rollout_stats['done_rate']:.4f}",
            "ep_count":    rollout_stats["ep_count"],
            "policy_loss": f"{float(update_info['policy_loss']):.6f}",
            "value_loss":  f"{float(update_info['value_loss']):.6f}",
            "entropy":     f"{float(update_info['entropy']):.4f}",
            "total_loss":  f"{float(update_info['total_loss']):.6f}",
            "rollout_sec": f"{rollout_sec:.2f}",
            "update_sec":  f"{update_sec:.2f}",
        }

        # CSV
        with open(self.csv_path, "a") as f:
            f.write(",".join(str(row[k]) for k in [
                "update", "timesteps", "wall_time", "rew_mean", "rew_min",
                "rew_max", "done_rate", "ep_count", "policy_loss",
                "value_loss", "entropy", "total_loss", "rollout_sec",
                "update_sec"]) + "\n")

        # TensorBoard
        if self.tb_writer:
            step = timesteps
            self.tb_writer.add_scalar("reward/mean", rollout_stats["rew_mean"], step)
            self.tb_writer.add_scalar("reward/min",  rollout_stats["rew_min"],  step)
            self.tb_writer.add_scalar("reward/max",  rollout_stats["rew_max"],  step)
            self.tb_writer.add_scalar("episode/done_rate", rollout_stats["done_rate"], step)
            self.tb_writer.add_scalar("episode/count",     rollout_stats["ep_count"],  step)
            self.tb_writer.add_scalar("loss/policy",  float(update_info["policy_loss"]), step)
            self.tb_writer.add_scalar("loss/value",   float(update_info["value_loss"]),  step)
            self.tb_writer.add_scalar("loss/entropy", float(update_info["entropy"]),     step)
            self.tb_writer.add_scalar("loss/total",   float(update_info["total_loss"]),  step)
            if "ratio_mean" in update_info:
                self.tb_writer.add_scalar("debug/ratio_mean", float(update_info["ratio_mean"]), step)
                self.tb_writer.add_scalar("debug/ratio_max",  float(update_info["ratio_max"]),  step)
            self.tb_writer.add_scalar("timing/rollout_sec", rollout_sec, step)
            self.tb_writer.add_scalar("timing/update_sec",  update_sec,  step)
            self.tb_writer.flush()

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"Spot Navigation Training — MJX + PPO")
    print(f"=" * 60)
    print(f"  n_envs:        {args.n_envs}")
    print(f"  n_steps:       {args.n_steps}")
    print(f"  total_updates: {args.total_updates}")
    print(f"  lr:            {args.lr}")
    print(f"  seed:          {args.seed}")
    print(f"  depth_noise:   {not args.no_noise}")
    print(f"  timesteps/upd: {args.n_envs * args.n_steps:,}")
    print(f"  total_steps:   {args.n_envs * args.n_steps * args.total_updates:,}")
    print(f"  JAX devices:   {jax.devices()}")
    print(f"  run_id:        {run_id}")
    print("=" * 60)

    # ── Directories ────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.ckpt_dir, run_id)
    tb_dir   = os.path.join(args.tb_dir, run_id)
    log_dir  = os.path.join("logs", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Initialize ─────────────────────────────────────────────────────
    print("\n[INIT] Creating environment...")
    t0 = time.time()
    env = SpotMJXEnv(
        n_envs        = args.n_envs,
        xml_path      = args.xml_path,
        noise_enabled = not args.no_noise,
        seed          = args.seed,
    )
    print(f"[INIT] Environment created in {time.time()-t0:.1f}s")

    print("[INIT] Creating PPO trainer...")
    trainer = PPOTrainer(
        n_envs  = args.n_envs,
        n_steps = args.n_steps,
        lr      = args.lr,
        seed    = args.seed,
    )

    if args.resume:
        print(f"[INIT] Resuming from {args.resume}")
        trainer.load(args.resume)

    logger = SimpleLogger(log_dir=log_dir, tb_dir=tb_dir)

    # ── Initial reset ──────────────────────────────────────────────────
    print("[INIT] Resetting environments...")
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    state, obs = env.reset(reset_key)
    print("[INIT] Reset complete. Starting training.\n")

    # ── Training loop ──────────────────────────────────────────────────
    total_timesteps = 0
    train_start     = time.time()
    best_rew_mean   = float("-inf")

    for update in range(1, args.total_updates + 1):
        # ── Collect rollout ────────────────────────────────────────────
        t_roll = time.time()
        state, obs, batch, rollout_stats = trainer.collect_rollout(env, state, obs)
        rollout_sec = time.time() - t_roll

        # ── PPO update ─────────────────────────────────────────────────
        t_upd = time.time()
        update_info = trainer.update(batch)
        update_sec = time.time() - t_upd

        total_timesteps += args.n_envs * args.n_steps
        wall_time = time.time() - train_start

        # ── Log ────────────────────────────────────────────────────────
        if update % args.log_interval == 0:
            rew_mean = rollout_stats["rew_mean"]
            logger.log(update, total_timesteps, wall_time,
                       rollout_stats, update_info, rollout_sec, update_sec)

            fps = (args.n_envs * args.n_steps) / (rollout_sec + update_sec)
            print(f"[{update:4d}/{args.total_updates}] "
                  f"t={total_timesteps:>10,} | "
                  f"rew={rew_mean:>8.3f} [{rollout_stats['rew_min']:.1f}, {rollout_stats['rew_max']:.1f}] | "
                  f"done={rollout_stats['done_rate']:.3f} ep={rollout_stats['ep_count']:>5} | "
                  f"loss={float(update_info['total_loss']):.4f} "
                  f"pi={float(update_info['policy_loss']):.4f} "
                  f"v={float(update_info['value_loss']):.4f} "
                  f"H={float(update_info['entropy']):.3f} | "
                  f"{fps:.0f} fps | "
                  f"roll={rollout_sec:.1f}s upd={update_sec:.1f}s")

            # Track best
            if rew_mean > best_rew_mean:
                best_rew_mean = rew_mean
                best_path = os.path.join(ckpt_dir, "best.pkl")
                trainer.save(best_path)

        # ── Checkpoint ─────────────────────────────────────────────────
        if update % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{update:05d}.pkl")
            trainer.save(ckpt_path)
            print(f"  [SAVE] {ckpt_path}")

        # ── Early NaN detection ────────────────────────────────────────
        if jnp.isnan(update_info["total_loss"]) or jnp.isinf(update_info["total_loss"]):
            print("\n[ERROR] NaN/Inf loss detected! Saving crash checkpoint and stopping.")
            crash_path = os.path.join(ckpt_dir, f"crash_{update:05d}.pkl")
            trainer.save(crash_path)
            break

    # ── Final save ─────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final.pkl")
    trainer.save(final_path)
    logger.close()

    total_time = time.time() - train_start
    print(f"\n{'=' * 60}")
    print(f"Training complete.")
    print(f"  Total updates:    {update}")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"  Wall time:        {total_time/60:.1f} min")
    print(f"  Best rew_mean:    {best_rew_mean:.4f}")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Log CSV:          {logger.csv_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
