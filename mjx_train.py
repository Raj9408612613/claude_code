"""
MJX Training Entry Point
=========================
Main training loop for Spot robot navigation with PPO.

Usage (Colab Pro / L4 / A100):
    python mjx_train.py --n_envs 512 --n_steps 2048 --total_updates 500
    python mjx_train.py --n_envs 512 --n_steps 1024 --total_updates 1000 --log_interval 1

For quick smoke test:
    python mjx_train.py --n_envs 64 --n_steps 128 --total_updates 5 --log_interval 1

GPU Memory Notes:
    By default JAX pre-allocates 90% of GPU VRAM at startup, so nvidia-smi
    will show ~70 GB "used" on a 80 GB GPU even if actual tensor usage is <2 GB.
    We set XLA_PYTHON_CLIENT_PREALLOCATE=false below so that reported memory
    reflects real usage.  Expected actual memory:
        128 envs:  ~0.7-1.5 GB
        512 envs:  ~2-4 GB
        2048 envs: ~8-15 GB
        4096 envs: ~15-30 GB

    Tip: prefer more envs × fewer steps (e.g. 512×512) over fewer envs × more
    steps (e.g. 128×2048).  Same sample count but fewer Python loop iterations,
    which is the main speed bottleneck.
"""

import os
# Must be set BEFORE importing JAX — switches from pre-allocating 90% of GPU
# VRAM to grow-on-demand, so nvidia-smi shows actual memory usage.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import sys
import time
import json
from datetime import datetime

import jax
import jax.numpy as jnp

from mjx_nav_env import SpotMJXEnv
from jax_ppo import PPOTrainer, compute_gae
from ppo_diagnostics import print_diagnostics


def parse_args():
    p = argparse.ArgumentParser(description="Train Spot navigation via PPO + MJX")
    # Environment
    p.add_argument("--n_envs",        type=int,   default=512,
                   help="Number of parallel environments")
    p.add_argument("--n_steps",       type=int,   default=2048,
                   help="Rollout length per update")
    p.add_argument("--xml_path",      type=str,   default="models/spot_scene_train.xml",
                   help="Path to MuJoCo XML scene (use spot_scene_train.xml for fast training, spot_scene.xml for visualization)")
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
    # Profiling
    p.add_argument("--profile",       type=int,   default=0, metavar="N",
                   help="Profile first N updates with per-component timing")
    p.add_argument("--render_interval", type=int,  default=4,
                   help="Render depth every N steps (1=every step, 4=skip 3/4)")
    p.add_argument("--warmup",        type=int,   default=2,
                   help="JIT warmup steps before training (0 to skip)")
    p.add_argument("--verbose_jit",   action="store_true",
                   help="Enable JAX JIT compilation logging")
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
            # Diagnostic scalars
            for diag_key in ["approx_kl", "clip_frac", "action_mean_abs",
                             "action_std_mean", "grad_norm", "grad_nan_frac"]:
                if diag_key in update_info:
                    self.tb_writer.add_scalar(f"diag/{diag_key}", float(update_info[diag_key]), step)
            # Rollout diagnostics
            diag = rollout_stats.get("_diag", {})
            for diag_key in ["ret_raw_mean", "ret_raw_std", "adv_mean", "adv_std",
                             "val_raw_mean", "val_raw_std", "explained_var",
                             "ret_scale_mean", "ret_scale_std"]:
                if diag_key in diag:
                    self.tb_writer.add_scalar(f"diag/{diag_key}", diag[diag_key], step)
            # Reward components
            for comp_name, comp_val in diag.get("reward_components", {}).items():
                self.tb_writer.add_scalar(f"reward_comp/{comp_name}", comp_val, step)
            self.tb_writer.add_scalar("timing/rollout_sec", rollout_sec, step)
            self.tb_writer.add_scalar("timing/update_sec",  update_sec,  step)
            self.tb_writer.flush()

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()


def report_gpu_memory(label=""):
    """Print actual GPU memory usage (not JAX's pre-allocated pool)."""
    try:
        for dev in jax.devices():
            stats = dev.memory_stats()
            if stats:
                used_gb = stats.get("bytes_in_use", 0) / 1e9
                peak_gb = stats.get("peak_bytes_in_use", 0) / 1e9
                limit_gb = stats.get("bytes_limit", 0) / 1e9
                print(f"  [GPU {label}] {used_gb:.2f} GB used, "
                      f"{peak_gb:.2f} GB peak, {limit_gb:.2f} GB limit")
    except Exception as e:
        print(f"  [GPU {label}] Could not read memory stats: {e}")


def warmup_jit(env, trainer, n_envs, n_steps, seed):
    """Force JIT compilation of all functions with per-component timing.

    Runs a few dummy steps so every JIT-compiled function gets compiled
    before the training loop starts.  Prints timing for each component
    so you can see exactly what's slow.
    """
    rng = jax.random.PRNGKey(seed)
    print("\n[WARMUP] Forcing JIT compilation (this is one-time)...")
    warmup_start = time.time()

    # 1. Reset
    print("  [1/5] reset()...", end=" ", flush=True)
    t0 = time.time()
    state, obs = env.reset(rng)
    jax.block_until_ready(obs["proprio"])
    print(f"{time.time()-t0:.1f}s")

    # 2. Inference (_inference_step)
    print("  [2/5] _inference_step (network forward + sample)...", end=" ", flush=True)
    t0 = time.time()
    action, log_prob, value, cnn_feat = trainer._sample_action(obs)
    jax.block_until_ready(action)
    print(f"{time.time()-t0:.1f}s")

    # 3. Physics step (_batch_step — usually the slowest)
    print("  [3/5] env.step (physics + render + reward)...", end=" ", flush=True)
    t0 = time.time()
    state, obs, reward, done, info = env.step(state, action)
    jax.block_until_ready(reward)
    print(f"{time.time()-t0:.1f}s")

    # 4. Auto-reset
    print("  [4/5] auto_reset...", end=" ", flush=True)
    t0 = time.time()
    rng, k = jax.random.split(rng)
    state, obs = env.auto_reset(state, obs, done, k)
    jax.block_until_ready(obs["proprio"])
    print(f"{time.time()-t0:.1f}s")

    # 5. GAE (compute_gae)
    print("  [5/5] compute_gae...", end=" ", flush=True)
    t0 = time.time()
    dummy_r = jnp.zeros((n_steps, n_envs))
    dummy_v = jnp.zeros((n_steps + 1, n_envs))
    dummy_d = jnp.zeros((n_steps, n_envs))
    compute_gae(dummy_r, dummy_v, dummy_d)
    print(f"{time.time()-t0:.1f}s")

    # Run a couple more cached steps to verify speed
    print("  [+] Cached steps (should be fast)...", end=" ", flush=True)
    t0 = time.time()
    for _ in range(3):
        action, _, _, _ = trainer._sample_action(obs)
        state, obs, reward, done, _ = env.step(state, action)
        rng, k = jax.random.split(rng)
        state, obs = env.auto_reset(state, obs, done, k)
        jax.block_until_ready(obs["proprio"])
    cached_time = time.time() - t0
    print(f"{cached_time:.1f}s ({cached_time/3*1000:.0f}ms/step)")

    total = time.time() - warmup_start
    print(f"[WARMUP] Complete in {total:.1f}s. Training will now start.\n")
    report_gpu_memory("after warmup")

    # Return a fresh state for training
    rng2 = jax.random.PRNGKey(seed + 1)
    return env.reset(rng2)


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
    print(f"  render_intv:   {args.render_interval} (depth every {args.render_interval} steps)")
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
        n_envs          = args.n_envs,
        xml_path        = args.xml_path,
        noise_enabled   = not args.no_noise,
        seed            = args.seed,
        render_interval = args.render_interval,
    )
    print(f"[INIT] Environment created in {time.time()-t0:.1f}s")
    report_gpu_memory("after env creation")

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

    # ── Verbose JIT logging ──────────────────────────────────────────
    if args.verbose_jit:
        jax.config.update("jax_log_compiles", True)
        print("[INIT] Verbose JIT logging enabled (JAX_LOG_COMPILES)")

    # ── JIT warmup or plain reset ─────────────────────────────────────
    if args.warmup > 0:
        state, obs = warmup_jit(
            env, trainer, args.n_envs, args.n_steps, args.seed,
        )
    else:
        print("[INIT] Resetting environments...")
        rng = jax.random.PRNGKey(args.seed)
        rng, reset_key = jax.random.split(rng)
        state, obs = env.reset(reset_key)
        report_gpu_memory("after reset")
        print("[INIT] Reset complete. Starting training.\n")

    # ── Training loop ──────────────────────────────────────────────────
    total_timesteps = 0
    train_start     = time.time()
    best_rew_mean   = float("-inf")

    for update in range(1, args.total_updates + 1):
        # ── Collect rollout ────────────────────────────────────────────
        do_profile = args.profile > 0 and update <= args.profile
        t_roll = time.time()
        state, obs, batch, rollout_stats = trainer.collect_rollout(
            env, state, obs, profile=do_profile,
        )
        rollout_sec = time.time() - t_roll

        # ── PPO update ─────────────────────────────────────────────────
        t_upd = time.time()
        update_info = trainer.update(batch)
        update_sec = time.time() - t_upd

        # ── Print profiling breakdown ─────────────────────────────────
        if do_profile:
            timing = rollout_stats.get("_timing", {})
            if timing:
                print(f"  [PROFILE update {update}] "
                      f"inference={timing.get('inference_sec', 0):.2f}s  "
                      f"env_step={timing.get('env_step_sec', 0):.2f}s  "
                      f"auto_reset={timing.get('auto_reset_sec', 0):.2f}s  "
                      f"gae_batch={timing.get('gae_batch_sec', 0):.2f}s  "
                      f"diag={timing.get('diag_sec', 0):.2f}s")
            report_gpu_memory(f"after update {update}")

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
                  f"policy_los={float(update_info['policy_loss']):.4f} "
                  f"vvalue_loss={float(update_info['value_loss']):.4f} "
                  f"entropy={float(update_info['entropy']):.3f} | "
                  f"{fps:.0f} fps | "
                  f"roll={rollout_sec:.1f}s upd={update_sec:.1f}s")

            # ── Diagnostics ────────────────────────────────────────────
            rollout_diag = rollout_stats.get("_diag", {})
            update_diag = {k: (float(v) if hasattr(v, 'shape') and v.ndim == 0 else
                              (v.tolist() if hasattr(v, 'tolist') and hasattr(v, 'ndim') and v.ndim > 0 else
                               float(v) if not isinstance(v, (list, dict)) else v))
                          for k, v in update_info.items()}
            print_diagnostics(update, rollout_diag, update_diag)

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
