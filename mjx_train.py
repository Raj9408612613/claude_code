"""
MJX Training Entry Point
=========================
Replaces train.py (Isaac Sim + Stable Baselines 3).

Run:
    python mjx_train.py                        # default 4096 envs
    python mjx_train.py --n_envs 1024 --timesteps 20000000
    python mjx_train.py --resume checkpoints/step_1000000.pkl

GPU usage:
    Physics:  MuJoCo MJX — jax.vmap over 4096 envs, single XLA call
    Rendering: NVIDIA Warp — all 5 cameras × all envs in one CUDA kernel
    Training:  JAX/Flax/Optax — fully JIT-compiled, no Python in hot path

Expected throughput (vs original Isaac Sim ~17ms/env/step on 1 vCPU):
    L4  GPU: ~70x speedup
    A100 GPU: ~150x speedup
    H100 GPU: ~300x speedup
"""

import os
import time
import argparse
import jax
import jax.numpy as jnp

from mjx_nav_env import SpotMJXEnv
from jax_ppo import PPOTrainer

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Spot with MJX + Warp + JAX PPO")
    p.add_argument("--n_envs",      type=int,   default=4096,        help="Parallel environments")
    p.add_argument("--timesteps",   type=int,   default=10_000_000,  help="Total env steps")
    p.add_argument("--n_steps",     type=int,   default=2048,        help="Rollout steps per update")
    p.add_argument("--lr",          type=float, default=3e-4,        help="Learning rate")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--xml",         type=str,   default="models/spot_scene.xml")
    p.add_argument("--save_dir",    type=str,   default="mjx_checkpoints")
    p.add_argument("--log_interval",type=int,   default=10,          help="Log every N updates")
    p.add_argument("--save_interval",type=int,  default=100,         help="Save every N updates")
    p.add_argument("--resume",      type=str,   default=None,        help="Path to checkpoint pkl")
    p.add_argument("--no_noise",    action="store_true",             help="Disable depth noise")
    return p.parse_args()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MJX + Warp Spot Training")
    print(f"{'='*60}")
    print(f"  Devices:     {jax.devices()}")
    print(f"  n_envs:      {args.n_envs}")
    print(f"  n_steps:     {args.n_steps}")
    print(f"  total steps: {args.timesteps:,}")
    print(f"  xml:         {args.xml}")
    print(f"{'='*60}\n")

    # ── Create environment ────────────────────────────────────────────
    print("Loading MJX environment...")
    env = SpotMJXEnv(
        n_envs        = args.n_envs,
        xml_path      = args.xml,
        noise_enabled = not args.no_noise,
        seed          = args.seed,
    )
    print(f"  nq={env.nq}, nv={env.nv}, action_dim={env.action_dim}")

    # ── Create trainer ────────────────────────────────────────────────
    print("Building JAX PPO trainer...")
    trainer = PPOTrainer(
        n_envs  = args.n_envs,
        n_steps = args.n_steps,
        lr      = args.lr,
        seed    = args.seed,
    )

    if args.resume:
        print(f"  Resuming from {args.resume}")
        trainer.load(args.resume)

    # ── Initial reset ─────────────────────────────────────────────────
    print("Resetting environments (first compile may take ~60s)...")
    rng   = jax.random.PRNGKey(args.seed)
    state, obs = env.reset(rng)
    print("  Done.\n")

    # ── Training loop ─────────────────────────────────────────────────
    total_env_steps = 0
    n_updates       = args.timesteps // (args.n_envs * args.n_steps)
    update          = 0
    t_start         = time.time()

    print(f"Starting training — {n_updates} updates × {args.n_envs * args.n_steps:,} steps/update")

    for update in range(1, n_updates + 1):
        t0 = time.time()

        # Collect rollout
        state, obs, batch = trainer.collect_rollout(env, state, obs)

        # PPO update
        info = trainer.update(batch)

        total_env_steps += args.n_envs * args.n_steps
        elapsed = time.time() - t_start
        fps     = total_env_steps / elapsed

        # ── Logging ───────────────────────────────────────────────────
        if update % args.log_interval == 0:
            print(
                f"[update {update:4d}/{n_updates}]  "
                f"steps={total_env_steps:9,}  "
                f"fps={fps:7,.0f}  "
                f"policy_loss={float(info['policy_loss']):.4f}  "
                f"value_loss={float(info['value_loss']):.4f}  "
                f"entropy={float(info['entropy']):.4f}"
            )

        # ── Checkpointing ─────────────────────────────────────────────
        if update % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{total_env_steps}.pkl")
            trainer.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # ── Auto-reset terminated envs ─────────────────────────────────
        # (full reset on all envs every n_steps — simpler than per-env reset)
        if update % 10 == 0:
            rng, k = jax.random.split(rng)
            state, obs = env.reset(k)

    # ── Final save ────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "final_model.pkl")
    trainer.save(final_path)
    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time/3600:.2f}h")
    print(f"Final model saved to {final_path}")
    print(f"Average FPS: {total_env_steps / total_time:,.0f}")


if __name__ == "__main__":
    main()
