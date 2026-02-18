"""
Training script for Spot with CNN Obstacle Detection

Uses the SpotObstacleEnv (Dict obs: image + proprioception) and the
SpotCNNExtractor custom feature extractor with PPO.

Optimized for GPU acceleration with larger batches and more parallel envs.

Usage:
    python train_obstacle.py --timesteps 2000000 --n_envs 8
    python train_obstacle.py --fast                          # quick iteration (200k steps)
    python train_obstacle.py --continue_from ./spot_obstacle_models/best_model.zip
"""

import os
import sys
import time
import argparse
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from spot_obstacle_env import SpotObstacleEnv
from cnn_policy import SpotCNNExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_device():
    """Auto-detect the best available device."""
    if th.cuda.is_available():
        name = th.cuda.get_device_name(0)
        mem = th.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU detected: {name} ({mem:.1f} GB)")
        return "cuda"
    print("  No GPU detected — using CPU")
    return "cpu"


class ThroughputCallback(BaseCallback):
    """Prints training throughput (steps/sec) periodically."""

    def __init__(self, print_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._last_time = None
        self._last_steps = 0

    def _on_training_start(self):
        self._last_time = time.time()
        self._last_steps = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_steps >= self.print_freq:
            now = time.time()
            elapsed = now - self._last_time
            delta = self.num_timesteps - self._last_steps
            sps = delta / max(elapsed, 1e-8)
            print(f"  [{self.num_timesteps:>10,} steps]  {sps:,.0f} steps/sec")
            self._last_time = now
            self._last_steps = self.num_timesteps
        return True


def make_env(rank, seed=0, camera_width=64, camera_height=64, n_obstacles=5,
             physics_substeps=5):
    """Create a monitored SpotObstacleEnv instance."""
    def _init():
        env = SpotObstacleEnv(
            render_mode=None,
            camera_width=camera_width,
            camera_height=camera_height,
            n_obstacles_active=n_obstacles,
            physics_substeps=physics_substeps,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESETS = {
    "fast": {
        "desc": "Quick iteration — 200k steps, fewer evals, fast feedback",
        "timesteps": 200_000,
        "n_envs": 8,
        "batch_size": 256,
        "n_steps": 2048,
        "n_epochs": 5,
        "eval_freq": 50_000,
        "checkpoint_freq": 100_000,
        "n_eval_episodes": 3,
        "physics_substeps": 4,
    },
    "medium": {
        "desc": "Balanced — 1M steps, moderate eval frequency",
        "timesteps": 1_000_000,
        "n_envs": 8,
        "batch_size": 256,
        "n_steps": 4096,
        "n_epochs": 10,
        "eval_freq": 25_000,
        "checkpoint_freq": 100_000,
        "n_eval_episodes": 5,
        "physics_substeps": 5,
    },
    "full": {
        "desc": "Full training — 2M steps, thorough evaluation",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "batch_size": 512,
        "n_steps": 4096,
        "n_epochs": 10,
        "eval_freq": 25_000,
        "checkpoint_freq": 50_000,
        "n_eval_episodes": 5,
        "physics_substeps": 5,
    },
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_obstacle(
    total_timesteps=2_000_000,
    n_envs=8,
    save_dir="./spot_obstacle_models",
    log_dir="./spot_obstacle_logs",
    camera_size=64,
    n_obstacles=5,
    learning_rate=3e-4,
    batch_size=256,
    n_steps=4096,
    n_epochs=10,
    eval_freq=25_000,
    checkpoint_freq=50_000,
    n_eval_episodes=5,
    device=None,
    physics_substeps=5,
):
    """
    Train Spot to navigate toward goals while avoiding obstacles using
    a CNN front-camera + proprioceptive state policy.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if device is None:
        device = detect_device()

    print("=" * 60)
    print("SPOT CNN OBSTACLE AVOIDANCE TRAINING")
    print("=" * 60)
    print(f"  Device:          {device}")
    print(f"  Timesteps:       {total_timesteps:,}")
    print(f"  Parallel envs:   {n_envs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Rollout (n_steps): {n_steps}")
    print(f"  PPO epochs:      {n_epochs}")
    print(f"  Camera:          {camera_size}x{camera_size} RGB")
    print(f"  Obstacles:       {n_obstacles}")
    print(f"  Physics substeps: {physics_substeps}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Eval every:      {eval_freq:,} steps")
    print(f"  Save dir:        {save_dir}")
    print(f"  Log dir:         {log_dir}")
    print("=" * 60)

    # Create vectorized environments
    env_fns = [
        make_env(i, camera_width=camera_size, camera_height=camera_size,
                 n_obstacles=n_obstacles, physics_substeps=physics_substeps)
        for i in range(n_envs)
    ]

    if n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        env = DummyVecEnv(env_fns)

    # Evaluation environment (single, deterministic)
    eval_env = DummyVecEnv([
        make_env(9999, camera_width=camera_size, camera_height=camera_size,
                 n_obstacles=n_obstacles, physics_substeps=physics_substeps)
    ])

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=save_dir,
        name_prefix="spot_obstacle",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    throughput_cb = ThroughputCallback(print_freq=max(total_timesteps // 20, 10_000))

    # Policy kwargs: use the custom CNN+MLP feature extractor
    policy_kwargs = dict(
        features_extractor_class=SpotCNNExtractor,
        features_extractor_kwargs=dict(
            cnn_output_dim=128,
            proprio_hidden_dim=64,
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=th.nn.ReLU,
    )

    print("\nInitializing PPO with MultiInputPolicy (CNN + MLP) ...")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy parameters: {total_params:,}")

    print("\nStarting training ...")
    print("=" * 60)

    t_start = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb, throughput_cb],
            tb_log_name="spot_obstacle_cnn",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    elapsed = time.time() - t_start
    avg_sps = total_timesteps / max(elapsed, 1)
    print(f"\nTraining finished in {elapsed / 60:.1f} min  ({avg_sps:,.0f} avg steps/sec)")

    final_path = os.path.join(save_dir, "spot_obstacle_final")
    model.save(final_path)
    print(f"Final model saved to: {final_path}")

    env.close()
    eval_env.close()
    return model


def continue_training(
    model_path,
    total_timesteps=1_000_000,
    n_envs=8,
    save_dir="./spot_obstacle_models",
    log_dir="./spot_obstacle_logs",
    camera_size=64,
    n_obstacles=5,
    batch_size=256,
    n_steps=4096,
    eval_freq=25_000,
    checkpoint_freq=50_000,
    n_eval_episodes=5,
    device=None,
    physics_substeps=5,
):
    """Resume training from a saved CNN obstacle model."""
    if device is None:
        device = detect_device()

    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")

    env_fns = [
        make_env(i, camera_width=camera_size, camera_height=camera_size,
                 n_obstacles=n_obstacles, physics_substeps=physics_substeps)
        for i in range(n_envs)
    ]
    if n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        env = DummyVecEnv(env_fns)

    eval_env = DummyVecEnv([
        make_env(9999, camera_width=camera_size, camera_height=camera_size,
                 n_obstacles=n_obstacles, physics_substeps=physics_substeps)
    ])

    custom_objects = {
        "policy_kwargs": dict(
            features_extractor_class=SpotCNNExtractor,
            features_extractor_kwargs=dict(
                cnn_output_dim=128,
                proprio_hidden_dim=64,
            ),
        ),
    }

    model = PPO.load(model_path, env=env, custom_objects=custom_objects,
                     device=device)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=save_dir,
        name_prefix="spot_obstacle_continued",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    throughput_cb = ThroughputCallback(print_freq=max(total_timesteps // 20, 10_000))

    print("\nContinuing training ...")
    t_start = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb, throughput_cb],
            tb_log_name="spot_obstacle_cnn_continued",
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    elapsed = time.time() - t_start
    print(f"\nContinued training finished in {elapsed / 60:.1f} min")

    final_path = os.path.join(save_dir, "spot_obstacle_continued_final")
    model.save(final_path)
    print(f"Final model saved to: {final_path}")

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spot with CNN Obstacle Detection")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (default depends on preset)")
    parser.add_argument("--n_envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="PPO minibatch size")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Rollout length per env per update")
    parser.add_argument("--save_dir", type=str, default="./spot_obstacle_models",
                        help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="./spot_obstacle_logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--camera_size", type=int, default=64,
                        help="Camera image size (square, default: 64)")
    parser.add_argument("--n_obstacles", type=int, default=5,
                        help="Number of active obstacles (default: 5)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="Force device (default: auto-detect)")
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Path to model checkpoint to continue training from")

    # Preset modes
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument("--fast", action="store_const", dest="preset",
                              const="fast",
                              help="Quick iteration: 200k steps, large batches")
    preset_group.add_argument("--medium", action="store_const", dest="preset",
                              const="medium",
                              help="Balanced: 1M steps, GPU-optimized batches")
    preset_group.add_argument("--full", action="store_const", dest="preset",
                              const="full",
                              help="Full training: 2M steps (default)")

    args = parser.parse_args()

    # Resolve preset
    preset_name = args.preset or "full"
    preset = PRESETS[preset_name]
    print(f"\nPreset: {preset_name} — {preset['desc']}")

    # CLI args override preset values
    timesteps       = args.timesteps      or preset["timesteps"]
    n_envs          = args.n_envs         or preset["n_envs"]
    batch_size      = args.batch_size     or preset["batch_size"]
    n_steps         = args.n_steps        or preset["n_steps"]
    eval_freq       = preset["eval_freq"]
    checkpoint_freq = preset["checkpoint_freq"]
    n_eval_episodes = preset["n_eval_episodes"]
    physics_substeps = preset["physics_substeps"]

    if args.continue_from:
        continue_training(
            model_path=args.continue_from,
            total_timesteps=timesteps,
            n_envs=n_envs,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            camera_size=args.camera_size,
            n_obstacles=args.n_obstacles,
            batch_size=batch_size,
            n_steps=n_steps,
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            n_eval_episodes=n_eval_episodes,
            device=args.device,
            physics_substeps=physics_substeps,
        )
    else:
        train_obstacle(
            total_timesteps=timesteps,
            n_envs=n_envs,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            camera_size=args.camera_size,
            n_obstacles=args.n_obstacles,
            learning_rate=args.lr,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=preset["n_epochs"],
            eval_freq=eval_freq,
            checkpoint_freq=checkpoint_freq,
            n_eval_episodes=n_eval_episodes,
            device=args.device,
            physics_substeps=physics_substeps,
        )

    print("\nTraining complete!")
