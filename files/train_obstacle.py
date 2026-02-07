"""
Training script for Spot with CNN Obstacle Detection

Uses the SpotObstacleEnv (Dict obs: image + proprioception) and the
SpotCNNExtractor custom feature extractor with PPO.

Usage:
    python train_obstacle.py --timesteps 2000000 --n_envs 4
    python train_obstacle.py --continue_from ./spot_obstacle_models/best_model.zip
"""

import os
import argparse
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from spot_obstacle_env import SpotObstacleEnv
from cnn_policy import SpotCNNExtractor


def make_env(rank, seed=0, camera_width=64, camera_height=64, n_obstacles=5):
    """Create a monitored SpotObstacleEnv instance."""
    def _init():
        env = SpotObstacleEnv(
            render_mode=None,
            camera_width=camera_width,
            camera_height=camera_height,
            n_obstacles_active=n_obstacles,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train_obstacle(
    total_timesteps=2_000_000,
    n_envs=4,
    save_dir="./spot_obstacle_models",
    log_dir="./spot_obstacle_logs",
    camera_size=64,
    n_obstacles=5,
    learning_rate=3e-4,
):
    """
    Train Spot to navigate toward goals while avoiding obstacles using
    a CNN front-camera + proprioceptive state policy.

    Args:
        total_timesteps: Total training steps.
        n_envs:          Number of parallel environments.
        save_dir:        Directory for model checkpoints.
        log_dir:         Directory for TensorBoard logs.
        camera_size:     Width and height of the camera image (square).
        n_obstacles:     Number of active obstacles per episode.
        learning_rate:   PPO learning rate.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("SPOT CNN OBSTACLE AVOIDANCE TRAINING")
    print("=" * 60)
    print(f"  Timesteps:       {total_timesteps:,}")
    print(f"  Parallel envs:   {n_envs}")
    print(f"  Camera:          {camera_size}x{camera_size} RGB")
    print(f"  Obstacles:       {n_obstacles}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Save dir:        {save_dir}")
    print(f"  Log dir:         {log_dir}")
    print("=" * 60)

    # Create vectorized environments
    # NOTE: rendering in sub-processes requires "spawn" start method on some
    # platforms. DummyVecEnv is safer for GPU-rendered envs when n_envs is small.
    env_fns = [
        make_env(i, camera_width=camera_size, camera_height=camera_size,
                 n_obstacles=n_obstacles)
        for i in range(n_envs)
    ]

    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Evaluation environment (single, deterministic)
    eval_env = DummyVecEnv([
        make_env(9999, camera_width=camera_size, camera_height=camera_size,
                 n_obstacles=n_obstacles)
    ])

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=save_dir,
        name_prefix="spot_obstacle",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

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
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )

    print("\nStarting training ...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            tb_log_name="spot_obstacle_cnn",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    final_path = os.path.join(save_dir, "spot_obstacle_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    env.close()
    eval_env.close()
    return model


def continue_training(
    model_path,
    total_timesteps=1_000_000,
    n_envs=4,
    save_dir="./spot_obstacle_models",
    log_dir="./spot_obstacle_logs",
    camera_size=64,
    n_obstacles=5,
):
    """Resume training from a saved CNN obstacle model."""
    print(f"Loading model from: {model_path}")

    env_fns = [
        make_env(i, camera_width=camera_size, camera_height=camera_size,
                 n_obstacles=n_obstacles)
        for i in range(n_envs)
    ]
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    eval_env = DummyVecEnv([
        make_env(9999, camera_width=camera_size, camera_height=camera_size,
                 n_obstacles=n_obstacles)
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

    model = PPO.load(model_path, env=env, custom_objects=custom_objects)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=save_dir,
        name_prefix="spot_obstacle_continued",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print("\nContinuing training ...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            tb_log_name="spot_obstacle_cnn_continued",
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    final_path = os.path.join(save_dir, "spot_obstacle_continued_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spot with CNN Obstacle Detection")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total training timesteps (default: 2M)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
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
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Path to model checkpoint to continue training from")

    args = parser.parse_args()

    if args.continue_from:
        continue_training(
            model_path=args.continue_from,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            camera_size=args.camera_size,
            n_obstacles=args.n_obstacles,
        )
    else:
        train_obstacle(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            camera_size=args.camera_size,
            n_obstacles=args.n_obstacles,
            learning_rate=args.lr,
        )

    print("\nTraining complete!")
