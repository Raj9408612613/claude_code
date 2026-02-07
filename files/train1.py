"""
Training script for Spot Quadruped Robot
Uses PPO (Proximal Policy Optimization) from Stable Baselines3
"""

# Suppress common harmless warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from spot_env import SpotEnv


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = SpotEnv(render_mode=None)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train_spot(
    total_timesteps=5_000_000,
    n_envs=8,
    save_dir="./spot_models",
    log_dir="./spot_logs"
):
    """
    Train Spot robot to walk towards goal
    
    Args:
        total_timesteps: Total number of training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        log_dir: Directory for logs
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Training with {n_envs} parallel environments")
    print(f"Total timesteps: {total_timesteps:,}")
    
    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(9999)])
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,  # Save every 50k steps
        save_path=save_dir,
        name_prefix='spot_model',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10_000 // n_envs,  # Evaluate every 10k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh  # Use PyTorch activation class
        ),
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    print("\nStarting training...")
    print("=" * 50)
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name="spot_ppo",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "spot_final_model")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


def continue_training(
    model_path,
    total_timesteps=1_000_000,
    n_envs=8,
    save_dir="./spot_models",
    log_dir="./spot_logs"
):
    """
    Continue training from a saved model
    
    Args:
        model_path: Path to saved model
        total_timesteps: Additional timesteps to train
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        log_dir: Directory for logs
    """
    
    print(f"Loading model from: {model_path}")
    
    # Create environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    
    eval_env = DummyVecEnv([make_env(9999)])
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=save_dir,
        name_prefix='spot_model_continued',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    print("\nContinuing training...")
    print("=" * 50)
    
    # Continue training
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name="spot_ppo_continued",
            progress_bar=True,
            reset_num_timesteps=False,  # Continue from current timestep
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "spot_continued_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Spot Robot')
    parser.add_argument('--timesteps', type=int, default=5_000_000,
                        help='Total training timesteps (default: 5M)')
    parser.add_argument('--n_envs', type=int, default=8,
                        help='Number of parallel environments (default: 8)')
    parser.add_argument('--save_dir', type=str, default='./spot_models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./spot_logs',
                        help='Directory for logs')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='Path to model to continue training from')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SPOT ROBOT TRAINING")
    print("=" * 50)
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Save directory: {args.save_dir}")
    print(f"Log directory: {args.log_dir}")
    print("=" * 50)
    
    if args.continue_from:
        model = continue_training(
            model_path=args.continue_from,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
    else:
        model = train_spot(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
    
    print("\nTraining complete!")