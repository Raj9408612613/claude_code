"""
Training script for Spot Quadruped Robot
Uses PPO (Proximal Policy Optimization) from Stable Baselines3
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import torch

# Allow importing from the same directory when run from Colab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spot_env import SpotEnv
from config import PPO_CONFIG, TRAINING_CONFIG, ENV_CONFIG


def make_env(rank, seed=0, max_episode_steps=None):
    """
    Utility function for multiprocessed env.
    """
    if max_episode_steps is None:
        max_episode_steps = ENV_CONFIG['max_episode_steps']

    def _init():
        env = SpotEnv(render_mode=None, max_episode_steps=max_episode_steps)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train_spot(
    total_timesteps=None,
    n_envs=None,
    save_dir=None,
    log_dir=None
):
    """
    Train Spot robot to walk towards goal.
    All parameters default to values from config.py.

    Args:
        total_timesteps: Total number of training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save models (use Google Drive path on Colab)
        log_dir: Directory for logs
    """
    # Use config defaults if not specified
    total_timesteps = total_timesteps or TRAINING_CONFIG['total_timesteps']
    n_envs = n_envs or TRAINING_CONFIG['n_envs']
    save_dir = save_dir or TRAINING_CONFIG['save_dir']
    log_dir = log_dir or TRAINING_CONFIG['log_dir']
    max_episode_steps = ENV_CONFIG['max_episode_steps']
    net_arch = PPO_CONFIG['policy_net_arch']

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Training with {n_envs} parallel environments")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Max episode steps: {max_episode_steps}")
    print(f"Network architecture: {net_arch}")

    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, max_episode_steps=max_episode_steps) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, max_episode_steps=max_episode_steps)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(9999, max_episode_steps=max_episode_steps)])

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG['checkpoint_freq'] // n_envs,
        save_path=save_dir,
        name_prefix='spot_model',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=TRAINING_CONFIG['eval_freq'] // n_envs,
        n_eval_episodes=TRAINING_CONFIG['n_eval_episodes'],
        deterministic=True,
        render=False,
    )

    # Create PPO model using config values
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=PPO_CONFIG['learning_rate'],
        n_steps=PPO_CONFIG['n_steps'],
        batch_size=PPO_CONFIG['batch_size'],
        n_epochs=PPO_CONFIG['n_epochs'],
        gamma=PPO_CONFIG['gamma'],
        gae_lambda=PPO_CONFIG['gae_lambda'],
        clip_range=PPO_CONFIG['clip_range'],
        clip_range_vf=None,
        ent_coef=PPO_CONFIG['ent_coef'],
        vf_coef=PPO_CONFIG['vf_coef'],
        max_grad_norm=PPO_CONFIG['max_grad_norm'],
        use_sde=PPO_CONFIG['use_sde'],
        policy_kwargs=dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
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
    total_timesteps=None,
    n_envs=None,
    save_dir=None,
    log_dir=None
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
    
    total_timesteps = total_timesteps or TRAINING_CONFIG['total_timesteps']
    n_envs = n_envs or TRAINING_CONFIG['n_envs']
    save_dir = save_dir or TRAINING_CONFIG['save_dir']
    log_dir = log_dir or TRAINING_CONFIG['log_dir']
    max_episode_steps = ENV_CONFIG['max_episode_steps']

    print(f"Loading model from: {model_path}")

    # Create environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, max_episode_steps=max_episode_steps) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, max_episode_steps=max_episode_steps)])

    eval_env = DummyVecEnv([make_env(9999, max_episode_steps=max_episode_steps)])
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG['checkpoint_freq'] // n_envs,
        save_path=save_dir,
        name_prefix='spot_model_continued',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=TRAINING_CONFIG['eval_freq'] // n_envs,
        n_eval_episodes=TRAINING_CONFIG['n_eval_episodes'],
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
    parser.add_argument('--timesteps', type=int, default=None,
                        help=f'Total training timesteps (default: {TRAINING_CONFIG["total_timesteps"]:,} from config)')
    parser.add_argument('--n_envs', type=int, default=None,
                        help=f'Number of parallel environments (default: {TRAINING_CONFIG["n_envs"]} from config)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for logs')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='Path to model to continue training from')

    args = parser.parse_args()

    timesteps = args.timesteps or TRAINING_CONFIG['total_timesteps']
    n_envs = args.n_envs or TRAINING_CONFIG['n_envs']
    save_dir = args.save_dir or TRAINING_CONFIG['save_dir']
    log_dir = args.log_dir or TRAINING_CONFIG['log_dir']

    print("=" * 50)
    print("SPOT ROBOT TRAINING")
    print("=" * 50)
    print(f"Total timesteps: {timesteps:,}")
    print(f"Max episode steps: {ENV_CONFIG['max_episode_steps']}")
    print(f"Network: {PPO_CONFIG['policy_net_arch']}")
    print(f"Parallel environments: {n_envs}")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {log_dir}")
    print("=" * 50)

    if args.continue_from:
        model = continue_training(
            model_path=args.continue_from,
            total_timesteps=timesteps,
            n_envs=n_envs,
            save_dir=save_dir,
            log_dir=log_dir
        )
    else:
        model = train_spot(
            total_timesteps=timesteps,
            n_envs=n_envs,
            save_dir=save_dir,
            log_dir=log_dir
        )

    print("\nTraining complete!")
