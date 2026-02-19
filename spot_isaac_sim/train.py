"""
Training Script for Spot Navigation with Depth Cameras
========================================================
Uses PPO with a custom CNN+MLP policy to train Spot robot
for autonomous navigation in randomized indoor environments.

Architecture:
    5 Depth Images (5×120×160) → CNN Encoder → 256-dim features
    Proprioception (37-dim)    → MLP Encoder → 64-dim features
                                                    ↓
                                              Combined (320-dim)
                                                    ↓
                                              MLP (256 → 128)
                                                    ↓
                                           Policy head → 12 actions
                                           Value head  → 1 value

Usage:
    # Basic training
    python -m spot_isaac_sim.train

    # Custom settings
    python -m spot_isaac_sim.train --timesteps 20000000 --n_envs 8

    # Continue from checkpoint
    python -m spot_isaac_sim.train --continue_from ./spot_isaac_models/best_model
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from . import config
from .navigation_env import SpotNavigationEnv


# ============================================================================
# CUSTOM CNN+MLP FEATURE EXTRACTOR
# ============================================================================

class DepthProprioExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observation space.

    Processes:
    1. Depth images (5, 120, 160) through a CNN → 256-dim
    2. Proprioception (37,) through an MLP → 64-dim
    3. Concatenates → 320-dim feature vector

    The CNN uses the depth images to understand spatial surroundings.
    The MLP encodes the robot's body state and goal direction.
    Combined, the policy knows both WHERE obstacles are and
    WHAT the robot's current state is.
    """

    def __init__(self, observation_space: spaces.Dict,
                 cnn_features_dim: int = 256,
                 proprio_features_dim: int = 64):
        # Compute total features dim
        features_dim = cnn_features_dim + proprio_features_dim
        super().__init__(observation_space, features_dim=features_dim)

        depth_space = observation_space["depth"]
        proprio_space = observation_space["proprio"]

        n_channels = depth_space.shape[0]   # 5 cameras
        height = depth_space.shape[1]       # 120
        width = depth_space.shape[2]        # 160

        # ── CNN for depth images ──
        # Input: (batch, 5, 120, 160) — 5 cameras as channels
        self.depth_cnn = nn.Sequential(
            # Conv1: 5 → 32 channels, 8×8 kernel, stride 4
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ELU(),

            # Conv2: 32 → 64 channels, 4×4 kernel, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ELU(),

            # Conv3: 64 → 64 channels, 3×3 kernel, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),

            nn.Flatten(),
        )

        # Compute CNN output size by doing a forward pass
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, height, width)
            cnn_out_size = self.depth_cnn(sample).shape[1]

        # Linear projection to features_dim
        self.depth_linear = nn.Sequential(
            nn.Linear(cnn_out_size, cnn_features_dim),
            nn.ELU(),
        )

        # ── MLP for proprioception ──
        proprio_dim = proprio_space.shape[0]  # 37
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ELU(),
            nn.Linear(128, proprio_features_dim),
            nn.ELU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through both encoders.

        Args:
            observations: Dict with "depth" and "proprio" tensors

        Returns:
            Combined feature vector (batch, features_dim)
        """
        depth = observations["depth"]
        proprio = observations["proprio"]

        # Normalize depth to [0, 1]
        max_range = config.CAMERA_RIG["max_range"]
        depth_normalized = depth / max_range

        # CNN forward
        depth_features = self.depth_cnn(depth_normalized)
        depth_features = self.depth_linear(depth_features)

        # MLP forward
        proprio_features = self.proprio_mlp(proprio)

        # Concatenate
        combined = torch.cat([depth_features, proprio_features], dim=1)
        return combined


# ============================================================================
# CURRICULUM CALLBACK
# ============================================================================

class CurriculumCallback(BaseCallback):
    """
    Logs curriculum stage transitions during training.
    The actual curriculum logic is in the environment.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._current_stage = None

    def _on_step(self) -> bool:
        # Log curriculum info from environment
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped
            if hasattr(env, '_total_timesteps'):
                stage = env._get_curriculum_stage()
                if stage and stage.get("name") != self._current_stage:
                    self._current_stage = stage["name"]
                    if self.verbose > 0:
                        print(f"\n[Curriculum] Advanced to: {self._current_stage}")
                    self.logger.record(
                        "curriculum/stage", self._current_stage
                    )

        return True


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def make_env(rank, seed=0, headless=True):
    """
    Factory function for creating training environments.

    Args:
        rank: Environment index (for parallel envs)
        seed: Random seed
        headless: Run without rendering

    Returns:
        Callable that creates a wrapped environment
    """
    def _init():
        env = SpotNavigationEnv(headless=headless)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train(
    total_timesteps=None,
    n_envs=None,
    save_dir=None,
    log_dir=None,
    headless=True,
    continue_from=None,
):
    """
    Train Spot navigation policy.

    Args:
        total_timesteps: Total training steps (default from config)
        n_envs: Number of parallel environments
        save_dir: Model save directory
        log_dir: TensorBoard log directory
        headless: Run without rendering
        continue_from: Path to model to continue training from
    """
    train_cfg = config.TRAINING
    total_timesteps = total_timesteps or train_cfg["total_timesteps"]
    n_envs = n_envs or train_cfg["n_envs"]
    save_dir = save_dir or train_cfg["save_dir"]
    log_dir = log_dir or train_cfg["log_dir"]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("SPOT NAVIGATION TRAINING — ISAAC SIM + DEPTH CAMERAS")
    print("=" * 60)
    print(f"  Total timesteps:     {total_timesteps:,}")
    print(f"  Parallel envs:       {n_envs}")
    print(f"  Save directory:      {save_dir}")
    print(f"  Log directory:       {log_dir}")
    print(f"  Headless:            {headless}")
    print(f"  Continue from:       {continue_from or 'scratch'}")
    print()
    print("  Camera config:")
    for cam in config.CAMERA_RIG["cameras"]:
        print(f"    {cam['name']:15s} → yaw={cam['rotation_deg'][2]:+6.1f}°")
    print()
    print("  Curriculum stages:")
    for stage in train_cfg["curriculum"]["stages"]:
        print(f"    {stage['name']:30s} @ {stage['timesteps']:>10,} steps")
    print("=" * 60)

    # ── Create environments ──
    print(f"\nCreating {n_envs} parallel environments...")
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, headless=headless) for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(0, headless=headless)])

    # Evaluation environment
    eval_env = DummyVecEnv([make_env(9999, headless=True)])

    # ── Callbacks ──
    checkpoint_callback = CheckpointCallback(
        save_freq=max(train_cfg["checkpoint_freq"] // n_envs, 1),
        save_path=save_dir,
        name_prefix="spot_nav",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(train_cfg["eval_freq"] // n_envs, 1),
        n_eval_episodes=train_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    curriculum_callback = CurriculumCallback(verbose=1)

    # ── Create or load model ──
    if continue_from:
        print(f"\nLoading model from: {continue_from}")
        model = PPO.load(continue_from, env=env)
    else:
        print("\nCreating new PPO model with CNN+MLP policy...")

        policy_kwargs = dict(
            features_extractor_class=DepthProprioExtractor,
            features_extractor_kwargs=dict(
                cnn_features_dim=train_cfg["cnn_features_dim"],
                proprio_features_dim=64,
            ),
            net_arch=dict(
                pi=train_cfg["combined_mlp_layers"],
                vf=train_cfg["combined_mlp_layers"],
            ),
            activation_fn=nn.ELU,
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=train_cfg["learning_rate"],
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            n_epochs=train_cfg["n_epochs"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            ent_coef=train_cfg["ent_coef"],
            vf_coef=train_cfg["vf_coef"],
            max_grad_norm=train_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # Print model summary
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(
        p.numel() for p in model.policy.parameters() if p.requires_grad
    )
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {model.device}")

    # ── Train ──
    print("\nStarting training...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, curriculum_callback],
            tb_log_name="spot_depth_nav",
            progress_bar=True,
            reset_num_timesteps=continue_from is None,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # ── Save final model ──
    final_path = os.path.join(save_dir, "spot_nav_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    env.close()
    eval_env.close()

    return model


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Spot Robot Navigation with Depth Cameras"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps (default: from config)",
    )
    parser.add_argument(
        "--n_envs", type=int, default=None,
        help="Number of parallel environments (default: from config)",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save models",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--continue_from", type=str, default=None,
        help="Path to model to continue training from",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable rendering (slower, for debugging)",
    )

    args = parser.parse_args()

    model = train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        headless=not args.render,
        continue_from=args.continue_from,
    )

    print("\nTraining complete!")
