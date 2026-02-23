"""
2-Hour Validation Training for Spot Navigation — G5.xlarge ($1.31/hr)
=====================================================================
Budget: ~$2.62 total | GPU: A10G 24GB | Goal: Validate pipeline works

This script is Phase 1 training. It will:
  1. Run a compressed 3-stage curriculum in 2 hours
  2. Auto-stop at the 2-hour mark (saves model before exit)
  3. Print health checks every 5 minutes so you know if it's learning
  4. Save checkpoints every 25k steps (nothing is wasted if it crashes)
  5. Print a final diagnostic: GO / NO-GO for full training

Expected results after 2 hours:
  - ~500k-800k timesteps completed
  - Spot should learn basic walking + moving toward goals
  - Reward should be trending upward
  - If reward is flat or negative → reward function or env has a bug

Usage:
    python -m spot_isaac_sim.train_2hr

    # With custom save location
    python -m spot_isaac_sim.train_2hr --save_dir /my/models --log_dir /my/logs

    # Continue a previous 2hr run
    python -m spot_isaac_sim.train_2hr --continue_from ./spot_2hr_models/best_model
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from . import config
from .train import DepthProprioExtractor, make_env


# ============================================================================
# CONSTANTS — 2 HOUR BUDGET
# ============================================================================

MAX_WALL_TIME_SECONDS = 2 * 60 * 60       # 2 hours hard limit
HEALTH_CHECK_INTERVAL = 5 * 60            # Print diagnostics every 5 minutes
TARGET_TIMESTEPS = 800_000                 # Optimistic target for 2 hours
CHECKPOINT_FREQ = 25_000                   # Save every 25k steps
EVAL_FREQ = 25_000                         # Evaluate every 25k steps
N_EVAL_EPISODES = 5                        # Quick evals (5 episodes, not 10)
N_ENVS = 2                                 # 2 parallel envs (safe for 24GB)


# ============================================================================
# COMPRESSED CURRICULUM — FITS IN ~800K STEPS
# ============================================================================

CURRICULUM_2HR = {
    "enabled": True,
    "stages": [
        {
            "name": "stage_1_learn_to_walk",
            "timesteps": 0,
            "obstacle_count": [0, 0],          # No obstacles — just walk
            "dynamic_count": [0, 0],
            "goal_distance": [1.5, 3.0],       # Very close goals
            "noise_multiplier": 0.3,           # Clean sensors
        },
        {
            "name": "stage_2_walk_to_goals",
            "timesteps": 200_000,
            "obstacle_count": [0, 3],          # A few obstacles
            "dynamic_count": [0, 0],
            "goal_distance": [2.0, 5.0],       # Medium goals
            "noise_multiplier": 0.5,
        },
        {
            "name": "stage_3_navigate_obstacles",
            "timesteps": 500_000,
            "obstacle_count": [3, 8],          # Real navigation
            "dynamic_count": [0, 1],           # Maybe one moving obstacle
            "goal_distance": [2.0, 7.0],
            "noise_multiplier": 0.75,
        },
    ],
}


# ============================================================================
# HEALTH CHECK CALLBACK — TELLS YOU IF TRAINING IS WORKING
# ============================================================================

class HealthCheckCallback(BaseCallback):
    """
    Prints clear diagnostics every 5 minutes.
    Tracks reward trends and warns if training looks broken.
    """

    def __init__(self, max_wall_time=MAX_WALL_TIME_SECONDS, verbose=1):
        super().__init__(verbose)
        self.max_wall_time = max_wall_time
        self.start_time = None
        self.last_check_time = None
        self.reward_history = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_mean_reward = -float("inf")
        self.checks_performed = 0

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_check_time = self.start_time
        print("\n" + "=" * 60)
        print("  2-HOUR TRAINING STARTED")
        print(f"  Auto-stop at: {self.max_wall_time / 3600:.1f} hours")
        print(f"  Health checks every: {HEALTH_CHECK_INTERVAL / 60:.0f} minutes")
        print("=" * 60 + "\n")

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time

        # ── Collect episode rewards from Monitor wrapper ──
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.reward_history.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    self.episode_count += 1

        # ── Time-based auto-stop ──
        if elapsed >= self.max_wall_time:
            print("\n" + "!" * 60)
            print("  2-HOUR TIME LIMIT REACHED — SAVING AND STOPPING")
            print("!" * 60)
            return False  # Stops training

        # ── Periodic health check ──
        if (time.time() - self.last_check_time) >= HEALTH_CHECK_INTERVAL:
            self._print_health_check(elapsed)
            self.last_check_time = time.time()

        return True

    def _print_health_check(self, elapsed):
        self.checks_performed += 1
        hours = elapsed / 3600
        minutes = elapsed / 60
        remaining = (self.max_wall_time - elapsed) / 60
        timesteps = self.num_timesteps
        cost_so_far = hours * 1.31

        print("\n" + "-" * 60)
        print(f"  HEALTH CHECK #{self.checks_performed}"
              f"  |  {minutes:.0f}min elapsed  |  {remaining:.0f}min remaining")
        print("-" * 60)
        print(f"  Timesteps:       {timesteps:>10,}")
        print(f"  Episodes:        {self.episode_count:>10,}")
        print(f"  Cost so far:     ${cost_so_far:>9.2f}")

        if len(self.reward_history) >= 5:
            recent = self.reward_history[-20:]
            mean_reward = np.mean(recent)
            std_reward = np.std(recent)
            mean_length = np.mean(self.episode_lengths[-20:])

            # Compare to earlier rewards
            if len(self.reward_history) >= 40:
                old = self.reward_history[-40:-20]
                old_mean = np.mean(old)
                trend = mean_reward - old_mean
                trend_str = f"+{trend:.1f}" if trend >= 0 else f"{trend:.1f}"
            else:
                trend_str = "N/A (need more data)"

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

            print(f"  Mean reward:     {mean_reward:>10.1f} (+/- {std_reward:.1f})")
            print(f"  Best mean:       {self.best_mean_reward:>10.1f}")
            print(f"  Reward trend:    {trend_str:>10s}")
            print(f"  Mean ep length:  {mean_length:>10.0f} steps")

            # Diagnosis
            if mean_reward > 10:
                status = "GOOD — reward is positive and meaningful"
            elif mean_reward > 0:
                status = "OK — learning something, reward is positive"
            elif len(self.reward_history) > 100 and mean_reward < -5:
                status = "WARNING — reward is negative, check reward function"
            else:
                status = "EARLY — still exploring, too soon to judge"

            print(f"  Status:          {status}")
        else:
            print(f"  Status:          WARMING UP (need more episodes)")

        print("-" * 60)

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        self._print_final_diagnostic(elapsed)

    def _print_final_diagnostic(self, elapsed):
        hours = elapsed / 3600
        total_cost = hours * 1.31

        print("\n")
        print("=" * 60)
        print("  FINAL DIAGNOSTIC — 2-HOUR TRAINING RUN")
        print("=" * 60)
        print(f"  Wall time:         {hours:.2f} hours")
        print(f"  Total cost:        ${total_cost:.2f}")
        print(f"  Timesteps:         {self.num_timesteps:,}")
        print(f"  Episodes:          {self.episode_count:,}")

        if len(self.reward_history) < 10:
            print("\n  RESULT: NOT ENOUGH DATA")
            print("  Too few episodes completed. Possible issues:")
            print("    - Isaac Sim environment is crashing")
            print("    - Episodes are extremely long (timeout too high)")
            print("    - Environment reset is hanging")
            print("=" * 60)
            return

        # Split rewards into first half and second half
        mid = len(self.reward_history) // 2
        first_half = np.mean(self.reward_history[:mid])
        second_half = np.mean(self.reward_history[mid:])
        overall_mean = np.mean(self.reward_history)
        overall_best = np.max(self.reward_history)

        print(f"\n  Reward (first half):   {first_half:.1f}")
        print(f"  Reward (second half):  {second_half:.1f}")
        print(f"  Reward (overall):      {overall_mean:.1f}")
        print(f"  Best single episode:   {overall_best:.1f}")
        print(f"  Total episodes:        {len(self.reward_history)}")

        # ── GO / NO-GO decision ──
        improving = second_half > first_half
        positive = second_half > 0
        significant_improvement = (second_half - first_half) > 2.0

        print("\n" + "-" * 60)

        if significant_improvement and positive:
            print("  VERDICT: GO FOR FULL TRAINING")
            print("  Reward is improving and positive.")
            print("  Next step: run full training on g5.4xlarge")
            print("  Estimated full training: 60-80 hours (~$103-137)")
        elif improving:
            print("  VERDICT: CAUTIOUS GO")
            print("  Reward is improving but slowly.")
            print("  Consider:")
            print("    - Run another 2hr session to confirm trend")
            print("    - Tweak reward weights (increase progress_weight)")
            print("    - Check if curriculum stages are advancing")
        elif positive and not improving:
            print("  VERDICT: PLATEAU — NEEDS TUNING")
            print("  Reward is positive but not improving.")
            print("  Consider:")
            print("    - Increase learning rate slightly (5e-4)")
            print("    - Increase entropy coefficient (0.02)")
            print("    - Check curriculum — may be stuck on one stage")
        else:
            print("  VERDICT: NO-GO — FIX BEFORE SCALING")
            print("  Reward is not improving or is negative.")
            print("  DO NOT run full training — you will waste money.")
            print("  Debug checklist:")
            print("    - Is the robot spawning correctly? (check height)")
            print("    - Is the goal reachable? (check placement)")
            print("    - Is the reward function returning sensible values?")
            print("    - Are observations normalized properly?")
            print("    - Try: increase alive_bonus, decrease collision_penalty")

        print("-" * 60)
        print("=" * 60)


# ============================================================================
# CURRICULUM CALLBACK (COMPRESSED 3-STAGE)
# ============================================================================

class CompressedCurriculumCallback(BaseCallback):
    """Logs curriculum transitions for the compressed 3-stage curriculum."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._current_stage = None

    def _on_step(self) -> bool:
        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0]
            if hasattr(env, "unwrapped"):
                env = env.unwrapped
            if hasattr(env, "_total_timesteps"):
                stage = env._get_curriculum_stage()
                if stage and stage.get("name") != self._current_stage:
                    self._current_stage = stage["name"]
                    if self.verbose > 0:
                        print(f"\n  [Curriculum] >>> {self._current_stage}")
                    self.logger.record("curriculum/stage", self._current_stage)
        return True


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_2hr(
    headless=True,
    continue_from=None,
    save_dir="./spot_2hr_models",
    log_dir="./spot_2hr_logs",
):
    """
    Run a 2-hour validation training session.

    This applies memory-optimized settings for g5.xlarge (24GB A10G)
    and a compressed curriculum designed to show learning signal in 2 hours.
    """

    # ── Apply 2-hour optimized config ──
    config.apply_overrides(
        total_timesteps=TARGET_TIMESTEPS,
        n_envs=N_ENVS,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        save_dir=save_dir,
        log_dir=log_dir,
        # Shorter episodes = faster learning signal
        max_episode_steps=500,
        # Slightly higher entropy for more exploration in 2 hours
        ent_coef=0.015,
    )

    # Inject compressed curriculum
    config.TRAINING["curriculum"] = CURRICULUM_2HR

    # ── Start Isaac Sim ──
    from . import ensure_isaac_sim
    ensure_isaac_sim(headless=headless)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_cfg = config.TRAINING

    print("\n" + "=" * 60)
    print("  SPOT 2-HOUR VALIDATION TRAINING")
    print("  GPU: g5.xlarge (A10G 24GB) — $1.31/hr")
    print("  Budget: ~$2.62 for 2 hours")
    print("=" * 60)
    print(f"  Target timesteps:    {TARGET_TIMESTEPS:,}")
    print(f"  Parallel envs:       {N_ENVS}")
    print(f"  Max episode steps:   500")
    print(f"  Checkpoint freq:     {CHECKPOINT_FREQ:,}")
    print(f"  Eval freq:           {EVAL_FREQ:,}")
    print(f"  Save directory:      {save_dir}")
    print(f"  Continue from:       {continue_from or 'scratch'}")
    print()
    print("  Compressed curriculum (3 stages):")
    for stage in CURRICULUM_2HR["stages"]:
        obs_lo, obs_hi = stage["obstacle_count"]
        goal_lo, goal_hi = stage["goal_distance"]
        print(f"    {stage['name']:30s} @ {stage['timesteps']:>8,} steps"
              f"  |  obs: {obs_lo}-{obs_hi}  |  goal: {goal_lo}-{goal_hi}m")
    print("=" * 60)

    # ── Create environments ──
    print(f"\nCreating {N_ENVS} parallel environments...")
    if N_ENVS > 1:
        env = SubprocVecEnv([
            make_env(i, headless=headless) for i in range(N_ENVS)
        ])
    else:
        env = DummyVecEnv([make_env(0, headless=headless)])

    eval_env = DummyVecEnv([make_env(9999, headless=True)])

    # ── Callbacks ──
    health_check = HealthCheckCallback(
        max_wall_time=MAX_WALL_TIME_SECONDS, verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=save_dir,
        name_prefix="spot_2hr",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    curriculum_cb = CompressedCurriculumCallback(verbose=1)

    # ── Create or load model ──
    if continue_from:
        print(f"\nLoading model from: {continue_from}")
        model = PPO.load(continue_from, env=env)
    else:
        print("\nCreating new PPO model (CNN+MLP)...")
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch.nn as nn

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

    # Model summary
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Device:               {model.device}")

    # ── Train (auto-stops at 2 hours via HealthCheckCallback) ──
    print("\nStarting 2-hour validation training...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=TARGET_TIMESTEPS,
            callback=[health_check, checkpoint_cb, eval_cb, curriculum_cb],
            tb_log_name="spot_2hr_validation",
            progress_bar=True,
            reset_num_timesteps=continue_from is None,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # ── Save final model ──
    final_path = os.path.join(save_dir, "spot_2hr_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    # ── List all saved files ──
    print("\nSaved files:")
    for f in sorted(os.listdir(save_dir)):
        fpath = os.path.join(save_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f:40s}  ({size_mb:.1f} MB)")

    env.close()
    eval_env.close()

    return model


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spot 2-Hour Validation Training (g5.xlarge)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./spot_2hr_models",
        help="Directory to save models (default: ./spot_2hr_models)",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./spot_2hr_logs",
        help="Directory for TensorBoard logs (default: ./spot_2hr_logs)",
    )
    parser.add_argument(
        "--continue_from", type=str, default=None,
        help="Path to model to continue training from",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable rendering (slower, for visual debugging)",
    )
    parser.add_argument(
        "--n_envs", type=int, default=N_ENVS,
        help=f"Number of parallel environments (default: {N_ENVS})",
    )
    parser.add_argument(
        "--max_hours", type=float, default=2.0,
        help="Maximum training time in hours (default: 2.0)",
    )

    args = parser.parse_args()

    # Allow overriding time limit
    if args.max_hours != 2.0:
        MAX_WALL_TIME_SECONDS = int(args.max_hours * 3600)
        print(f"[Override] Max training time: {args.max_hours} hours")

    # Allow overriding n_envs
    if args.n_envs != N_ENVS:
        N_ENVS = args.n_envs

    model = train_2hr(
        headless=not args.render,
        continue_from=args.continue_from,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    print("\n2-hour validation training complete!")
    print("Check the FINAL DIAGNOSTIC above for GO / NO-GO decision.")
