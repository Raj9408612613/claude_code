"""
Evaluation results for Spot Robot Training
Reads EvalCallback outputs and test_policy results, saves plots + CSV summary.

Usage:
    python eval_results.py                          # uses default paths
    python eval_results.py --model ./spot_models/best_model
    python eval_results.py --log_dir ./spot_logs --model ./spot_models/best_model
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime


def load_eval_data(log_dir="./spot_logs"):
    """Load evaluations.npz saved by EvalCallback during training."""
    eval_path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(eval_path):
        print(f"[WARNING] No evaluations.npz found at {eval_path}")
        print("  -> Make sure training ran with EvalCallback and log_dir is correct.")
        return None
    data = np.load(eval_path)
    return data  # keys: timesteps, results (rewards), ep_lengths


def plot_training_curves(eval_data, output_dir="./spot_results"):
    """Plot reward and episode length curves from EvalCallback data."""
    os.makedirs(output_dir, exist_ok=True)

    timesteps   = eval_data["timesteps"]
    results     = eval_data["results"]      # shape: (n_evals, n_eval_episodes)
    ep_lengths  = eval_data["ep_lengths"]   # shape: (n_evals, n_eval_episodes)

    mean_rewards = results.mean(axis=1)
    std_rewards  = results.std(axis=1)
    mean_lengths = ep_lengths.mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Reward curve
    axes[0].plot(timesteps, mean_rewards, label="Mean Reward", color="steelblue")
    axes[0].fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3, color="steelblue", label="±1 std"
    )
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Training Reward Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episode length curve
    axes[1].plot(timesteps, mean_lengths, label="Mean Episode Length", color="darkorange")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Episode Length Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Saved] Training curves -> {plot_path}")
    return plot_path


def run_test_episodes(model_path, n_episodes=10, output_dir="./spot_results"):
    """Run test episodes with trained model and return stats."""
    try:
        from stable_baselines3 import PPO
        from spot_env import SpotEnv
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return None

    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)
    env = SpotEnv(render_mode=None)

    rows = []
    success_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward, steps, done = 0, 0, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

        final_dist = info["distance_to_goal"]
        success = final_dist < 0.5
        if success:
            success_count += 1

        rows.append({
            "episode":       ep + 1,
            "reward":        round(ep_reward, 3),
            "steps":         steps,
            "final_dist_m":  round(final_dist, 3),
            "success":       int(success),
        })
        print(f"  Ep {ep+1:>2}: reward={ep_reward:7.2f}  steps={steps:4d}  "
              f"dist={final_dist:.2f}m  {'✓' if success else '✗'}")

    env.close()

    rewards = [r["reward"] for r in rows]
    lengths = [r["steps"]  for r in rows]

    summary = {
        "episodes":          n_episodes,
        "success_rate":      round(success_count / n_episodes, 3),
        "mean_reward":       round(np.mean(rewards), 3),
        "std_reward":        round(np.std(rewards), 3),
        "min_reward":        round(np.min(rewards), 3),
        "max_reward":        round(np.max(rewards), 3),
        "mean_ep_length":    round(np.mean(lengths), 1),
    }

    # Save episode CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "test_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] Episode data   -> {csv_path}")

    return summary, rows


def save_summary(summary, eval_data=None, output_dir="./spot_results"):
    """Save a plain-text summary report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "summary_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("SPOT ROBOT TRAINING — RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        if eval_data is not None:
            timesteps  = eval_data["timesteps"]
            results    = eval_data["results"]
            f.write("--- Training Evaluation (EvalCallback) ---\n")
            f.write(f"  Total eval checkpoints : {len(timesteps)}\n")
            f.write(f"  Final timestep          : {timesteps[-1]:,}\n")
            f.write(f"  Final mean reward       : {results[-1].mean():.2f} "
                    f"± {results[-1].std():.2f}\n")
            f.write(f"  Best mean reward        : {results.mean(axis=1).max():.2f} "
                    f"(at step {timesteps[results.mean(axis=1).argmax()]:,})\n\n")

        if summary:
            f.write("--- Test Policy Results ---\n")
            for k, v in summary.items():
                f.write(f"  {k:<22}: {v}\n")

    print(f"[Saved] Summary report  -> {report_path}")
    return report_path


def plot_test_results(rows, output_dir="./spot_results"):
    """Bar + scatter plots from test episodes."""
    episodes = [r["episode"]  for r in rows]
    rewards  = [r["reward"]   for r in rows]
    dists    = [r["final_dist_m"] for r in rows]
    colors   = ["green" if r["success"] else "red" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(episodes, rewards, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(np.mean(rewards), color="blue", linestyle="--", label=f"Mean: {np.mean(rewards):.1f}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Test Episode Rewards\n(green=success, red=fail)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].scatter(episodes, dists, c=colors, edgecolors="black", linewidths=0.5, s=80)
    axes[1].axhline(0.5, color="blue", linestyle="--", label="Success threshold (0.5m)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Final Distance to Goal (m)")
    axes[1].set_title("Final Distance to Goal")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "test_results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] Test result plot-> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spot Robot — Evaluation Results")
    parser.add_argument("--model",      type=str, default="./spot_models/best_model",
                        help="Path to trained model (no .zip extension needed)")
    parser.add_argument("--log_dir",    type=str, default="./spot_logs",
                        help="Directory containing evaluations.npz")
    parser.add_argument("--output_dir", type=str, default="./spot_results",
                        help="Where to save results")
    parser.add_argument("--episodes",   type=int, default=10,
                        help="Test episodes to run (default: 10)")
    parser.add_argument("--skip_test",  action="store_true",
                        help="Only plot training curves, skip live test episodes")
    args = parser.parse_args()

    print("=" * 50)
    print("SPOT ROBOT — EVALUATION RESULTS")
    print("=" * 50)

    # 1. Training curves from EvalCallback
    eval_data = load_eval_data(args.log_dir)
    if eval_data is not None:
        plot_training_curves(eval_data, args.output_dir)

    # 2. Live test episodes
    summary, rows = None, None
    if not args.skip_test:
        result = run_test_episodes(args.model, args.episodes, args.output_dir)
        if result:
            summary, rows = result
            plot_test_results(rows, args.output_dir)

            print("\n--- Summary ---")
            for k, v in summary.items():
                print(f"  {k:<22}: {v}")

    # 3. Text report
    save_summary(summary, eval_data, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}/")
