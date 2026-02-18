"""
Full Pipeline: Train -> Evaluate -> Visualize in one command.

Designed for fast iteration on GPU-enabled machines (Colab, etc.).

Usage:
    python run_pipeline.py --fast                  # quick 200k-step cycle
    python run_pipeline.py --medium                # balanced 1M-step cycle
    python run_pipeline.py --full                  # full 2M-step training
    python run_pipeline.py --eval_only --model ./spot_obstacle_models/best_model.zip
    python run_pipeline.py --fast --skip_viz       # train+eval, skip GIF
"""

import os
import sys
import time
import argparse

# Ensure the files directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_train(args, preset_flag):
    """Run training with the chosen preset."""
    banner("STAGE 1 / 3 — TRAINING")

    from train_obstacle import train_obstacle, PRESETS

    preset = PRESETS[preset_flag]
    print(f"Preset: {preset_flag} — {preset['desc']}")

    model = train_obstacle(
        total_timesteps=args.timesteps or preset["timesteps"],
        n_envs=args.n_envs or preset["n_envs"],
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        camera_size=args.camera_size,
        n_obstacles=args.n_obstacles,
        learning_rate=args.lr,
        batch_size=args.batch_size or preset["batch_size"],
        n_steps=args.n_steps or preset["n_steps"],
        n_epochs=preset["n_epochs"],
        eval_freq=preset["eval_freq"],
        checkpoint_freq=preset["checkpoint_freq"],
        n_eval_episodes=preset["n_eval_episodes"],
        device=args.device,
        physics_substeps=preset["physics_substeps"],
    )

    # Return path to best model (or final if best doesn't exist)
    best = os.path.join(args.save_dir, "best_model.zip")
    final = os.path.join(args.save_dir, "spot_obstacle_final.zip")
    return best if os.path.exists(best) else final


def run_eval(model_path, n_episodes=10, output_dir="./spot_obstacle_results"):
    """Evaluate the trained model and save results."""
    banner("STAGE 2 / 3 — EVALUATION")

    from eval_obstacle import evaluate_obstacle_model

    return evaluate_obstacle_model(
        model_path=model_path,
        n_episodes=n_episodes,
        output_dir=output_dir,
    )


def run_quick_eval(model_path, n_episodes=5):
    """Lightweight eval that prints stats without saving files."""
    banner("STAGE 2 / 3 — QUICK EVALUATION")

    import numpy as np
    from stable_baselines3 import PPO
    from spot_obstacle_env import SpotObstacleEnv
    from stable_baselines3.common.monitor import Monitor
    from cnn_policy import SpotCNNExtractor

    custom_objects = {
        "policy_kwargs": dict(
            features_extractor_class=SpotCNNExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=128, proprio_hidden_dim=64),
        ),
    }
    model = PPO.load(model_path, custom_objects=custom_objects)

    env = SpotObstacleEnv(render_mode=None, camera_width=64, camera_height=64)

    rewards, lengths, successes, collisions = [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward, done = 0.0, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        dist = info["distance_to_goal"]
        success = dist < 0.5
        rewards.append(ep_reward)
        lengths.append(env.current_step)
        successes.append(success)
        collisions.append(info["obstacle_collisions"])

        status = "OK" if success else "FAIL"
        print(f"  Ep {ep+1:>2}: reward={ep_reward:8.1f}  steps={env.current_step:4d}  "
              f"dist={dist:.2f}m  collisions={info['obstacle_collisions']:2d}  [{status}]")

    env.close()

    print(f"\n  Success rate:     {sum(successes)}/{n_episodes} "
          f"({100*sum(successes)/n_episodes:.0f}%)")
    print(f"  Mean reward:      {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Mean ep length:   {np.mean(lengths):.0f} steps")
    print(f"  Mean collisions:  {np.mean(collisions):.1f}")

    return {
        "success_rate": sum(successes) / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_collisions": float(np.mean(collisions)),
    }


def run_viz(model_path, output_path="spot_obstacle_dual.gif", max_steps=300):
    """Generate a visualization GIF."""
    banner("STAGE 3 / 3 — VISUALIZATION")

    from test_obstacle_visual import run_visualization

    run_visualization(
        model_path=model_path,
        output_path=output_path,
        n_episodes=1,
        fps=25,
        max_steps=max_steps,
    )
    print(f"  Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: Train -> Evaluate -> Visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --fast                   # Quick 200k iteration
  python run_pipeline.py --medium                 # 1M balanced run
  python run_pipeline.py --full                   # Full 2M training
  python run_pipeline.py --eval_only --model best_model.zip
  python run_pipeline.py --fast --skip_viz        # No visualization
        """,
    )

    # Preset
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--fast", action="store_const", dest="preset", const="fast",
                        help="Quick iteration (200k steps)")
    preset.add_argument("--medium", action="store_const", dest="preset", const="medium",
                        help="Balanced (1M steps)")
    preset.add_argument("--full", action="store_const", dest="preset", const="full",
                        help="Full training (2M steps, default)")

    # Training overrides
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--camera_size", type=int, default=64)
    parser.add_argument("--n_obstacles", type=int, default=5)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])

    # Directories
    parser.add_argument("--save_dir", type=str, default="./spot_obstacle_models")
    parser.add_argument("--log_dir", type=str, default="./spot_obstacle_logs")
    parser.add_argument("--output_dir", type=str, default="./spot_obstacle_results")

    # Pipeline control
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only evaluate + visualize")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (required for --eval_only)")
    parser.add_argument("--skip_viz", action="store_true",
                        help="Skip visualization GIF generation")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--viz_steps", type=int, default=300,
                        help="Max steps in visualization (default: 300)")

    args = parser.parse_args()
    preset_flag = args.preset or "full"

    t_pipeline = time.time()

    # ----- TRAINING -----
    if args.eval_only:
        if args.model is None:
            parser.error("--model is required with --eval_only")
        model_path = args.model
    else:
        model_path = run_train(args, preset_flag)

    # ----- EVALUATION -----
    run_quick_eval(model_path, n_episodes=args.eval_episodes)

    # ----- VISUALIZATION -----
    if not args.skip_viz:
        viz_path = os.path.join(args.output_dir, "spot_obstacle_dual.gif")
        os.makedirs(args.output_dir, exist_ok=True)
        run_viz(model_path, output_path=viz_path, max_steps=args.viz_steps)

    elapsed = time.time() - t_pipeline
    banner(f"PIPELINE COMPLETE — {elapsed / 60:.1f} min total")
    print(f"  Model:  {model_path}")
    if not args.skip_viz:
        print(f"  Visual: {viz_path}")
    print()


if __name__ == "__main__":
    main()
