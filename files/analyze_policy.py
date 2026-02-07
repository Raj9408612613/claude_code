"""
Analyze and visualize trained Spot robot behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from spot_env import SpotEnv
import os


def analyze_policy_behavior(model_path, n_episodes=20, save_plots=True):
    """
    Analyze trained policy behavior and create visualizations
    
    Args:
        model_path: Path to trained model
        n_episodes: Number of episodes to analyze
        save_plots: Whether to save plots to files
    """
    
    print(f"Analyzing policy from: {model_path}")
    print(f"Running {n_episodes} episodes for analysis...")
    print("=" * 50)
    
    # Load model
    model = PPO.load(model_path)
    env = SpotEnv(render_mode=None)
    
    # Collect data
    all_rewards = []
    all_distances = []
    all_heights = []
    all_tilts = []
    all_velocities = []
    all_joint_positions = []
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_distances = []
        episode_heights = []
        episode_tilts = []
        episode_velocities = []
        episode_joints = []
        steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Collect metrics
            episode_distances.append(info['distance_to_goal'])
            episode_heights.append(env.data.qpos[2])
            
            # Calculate tilt
            quat = env.data.qpos[3:7]
            w, x, y, z = quat
            roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
            tilt = np.sqrt(roll**2 + pitch**2)
            episode_tilts.append(tilt)
            
            # Velocity
            velocity = np.linalg.norm(env.data.qvel[0:2])
            episode_velocities.append(velocity)
            
            # Joint positions
            episode_joints.append(env.data.qpos[7:19].copy())
        
        # Episode finished
        all_rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Store episode data
        all_distances.append(episode_distances)
        all_heights.append(episode_heights)
        all_tilts.append(episode_tilts)
        all_velocities.append(episode_velocities)
        all_joint_positions.append(episode_joints)
        
        # Check success
        if info['distance_to_goal'] < 0.5:
            success_count += 1
        
        print(f"Episode {episode+1}/{n_episodes}: Reward={episode_reward:.1f}, "
              f"Length={steps}, Success={'Yes' if info['distance_to_goal'] < 0.5 else 'No'}")
    
    env.close()
    
    # Create visualizations
    print("\nGenerating analysis plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Spot Robot Policy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards Distribution
    ax = axes[0, 0]
    ax.hist(episode_rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
               label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode Lengths Distribution
    ax = axes[0, 1]
    ax.hist(episode_lengths, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(episode_lengths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Distance to Goal Over Time (sample episodes)
    ax = axes[1, 0]
    for i in range(min(5, len(all_distances))):
        ax.plot(all_distances[i], alpha=0.6, label=f'Episode {i+1}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Goal (m)')
    ax.set_title('Distance to Goal Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Body Height Over Time
    ax = axes[1, 1]
    for i in range(min(5, len(all_heights))):
        ax.plot(all_heights[i], alpha=0.6, label=f'Episode {i+1}')
    ax.axhline(0.35, color='red', linestyle='--', label='Target height', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Height (m)')
    ax.set_title('Body Height Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Body Tilt Over Time
    ax = axes[2, 0]
    for i in range(min(5, len(all_tilts))):
        ax.plot(np.rad2deg(all_tilts[i]), alpha=0.6, label=f'Episode {i+1}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tilt (degrees)')
    ax.set_title('Body Tilt Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Velocity Over Time
    ax = axes[2, 1]
    for i in range(min(5, len(all_velocities))):
        ax.plot(all_velocities[i], alpha=0.6, label=f'Episode {i+1}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Forward Velocity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = os.path.dirname(model_path)
        if not output_dir:
            output_dir = '.'
        plot_path = os.path.join(output_dir, 'policy_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Episodes analyzed: {n_episodes}")
    print(f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"\nRewards:")
    print(f"  Mean: {np.mean(episode_rewards):.2f}")
    print(f"  Std: {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    print(f"\nEpisode Lengths:")
    print(f"  Mean: {np.mean(episode_lengths):.1f} steps")
    print(f"  Std: {np.std(episode_lengths):.1f} steps")
    print(f"\nAverage Metrics (successful episodes only):")
    if success_count > 0:
        successful_heights = [all_heights[i] for i in range(n_episodes) 
                            if all_distances[i][-1] < 0.5]
        successful_velocities = [all_velocities[i] for i in range(n_episodes) 
                               if all_distances[i][-1] < 0.5]
        
        avg_heights = [np.mean(ep) for ep in successful_heights]
        avg_velocities = [np.mean(ep) for ep in successful_velocities]
        
        print(f"  Average height: {np.mean(avg_heights):.3f}m")
        print(f"  Average velocity: {np.mean(avg_velocities):.3f}m/s")
    print("=" * 50)
    
    return {
        'success_rate': success_count / n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'mean_length': np.mean(episode_lengths),
    }


def compare_models(model_paths, labels=None, n_episodes=10):
    """
    Compare multiple trained models
    
    Args:
        model_paths: List of paths to models
        labels: List of labels for each model
        n_episodes: Number of episodes per model
    """
    
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(model_paths))]
    
    print("Comparing models...")
    print("=" * 50)
    
    results = []
    
    for model_path, label in zip(model_paths, labels):
        print(f"\nEvaluating {label}...")
        model = PPO.load(model_path)
        env = SpotEnv(render_mode=None)
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for _ in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            if info['distance_to_goal'] < 0.5:
                success_count += 1
        
        env.close()
        
        results.append({
            'label': label,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / n_episodes,
            'mean_length': np.mean(episode_lengths),
        })
        
        print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Success rate: {100*success_count/n_episodes:.1f}%")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rewards comparison
    ax = axes[0]
    means = [r['mean_reward'] for r in results]
    stds = [r['std_reward'] for r in results]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Model Comparison: Rewards')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Success rate comparison
    ax = axes[1]
    success_rates = [100 * r['success_rate'] for r in results]
    ax.bar(x, success_rates, color='lightgreen', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Model Comparison: Success Rate')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to: model_comparison.png")
    plt.show()
    
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"\n{r['label']}:")
        print(f"  Mean reward: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"  Success rate: {100*r['success_rate']:.1f}%")
        print(f"  Mean length: {r['mean_length']:.1f} steps")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Spot Robot Policy')
    parser.add_argument('model_path', type=str, nargs='+',
                        help='Path(s) to trained model(s)')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes for analysis (default: 20)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple models')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Labels for models (when comparing)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots')
    
    args = parser.parse_args()
    
    if args.compare and len(args.model_path) > 1:
        compare_models(
            args.model_path,
            labels=args.labels,
            n_episodes=args.episodes
        )
    else:
        analyze_policy_behavior(
            args.model_path[0],
            n_episodes=args.episodes,
            save_plots=not args.no_save
        )
