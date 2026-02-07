"""
Test and visualize trained Spot robot
"""

import numpy as np
import time
from stable_baselines3 import PPO
from spot_env import SpotEnv


def test_policy(model_path, n_episodes=5, render=True):
    """
    Test a trained policy
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to run
        render: Whether to render visualization
    """
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with rendering
    render_mode = 'human' if render else None
    env = SpotEnv(render_mode=render_mode)
    
    print(f"\nRunning {n_episodes} test episodes...")
    print("=" * 50)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print(f"Goal position: ({info['goal_position'][0]:.2f}, {info['goal_position'][1]:.2f})")
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Print progress every 50 steps
            if steps % 50 == 0:
                dist = info['distance_to_goal']
                pos = info['base_position']
                print(f"  Step {steps}: Position ({pos[0]:.2f}, {pos[1]:.2f}), "
                      f"Distance to goal: {dist:.2f}m, Reward: {episode_reward:.2f}")
            
            if render:
                time.sleep(0.02)  # Slow down for visualization
        
        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        final_dist = info['distance_to_goal']
        success = final_dist < 0.5
        if success:
            success_count += 1
        
        print(f"\n  Episode finished!")
        print(f"  Total steps: {steps}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Final distance to goal: {final_dist:.2f}m")
        print(f"  Status: {'SUCCESS!' if success else 'Failed'}")
        print("=" * 50)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Episodes: {n_episodes}")
    print(f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print("=" * 50)
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_count / n_episodes,
    }


def test_random_policy(n_episodes=3, render=True):
    """
    Test random actions to see baseline performance
    """
    
    print("Testing RANDOM policy (baseline)...")
    print("=" * 50)
    
    render_mode = 'human' if render else None
    env = SpotEnv(render_mode=render_mode)
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done and steps < 200:  # Limit steps for random
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            if render:
                time.sleep(0.02)
        
        episode_rewards.append(episode_reward)
        print(f"  Total reward: {episode_reward:.2f}")
    
    print(f"\nRandom policy average reward: {np.mean(episode_rewards):.2f}")
    print("=" * 50)
    
    env.close()


def interactive_test(model_path):
    """
    Interactive testing - press Enter to reset to new goal
    """
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    env = SpotEnv(render_mode='human')
    
    print("\nInteractive mode - Press Ctrl+C to exit")
    print("Robot will walk towards randomly generated goals")
    print("=" * 50)
    
    try:
        while True:
            obs, info = env.reset()
            print(f"\nNew goal: ({info['goal_position'][0]:.2f}, {info['goal_position'][1]:.2f})")
            
            done = False
            steps = 0
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                time.sleep(0.02)
                
                if steps % 100 == 0:
                    dist = info['distance_to_goal']
                    print(f"  Step {steps}, Distance: {dist:.2f}m")
            
            final_dist = info['distance_to_goal']
            success = final_dist < 0.5
            
            print(f"\nEpisode finished: {steps} steps, Reward: {total_reward:.2f}")
            print(f"Final distance: {final_dist:.2f}m - {'SUCCESS!' if success else 'Failed'}")
            print("\nStarting new episode in 2 seconds...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nExiting interactive mode...")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Spot Robot')
    parser.add_argument('model_path', type=str, 
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--random', action='store_true',
                        help='Test random policy instead')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode (continuous testing)')
    
    args = parser.parse_args()
    
    if args.random:
        test_random_policy(
            n_episodes=args.episodes,
            render=not args.no_render
        )
    elif args.interactive:
        interactive_test(args.model_path)
    else:
        test_policy(
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render
        )
