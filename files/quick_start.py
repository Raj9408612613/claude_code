"""
Quick Start Script - Verify installation and test environment
"""

import sys
import time


def check_imports():
    """Check if all required packages are installed"""
    print("Checking package installations...")
    print("=" * 50)
    
    packages = {
        'gymnasium': 'Gymnasium (RL environment)',
        'mujoco': 'MuJoCo (Physics simulator)',
        'numpy': 'NumPy',
        'stable_baselines3': 'Stable Baselines3 (RL algorithms)',
    }
    
    all_installed = True
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✓ {description}")
        except ImportError:
            print(f"✗ {description} - NOT INSTALLED")
            all_installed = False
    
    print("=" * 50)
    
    if not all_installed:
        print("\n❌ Some packages are missing!")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed!")
        return True


def test_environment():
    """Test if the environment can be created"""
    print("\nTesting environment creation...")
    print("=" * 50)
    
    try:
        from spot_env import SpotEnv
        
        # Create environment
        env = SpotEnv(render_mode=None)
        print("✓ Environment created successfully")
        
        # Reset environment
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Goal position: ({info['goal_position'][0]:.2f}, {info['goal_position'][1]:.2f})")
        
        # Take random actions
        print("\n✓ Testing random actions...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward = {reward:.3f}, terminated = {terminated}")
            
            if terminated or truncated:
                print("  Episode ended, resetting...")
                obs, info = env.reset()
        
        env.close()
        print("\n✅ Environment test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Environment test failed!")
        print(f"Error: {e}")
        return False


def quick_training_test():
    """Run a very short training test"""
    print("\nRunning quick training test (10k steps)...")
    print("=" * 50)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from spot_env import SpotEnv
        
        # Create single environment
        env = DummyVecEnv([lambda: Monitor(SpotEnv())])
        
        # Create PPO model
        print("Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            verbose=0,
        )
        
        # Train for 10k steps
        print("Training for 10,000 steps (this will take 1-2 minutes)...")
        start_time = time.time()
        
        model.learn(total_timesteps=10_000, progress_bar=True)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed:.1f} seconds")
        print(f"  Speed: {10_000/elapsed:.0f} steps/second")
        
        # Test the policy
        print("\nTesting trained policy...")
        obs = env.reset()
        total_reward = 0
        
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            if done[0]:
                break
        
        print(f"✓ Policy test complete, reward: {total_reward:.2f}")
        
        env.close()
        print("\n✅ Quick training test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training test failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("SPOT ROBOT - QUICK START VERIFICATION")
    print("=" * 50)
    print()
    
    # Step 1: Check imports
    if not check_imports():
        print("\n⚠️  Please install missing packages before continuing.")
        sys.exit(1)
    
    # Step 2: Test environment
    if not test_environment():
        print("\n⚠️  Environment test failed. Check the error above.")
        sys.exit(1)
    
    # Step 3: Ask user if they want to do training test
    print("\n" + "=" * 50)
    response = input("\nRun quick training test? (takes 1-2 minutes) [y/N]: ")
    
    if response.lower() in ['y', 'yes']:
        if not quick_training_test():
            print("\n⚠️  Training test failed. Check the error above.")
            sys.exit(1)
    else:
        print("\nSkipping training test.")
    
    # All tests passed
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("\nYou're ready to start training!")
    print("\nNext steps:")
    print("  1. Start training: python train_spot.py")
    print("  2. Monitor progress: tensorboard --logdir spot_logs/")
    print("  3. Test model: python test_spot.py spot_models/best_model.zip")
    print("\nFor more info, see README.md")
    print("=" * 50)


if __name__ == "__main__":
    main()
