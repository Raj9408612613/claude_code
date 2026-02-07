# Spot Quadruped Robot - Reinforcement Learning Training

Train a Spot-like quadruped robot to walk towards goals using deep reinforcement learning (PPO algorithm).

##  Project Overview

This project trains a simulated quadruped robot to:
- **Walk forward** without falling
- **Navigate towards randomly placed goals** in the environment
- **Maintain balance** while moving
- Learn through trial and error using RL

## 📁 Project Structure

```
.
├── spot_env.py           # Gymnasium environment definition
├── spot_scene.xml        # MuJoCo robot model and scene
├── train_spot.py         # Training script
├── test_spot.py          # Testing and visualization script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

##  Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Start Training

```bash
# Basic training (5M timesteps with 8 parallel environments)
python train_spot.py

# Custom training parameters
python train_spot.py --timesteps 10000000 --n_envs 16 --save_dir ./my_models
```

### 3. Test the Trained Robot

```bash
# Test a trained model (with visualization)
python test_spot.py spot_models/best_model.zip --episodes 10

# Test without rendering (faster)
python test_spot.py spot_models/best_model.zip --no-render

# Interactive mode (continuous episodes)
python test_spot.py spot_models/best_model.zip --interactive
```

##  Robot Description

### Physical Properties
- **Type**: Quadruped (4 legs)
- **Degrees of Freedom**: 12 joints (3 per leg: hip, thigh, calf)
- **Mass**: ~15 kg total
- **Dimensions**: 0.6m x 0.3m x 0.35m (L x W x H)

### Joint Structure (per leg)
1. **Hip joint**: Abduction/adduction (side-to-side)
2. **Thigh joint**: Flexion/extension
3. **Calf joint**: Flexion/extension

##  Learning Setup

### Observation Space (37 dimensions)
- Joint positions (12): Current angle of each joint
- Joint velocities (12): Angular velocity of each joint
- Body orientation (4): Quaternion representation
- Body linear velocity (3): Movement speed in x, y, z
- Body angular velocity (3): Rotation speed
- Goal direction (2): Normalized vector pointing to goal
- Body height (1): Distance from ground

### Action Space (12 dimensions)
- Target position for each joint (normalized -1 to 1)
- Mapped to actual joint limits via PD controller

### Reward Function

The reward encourages the robot to:

1. **Move towards goal** (+2.0 per meter of progress)
2. **Stay upright** (penalize roll/pitch)
3. **Maintain proper height** (target: 0.35m)
4. **Use energy efficiently** (small penalty for large velocities)
5. **Reach the goal** (+100 bonus when within 0.5m)
6. **Stay alive** (+0.5 per step)

### Termination Conditions

Episode ends when:
- Robot falls (height < 0.15m)
- Robot flips over (roll/pitch > 90°)
- Goal is reached (distance < 0.5m)
- Maximum steps reached (1000 steps)

##  Training Details

### Algorithm: PPO (Proximal Policy Optimization)

**Key hyperparameters:**
- Learning rate: 3e-4
- Batch size: 64
- Network architecture: 2 hidden layers × 256 neurons
- Parallel environments: 8 (default)
- Training timesteps: 5M (default)

**Why PPO?**
- Stable and sample-efficient
- Works well for continuous control
- Industry standard for robotics

### Training Progress

Expected training timeline (on modern CPU/GPU):
- **100K steps**: Robot learns to stand and balance
- **500K steps**: Basic forward walking emerges
- **2M steps**: Stable walking and goal-directed movement
- **5M steps**: Efficient navigation and goal reaching

Monitor training with TensorBoard:
```bash
tensorboard --logdir spot_logs/
```

##  Usage Examples

### Train from scratch
```bash
python train_spot.py --timesteps 5000000 --n_envs 8
```

### Continue training from checkpoint
```bash
python train_spot.py --continue_from spot_models/spot_model_1000000_steps.zip --timesteps 2000000
```

### Quick test (3 episodes, no rendering)
```bash
python test_spot.py spot_models/best_model.zip --episodes 3 --no-render
```

### Watch the robot walk
```bash
python test_spot.py spot_models/best_model.zip --interactive
```

## 🔧 Customization

### Modify the robot
Edit `spot_scene.xml` to change:
- Robot dimensions
- Joint limits
- Mass distribution
- Friction parameters

### Tune the reward function
Edit `_calculate_reward()` in `spot_env.py` to:
- Weight different objectives
- Add new behaviors (e.g., energy efficiency)
- Change goal reaching criteria

### Adjust training
Modify hyperparameters in `train_spot.py`:
- Network size
- Learning rate
- Number of parallel environments
- Rollout length

##  Expected Results

After successful training, the robot should:
-  Walk forward smoothly
-  Navigate to goals 5-8 meters away
-  Success rate > 80% on goal reaching
-  Average episode reward > 100
-  Maintain balance on flat terrain

##  Troubleshooting

### Robot immediately falls
- Increase training time
- Check that `spot_scene.xml` is in the same directory
- Verify joint limits are reasonable

### Training is very slow
- Reduce `n_envs` if running out of memory
- Use GPU if available (PPO will automatically use it)
- Reduce `n_steps` in PPO configuration

### Robot doesn't reach goals
- Train longer (try 10M steps)
- Increase goal reaching bonus in reward function
- Decrease goal tolerance distance

##  Next Steps: Sim-to-Real Transfer

To deploy on a real robot:

1. **Domain Randomization**
   - Add noise to observations
   - Randomize physics parameters
   - Vary ground friction

2. **System Identification**
   - Measure real robot's mass/inertia
   - Update simulation to match

3. **Real Robot Interface**
   - Create ROS wrapper for trained policy
   - Map simulation actions to motor commands
   - Add safety checks

4. **Fine-tuning**
   - Collect real robot data
   - Use sim policy as initialization
   - Continue training with real data

##  Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Boston Dynamics Spot](https://www.bostondynamics.com/products/spot)

##  Contributing

Want to improve this? Try:
- Adding obstacle avoidance
- Implementing terrain adaptation
- Adding vision-based navigation
- Creating more complex tasks (climbing stairs)

##  License

MIT License - Feel free to use and modify!

---

**Happy Training! **
