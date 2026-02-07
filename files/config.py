"""
Configuration file for Spot Robot training
Edit these parameters to customize your training
"""

# ============================================
# ENVIRONMENT CONFIGURATION
# ============================================

ENV_CONFIG = {
    # Maximum steps per episode
    'max_episode_steps': 1000,
    
    # Goal position parameters
    'goal_distance_min': 3.0,  # Minimum distance to goal (meters)
    'goal_distance_max': 8.0,  # Maximum distance to goal (meters)
    'goal_tolerance': 0.5,     # Success threshold (meters)
    
    # Initial robot state
    'initial_height': 0.35,    # Starting body height (meters)
    'joint_noise_std': 0.1,    # Random noise added to initial joint positions
    
    # Physics parameters
    'simulation_dt': 0.002,    # Simulation timestep (seconds)
    'control_freq': 50,        # Control frequency (Hz)
}


# ============================================
# REWARD FUNCTION WEIGHTS
# ============================================

REWARD_WEIGHTS = {
    # Moving towards goal
    'progress_reward': 2.0,
    
    # Staying upright (penalize tilt)
    'tilt_penalty': 0.5,
    
    # Maintaining proper height
    'height_penalty': 2.0,
    'target_height': 0.35,
    
    # Energy efficiency
    'energy_penalty': 0.005,
    
    # Control smoothness
    'control_penalty': 0.001,
    
    # Goal reached bonus
    'goal_bonus': 100.0,
    
    # Alive bonus (per step)
    'alive_bonus': 0.5,
}


# ============================================
# TERMINATION CONDITIONS
# ============================================

TERMINATION_CONFIG = {
    # Minimum body height before termination (meters)
    'min_height': 0.15,
    
    # Maximum allowed tilt (radians)
    'max_tilt': 1.57,  # π/2 radians = 90 degrees
}


# ============================================
# PPO TRAINING HYPERPARAMETERS
# ============================================

PPO_CONFIG = {
    # Learning rate
    'learning_rate': 3e-4,
    
    # Number of steps to collect per environment per update
    'n_steps': 2048,
    
    # Minibatch size
    'batch_size': 64,
    
    # Number of epochs to optimize the policy
    'n_epochs': 10,
    
    # Discount factor
    'gamma': 0.99,
    
    # GAE lambda
    'gae_lambda': 0.95,
    
    # Clipping parameter
    'clip_range': 0.2,
    
    # Entropy coefficient (exploration)
    'ent_coef': 0.01,
    
    # Value function coefficient
    'vf_coef': 0.5,
    
    # Max gradient norm
    'max_grad_norm': 0.5,
    
    # Neural network architecture
    'policy_net_arch': [256, 256],  # Hidden layers for policy network
    'value_net_arch': [256, 256],   # Hidden layers for value network
    
    # Use State Dependent Exploration (SDE)
    'use_sde': False,
}


# ============================================
# TRAINING CONFIGURATION
# ============================================

TRAINING_CONFIG = {
    # Total number of training steps
    'total_timesteps': 5_000_000,
    
    # Number of parallel environments
    'n_envs': 8,
    
    # Save checkpoint every N steps (per environment)
    'checkpoint_freq': 50_000,
    
    # Evaluate every N steps (per environment)
    'eval_freq': 10_000,
    
    # Number of evaluation episodes
    'n_eval_episodes': 5,
    
    # Directories
    'save_dir': './spot_models',
    'log_dir': './spot_logs',
}


# ============================================
# ROBOT JOINT LIMITS
# ============================================

JOINT_LIMITS = {
    # Format: [hip, thigh, calf] for each leg
    'lower_limits': [
        -0.8, -2.8, -0.5,  # Front left
        -0.8, -2.8, -0.5,  # Front right
        -0.8, -2.8, -0.5,  # Rear left
        -0.8, -2.8, -0.5,  # Rear right
    ],
    'upper_limits': [
        0.8, 0.8, 2.8,     # Front left
        0.8, 0.8, 2.8,     # Front right
        0.8, 0.8, 2.8,     # Rear left
        0.8, 0.8, 2.8,     # Rear right
    ],
}


# ============================================
# CURRICULUM LEARNING (OPTIONAL)
# ============================================
# Gradually increase difficulty during training

CURRICULUM_CONFIG = {
    'enabled': False,
    
    # Start with closer goals, gradually increase distance
    'goal_distance_schedule': {
        0: (2.0, 4.0),           # First 0 steps: 2-4m
        1_000_000: (3.0, 6.0),   # After 1M steps: 3-6m
        2_000_000: (3.0, 8.0),   # After 2M steps: 3-8m
    },
    
    # Start on flat terrain, add noise later
    'terrain_noise_schedule': {
        0: 0.0,                  # First 0 steps: flat
        1_500_000: 0.02,         # After 1.5M steps: small bumps
        3_000_000: 0.05,         # After 3M steps: larger bumps
    },
}


# ============================================
# DOMAIN RANDOMIZATION (FOR SIM-TO-REAL)
# ============================================

DOMAIN_RANDOMIZATION = {
    'enabled': False,
    
    # Randomize robot mass (±percentage)
    'mass_randomization': 0.1,  # ±10%
    
    # Randomize joint friction
    'friction_randomization': 0.2,  # ±20%
    
    # Add sensor noise
    'observation_noise': 0.01,
    
    # Randomize motor strength
    'actuator_strength_randomization': 0.1,  # ±10%
    
    # Randomize ground friction
    'ground_friction_randomization': 0.3,  # ±30%
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_config():
    """Get all configuration as a single dictionary"""
    return {
        'env': ENV_CONFIG,
        'rewards': REWARD_WEIGHTS,
        'termination': TERMINATION_CONFIG,
        'ppo': PPO_CONFIG,
        'training': TRAINING_CONFIG,
        'joint_limits': JOINT_LIMITS,
        'curriculum': CURRICULUM_CONFIG,
        'domain_randomization': DOMAIN_RANDOMIZATION,
    }


def print_config():
    """Pretty print all configuration"""
    print("=" * 60)
    print("SPOT ROBOT TRAINING CONFIGURATION")
    print("=" * 60)
    
    config = get_config()
    for section_name, section_config in config.items():
        print(f"\n{section_name.upper().replace('_', ' ')}:")
        print("-" * 60)
        for key, value in section_config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_config()
