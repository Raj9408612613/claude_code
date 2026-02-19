"""
Central configuration for Spot Isaac Sim training.
All parameters in one place for easy tuning.
"""

import math

# ============================================================================
# DEPTH CAMERA CONFIGURATION
# ============================================================================

# 5 cameras providing ~330° coverage with overlap in front
# Each camera: (name, position_offset, rotation_euler_deg, fov_h, fov_v)
#
# Layout (top view of Spot):
#
#         Front-Left (70°)     Front (0°)     Front-Right (-70°)
#                  \              |              /
#                   \             |             /
#                    +============+============+
#                    |         SPOT BODY       |
#                    +============+============+
#                   /                           \
#                  /                             \
#         Rear-Left (145°)              Rear-Right (-145°)

CAMERA_RIG = {
    "cameras": [
        {
            "name": "front_center",
            "position": [0.32, 0.0, 0.05],      # Front of body, centered
            "rotation_deg": [0.0, -10.0, 0.0],   # Slight downward tilt
            "description": "Primary forward-facing camera",
        },
        {
            "name": "front_left",
            "position": [0.20, 0.16, 0.05],      # Front-left corner
            "rotation_deg": [0.0, -10.0, 70.0],   # 70° left
            "description": "Left peripheral camera",
        },
        {
            "name": "front_right",
            "position": [0.20, -0.16, 0.05],     # Front-right corner
            "rotation_deg": [0.0, -10.0, -70.0],  # 70° right
            "description": "Right peripheral camera",
        },
        {
            "name": "rear_left",
            "position": [-0.28, 0.14, 0.05],     # Rear-left
            "rotation_deg": [0.0, -10.0, 145.0],  # 145° left (rear-facing)
            "description": "Rear-left coverage camera",
        },
        {
            "name": "rear_right",
            "position": [-0.28, -0.14, 0.05],    # Rear-right
            "rotation_deg": [0.0, -10.0, -145.0], # 145° right (rear-facing)
            "description": "Rear-right coverage camera",
        },
    ],

    # Shared camera properties
    "resolution": [160, 120],       # Width x Height (low res for fast training)
    "horizontal_fov": 87.0,         # Degrees (matches Intel RealSense D435)
    "vertical_fov": 58.0,           # Degrees
    "min_range": 0.1,               # Meters — closer readings are clipped
    "max_range": 10.0,              # Meters — farther readings are max_range
    "update_rate": 30,              # Hz — camera frame rate
}


# ============================================================================
# DEPTH CAMERA NOISE MODEL
# ============================================================================
# Mimics real depth camera imperfections (Intel RealSense D435 profile)

DEPTH_NOISE = {
    "enabled": True,

    # Gaussian noise on depth readings (proportional to distance)
    "gaussian_std_base": 0.005,     # 5mm base noise at 1m
    "gaussian_std_scale": 0.01,     # Noise grows with distance: std = base + scale * depth

    # Random pixel dropout (no reading returned)
    "dropout_rate": 0.02,           # 2% of pixels randomly return 0 (no reading)

    # Edge noise (depth discontinuities are noisy on real cameras)
    "edge_noise_enabled": True,
    "edge_noise_std": 0.03,         # 3cm noise at depth edges

    # Quantization (real sensors have finite depth resolution)
    "quantization_step": 0.001,     # 1mm depth quantization

    # Temporal flicker (random pixels change frame-to-frame)
    "temporal_flicker_rate": 0.005, # 0.5% of pixels flicker per frame
}


# ============================================================================
# ENVIRONMENT / ROOM CONFIGURATION
# ============================================================================

# Pre-made environment USD paths (NVIDIA Isaac Sim assets)
ENVIRONMENTS = {
    "rooms": [
        {
            "name": "simple_office",
            "usd_path": "/Isaac/Environments/Simple_Room/simple_room.usd",
            "description": "Small office room with basic layout",
            "approx_size": [8.0, 8.0],
        },
        {
            "name": "office",
            "usd_path": "/Isaac/Environments/Office/office.usd",
            "description": "Full office with desks and partitions",
            "approx_size": [15.0, 12.0],
        },
        {
            "name": "hospital",
            "usd_path": "/Isaac/Environments/Hospital/hospital.usd",
            "description": "Hospital corridor with rooms",
            "approx_size": [20.0, 10.0],
        },
        {
            "name": "warehouse_basic",
            "usd_path": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
            "description": "Simple warehouse with shelving",
            "approx_size": [20.0, 15.0],
        },
        {
            "name": "warehouse_full",
            "usd_path": "/Isaac/Environments/Warehouse/full_warehouse.usd",
            "description": "Full warehouse environment",
            "approx_size": [30.0, 20.0],
        },
    ],

    # Procedural room (fallback if pre-made rooms unavailable)
    "procedural": {
        "width_range": [5.0, 15.0],
        "length_range": [5.0, 15.0],
        "wall_height": 3.0,
        "wall_thickness": 0.15,
    },
}

# Furniture / obstacle assets to scatter in rooms
OBSTACLE_ASSETS = {
    "static": [
        {"name": "chair",      "usd": "/Isaac/Props/Chairs/Chair_01.usd",
         "scale_range": [0.8, 1.2], "weight": 3},
        {"name": "table",      "usd": "/Isaac/Props/Tables/Table_01.usd",
         "scale_range": [0.8, 1.1], "weight": 2},
        {"name": "shelf",      "usd": "/Isaac/Props/Shelves/Shelf_01.usd",
         "scale_range": [0.9, 1.1], "weight": 2},
        {"name": "desk",       "usd": "/Isaac/Props/Tables/Desk_01.usd",
         "scale_range": [0.9, 1.1], "weight": 2},
        {"name": "cabinet",    "usd": "/Isaac/Props/Cabinets/Cabinet_01.usd",
         "scale_range": [0.9, 1.1], "weight": 1},
        {"name": "box_small",  "usd": "/Isaac/Props/Boxes/CardboardBox_A.usd",
         "scale_range": [0.5, 1.5], "weight": 3},
        {"name": "box_large",  "usd": "/Isaac/Props/Boxes/CardboardBox_B.usd",
         "scale_range": [0.8, 2.0], "weight": 2},
        {"name": "barrel",     "usd": "/Isaac/Props/Barrels/Barrel_01.usd",
         "scale_range": [0.8, 1.2], "weight": 1},
        {"name": "trash_bin",  "usd": "/Isaac/Props/Bins/TrashBin_01.usd",
         "scale_range": [0.8, 1.2], "weight": 1},
        {"name": "cone",       "usd": "/Isaac/Props/Safety/Cone_01.usd",
         "scale_range": [0.8, 1.2], "weight": 1},
    ],

    # Primitive fallbacks (used if USD assets not available)
    "primitives": [
        {"name": "box",      "type": "Cube",     "size_range": [0.2, 1.0], "weight": 3},
        {"name": "cylinder", "type": "Cylinder",  "size_range": [0.15, 0.5], "weight": 2},
        {"name": "sphere",   "type": "Sphere",    "size_range": [0.2, 0.6], "weight": 1},
    ],

    # How many obstacles to place per episode
    "count_range": [5, 25],

    # Minimum distance between obstacles and from robot start
    "min_obstacle_spacing": 0.6,
    "min_dist_from_robot": 1.0,
    "min_dist_from_goal": 0.8,
}


# ============================================================================
# DYNAMIC OBSTACLES (MOVING OBJECTS — SIMULATES PEOPLE, CARTS)
# ============================================================================

DYNAMIC_OBSTACLES = {
    "enabled": True,

    # Number of moving obstacles per episode
    "count_range": [0, 5],

    "types": [
        {
            "name": "walking_person",
            "usd": "/Isaac/People/Characters/Character_01.usd",
            "speed_range": [0.5, 1.5],       # m/s — walking speed
            "size": [0.4, 0.4, 1.7],          # Approximate bounding box
            "weight": 3,
        },
        {
            "name": "cart",
            "usd": "/Isaac/Props/Carts/Cart_01.usd",
            "speed_range": [0.3, 1.0],
            "size": [0.6, 0.4, 0.9],
            "weight": 1,
        },
    ],

    # Movement patterns
    "movement": {
        "type": "random_waypoints",     # "random_waypoints" or "linear"
        "waypoint_radius": 3.0,         # Meters — how far each waypoint is
        "direction_change_prob": 0.02,   # Per-step probability of changing direction
        "pause_prob": 0.005,             # Per-step probability of stopping briefly
        "pause_duration_range": [0.5, 3.0],  # Seconds
    },

    # Primitive fallback (cylinder that moves)
    "primitive_fallback": {
        "type": "Cylinder",
        "radius": 0.25,
        "height": 1.7,
    },
}


# ============================================================================
# DOMAIN RANDOMIZATION — ALL 10 CATEGORIES
# ============================================================================

DOMAIN_RANDOMIZATION = {
    # 1. Obstacle positions/sizes
    "obstacle_position": {
        "enabled": True,
        # Positions are always randomized (see OBSTACLE_ASSETS.count_range)
        # Scale variation is in OBSTACLE_ASSETS[].scale_range
    },

    # 2. Room shape/size
    "room_shape": {
        "enabled": True,
        # Selects random room from ENVIRONMENTS.rooms each episode
        # For procedural rooms, size varies per ENVIRONMENTS.procedural
    },

    # 3. Number of obstacles
    "obstacle_count": {
        "enabled": True,
        # Range defined in OBSTACLE_ASSETS.count_range
    },

    # 4. Floor texture/material + surface curvature
    "floor_surface": {
        "enabled": True,
        "textures": [
            "concrete_polished", "concrete_rough", "wood_planks",
            "tile_ceramic", "tile_marble", "carpet_grey",
            "carpet_blue", "linoleum", "rubber_mat",
        ],
        "friction_range": [0.4, 1.2],         # Coefficient of friction
        # Surface curvature for balance robustness
        "curvature_enabled": True,
        "curvature_amplitude_range": [0.0, 0.03],  # Meters — max bump height
        "curvature_frequency_range": [0.5, 2.0],    # Bumps per meter
        "slope_angle_range": [0.0, 5.0],             # Degrees — overall tilt
    },

    # 5. Lighting conditions
    "lighting": {
        "enabled": True,
        "intensity_range": [200.0, 2000.0],    # Lux
        "color_temp_range": [3000.0, 6500.0],  # Kelvin (warm to cool white)
        "num_lights_range": [1, 4],
        "shadow_enabled": True,
        # Depth cameras are less affected by lighting than RGB,
        # but structured-light depth sensors (RealSense) ARE affected
        # by bright IR sources and direct sunlight
        "ambient_range": [0.1, 0.5],
    },

    # 6. Obstacle shapes (diverse objects)
    "obstacle_shapes": {
        "enabled": True,
        # Uses both USD assets and primitives from OBSTACLE_ASSETS
        # Weighted random selection for diversity
    },

    # 7. Sensor noise levels
    "sensor_noise": {
        "enabled": True,
        "noise_multiplier_range": [0.5, 2.0],  # Scales DEPTH_NOISE values
        "dropout_multiplier_range": [0.5, 3.0],
        # Per-episode randomization: some episodes are clean, some very noisy
    },

    # 8. Robot starting position
    "robot_start": {
        "enabled": True,
        "random_position": True,       # Random valid position in room
        "random_orientation": True,    # Random yaw (0-360°)
        "height_variation": 0.02,      # ±2cm start height variation
    },

    # 9. Goal position
    "goal_position": {
        "enabled": True,
        "distance_range": [2.0, 10.0], # Meters from robot
        "must_be_reachable": True,      # Path must exist (no goal inside walls)
    },

    # 10. Dynamic obstacles (moving)
    "dynamic_obstacles": {
        "enabled": True,
        # Configuration in DYNAMIC_OBSTACLES section above
    },
}


# ============================================================================
# ROBOT PHYSICAL PROPERTIES
# ============================================================================

SPOT_ROBOT = {
    "usd_path": "/Isaac/Robots/BostonDynamics/spot/spot.usd",

    # Fallback: use URDF if USD not available
    "urdf_fallback": None,

    # Physical properties
    "mass": 14.0,                       # kg (Spot mini approximate)
    "body_dimensions": [0.6, 0.3, 0.16],  # L x W x H of body
    "standing_height": 0.5,             # Meters from ground to body center

    # Joint configuration
    "n_joints": 12,                     # 3 per leg × 4 legs
    "joint_names": [
        "fl_hx", "fl_hy", "fl_kn",     # Front-left: hip_x, hip_y, knee
        "fr_hx", "fr_hy", "fr_kn",     # Front-right
        "hl_hx", "hl_hy", "hl_kn",     # Hind-left
        "hr_hx", "hr_hy", "hr_kn",     # Hind-right
    ],

    # Standing pose (default joint angles in radians)
    "standing_pose": [
        0.0, 0.8, -1.6,    # Front-left
        0.0, 0.8, -1.6,    # Front-right
        0.0, 0.8, -1.6,    # Hind-left
        0.0, 0.8, -1.6,    # Hind-right
    ],

    # Joint limits (radians)
    "joint_lower_limits": [
        -0.8, -0.6, -2.8,
        -0.8, -0.6, -2.8,
        -0.8, -0.6, -2.8,
        -0.8, -0.6, -2.8,
    ],
    "joint_upper_limits": [
        0.8, 2.4, -0.5,
        0.8, 2.4, -0.5,
        0.8, 2.4, -0.5,
        0.8, 2.4, -0.5,
    ],

    # PD controller gains
    "kp": 100.0,   # Proportional gain
    "kd": 10.0,    # Derivative gain

    # Mass randomization for domain randomization
    "mass_variation": 0.15,     # ±15%
    "friction_variation": 0.3,  # ±30%
}


# ============================================================================
# REWARD CONFIGURATION
# ============================================================================

REWARD_CONFIG = {
    # Goal reaching
    "goal_reached_bonus": 200.0,
    "goal_tolerance": 0.5,              # Meters

    # Progress toward goal
    "progress_weight": 5.0,             # Reward per meter of progress

    # Collision penalty
    "collision_penalty": -10.0,
    "near_collision_penalty": -2.0,     # Within 0.3m of obstacle
    "near_collision_threshold": 0.3,    # Meters

    # Stability rewards
    "upright_weight": -1.0,             # Penalty per radian of tilt
    "height_weight": -3.0,              # Penalty for height deviation
    "target_height": 0.5,              # Target body height

    # Efficiency
    "energy_weight": -0.005,            # Penalty for joint velocity
    "smoothness_weight": -0.002,        # Penalty for action change
    "alive_bonus": 0.5,                 # Per-step survival bonus

    # Orientation toward goal (encourage facing the goal)
    "heading_weight": 0.3,              # Reward for facing goal direction
}


# ============================================================================
# TERMINATION CONDITIONS
# ============================================================================

TERMINATION = {
    "min_height": 0.2,                  # Body too low → fallen
    "max_tilt_rad": math.pi / 3,        # 60° tilt → fallen
    "max_episode_steps": 1000,
    "terminate_on_collision": False,     # Don't end episode on collision
    "terminate_on_goal": True,          # End when goal reached
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING = {
    # PPO hyperparameters
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # Network architecture
    # CNN processes depth images → feature vector
    # MLP processes proprioception → combined with CNN features
    "cnn_features_dim": 256,            # Output dim of CNN encoder
    "mlp_proprio_layers": [128, 64],    # Proprioception MLP
    "combined_mlp_layers": [256, 128],  # Combined feature MLP
    "activation": "elu",

    # Training scale
    "total_timesteps": 10_000_000,
    "n_envs": 4,                        # Parallel envs (limited by GPU memory)
    "checkpoint_freq": 100_000,
    "eval_freq": 50_000,
    "n_eval_episodes": 10,

    # Directories
    "save_dir": "./spot_isaac_models",
    "log_dir": "./spot_isaac_logs",

    # Curriculum: gradually increase difficulty
    "curriculum": {
        "enabled": True,
        "stages": [
            {
                "name": "stage_1_empty_room",
                "timesteps": 0,
                "obstacle_count": [0, 3],
                "dynamic_count": [0, 0],
                "goal_distance": [2.0, 5.0],
                "noise_multiplier": 0.5,
            },
            {
                "name": "stage_2_static_obstacles",
                "timesteps": 1_000_000,
                "obstacle_count": [3, 10],
                "dynamic_count": [0, 0],
                "goal_distance": [2.0, 7.0],
                "noise_multiplier": 0.75,
            },
            {
                "name": "stage_3_more_obstacles",
                "timesteps": 3_000_000,
                "obstacle_count": [5, 20],
                "dynamic_count": [0, 2],
                "goal_distance": [3.0, 10.0],
                "noise_multiplier": 1.0,
            },
            {
                "name": "stage_4_full_difficulty",
                "timesteps": 6_000_000,
                "obstacle_count": [5, 25],
                "dynamic_count": [0, 5],
                "goal_distance": [3.0, 10.0],
                "noise_multiplier": 1.5,
            },
        ],
    },
}


# ============================================================================
# OBSERVATION SPACE DEFINITION
# ============================================================================

OBSERVATION = {
    # Depth images from 5 cameras → stacked as channels
    "depth_shape": [5, 120, 160],       # [n_cameras, height, width]

    # Proprioceptive state (flat vector fed alongside depth)
    "proprio_dim": 37,
    # Breakdown:
    #   Joint positions: 12
    #   Joint velocities: 12
    #   Body orientation (quaternion): 4
    #   Body linear velocity: 3
    #   Body angular velocity: 3
    #   Goal direction (relative): 2
    #   Goal distance: 1
    #   Total: 37
}

# ============================================================================
# ACTION SPACE
# ============================================================================

ACTION = {
    "dim": 12,                          # 12 joint position targets
    "type": "continuous",               # Continuous control
    "control_mode": "position",         # Position-based PD control
    "control_frequency": 50,            # Hz
}


# ============================================================================
# OVERRIDE SYSTEM
# ============================================================================
# Call apply_overrides() to change any parameter at runtime.
# If a value is not provided (None), the default above is kept.
#
# Example (in Colab notebook or script):
#   from spot_isaac_sim import config
#   config.apply_overrides(
#       total_timesteps=500_000,
#       max_episode_steps=500,
#       combined_mlp_layers=[128, 128],
#   )

# Map of override key → (config_dict_name, key_in_dict)
_OVERRIDE_MAP = {
    # Training
    "total_timesteps":      (TRAINING, "total_timesteps"),
    "n_envs":               (TRAINING, "n_envs"),
    "learning_rate":        (TRAINING, "learning_rate"),
    "n_steps":              (TRAINING, "n_steps"),
    "batch_size":           (TRAINING, "batch_size"),
    "n_epochs":             (TRAINING, "n_epochs"),
    "gamma":                (TRAINING, "gamma"),
    "gae_lambda":           (TRAINING, "gae_lambda"),
    "clip_range":           (TRAINING, "clip_range"),
    "ent_coef":             (TRAINING, "ent_coef"),
    "vf_coef":              (TRAINING, "vf_coef"),
    "max_grad_norm":        (TRAINING, "max_grad_norm"),
    "checkpoint_freq":      (TRAINING, "checkpoint_freq"),
    "eval_freq":            (TRAINING, "eval_freq"),
    "n_eval_episodes":      (TRAINING, "n_eval_episodes"),
    "save_dir":             (TRAINING, "save_dir"),
    "log_dir":              (TRAINING, "log_dir"),
    "cnn_features_dim":     (TRAINING, "cnn_features_dim"),
    "mlp_proprio_layers":   (TRAINING, "mlp_proprio_layers"),
    "combined_mlp_layers":  (TRAINING, "combined_mlp_layers"),
    "curriculum_enabled":   None,  # handled specially
    # Termination / Environment
    "max_episode_steps":    (TERMINATION, "max_episode_steps"),
    "min_height":           (TERMINATION, "min_height"),
    # Reward
    "goal_reached_bonus":   (REWARD_CONFIG, "goal_reached_bonus"),
    "progress_weight":      (REWARD_CONFIG, "progress_weight"),
    "collision_penalty":    (REWARD_CONFIG, "collision_penalty"),
    "alive_bonus":          (REWARD_CONFIG, "alive_bonus"),
    # Goal distance
    "goal_distance_range":  None,  # handled specially
}


def apply_overrides(**kwargs):
    """
    Override any config parameter at runtime.
    Pass only the values you want to change; everything else keeps its default.

    Example:
        config.apply_overrides(
            total_timesteps=500_000,
            max_episode_steps=500,
            combined_mlp_layers=[128, 128],
            n_envs=4,
            checkpoint_freq=50_000,
            goal_distance_range=[2.0, 5.0],
            curriculum_enabled=False,
            save_dir="/content/drive/MyDrive/robot_training/models",
            log_dir="/content/drive/MyDrive/robot_training/logs",
        )
    """
    changed = []

    for key, value in kwargs.items():
        if value is None:
            continue

        # Special cases
        if key == "curriculum_enabled":
            TRAINING["curriculum"]["enabled"] = value
            changed.append(f"  curriculum.enabled = {value}")
            continue

        if key == "goal_distance_range":
            DOMAIN_RANDOMIZATION["goal_position"]["distance_range"] = list(value)
            changed.append(f"  goal_distance_range = {value}")
            continue

        # Standard lookup
        if key in _OVERRIDE_MAP:
            target_dict, target_key = _OVERRIDE_MAP[key]
            old_val = target_dict[target_key]
            target_dict[target_key] = value
            changed.append(f"  {key}: {old_val} -> {value}")
        else:
            print(f"[config] WARNING: unknown override key '{key}', ignored")

    if changed:
        print(f"[config] Applied {len(changed)} override(s):")
        for line in changed:
            print(line)


def print_active_config():
    """Print the currently active training-relevant parameters."""
    print("=" * 60)
    print("ACTIVE CONFIGURATION")
    print("=" * 60)
    print(f"  total_timesteps:      {TRAINING['total_timesteps']:,}")
    print(f"  max_episode_steps:    {TERMINATION['max_episode_steps']}")
    print(f"  n_envs:               {TRAINING['n_envs']}")
    print(f"  learning_rate:        {TRAINING['learning_rate']}")
    print(f"  batch_size:           {TRAINING['batch_size']}")
    print(f"  n_steps:              {TRAINING['n_steps']}")
    print(f"  n_epochs:             {TRAINING['n_epochs']}")
    print(f"  combined_mlp_layers:  {TRAINING['combined_mlp_layers']}")
    print(f"  cnn_features_dim:     {TRAINING['cnn_features_dim']}")
    print(f"  checkpoint_freq:      {TRAINING['checkpoint_freq']:,}")
    print(f"  eval_freq:            {TRAINING['eval_freq']:,}")
    print(f"  curriculum_enabled:   {TRAINING['curriculum']['enabled']}")
    print(f"  goal_distance_range:  {DOMAIN_RANDOMIZATION['goal_position']['distance_range']}")
    print(f"  save_dir:             {TRAINING['save_dir']}")
    print(f"  log_dir:              {TRAINING['log_dir']}")
    print("=" * 60)
