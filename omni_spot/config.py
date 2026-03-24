"""
Environment & Training Configuration
======================================
Ported from config.py + jax_ppo.py constants.
All hyperparameters preserved exactly from the JAX version.
"""

# ── PPO Hyperparameters ──────────────────────────────────────────────────────
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
ENT_COEF     = 0.01
VF_COEF      = 0.5
MAX_GRAD     = 0.5
LR           = 3e-4
N_EPOCHS     = 4
MINIBATCH_SZ = 512
TARGET_KL    = 0.03

# ── Network dimensions ──────────────────────────────────────────────────────
CNN_FEAT_DIM = 256
PROPRIO_DIM  = 37
ACTION_DIM   = 12
LOG_STD_MIN  = -5.0
LOG_STD_MAX  =  2.0

# ── Reward weights (from jax_reward.py) ──────────────────────────────────────
GOAL_BONUS       =  200.0
GOAL_TOL         =    0.5    # metres
PROGRESS_W       =    5.0
COLLISION_PEN    =  -10.0
NEAR_COLL_PEN    =   -2.0
NEAR_COLL_THRESH =    0.35   # metres
UPRIGHT_W        =   -1.0
HEIGHT_W         =   -3.0
TARGET_HEIGHT    =    0.46   # real Spot standing height (m)
ENERGY_W         =  -0.005
SMOOTH_W         =  -0.002
ALIVE_BONUS      =    0.5
HEADING_W        =    0.3

# ── Physics ──────────────────────────────────────────────────────────────────
PHYSICS_DT        = 0.005    # PhysX sim timestep (200 Hz)
CONTROL_DT        = 0.02     # RL control rate (50 Hz) = 4 substeps
PHYSICS_SUBSTEPS  = 4

# ── Joint limits (real Spot SDK values from MuJoCo Menagerie) ────────────────
JOINT_LOWER = [
    -0.785398, -0.898845, -2.7929,   # fl: hx, hy, kn
    -0.785398, -0.898845, -2.7929,   # fr
    -0.785398, -0.898845, -2.7929,   # hl
    -0.785398, -0.898845, -2.7929,   # hr
]
JOINT_UPPER = [
     0.785398,  2.29511,  -0.254402, # fl
     0.785398,  2.24363,  -0.255648, # fr
     0.785398,  2.29511,  -0.247067, # hl
     0.785398,  2.29511,  -0.248282, # hr
]
STANDING_POSE = [0.0, 1.04, -1.8] * 4  # home keyframe (12 joints)

# ── Room ─────────────────────────────────────────────────────────────────────
ROOM_HALF = 4.5   # 10x10 m room, keep 0.5m buffer from walls

# ── Camera ───────────────────────────────────────────────────────────────────
N_CAMS   = 5
CAM_H    = 120
CAM_W    = 160
H_FOV    = 87.0    # degrees
V_FOV    = 58.0    # degrees
MIN_DEPTH = 0.1
MAX_DEPTH = 10.0

# ── Obstacles ────────────────────────────────────────────────────────────────
N_STATIC   = 25
N_DYNAMIC  = 5
N_HUMANOID = 1
N_OBS      = N_STATIC + N_DYNAMIC + N_HUMANOID  # 31
HUMANOID_MOCAP_IDX = N_STATIC + N_DYNAMIC       # = 30 (last entry)

# Obstacle half-sizes (x, y, z) — ported from warp_cameras.py _OBS_HALF_SIZES
# 25 static boxes + 5 dynamic cylinders (approx as boxes) + 1 humanoid AABB
OBS_HALF_SIZES = [
    # 25 static boxes (varied sizes)
    [0.30, 0.30, 0.50], [0.25, 0.40, 0.40], [0.20, 0.20, 0.80], [0.40, 0.20, 0.45],
    [0.35, 0.35, 0.35], [0.30, 0.30, 0.50], [0.50, 0.20, 0.60], [0.20, 0.50, 0.40],
    [0.40, 0.40, 0.30], [0.30, 0.25, 0.70], [0.20, 0.30, 0.50], [0.35, 0.35, 0.40],
    [0.45, 0.20, 0.45], [0.30, 0.40, 0.30], [0.25, 0.25, 0.60], [0.30, 0.30, 0.50],
    [0.40, 0.30, 0.35], [0.20, 0.40, 0.55], [0.35, 0.20, 0.40], [0.30, 0.30, 0.50],
    [0.40, 0.40, 0.40], [0.25, 0.35, 0.45], [0.30, 0.25, 0.50], [0.45, 0.30, 0.35],
    [0.20, 0.20, 0.70],
    # 5 dynamic cylinders (approximated as boxes)
    [0.25, 0.25, 0.85], [0.25, 0.25, 0.85], [0.25, 0.25, 0.85],
    [0.25, 0.25, 0.85], [0.25, 0.25, 0.85],
    # 1 humanoid AABB (centred at torso, z=1.0, covers 0-2m height)
    [0.30, 0.30, 1.00],
]

# ── Humanoid walking obstacle ────────────────────────────────────────────────
HUMANOID_OBSTACLE = {
    "enabled":        True,
    "speed":          0.8,     # m/s walking speed
    "stride_freq":    1.2,     # Hz
    "patrol_radius":  1.5,     # metres from goal centre to each waypoint
    "mocap_z":        1.0,     # world-frame z of the torso
    "wp_switch_dist": 0.2,     # switch waypoint when closer than this (m)
}
