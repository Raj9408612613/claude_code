"""
JAX-native Reward Function
===========================
Replaces reward.py (NumPy/class-based).

All functions are pure JAX — JIT-compiled and vmapped over the batch dim.
No Python loops in the hot path.

Matches the reward coefficients in spot_isaac_sim/config.py REWARD_CONFIG.
"""

import jax
import jax.numpy as jnp

# ── Reward weights (from config.py REWARD_CONFIG) ────────────────────────────
GOAL_BONUS          =  200.0
GOAL_TOL            =    0.5   # metres
PROGRESS_W          =    5.0
COLLISION_PEN       =  -10.0
NEAR_COLL_PEN       =   -2.0
NEAR_COLL_THRESH    =    0.3   # metres
UPRIGHT_W           =   -1.0
HEIGHT_W            =   -3.0
TARGET_HEIGHT       =    0.5
ENERGY_W            =  -0.005
SMOOTH_W            =  -0.002
ALIVE_BONUS         =    0.5
HEADING_W           =    0.3


@jax.jit
def compute_reward(
    robot_pos:       jnp.ndarray,   # (B, 3)
    robot_quat:      jnp.ndarray,   # (B, 4)  w,x,y,z
    goal_pos:        jnp.ndarray,   # (B, 2)
    prev_robot_pos:  jnp.ndarray,   # (B, 3)
    joint_vel:       jnp.ndarray,   # (B, 12)
    action:          jnp.ndarray,   # (B, 12)
    prev_action:     jnp.ndarray,   # (B, 12)
    min_obs_dist:    jnp.ndarray,   # (B,)
    has_collision:   jnp.ndarray,   # (B,) bool
    prev_dist_goal:  jnp.ndarray,   # (B,)
):
    """
    Fully batched reward computation.

    Returns:
        total:         (B,) float32
        info:          dict of (B,) reward components
        new_dist_goal: (B,) updated distance to goal (carry forward)
    """
    robot_xy   = robot_pos[:, :2]
    prev_xy    = prev_robot_pos[:, :2]
    goal_xy    = goal_pos                        # (B, 2)

    # ── 1. Progress ───────────────────────────────────────────────────
    dist_goal = jnp.linalg.norm(goal_xy - robot_xy, axis=-1)   # (B,)
    progress  = prev_dist_goal - dist_goal
    r_progress = progress * PROGRESS_W

    # ── 2. Goal reached bonus ─────────────────────────────────────────
    goal_reached = dist_goal < GOAL_TOL                         # (B,) bool
    r_goal = jnp.where(goal_reached, GOAL_BONUS, 0.0)

    # ── 3. Collision penalties ────────────────────────────────────────
    r_collision  = jnp.where(has_collision, COLLISION_PEN, 0.0)
    r_near       = jnp.where(min_obs_dist < NEAR_COLL_THRESH, NEAR_COLL_PEN, 0.0)

    # ── 4. Upright (tilt from quaternion) ─────────────────────────────
    # tilt = angle between body z-axis and world z-axis
    # body_z in world = R(quat) @ [0,0,1]; dot with [0,0,1] = cos(tilt)
    # From quaternion (w,x,y,z): body_z.z = 1 - 2*(x^2 + y^2)
    w, x, y, z = (robot_quat[:, i] for i in range(4))
    cos_tilt = 1.0 - 2.0 * (x**2 + y**2)
    tilt_rad = jnp.arccos(jnp.clip(cos_tilt, -1.0, 1.0))
    r_upright = tilt_rad * UPRIGHT_W

    # ── 5. Height deviation ───────────────────────────────────────────
    height_dev = jnp.abs(robot_pos[:, 2] - TARGET_HEIGHT)
    r_height   = height_dev * HEIGHT_W

    # ── 6. Energy (joint velocity magnitude) ─────────────────────────
    r_energy = jnp.sum(joint_vel**2, axis=-1) * ENERGY_W

    # ── 7. Smoothness (action change) ─────────────────────────────────
    r_smooth = jnp.sum((action - prev_action)**2, axis=-1) * SMOOTH_W

    # ── 8. Alive bonus ────────────────────────────────────────────────
    r_alive = jnp.full(robot_pos.shape[0], ALIVE_BONUS)

    # ── 9. Heading reward (face toward goal) ──────────────────────────
    goal_dir = goal_xy - robot_xy                               # (B, 2)
    goal_dir_norm = goal_dir / (jnp.linalg.norm(goal_dir, axis=-1, keepdims=True) + 1e-8)
    # Forward dir from quaternion: x-component of body x-axis in world frame
    # body_x.x = 1 - 2*(y^2 + z^2),  body_x.y = 2*(x*y + w*z)
    fwd_x = 1.0 - 2.0 * (y**2 + z**2)
    fwd_y = 2.0 * (x*y + w*z)
    fwd_norm = jnp.stack([fwd_x, fwd_y], axis=-1)
    fwd_norm = fwd_norm / (jnp.linalg.norm(fwd_norm, axis=-1, keepdims=True) + 1e-8)
    heading_dot = jnp.sum(fwd_norm * goal_dir_norm, axis=-1)
    r_heading = heading_dot * HEADING_W

    # ── Sanitize individual components before summing ─────────────────
    # Prevents one exploding term (e.g. huge joint vel) from corrupting gradients
    r_progress = jnp.clip(r_progress, -5.0, 5.0)
    r_energy   = jnp.clip(r_energy,   -2.0, 0.0)
    r_smooth   = jnp.clip(r_smooth,   -2.0, 0.0)

    # ── Total ─────────────────────────────────────────────────────────
    total = (r_progress + r_goal + r_collision + r_near
             + r_upright + r_height + r_energy + r_smooth
             + r_alive + r_heading)

    # ── Guard: replace NaN/inf with 0 and clip to finite range ────────
    # Upper bound 210 preserves the 200-point goal bonus
    total = jnp.where(jnp.isfinite(total), total, 0.0)
    total = jnp.clip(total, -20.0, 210.0)

    info = {
        "r_progress":  r_progress,
        "r_goal":      r_goal,
        "r_collision": r_collision,
        "r_near":      r_near,
        "r_upright":   r_upright,
        "r_height":    r_height,
        "r_energy":    r_energy,
        "r_smooth":    r_smooth,
        "r_alive":     r_alive,
        "r_heading":   r_heading,
        "dist_goal":   dist_goal,
    }
    return total, info, dist_goal


@jax.jit
def check_termination(
    robot_pos:   jnp.ndarray,   # (B, 3)
    robot_quat:  jnp.ndarray,   # (B, 4)
    goal_pos:    jnp.ndarray,   # (B, 2)
    step_count:  jnp.ndarray,   # (B,) int32
    max_steps:   int = 1000,
    min_height:  float = 0.2,
    max_tilt:    float = 1.0472,   # pi/3
):
    """
    Returns terminated (B,) bool — episode is over.
    Fallen: height < min_height or tilt > max_tilt.
    Goal: dist < GOAL_TOL.
    Timeout: step_count >= max_steps.
    """
    w, x, y, z = (robot_quat[:, i] for i in range(4))
    cos_tilt = 1.0 - 2.0 * (x**2 + y**2)
    tilt = jnp.arccos(jnp.clip(cos_tilt, -1.0, 1.0))

    fallen  = (robot_pos[:, 2] < min_height) | (tilt > max_tilt)
    at_goal = jnp.linalg.norm(goal_pos - robot_pos[:, :2], axis=-1) < GOAL_TOL
    timeout = step_count >= max_steps
    return fallen | at_goal | timeout
