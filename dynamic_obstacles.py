"""
Dynamic Obstacle Classes
========================
CPU-side utility classes for single-env simulation or testing.
The batched MJX training loop uses the equivalent JAX ops in mjx_nav_env.py.

HumanoidObstacle — kinematic biped that patrols near the episode goal.
  - Driven by waypoint system (goal_pos ± patrol_radius)
  - Sinusoidal limb offsets for visual walking animation
  - No RL policy or physics articulation needed
"""

import math
import numpy as np


class HumanoidObstacle:
    """
    Kinematic humanoid obstacle that walks back and forth near the goal.

    Patrol path: goal_pos +/- patrol_radius along the X axis.
    The torso mocap body is moved each step; limb animation is visual-only
    (sinusoidal offsets that can be read for rendering but are not separate
    MJX mocap bodies in the batched env).

    Args:
        goal_pos (np.ndarray): 2-D goal position [x, y] in world frame.
        speed (float): Walking speed in m/s.
        stride_freq (float): Stride frequency in Hz.
        patrol_radius (float): Half-distance of patrol path from goal (m).
    """

    def __init__(
        self,
        goal_pos: np.ndarray,
        speed: float = 0.8,
        stride_freq: float = 1.2,
        patrol_radius: float = 1.5,
    ):
        self.speed = speed
        self.stride_freq = stride_freq
        self.patrol_radius = patrol_radius
        self.t = 0.0

        # Patrol between two waypoints either side of the goal along X
        offset = np.array([patrol_radius, 0.0], dtype=np.float32)
        self.waypoints = [
            (goal_pos + offset).copy(),
            (goal_pos - offset).copy(),
        ]
        self.wp_idx = 0
        self.position = self.waypoints[0].copy()

    # ------------------------------------------------------------------ #
    def update(self, dt: float) -> None:
        """Advance humanoid position and internal clock by dt seconds."""
        self.t += dt
        target = self.waypoints[self.wp_idx]
        diff = target - self.position
        dist = np.linalg.norm(diff)
        if dist < 0.2:
            self.wp_idx = 1 - self.wp_idx       # toggle between wp 0 and 1
        else:
            self.position += (diff / dist) * self.speed * dt

    # ------------------------------------------------------------------ #
    def get_mocap_pos(self) -> np.ndarray:
        """
        World-frame position for the humanoid torso mocap body.

        z=1.0 places the torso centre at 1 m above floor; the lower-leg
        bottoms reach approximately z=-0.11 (slightly embedded — acceptable
        for a simplified kinematic model).

        Returns:
            np.ndarray: shape (3,) — [x, y, z]
        """
        return np.array([self.position[0], self.position[1], 1.0],
                        dtype=np.float32)

    # ------------------------------------------------------------------ #
    def get_limb_offsets(self):
        """
        Sinusoidal limb offsets for visual walking animation.

        These represent Y-axis angular swings for each limb segment in the
        body frame. They are informational only — the batched MJX environment
        uses a single rigid mocap body (no per-limb mocap update).

        Returns:
            tuple: (lul_y, rul_y, lll_y, rll_y)
                   left-upper-leg, right-upper-leg,
                   left-lower-leg, right-lower-leg offsets (radians / metres)
        """
        phase = self.t * self.stride_freq * 2.0 * math.pi
        lul_y = 0.15 * math.sin(phase)           # left upper leg swing
        rul_y = -0.15 * math.sin(phase)          # right upper leg (opposite)
        lll_y = 0.10 * math.sin(phase - 0.5)    # lower leg follow-through
        rll_y = -0.10 * math.sin(phase - 0.5)
        return lul_y, rul_y, lll_y, rll_y

    # ------------------------------------------------------------------ #
    def reset(self, new_goal_pos: np.ndarray) -> None:
        """Reset patrol path for a new episode goal position."""
        offset = np.array([self.patrol_radius, 0.0], dtype=np.float32)
        self.waypoints = [
            (new_goal_pos + offset).copy(),
            (new_goal_pos - offset).copy(),
        ]
        self.wp_idx = 0
        self.position = self.waypoints[0].copy()
        self.t = 0.0
