"""
Reward Function for Spot Navigation
=====================================
Multi-component reward that teaches the robot to:
1. Navigate toward the goal efficiently
2. Avoid obstacles (static and dynamic)
3. Stay balanced and upright
4. Use energy efficiently
5. Maintain smooth, natural movement

The reward is carefully shaped so the robot learns priorities:
    Safety (no collisions) > Goal reaching > Efficiency > Smoothness
"""

import numpy as np
from . import config


class RewardComputer:
    """
    Computes reward for navigation task each simulation step.

    Usage:
        reward_fn = RewardComputer()

        # Each step:
        reward, info = reward_fn.compute(
            robot_pos=..., robot_quat=..., goal_pos=...,
            prev_robot_pos=..., joint_vel=..., action=...,
            prev_action=..., min_obstacle_dist=...,
        )
    """

    def __init__(self, reward_config=None):
        self.cfg = reward_config or config.REWARD_CONFIG
        self.prev_distance_to_goal = None

    def reset(self, robot_pos, goal_pos):
        """Reset reward state for new episode."""
        self.prev_distance_to_goal = np.linalg.norm(
            np.array(goal_pos) - np.array(robot_pos[:2])
        )

    def compute(self, robot_pos, robot_quat, goal_pos,
                prev_robot_pos, joint_vel, action, prev_action,
                min_obstacle_dist, has_collision):
        """
        Compute total reward and component breakdown.

        Args:
            robot_pos: (x, y, z) current position
            robot_quat: (w, x, y, z) body orientation quaternion
            goal_pos: (x, y) goal position
            prev_robot_pos: (x, y, z) previous position
            joint_vel: (12,) joint velocities
            action: (12,) current action
            prev_action: (12,) previous action
            min_obstacle_dist: Closest distance to any obstacle (meters)
            has_collision: Boolean — did the robot collide this step?

        Returns:
            total_reward: float
            info: dict with reward component breakdown
        """
        robot_xy = np.array(robot_pos[:2])
        goal_xy = np.array(goal_pos[:2])

        # ── 1. Progress toward goal ──
        distance_to_goal = np.linalg.norm(goal_xy - robot_xy)
        progress = self.prev_distance_to_goal - distance_to_goal
        progress_reward = progress * self.cfg["progress_weight"]
        self.prev_distance_to_goal = distance_to_goal

        # ── 2. Goal reached bonus ──
        goal_bonus = 0.0
        goal_reached = distance_to_goal < self.cfg["goal_tolerance"]
        if goal_reached:
            goal_bonus = self.cfg["goal_reached_bonus"]

        # ── 3. Collision penalty ──
        collision_penalty = 0.0
        if has_collision:
            collision_penalty = self.cfg["collision_penalty"]

        # ── 4. Near-collision penalty (proximity warning) ──
        near_collision_penalty = 0.0
        if min_obstacle_dist < self.cfg["near_collision_threshold"]:
            # Penalty increases as robot gets closer to obstacles
            proximity_factor = 1.0 - (
                min_obstacle_dist / self.cfg["near_collision_threshold"]
            )
            near_collision_penalty = (
                self.cfg["near_collision_penalty"] * proximity_factor
            )

        # ── 5. Upright stability reward ──
        w, x, y, z = robot_quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        tilt_magnitude = np.sqrt(roll**2 + pitch**2)
        upright_penalty = tilt_magnitude * self.cfg["upright_weight"]

        # ── 6. Height maintenance ──
        height = robot_pos[2]
        height_error = abs(height - self.cfg["target_height"])
        height_penalty = height_error * self.cfg["height_weight"]

        # ── 7. Energy efficiency ──
        energy = np.sum(np.square(joint_vel))
        energy_penalty = energy * self.cfg["energy_weight"]

        # ── 8. Action smoothness ──
        if prev_action is not None:
            action_diff = np.sum(np.square(action - prev_action))
        else:
            action_diff = 0.0
        smoothness_penalty = action_diff * self.cfg["smoothness_weight"]

        # ── 9. Heading reward (face toward goal) ──
        # Extract yaw from quaternion
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        goal_direction = np.arctan2(
            goal_xy[1] - robot_xy[1],
            goal_xy[0] - robot_xy[0],
        )
        heading_error = abs(self._angle_diff(yaw, goal_direction))
        # Reward for facing goal (max 1.0 when perfectly aligned)
        heading_reward = (1.0 - heading_error / np.pi) * self.cfg["heading_weight"]

        # ── 10. Alive bonus ──
        alive_bonus = self.cfg["alive_bonus"]

        # ── Total ──
        total_reward = (
            progress_reward
            + goal_bonus
            + collision_penalty
            + near_collision_penalty
            + upright_penalty
            + height_penalty
            + energy_penalty
            + smoothness_penalty
            + heading_reward
            + alive_bonus
        )

        info = {
            "reward_progress": progress_reward,
            "reward_goal_bonus": goal_bonus,
            "reward_collision": collision_penalty,
            "reward_near_collision": near_collision_penalty,
            "reward_upright": upright_penalty,
            "reward_height": height_penalty,
            "reward_energy": energy_penalty,
            "reward_smoothness": smoothness_penalty,
            "reward_heading": heading_reward,
            "reward_alive": alive_bonus,
            "reward_total": total_reward,
            "distance_to_goal": distance_to_goal,
            "min_obstacle_dist": min_obstacle_dist,
            "tilt_rad": tilt_magnitude,
            "body_height": height,
            "goal_reached": goal_reached,
        }

        return total_reward, info

    @staticmethod
    def _angle_diff(a, b):
        """Compute shortest angular difference between two angles."""
        diff = a - b
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
