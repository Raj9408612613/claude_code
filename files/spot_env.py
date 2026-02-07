"""
Spot Quadruped Gymnasium Environment
Trains a quadruped robot to walk towards a goal without falling
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class SpotEnv(gym.Env):
    """Custom Environment for training Spot to walk towards a goal"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}
    
    def __init__(self, render_mode=None, max_episode_steps=1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Load MuJoCo model
        xml_path = os.path.join(os.path.dirname(__file__), "spot_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint indices (12 joints: 3 per leg x 4 legs)
        self.n_joints = 12
        
        # Action space: joint position targets (normalized -1 to 1)
        # Will be scaled to actual joint limits
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.n_joints,), 
            dtype=np.float32
        )
        
        # Observation space includes:
        # - Joint positions (12)
        # - Joint velocities (12)
        # - Body orientation (4 quaternion)
        # - Body linear velocity (3)
        # - Body angular velocity (3)
        # - Goal direction (2: x, y relative)
        # - Body height (1)
        # Total: 37 dimensions
        obs_dim = 12 + 12 + 4 + 3 + 3 + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Joint limits for scaling actions
        self.joint_limits_lower = np.array([
            -0.8, -2.8, -0.5,  # Front left: hip, thigh, calf
            -0.8, -2.8, -0.5,  # Front right
            -0.8, -2.8, -0.5,  # Rear left
            -0.8, -2.8, -0.5,  # Rear right
        ])
        self.joint_limits_upper = np.array([
            0.8, 0.8, 2.8,     # Front left
            0.8, 0.8, 2.8,     # Front right
            0.8, 0.8, 2.8,     # Rear left
            0.8, 0.8, 2.8,     # Rear right
        ])
        
        # Goal position
        self.goal_position = np.array([0.0, 0.0])
        self.goal_tolerance = 0.5  # meters
        
        # For rendering
        if self.render_mode == 'human':
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None
            
        # Previous position for velocity calculation
        self.prev_base_pos = None
        
    def _get_obs(self):
        """Get current observation"""
        # Joint positions and velocities
        joint_pos = self.data.qpos[7:19]  # Skip base position/orientation
        joint_vel = self.data.qvel[6:18]  # Skip base velocities
        
        # Base orientation (quaternion)
        base_quat = self.data.qpos[3:7]
        
        # Base velocities
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        
        # Body position
        base_pos = self.data.qpos[0:2]  # x, y only
        
        # Goal direction (relative to robot)
        goal_direction = self.goal_position - base_pos
        goal_distance = np.linalg.norm(goal_direction)
        if goal_distance > 0:
            goal_direction = goal_direction / goal_distance
        
        # Body height
        body_height = self.data.qpos[2:3]
        
        # Concatenate all observations
        obs = np.concatenate([
            joint_pos,
            joint_vel,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            goal_direction,
            body_height,
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self):
        """Additional info for debugging"""
        base_pos = self.data.qpos[0:2]
        distance_to_goal = np.linalg.norm(self.goal_position - base_pos)
        
        return {
            'distance_to_goal': distance_to_goal,
            'base_position': base_pos.copy(),
            'goal_position': self.goal_position.copy(),
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set random initial joint positions (standing pose with noise)
        standing_pose = np.array([
            0.0, -0.9, 1.8,  # Front left
            0.0, -0.9, 1.8,  # Front right
            0.0, -0.9, 1.8,  # Rear left
            0.0, -0.9, 1.8,  # Rear right
        ])
        joint_noise = self.np_random.uniform(-0.1, 0.1, size=12)
        self.data.qpos[7:19] = standing_pose + joint_noise
        
        # Set initial height
        self.data.qpos[2] = 0.35  # Body height
        
        # Random goal position
        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(3.0, 8.0)
        self.goal_position = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ])
        
        # Forward simulation to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step = 0
        self.prev_base_pos = self.data.qpos[0:2].copy()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Scale action to joint limits
        action = np.clip(action, -1, 1)
        target_joint_pos = (
            self.joint_limits_lower + 
            (action + 1) * 0.5 * (self.joint_limits_upper - self.joint_limits_lower)
        )
        
        # Apply control (PD controller is built into MuJoCo actuators)
        self.data.ctrl[:] = target_joint_pos
        
        # Step simulation (multiple substeps for stability)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_obs()
        info = self._get_info()
        
        # Calculate reward
        reward, reward_info = self._calculate_reward()
        info.update(reward_info)
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        self.current_step += 1
        
        # Update previous position
        self.prev_base_pos = self.data.qpos[0:2].copy()
        
        # Rendering
        if self.render_mode == 'human' and self.viewer is not None:
            self.viewer.sync()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """
        Reward function with multiple components:
        1. Moving towards goal
        2. Staying upright
        3. Energy efficiency
        4. Reaching goal
        """
        base_pos = self.data.qpos[0:2]
        base_height = self.data.qpos[2]
        base_quat = self.data.qpos[3:7]
        
        # 1. Progress towards goal
        distance_to_goal = np.linalg.norm(self.goal_position - base_pos)
        prev_distance = np.linalg.norm(self.goal_position - self.prev_base_pos)
        progress_reward = (prev_distance - distance_to_goal) * 2.0
        
        # 2. Stay upright (penalize tilt)
        # Convert quaternion to roll, pitch
        w, x, y, z = base_quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = np.arcsin(2*(w*y - z*x))
        tilt_penalty = -(abs(roll) + abs(pitch)) * 0.5
        
        # 3. Height reward (encourage staying at proper height)
        target_height = 0.35
        height_reward = -abs(base_height - target_height) * 2.0
        
        # 4. Energy penalty (penalize large joint velocities)
        joint_vel = self.data.qvel[6:18]
        energy_penalty = -np.sum(np.square(joint_vel)) * 0.005
        
        # 5. Control smoothness (penalize large actions)
        control_penalty = -np.sum(np.square(self.data.ctrl)) * 0.001
        
        # 6. Goal reached bonus
        goal_bonus = 0.0
        if distance_to_goal < self.goal_tolerance:
            goal_bonus = 100.0
        
        # 7. Alive bonus (encourage staying alive)
        alive_bonus = 0.5
        
        # Total reward
        total_reward = (
            progress_reward +
            tilt_penalty +
            height_reward +
            energy_penalty +
            control_penalty +
            goal_bonus +
            alive_bonus
        )
        
        reward_info = {
            'progress_reward': progress_reward,
            'tilt_penalty': tilt_penalty,
            'height_reward': height_reward,
            'energy_penalty': energy_penalty,
            'goal_bonus': goal_bonus,
            'total_reward': total_reward,
        }
        
        return total_reward, reward_info
    
    def _is_terminated(self):
        """Check if episode should terminate"""
        base_height = self.data.qpos[2]
        base_quat = self.data.qpos[3:7]
        
        # Terminate if robot falls (too low or flipped)
        if base_height < 0.15:  # Too low
            return True
        
        # Check if severely tilted
        w, x, y, z = base_quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        
        if abs(roll) > np.pi/2 or abs(pitch) > np.pi/2:
            return True
        
        # Check if goal reached
        base_pos = self.data.qpos[0:2]
        distance_to_goal = np.linalg.norm(self.goal_position - base_pos)
        if distance_to_goal < self.goal_tolerance:
            return True
        
        return False
    
    def render(self):
        if self.render_mode == 'human' and self.viewer is not None:
            self.viewer.sync()
        return None
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
