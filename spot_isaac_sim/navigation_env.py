"""
Spot Navigation Environment for Isaac Sim
===========================================
Gymnasium-compatible RL environment that:
- Runs Spot robot in Isaac Sim with physics
- Captures 5 depth camera images each step
- Provides proprioceptive observations (joint state, body pose)
- Applies domain randomization each episode
- Handles dynamic obstacles
- Computes multi-component reward

Observation space:
    Dict({
        "depth": Box(0, max_range, shape=(5, 120, 160)),  # 5 cameras
        "proprio": Box(-inf, inf, shape=(37,)),            # Body state
    })

Action space:
    Box(-1, 1, shape=(12,))  # 12 joint position targets (normalized)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.isaac.core.utils.prims as prim_utils
    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False

from . import config
from .depth_cameras import DepthCameraRig
from .environment import EnvironmentBuilder
from .dynamic_obstacles import DynamicObstacleManager
from .reward import RewardComputer


class SpotNavigationEnv(gym.Env):
    """
    Isaac Sim environment for training Spot with depth camera navigation.

    This environment:
    1. Loads a random room each episode
    2. Places random obstacles
    3. Spawns dynamic (moving) obstacles
    4. Captures depth images from 5 cameras
    5. Returns depth + proprioception as observations
    6. Rewards progress toward goal while avoiding obstacles

    Usage:
        env = SpotNavigationEnv(headless=True)
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, headless=True, render_mode=None):
        super().__init__()

        self.headless = headless
        self.render_mode = render_mode

        # Configuration
        self.robot_cfg = config.SPOT_ROBOT
        self.cam_cfg = config.CAMERA_RIG
        self.obs_cfg = config.OBSERVATION
        self.act_cfg = config.ACTION
        self.term_cfg = config.TERMINATION
        self.rand_cfg = config.DOMAIN_RANDOMIZATION
        self.train_cfg = config.TRAINING

        # ── Observation space ──
        # Depth images: 5 cameras × 120H × 160W, values in [0, max_range]
        depth_shape = tuple(self.obs_cfg["depth_shape"])
        max_range = self.cam_cfg["max_range"]

        # Proprioception: 37-dim vector
        proprio_dim = self.obs_cfg["proprio_dim"]

        self.observation_space = spaces.Dict({
            "depth": spaces.Box(
                low=0.0, high=max_range,
                shape=depth_shape, dtype=np.float32,
            ),
            "proprio": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(proprio_dim,), dtype=np.float32,
            ),
        })

        # ── Action space ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.act_cfg["dim"],), dtype=np.float32,
        )

        # ── Internal state ──
        self.current_step = 0
        self.goal_position = None
        self.robot_spawn = None
        self.prev_action = None
        self.prev_robot_pos = None
        self._total_timesteps = 0  # Across all episodes (for curriculum)

        # ── Joint limits for action scaling ──
        self.joint_lower = np.array(
            self.robot_cfg["joint_lower_limits"], dtype=np.float32
        )
        self.joint_upper = np.array(
            self.robot_cfg["joint_upper_limits"], dtype=np.float32
        )

        # ── Sub-systems ──
        self.world = None
        self.robot = None
        self.camera_rig = None
        self.env_builder = None
        self.dynamic_manager = None
        self.reward_computer = RewardComputer()

        # Initialize simulation
        self._setup_simulation()

    def _setup_simulation(self):
        """Initialize Isaac Sim world and all subsystems."""
        if HAS_ISAAC:
            self.world = World(
                stage_units_in_meters=1.0,
                physics_dt=1.0 / 500.0,        # 500Hz physics
                rendering_dt=1.0 / 30.0,        # 30Hz rendering
            )

            # Load Spot robot
            add_reference_to_stage(
                usd_path=self.robot_cfg["usd_path"],
                prim_path="/World/Spot",
            )
            self.robot = Robot(prim_path="/World/Spot")
            self.world.scene.add(self.robot)

            # Initialize camera rig
            self.camera_rig = DepthCameraRig(
                robot_prim_path="/World/Spot/base_link"
            )
            self.camera_rig.initialize()

            self.world.reset()
        else:
            # Mock mode for testing without Isaac Sim
            self.camera_rig = DepthCameraRig(
                robot_prim_path="/World/Spot/base_link"
            )
            self.camera_rig.initialize()

        # Environment builder (works with or without Isaac Sim)
        self.env_builder = EnvironmentBuilder()
        self.env_builder.initialize()

        print("[SpotNavigationEnv] Initialization complete")
        print(f"  Observation space: depth={self.obs_cfg['depth_shape']}, "
              f"proprio={self.obs_cfg['proprio_dim']}")
        print(f"  Action space: {self.act_cfg['dim']} joints")
        print(f"  Isaac Sim available: {HAS_ISAAC}")

    def reset(self, seed=None, options=None):
        """
        Reset environment for a new episode.

        1. Select random room
        2. Place random obstacles
        3. Spawn dynamic obstacles
        4. Randomize sensor noise
        5. Place robot at random position
        6. Place goal at random reachable position

        Returns:
            observation: Dict with "depth" and "proprio"
            info: Dict with episode metadata
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_action = None

        # ── Get curriculum stage ──
        curriculum_stage = self._get_curriculum_stage()

        # ── Build randomized environment ──
        self.robot_spawn, self.goal_position = self.env_builder.build_episode(
            curriculum_stage=curriculum_stage,
        )

        # ── Spawn dynamic obstacles ──
        room_bounds = self.env_builder.get_room_bounds()
        self.dynamic_manager = DynamicObstacleManager(room_bounds=room_bounds)

        if curriculum_stage and "dynamic_count" in curriculum_stage:
            dyn_range = curriculum_stage["dynamic_count"]
        else:
            dyn_range = config.DYNAMIC_OBSTACLES["count_range"]
        n_dynamic = self._rng_integers(dyn_range[0], dyn_range[1] + 1)

        self.dynamic_manager.spawn_episode(
            n_dynamic=n_dynamic,
            robot_pos=self.robot_spawn,
            goal_pos=self.goal_position,
        )

        # ── Randomize sensor noise ──
        if self.rand_cfg["sensor_noise"]["enabled"]:
            noise_mult = self._rng_uniform(
                *self.rand_cfg["sensor_noise"]["noise_multiplier_range"]
            )
            dropout_mult = self._rng_uniform(
                *self.rand_cfg["sensor_noise"]["dropout_multiplier_range"]
            )
            if curriculum_stage and "noise_multiplier" in curriculum_stage:
                noise_mult *= curriculum_stage["noise_multiplier"]
            self.camera_rig.set_noise_multipliers(noise_mult, dropout_mult)

        # ── Set robot initial state ──
        if HAS_ISAAC and self.robot is not None:
            self._set_robot_pose(self.robot_spawn)
            self.world.step(render=not self.headless)

        self.prev_robot_pos = np.array([
            self.robot_spawn[0], self.robot_spawn[1], self.robot_spawn[2]
        ])

        # ── Reset reward computer ──
        self.reward_computer.reset(
            robot_pos=self.prev_robot_pos,
            goal_pos=self.goal_position,
        )

        # ── Get initial observation ──
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one environment step.

        1. Apply action to robot joints
        2. Step physics simulation
        3. Update dynamic obstacles
        4. Capture depth images
        5. Compute reward
        6. Check termination

        Args:
            action: np.ndarray shape (12,), values in [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── Scale action to joint limits ──
        target_joints = (
            self.joint_lower
            + (action + 1.0) * 0.5 * (self.joint_upper - self.joint_lower)
        )

        # ── Apply to robot ──
        if HAS_ISAAC and self.robot is not None:
            self.robot.set_joint_positions(target_joints)

            # Step simulation (multiple substeps for stability)
            control_dt = 1.0 / self.act_cfg["control_frequency"]
            physics_dt = 1.0 / 500.0
            n_substeps = max(1, int(control_dt / physics_dt))
            for _ in range(n_substeps):
                self.world.step(render=False)

        # ── Update dynamic obstacles ──
        dt = 1.0 / self.act_cfg["control_frequency"]
        self.dynamic_manager.step(dt)

        # ── Get robot state ──
        robot_pos, robot_quat, joint_pos, joint_vel = self._get_robot_state()

        # ── Compute minimum distance to obstacles ──
        min_obs_dist, has_collision = self._compute_obstacle_distances(
            robot_pos[:2]
        )

        # ── Compute reward ──
        reward, reward_info = self.reward_computer.compute(
            robot_pos=robot_pos,
            robot_quat=robot_quat,
            goal_pos=self.goal_position,
            prev_robot_pos=self.prev_robot_pos,
            joint_vel=joint_vel,
            action=action,
            prev_action=self.prev_action,
            min_obstacle_dist=min_obs_dist,
            has_collision=has_collision,
        )

        # ── Check termination ──
        terminated = self._check_termination(
            robot_pos, robot_quat, reward_info.get("goal_reached", False)
        )
        self.current_step += 1
        self._total_timesteps += 1
        truncated = self.current_step >= self.term_cfg["max_episode_steps"]

        # ── Get observation ──
        observation = self._get_observation()

        # ── Update state ──
        self.prev_action = action.copy()
        self.prev_robot_pos = robot_pos.copy()

        # ── Info ──
        info = self._get_info()
        info.update(reward_info)

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Build observation dict with depth images and proprioception.

        Returns:
            dict: {
                "depth": (5, 120, 160) float32,
                "proprio": (37,) float32
            }
        """
        # Depth images from 5 cameras
        depth = self.camera_rig.capture()

        # Proprioceptive state
        robot_pos, robot_quat, joint_pos, joint_vel = self._get_robot_state()

        # Body velocities
        if self.prev_robot_pos is not None:
            dt = 1.0 / self.act_cfg["control_frequency"]
            lin_vel = (robot_pos - self.prev_robot_pos) / max(dt, 1e-6)
        else:
            lin_vel = np.zeros(3, dtype=np.float32)

        ang_vel = self._get_angular_velocity()

        # Goal direction (relative to robot)
        goal_xy = np.array(self.goal_position[:2])
        robot_xy = robot_pos[:2]
        goal_vec = goal_xy - robot_xy
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 0:
            goal_dir = goal_vec / goal_dist
        else:
            goal_dir = np.zeros(2)

        # Concatenate proprioception
        proprio = np.concatenate([
            joint_pos,              # 12
            joint_vel,              # 12
            robot_quat,             # 4
            lin_vel,                # 3
            ang_vel,                # 3
            goal_dir,               # 2
            [goal_dist],            # 1
        ]).astype(np.float32)       # Total: 37

        return {
            "depth": depth,
            "proprio": proprio,
        }

    def _get_robot_state(self):
        """
        Get current robot state from simulation.

        Returns:
            robot_pos: (3,) position
            robot_quat: (4,) quaternion [w, x, y, z]
            joint_pos: (12,) joint angles
            joint_vel: (12,) joint velocities
        """
        if HAS_ISAAC and self.robot is not None:
            pos, quat = self.robot.get_world_pose()
            joint_pos = self.robot.get_joint_positions()
            joint_vel = self.robot.get_joint_velocities()
            return (
                np.array(pos, dtype=np.float32),
                np.array(quat, dtype=np.float32),
                np.array(joint_pos[:12], dtype=np.float32),
                np.array(joint_vel[:12], dtype=np.float32),
            )
        else:
            # Mock state for testing
            return (
                np.array(self.prev_robot_pos or [0, 0, 0.5], dtype=np.float32),
                np.array([1, 0, 0, 0], dtype=np.float32),
                np.array(self.robot_cfg["standing_pose"], dtype=np.float32),
                np.zeros(12, dtype=np.float32),
            )

    def _get_angular_velocity(self):
        """Get body angular velocity."""
        if HAS_ISAAC and self.robot is not None:
            vel = self.robot.get_angular_velocity()
            return np.array(vel, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _set_robot_pose(self, spawn):
        """Set robot position and orientation in simulation."""
        x, y, z, yaw = spawn

        # Yaw to quaternion
        w = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)
        quat = np.array([w, 0, 0, qz])

        self.robot.set_world_pose(
            position=np.array([x, y, z]),
            orientation=quat,
        )

        # Set standing pose
        standing = np.array(self.robot_cfg["standing_pose"], dtype=np.float32)
        self.robot.set_joint_positions(standing)
        self.robot.set_joint_velocities(np.zeros(12, dtype=np.float32))

    def _compute_obstacle_distances(self, robot_xy):
        """
        Compute minimum distance to any obstacle.

        Args:
            robot_xy: (2,) robot x,y position

        Returns:
            min_dist: float — closest obstacle distance
            has_collision: bool — is robot overlapping an obstacle?
        """
        robot_xy = np.array(robot_xy)
        min_dist = float("inf")
        collision_threshold = 0.3  # Robot body radius

        # Static obstacles
        static_positions = self.env_builder.get_obstacle_positions()
        for pos in static_positions:
            dist = np.linalg.norm(robot_xy - np.array(pos))
            min_dist = min(min_dist, dist)

        # Dynamic obstacles
        dynamic_positions = self.dynamic_manager.get_positions()
        for pos in dynamic_positions:
            dist = np.linalg.norm(robot_xy - np.array(pos))
            min_dist = min(min_dist, dist)

        has_collision = min_dist < collision_threshold

        if min_dist == float("inf"):
            min_dist = 10.0  # No obstacles

        return min_dist, has_collision

    def _check_termination(self, robot_pos, robot_quat, goal_reached):
        """Check if episode should terminate."""
        # Height check (fallen)
        if robot_pos[2] < self.term_cfg["min_height"]:
            return True

        # Tilt check (fallen over)
        w, x, y, z = robot_quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        if abs(roll) > self.term_cfg["max_tilt_rad"]:
            return True
        if abs(pitch) > self.term_cfg["max_tilt_rad"]:
            return True

        # Goal reached
        if self.term_cfg["terminate_on_goal"] and goal_reached:
            return True

        return False

    def _get_curriculum_stage(self):
        """
        Get current curriculum stage based on total timesteps.

        Returns curriculum parameters or None if curriculum is disabled.
        """
        curriculum = self.train_cfg["curriculum"]
        if not curriculum["enabled"]:
            return None

        current_stage = None
        for stage in curriculum["stages"]:
            if self._total_timesteps >= stage["timesteps"]:
                current_stage = stage

        return current_stage

    def _get_info(self):
        """Build info dict for debugging/logging."""
        return {
            "step": self.current_step,
            "total_timesteps": self._total_timesteps,
            "goal_position": self.goal_position,
            "robot_spawn": self.robot_spawn,
            "room": (self.env_builder._current_room["name"]
                     if self.env_builder._current_room else "unknown"),
            "n_static_obstacles": len(self.env_builder._placed_obstacles),
            "n_dynamic_obstacles": len(self.dynamic_manager.obstacles)
            if self.dynamic_manager else 0,
        }

    def _rng_uniform(self, low, high):
        """Generate uniform random number using env's RNG."""
        if hasattr(self, 'np_random'):
            return self.np_random.uniform(low, high)
        return np.random.uniform(low, high)

    def _rng_integers(self, low, high):
        """Generate random integer using env's RNG."""
        if hasattr(self, 'np_random'):
            return self.np_random.integers(low, high)
        return np.random.randint(low, high)

    def render(self):
        """Render is handled by Isaac Sim viewport."""
        if HAS_ISAAC and self.world is not None:
            self.world.step(render=True)

    def close(self):
        """Clean up simulation."""
        if HAS_ISAAC and self.world is not None:
            self.world.stop()
