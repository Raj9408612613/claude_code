"""
Spot Quadruped Environment with CNN-based Obstacle Detection

Extends the base SpotEnv to add:
- Random obstacles placed between robot and goal
- Front-facing camera mounted on the robot body
- Dict observation space: camera image + proprioceptive state
- Collision penalty and proximity-based avoidance rewards
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class SpotObstacleEnv(gym.Env):
    """
    Spot environment with visual obstacle detection via front camera.

    Observation space (Dict):
        "image":          (H, W, 3) uint8  - RGB from front camera
        "proprioception": (37,)    float32 - joint/body state + goal direction

    The CNN processes the camera feed to detect obstacles while the
    proprioceptive vector provides body state for locomotion control.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Number of obstacle bodies defined in the XML
    N_OBSTACLES = 8

    def __init__(
        self,
        render_mode=None,
        max_episode_steps=1000,
        camera_width=64,
        camera_height=64,
        n_obstacles_active=5,
        obstacle_zone_min=1.5,
        obstacle_zone_max=7.0,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.n_obstacles_active = min(n_obstacles_active, self.N_OBSTACLES)
        self.obstacle_zone_min = obstacle_zone_min
        self.obstacle_zone_max = obstacle_zone_max

        # Load MuJoCo model
        xml_path = os.path.join(os.path.dirname(__file__), "spot_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Pre-allocate rendering buffers
        self.renderer = mujoco.Renderer(self.model, self.camera_height, self.camera_width)

        # Cache body and geom IDs for obstacles
        self.obstacle_body_ids = []
        self.obstacle_geom_ids = []
        for i in range(self.N_OBSTACLES):
            self.obstacle_body_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            )
            self.obstacle_geom_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_geom_{i}")
            )

        # Cache robot body/geom IDs for collision checking
        self.robot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self.robot_geom_names = [
            "body",
            "fl_hip_geom", "fl_thigh_geom", "fl_calf_geom", "fl_foot",
            "fr_hip_geom", "fr_thigh_geom", "fr_calf_geom", "fr_foot",
            "rl_hip_geom", "rl_thigh_geom", "rl_calf_geom", "rl_foot",
            "rr_hip_geom", "rr_thigh_geom", "rr_calf_geom", "rr_foot",
        ]
        self.robot_geom_ids = set()
        for name in self.robot_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.robot_geom_ids.add(gid)

        self.obstacle_geom_id_set = set(self.obstacle_geom_ids)

        # Camera ID
        self.front_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera"
        )

        # Joint setup
        self.n_joints = 12
        self.joint_limits_lower = np.array([
            -0.8, -2.8, -0.5,
            -0.8, -2.8, -0.5,
            -0.8, -2.8, -0.5,
            -0.8, -2.8, -0.5,
        ])
        self.joint_limits_upper = np.array([
            0.8, 0.8, 2.8,
            0.8, 0.8, 2.8,
            0.8, 0.8, 2.8,
            0.8, 0.8, 2.8,
        ])

        # --- Action space (same as base) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # --- Dict observation space ---
        proprioception_dim = 12 + 12 + 4 + 3 + 3 + 2 + 1  # 37
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8,
            ),
            "proprioception": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(proprioception_dim,),
                dtype=np.float32,
            ),
        })

        # Goal
        self.goal_position = np.array([0.0, 0.0])
        self.goal_tolerance = 0.5

        # State tracking
        self.prev_base_pos = None
        self.obstacle_collision_count = 0

        # For human rendering
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_proprioception(self):
        """37-dim proprioceptive state (identical to base SpotEnv)."""
        joint_pos = self.data.qpos[7:19]
        joint_vel = self.data.qvel[6:18]
        base_quat = self.data.qpos[3:7]
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        base_pos = self.data.qpos[0:2]

        goal_direction = self.goal_position - base_pos
        goal_distance = np.linalg.norm(goal_direction)
        if goal_distance > 0:
            goal_direction = goal_direction / goal_distance

        body_height = self.data.qpos[2:3]

        return np.concatenate([
            joint_pos, joint_vel, base_quat,
            base_lin_vel, base_ang_vel,
            goal_direction, body_height,
        ]).astype(np.float32)

    def _get_camera_image(self):
        """Render RGB image from the robot's front-facing camera."""
        self.renderer.update_scene(self.data, camera=self.front_camera_id)
        img = self.renderer.render()
        return img  # (H, W, 3) uint8

    def _get_obs(self):
        return {
            "image": self._get_camera_image(),
            "proprioception": self._get_proprioception(),
        }

    def _get_info(self):
        base_pos = self.data.qpos[0:2]
        distance_to_goal = np.linalg.norm(self.goal_position - base_pos)
        return {
            "distance_to_goal": distance_to_goal,
            "base_position": base_pos.copy(),
            "goal_position": self.goal_position.copy(),
            "obstacle_collisions": self.obstacle_collision_count,
        }

    # ------------------------------------------------------------------
    # Obstacle placement
    # ------------------------------------------------------------------

    def _randomize_obstacles(self):
        """Place obstacles randomly between robot start and goal."""
        robot_pos = np.array([0.0, 0.0])
        goal_dir = self.goal_position - robot_pos
        goal_dist = np.linalg.norm(goal_dir)
        goal_unit = goal_dir / max(goal_dist, 1e-6)
        # perpendicular direction
        perp = np.array([-goal_unit[1], goal_unit[0]])

        for i in range(self.N_OBSTACLES):
            body_id = self.obstacle_body_ids[i]

            if i < self.n_obstacles_active:
                # Place along the path with lateral scatter
                frac = self.np_random.uniform(0.2, 0.85)
                lateral = self.np_random.uniform(-1.5, 1.5)
                pos = robot_pos + goal_dir * frac + perp * lateral
                # Clamp within zone limits
                dist = np.linalg.norm(pos)
                if dist < self.obstacle_zone_min:
                    pos = pos / max(dist, 1e-6) * self.obstacle_zone_min
                elif dist > self.obstacle_zone_max:
                    pos = pos / max(dist, 1e-6) * self.obstacle_zone_max

                self.model.body_pos[body_id][0] = pos[0]
                self.model.body_pos[body_id][1] = pos[1]
                self.model.body_pos[body_id][2] = 0.15  # on ground
            else:
                # Move inactive obstacles far away and underground
                self.model.body_pos[body_id][0] = 100.0 + i
                self.model.body_pos[body_id][1] = 100.0
                self.model.body_pos[body_id][2] = -5.0

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_obstacle_collision(self):
        """Return True if robot is in contact with any obstacle."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            robot_involved = g1 in self.robot_geom_ids or g2 in self.robot_geom_ids
            obs_involved = g1 in self.obstacle_geom_id_set or g2 in self.obstacle_geom_id_set
            if robot_involved and obs_involved:
                return True
        return False

    def _min_obstacle_distance(self):
        """Return the minimum distance from robot body to any active obstacle."""
        base_pos = self.data.qpos[0:2]
        min_dist = float("inf")
        for i in range(self.n_obstacles_active):
            body_id = self.obstacle_body_ids[i]
            obs_pos = self.data.xpos[body_id][0:2]
            dist = np.linalg.norm(base_pos - obs_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # ------------------------------------------------------------------
    # Core environment methods
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Standing pose with noise
        standing_pose = np.array([
            0.0, -0.9, 1.8,
            0.0, -0.9, 1.8,
            0.0, -0.9, 1.8,
            0.0, -0.9, 1.8,
        ])
        joint_noise = self.np_random.uniform(-0.1, 0.1, size=12)
        self.data.qpos[7:19] = standing_pose + joint_noise
        self.data.qpos[2] = 0.35

        # Random goal
        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(3.0, 8.0)
        self.goal_position = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
        ])

        # Randomize obstacle positions
        self._randomize_obstacles()

        # Stabilisation steps
        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.current_step = 0
        self.prev_base_pos = self.data.qpos[0:2].copy()
        self.obstacle_collision_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -1, 1)
        target_joint_pos = (
            self.joint_limits_lower
            + (action + 1) * 0.5 * (self.joint_limits_upper - self.joint_limits_lower)
        )

        self.data.ctrl[:] = target_joint_pos

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()

        reward, reward_info = self._calculate_reward()
        info.update(reward_info)

        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps

        self.current_step += 1
        self.prev_base_pos = self.data.qpos[0:2].copy()

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _calculate_reward(self):
        base_pos = self.data.qpos[0:2]
        base_height = self.data.qpos[2]
        base_quat = self.data.qpos[3:7]

        # 1. Progress toward goal
        distance_to_goal = np.linalg.norm(self.goal_position - base_pos)
        prev_distance = np.linalg.norm(self.goal_position - self.prev_base_pos)
        progress_reward = (prev_distance - distance_to_goal) * 2.0

        # 2. Tilt penalty
        w, x, y, z = base_quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        tilt_penalty = -(abs(roll) + abs(pitch)) * 0.5

        # 3. Height reward
        target_height = 0.35
        height_reward = -abs(base_height - target_height) * 2.0

        # 4. Energy penalty
        joint_vel = self.data.qvel[6:18]
        energy_penalty = -np.sum(np.square(joint_vel)) * 0.005

        # 5. Control penalty
        control_penalty = -np.sum(np.square(self.data.ctrl)) * 0.001

        # 6. Goal bonus
        goal_bonus = 0.0
        if distance_to_goal < self.goal_tolerance:
            goal_bonus = 100.0

        # 7. Alive bonus
        alive_bonus = 0.5

        # ---------- Obstacle-specific rewards ----------

        # 8. Collision penalty
        collision = self._check_obstacle_collision()
        collision_penalty = 0.0
        if collision:
            collision_penalty = -5.0
            self.obstacle_collision_count += 1

        # 9. Proximity penalty (soft repulsion when getting close)
        min_obs_dist = self._min_obstacle_distance()
        proximity_penalty = 0.0
        proximity_threshold = 0.6  # start penalising within 0.6m
        if min_obs_dist < proximity_threshold:
            # Linearly increasing penalty as robot approaches obstacle
            proximity_penalty = -(1.0 - min_obs_dist / proximity_threshold) * 1.0

        # Total
        total_reward = (
            progress_reward
            + tilt_penalty
            + height_reward
            + energy_penalty
            + control_penalty
            + goal_bonus
            + alive_bonus
            + collision_penalty
            + proximity_penalty
        )

        reward_info = {
            "progress_reward": progress_reward,
            "tilt_penalty": tilt_penalty,
            "height_reward": height_reward,
            "energy_penalty": energy_penalty,
            "goal_bonus": goal_bonus,
            "collision_penalty": collision_penalty,
            "proximity_penalty": proximity_penalty,
            "min_obstacle_dist": min_obs_dist,
            "total_reward": total_reward,
        }

        return total_reward, reward_info

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_terminated(self):
        base_height = self.data.qpos[2]
        base_quat = self.data.qpos[3:7]

        if base_height < 0.15:
            return True

        w, x, y, z = base_quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))

        if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
            return True

        base_pos = self.data.qpos[0:2]
        distance_to_goal = np.linalg.norm(self.goal_position - base_pos)
        if distance_to_goal < self.goal_tolerance:
            return True

        return False

    # ------------------------------------------------------------------
    # Rendering & cleanup
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
        return None

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()
