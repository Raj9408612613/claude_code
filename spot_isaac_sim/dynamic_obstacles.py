"""
Dynamic (Moving) Obstacles for Spot Training
==============================================
Simulates people walking, carts moving, etc.

Each dynamic obstacle:
- Spawns at a random position
- Moves toward random waypoints at a random speed
- Can pause briefly (simulating a person stopping)
- Uses collision geometry so the robot must avoid it

This teaches the robot to handle unpredictable moving objects
in real-world environments (hallways, offices, hospitals).
"""

import numpy as np

try:
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.prims import XFormPrim
    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False

from . import config


class DynamicObstacleManager:
    """
    Spawns and updates moving obstacles each simulation step.

    Usage:
        manager = DynamicObstacleManager(room_bounds=[-5, 5, -5, 5])
        manager.spawn_episode(n_dynamic=3)

        # Each step:
        manager.step(dt=0.02)
        positions = manager.get_positions()
    """

    def __init__(self, room_bounds, dynamic_config=None):
        """
        Args:
            room_bounds: [x_min, x_max, y_min, y_max]
            dynamic_config: Override default DYNAMIC_OBSTACLES config
        """
        self.room_bounds = room_bounds
        self.cfg = dynamic_config or config.DYNAMIC_OBSTACLES

        self._rng = np.random.default_rng()
        self.obstacles = []  # List of DynamicObstacle instances

    def spawn_episode(self, n_dynamic=None, robot_pos=None, goal_pos=None):
        """
        Spawn dynamic obstacles for a new episode.

        Args:
            n_dynamic: Number of obstacles (None = random from config range)
            robot_pos: (x, y) robot position to avoid spawning on top of
            goal_pos: (x, y) goal position to avoid
        """
        # Clear previous
        self._clear()

        if not self.cfg["enabled"]:
            return

        if n_dynamic is None:
            n_dynamic = self._rng.integers(*self.cfg["count_range"])

        # Build weighted type pool
        type_pool = []
        for obs_type in self.cfg["types"]:
            type_pool.extend([obs_type] * obs_type["weight"])

        robot_xy = np.array(robot_pos[:2]) if robot_pos else np.array([0, 0])
        goal_xy = np.array(goal_pos[:2]) if goal_pos else np.array([0, 0])

        for i in range(n_dynamic):
            obs_type = self._rng.choice(type_pool)

            # Find valid spawn position
            pos = self._find_spawn_pos(robot_xy, goal_xy)
            if pos is None:
                continue

            speed = self._rng.uniform(*obs_type["speed_range"])

            obstacle = DynamicObstacle(
                obstacle_id=i,
                position=pos,
                speed=speed,
                obs_type=obs_type,
                room_bounds=self.room_bounds,
                movement_cfg=self.cfg["movement"],
                rng=self._rng,
            )

            # Create visual in Isaac Sim
            if HAS_ISAAC:
                self._create_prim(obstacle, obs_type)

            self.obstacles.append(obstacle)

    def step(self, dt):
        """
        Update all dynamic obstacle positions.

        Args:
            dt: Time step in seconds
        """
        for obs in self.obstacles:
            obs.update(dt)

            # Update Isaac Sim prim position
            if HAS_ISAAC:
                prim_path = f"/World/DynamicObstacles/dynamic_{obs.obstacle_id}"
                if prim_utils.is_prim_path_valid(prim_path):
                    prim = XFormPrim(prim_path)
                    prim.set_world_pose(
                        position=np.array([obs.position[0], obs.position[1],
                                           obs.height / 2]),
                    )

    def get_positions(self):
        """
        Get current positions of all dynamic obstacles.

        Returns:
            list of (x, y) tuples
        """
        return [obs.position.copy() for obs in self.obstacles]

    def get_velocities(self):
        """
        Get current velocities of all dynamic obstacles.

        Returns:
            list of (vx, vy) tuples
        """
        return [obs.velocity.copy() for obs in self.obstacles]

    def _find_spawn_pos(self, robot_xy, goal_xy, min_dist=2.0):
        """Find a spawn position not too close to robot or goal."""
        margin = 1.0
        for _ in range(50):
            x = self._rng.uniform(
                self.room_bounds[0] + margin,
                self.room_bounds[1] - margin,
            )
            y = self._rng.uniform(
                self.room_bounds[2] + margin,
                self.room_bounds[3] - margin,
            )
            pos = np.array([x, y])

            if np.linalg.norm(pos - robot_xy) < min_dist:
                continue
            if np.linalg.norm(pos - goal_xy) < min_dist:
                continue

            return pos

        return None

    def _create_prim(self, obstacle, obs_type):
        """Create the visual/collision prim in Isaac Sim."""
        prim_path = (
            f"/World/DynamicObstacles/dynamic_{obstacle.obstacle_id}"
        )

        fallback = self.cfg["primitive_fallback"]
        prim_utils.create_prim(
            prim_path=prim_path,
            prim_type=fallback["type"],
            position=[
                obstacle.position[0],
                obstacle.position[1],
                fallback["height"] / 2,
            ],
            scale=[
                fallback["radius"],
                fallback["radius"],
                fallback["height"] / 2,
            ],
            attributes={"primvars:displayColor": [(0.8, 0.2, 0.2)]},
        )

    def _clear(self):
        """Remove all dynamic obstacles."""
        if HAS_ISAAC:
            for obs in self.obstacles:
                prim_path = (
                    f"/World/DynamicObstacles/dynamic_{obs.obstacle_id}"
                )
                if prim_utils.is_prim_path_valid(prim_path):
                    prim_utils.delete_prim(prim_path)
        self.obstacles = []


class DynamicObstacle:
    """
    A single moving obstacle with waypoint-based navigation.

    Movement pattern:
    1. Pick a random waypoint within room bounds
    2. Move toward it at configured speed
    3. When reached, pick a new waypoint
    4. Randomly pause (simulates a person stopping to look at phone, etc.)
    """

    def __init__(self, obstacle_id, position, speed, obs_type,
                 room_bounds, movement_cfg, rng):
        self.obstacle_id = obstacle_id
        self.position = np.array(position, dtype=np.float64)
        self.speed = speed
        self.height = obs_type["size"][2] if "size" in obs_type else 1.7
        self.room_bounds = room_bounds
        self.move_cfg = movement_cfg
        self._rng = rng

        # Current movement state
        self.velocity = np.zeros(2, dtype=np.float64)
        self.target_waypoint = self._pick_waypoint()
        self.is_paused = False
        self.pause_timer = 0.0

    def update(self, dt):
        """
        Update obstacle position for one time step.

        Args:
            dt: Time step in seconds
        """
        # Handle pause state
        if self.is_paused:
            self.pause_timer -= dt
            if self.pause_timer <= 0:
                self.is_paused = False
                self.target_waypoint = self._pick_waypoint()
            self.velocity = np.zeros(2)
            return

        # Random chance to pause
        if self._rng.random() < self.move_cfg["pause_prob"]:
            self.is_paused = True
            self.pause_timer = self._rng.uniform(
                *self.move_cfg["pause_duration_range"]
            )
            self.velocity = np.zeros(2)
            return

        # Random chance to change direction
        if self._rng.random() < self.move_cfg["direction_change_prob"]:
            self.target_waypoint = self._pick_waypoint()

        # Move toward waypoint
        direction = self.target_waypoint - self.position
        distance = np.linalg.norm(direction)

        if distance < 0.3:
            # Reached waypoint, pick new one
            self.target_waypoint = self._pick_waypoint()
            direction = self.target_waypoint - self.position
            distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance
            self.velocity = direction * self.speed
            self.position += self.velocity * dt

        # Clamp to room bounds
        margin = 0.5
        self.position[0] = np.clip(
            self.position[0],
            self.room_bounds[0] + margin,
            self.room_bounds[1] - margin,
        )
        self.position[1] = np.clip(
            self.position[1],
            self.room_bounds[2] + margin,
            self.room_bounds[3] - margin,
        )

    def _pick_waypoint(self):
        """Pick a random waypoint within room bounds."""
        margin = 1.0
        radius = self.move_cfg["waypoint_radius"]

        # Waypoint near current position (within radius)
        angle = self._rng.uniform(0, 2 * np.pi)
        dist = self._rng.uniform(1.0, radius)

        wp = self.position + np.array([
            dist * np.cos(angle),
            dist * np.sin(angle),
        ])

        # Clamp to bounds
        wp[0] = np.clip(wp[0],
                         self.room_bounds[0] + margin,
                         self.room_bounds[1] - margin)
        wp[1] = np.clip(wp[1],
                         self.room_bounds[2] + margin,
                         self.room_bounds[3] - margin)

        return wp
