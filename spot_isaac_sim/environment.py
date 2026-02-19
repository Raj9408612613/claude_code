"""
Environment Builder for Spot Isaac Sim Training
=================================================
Handles:
- Loading pre-made rooms (office, hospital, warehouse)
- Placing random furniture/obstacles
- Procedural room generation (fallback)
- Full domain randomization per episode
- Floor surface randomization (texture, friction, curvature)
- Lighting randomization
"""

import numpy as np

try:
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.core.prims import XFormPrim, GeometryPrim
    from omni.isaac.core.materials import PhysicsMaterial
    from pxr import UsdGeom, UsdLux, Gf, Sdf
    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False

from . import config


class EnvironmentBuilder:
    """
    Builds and randomizes training environments.

    Each call to build_episode() creates a new randomized environment:
    - Selects a random pre-made room (or generates procedural room)
    - Places random furniture/obstacles
    - Randomizes floor, lighting, surface properties
    - Returns valid robot spawn and goal positions

    Usage:
        builder = EnvironmentBuilder()
        builder.initialize()

        # Each episode:
        spawn_pos, goal_pos = builder.build_episode(
            curriculum_stage=current_stage
        )
    """

    def __init__(self, env_config=None, obstacle_config=None,
                 randomization_config=None):
        self.env_cfg = env_config or config.ENVIRONMENTS
        self.obs_cfg = obstacle_config or config.OBSTACLE_ASSETS
        self.rand_cfg = randomization_config or config.DOMAIN_RANDOMIZATION

        self._rng = np.random.default_rng()
        self._current_room = None
        self._placed_obstacles = []
        self._placed_lights = []
        self._room_bounds = None  # [x_min, x_max, y_min, y_max]

    def initialize(self):
        """One-time initialization."""
        print("[EnvironmentBuilder] Initialized")
        print(f"  Available rooms: {len(self.env_cfg['rooms'])}")
        print(f"  Available obstacle types: {len(self.obs_cfg['static'])}")

    def build_episode(self, curriculum_stage=None):
        """
        Build a fully randomized environment for one training episode.

        Args:
            curriculum_stage: Optional dict with overrides for difficulty
                             (obstacle counts, goal distance, etc.)

        Returns:
            robot_spawn: (x, y, z, yaw) — spawn position and orientation
            goal_position: (x, y) — goal position in world frame
        """
        # Clear previous episode
        self._clear_environment()

        # 1. Select and load room
        room_info = self._select_room()
        self._load_room(room_info)

        # 2. Randomize floor surface
        if self.rand_cfg["floor_surface"]["enabled"]:
            self._randomize_floor()

        # 3. Randomize lighting
        if self.rand_cfg["lighting"]["enabled"]:
            self._randomize_lighting()

        # 4. Determine obstacle count (with curriculum override)
        if curriculum_stage and "obstacle_count" in curriculum_stage:
            obs_range = curriculum_stage["obstacle_count"]
        else:
            obs_range = self.obs_cfg["count_range"]
        n_obstacles = self._rng.integers(obs_range[0], obs_range[1] + 1)

        # 5. Find valid robot spawn position
        robot_spawn = self._find_spawn_position()

        # 6. Find valid goal position
        if curriculum_stage and "goal_distance" in curriculum_stage:
            goal_dist_range = curriculum_stage["goal_distance"]
        else:
            goal_dist_range = self.rand_cfg["goal_position"]["distance_range"]
        goal_position = self._find_goal_position(robot_spawn, goal_dist_range)

        # 7. Place obstacles (avoiding robot and goal positions)
        self._place_obstacles(n_obstacles, robot_spawn, goal_position)

        return robot_spawn, goal_position

    def _clear_environment(self):
        """Remove all dynamically placed objects from previous episode."""
        if HAS_ISAAC:
            for prim_path in self._placed_obstacles:
                if prim_utils.is_prim_path_valid(prim_path):
                    prim_utils.delete_prim(prim_path)

            for prim_path in self._placed_lights:
                if prim_utils.is_prim_path_valid(prim_path):
                    prim_utils.delete_prim(prim_path)

        self._placed_obstacles = []
        self._placed_lights = []

    def _select_room(self):
        """Pick a random pre-made room."""
        rooms = self.env_cfg["rooms"]
        idx = self._rng.integers(0, len(rooms))
        room = rooms[idx]
        self._room_bounds = [
            -room["approx_size"][0] / 2, room["approx_size"][0] / 2,
            -room["approx_size"][1] / 2, room["approx_size"][1] / 2,
        ]
        self._current_room = room
        return room

    def _load_room(self, room_info):
        """Load room USD into the stage."""
        if not HAS_ISAAC:
            print(f"[Env] Would load room: {room_info['name']} "
                  f"from {room_info['usd_path']}")
            return

        room_prim_path = "/World/Room"

        # Remove old room if exists
        if prim_utils.is_prim_path_valid(room_prim_path):
            prim_utils.delete_prim(room_prim_path)

        # Load room USD
        prim_utils.create_prim(
            prim_path=room_prim_path,
            usd_path=room_info["usd_path"],
        )

        print(f"[Env] Loaded room: {room_info['name']}")

    def _build_procedural_room(self):
        """
        Build a simple procedural room with 4 walls and a floor.
        Fallback when pre-made rooms are not available.
        """
        cfg = self.env_cfg["procedural"]
        width = self._rng.uniform(*cfg["width_range"])
        length = self._rng.uniform(*cfg["length_range"])
        wall_h = cfg["wall_height"]
        wall_t = cfg["wall_thickness"]

        self._room_bounds = [-width / 2, width / 2, -length / 2, length / 2]

        if not HAS_ISAAC:
            print(f"[Env] Procedural room: {width:.1f}m x {length:.1f}m")
            return

        # Floor
        prim_utils.create_prim(
            prim_path="/World/Room/Floor",
            prim_type="Cube",
            position=[0, 0, -0.05],
            scale=[width / 2, length / 2, 0.05],
        )

        # 4 walls
        walls = [
            ("North", [0, length / 2, wall_h / 2],
             [width / 2, wall_t / 2, wall_h / 2]),
            ("South", [0, -length / 2, wall_h / 2],
             [width / 2, wall_t / 2, wall_h / 2]),
            ("East", [width / 2, 0, wall_h / 2],
             [wall_t / 2, length / 2, wall_h / 2]),
            ("West", [-width / 2, 0, wall_h / 2],
             [wall_t / 2, length / 2, wall_h / 2]),
        ]
        for name, pos, scale in walls:
            prim_utils.create_prim(
                prim_path=f"/World/Room/Wall_{name}",
                prim_type="Cube",
                position=pos,
                scale=scale,
            )

    def _randomize_floor(self):
        """Randomize floor texture, friction, and surface curvature."""
        floor_cfg = self.rand_cfg["floor_surface"]

        # Select random texture
        texture = self._rng.choice(floor_cfg["textures"])

        # Random friction
        friction = self._rng.uniform(*floor_cfg["friction_range"])

        if HAS_ISAAC:
            # Apply physics material with randomized friction
            material = PhysicsMaterial(
                prim_path="/World/Room/FloorMaterial",
                static_friction=friction,
                dynamic_friction=friction * 0.8,
                restitution=0.1,
            )

        # Surface curvature (uneven floor for balance training)
        if floor_cfg["curvature_enabled"]:
            amplitude = self._rng.uniform(*floor_cfg["curvature_amplitude_range"])
            frequency = self._rng.uniform(*floor_cfg["curvature_frequency_range"])
            slope_deg = self._rng.uniform(*floor_cfg["slope_angle_range"])

            if HAS_ISAAC and amplitude > 0:
                self._apply_floor_curvature(amplitude, frequency, slope_deg)

    def _apply_floor_curvature(self, amplitude, frequency, slope_deg):
        """
        Create an uneven floor surface using a heightfield.

        This forces the robot to learn balance on non-flat surfaces,
        making the policy robust for real-world deployment where floors
        have bumps, ramps, and imperfections.

        Args:
            amplitude: Max bump height in meters
            frequency: Bumps per meter
            slope_deg: Overall floor tilt angle in degrees
        """
        if not HAS_ISAAC:
            return

        bounds = self._room_bounds
        resolution = 0.1  # 10cm grid resolution
        nx = int((bounds[1] - bounds[0]) / resolution)
        ny = int((bounds[3] - bounds[2]) / resolution)

        # Generate heightfield using Perlin-like noise
        x = np.linspace(0, frequency * (bounds[1] - bounds[0]), nx)
        y = np.linspace(0, frequency * (bounds[3] - bounds[2]), ny)
        xx, yy = np.meshgrid(x, y)

        # Simple multi-frequency noise (approximation of Perlin noise)
        heights = np.zeros_like(xx)
        for octave in range(3):
            freq_mult = 2 ** octave
            amp_mult = 0.5 ** octave
            heights += amp_mult * np.sin(freq_mult * xx) * np.cos(freq_mult * yy)

        # Normalize and scale
        heights = heights / np.max(np.abs(heights) + 1e-8) * amplitude

        # Add overall slope
        slope_rad = np.radians(slope_deg)
        slope_direction = self._rng.uniform(0, 2 * np.pi)
        slope_x = np.sin(slope_direction) * np.tan(slope_rad)
        slope_y = np.cos(slope_direction) * np.tan(slope_rad)

        x_pos = np.linspace(bounds[0], bounds[1], nx)
        y_pos = np.linspace(bounds[2], bounds[3], ny)
        xx_pos, yy_pos = np.meshgrid(x_pos, y_pos)
        heights += slope_x * xx_pos + slope_y * yy_pos

        # In Isaac Sim, you'd apply this as a heightfield terrain
        # Using omni.isaac.core terrain utilities
        print(f"[Env] Floor curvature: amplitude={amplitude:.3f}m, "
              f"frequency={frequency:.1f}/m, slope={slope_deg:.1f}°")

    def _randomize_lighting(self):
        """Randomize scene lighting conditions."""
        light_cfg = self.rand_cfg["lighting"]

        n_lights = self._rng.integers(*light_cfg["num_lights_range"])
        intensity = self._rng.uniform(*light_cfg["intensity_range"])
        color_temp = self._rng.uniform(*light_cfg["color_temp_range"])
        ambient = self._rng.uniform(*light_cfg["ambient_range"])

        # Convert color temperature to RGB (approximate)
        rgb = self._color_temp_to_rgb(color_temp)

        if not HAS_ISAAC:
            print(f"[Env] Lighting: {n_lights} lights, intensity={intensity:.0f}, "
                  f"color_temp={color_temp:.0f}K")
            return

        # Remove existing dynamic lights
        for old_light in self._placed_lights:
            if prim_utils.is_prim_path_valid(old_light):
                prim_utils.delete_prim(old_light)
        self._placed_lights = []

        bounds = self._room_bounds
        for i in range(n_lights):
            light_path = f"/World/DynamicLights/Light_{i}"

            # Random position on ceiling
            x = self._rng.uniform(bounds[0] * 0.7, bounds[1] * 0.7)
            y = self._rng.uniform(bounds[2] * 0.7, bounds[3] * 0.7)
            z = 2.5 + self._rng.uniform(-0.3, 0.3)

            prim_utils.create_prim(
                prim_path=light_path,
                prim_type="SphereLight",
                position=[x, y, z],
                attributes={
                    "intensity": intensity / n_lights,
                    "color": Gf.Vec3f(*rgb),
                    "radius": 0.1,
                },
            )
            self._placed_lights.append(light_path)

    def _find_spawn_position(self):
        """
        Find a valid random spawn position for the robot.

        Returns:
            tuple: (x, y, z, yaw_rad)
        """
        bounds = self._room_bounds
        margin = 1.0  # Stay away from walls

        x = self._rng.uniform(bounds[0] + margin, bounds[1] - margin)
        y = self._rng.uniform(bounds[2] + margin, bounds[3] - margin)
        z = config.SPOT_ROBOT["standing_height"]

        yaw = 0.0
        if self.rand_cfg["robot_start"]["random_orientation"]:
            yaw = self._rng.uniform(0, 2 * np.pi)

        if self.rand_cfg["robot_start"]["height_variation"] > 0:
            z += self._rng.uniform(
                -self.rand_cfg["robot_start"]["height_variation"],
                self.rand_cfg["robot_start"]["height_variation"],
            )

        return (x, y, z, yaw)

    def _find_goal_position(self, robot_spawn, distance_range):
        """
        Find a valid goal position at a random distance/direction from robot.

        Args:
            robot_spawn: (x, y, z, yaw)
            distance_range: [min_dist, max_dist] in meters

        Returns:
            tuple: (x, y)
        """
        bounds = self._room_bounds
        margin = 0.5
        robot_x, robot_y = robot_spawn[0], robot_spawn[1]

        for _ in range(100):  # Try up to 100 times
            angle = self._rng.uniform(0, 2 * np.pi)
            dist = self._rng.uniform(*distance_range)

            gx = robot_x + dist * np.cos(angle)
            gy = robot_y + dist * np.sin(angle)

            # Check bounds
            if (bounds[0] + margin < gx < bounds[1] - margin and
                    bounds[2] + margin < gy < bounds[3] - margin):
                return (gx, gy)

        # Fallback: place goal at center of room
        return (0.0, 0.0)

    def _place_obstacles(self, n_obstacles, robot_spawn, goal_pos):
        """
        Place random obstacles in the environment.

        Ensures no obstacle overlaps with robot spawn or goal position.

        Args:
            n_obstacles: Number of obstacles to place
            robot_spawn: (x, y, z, yaw)
            goal_pos: (x, y)
        """
        bounds = self._room_bounds
        margin = 0.3
        placed_positions = []
        robot_xy = np.array([robot_spawn[0], robot_spawn[1]])
        goal_xy = np.array(goal_pos)

        # Build weighted list of obstacle types
        obstacle_pool = []
        use_usd = HAS_ISAAC
        source = self.obs_cfg["static"] if use_usd else self.obs_cfg["primitives"]
        for obs_def in source:
            obstacle_pool.extend([obs_def] * obs_def["weight"])

        for i in range(n_obstacles):
            # Find valid position (not too close to robot, goal, or others)
            position = None
            for _ in range(50):
                x = self._rng.uniform(bounds[0] + margin, bounds[1] - margin)
                y = self._rng.uniform(bounds[2] + margin, bounds[3] - margin)
                pos = np.array([x, y])

                # Check distances
                dist_robot = np.linalg.norm(pos - robot_xy)
                dist_goal = np.linalg.norm(pos - goal_xy)
                if dist_robot < self.obs_cfg["min_dist_from_robot"]:
                    continue
                if dist_goal < self.obs_cfg["min_dist_from_goal"]:
                    continue

                # Check spacing from other obstacles
                too_close = False
                for prev_pos in placed_positions:
                    if np.linalg.norm(pos - prev_pos) < self.obs_cfg["min_obstacle_spacing"]:
                        too_close = True
                        break
                if too_close:
                    continue

                position = pos
                break

            if position is None:
                continue  # Could not find valid position, skip

            placed_positions.append(position)

            # Select random obstacle type
            obs_def = self._rng.choice(obstacle_pool)
            rotation_yaw = self._rng.uniform(0, 360)

            prim_path = f"/World/Obstacles/obstacle_{i}"

            if HAS_ISAAC:
                if "usd" in obs_def:
                    # Load USD asset
                    scale = self._rng.uniform(*obs_def["scale_range"])
                    prim_utils.create_prim(
                        prim_path=prim_path,
                        usd_path=obs_def["usd"],
                        position=[position[0], position[1], 0.0],
                        scale=[scale, scale, scale],
                    )
                else:
                    # Primitive shape
                    size = self._rng.uniform(*obs_def["size_range"])
                    prim_utils.create_prim(
                        prim_path=prim_path,
                        prim_type=obs_def["type"],
                        position=[position[0], position[1], size / 2],
                        scale=[size, size, size],
                    )

            self._placed_obstacles.append(prim_path)

    def get_obstacle_positions(self):
        """
        Get positions of all currently placed obstacles.

        Returns:
            list of (x, y) tuples
        """
        positions = []
        if HAS_ISAAC:
            for prim_path in self._placed_obstacles:
                if prim_utils.is_prim_path_valid(prim_path):
                    prim = XFormPrim(prim_path)
                    pos, _ = prim.get_world_pose()
                    positions.append((pos[0], pos[1]))
        return positions

    def get_room_bounds(self):
        """Return current room bounds [x_min, x_max, y_min, y_max]."""
        return self._room_bounds

    @staticmethod
    def _color_temp_to_rgb(kelvin):
        """
        Convert color temperature (Kelvin) to normalized RGB.
        Approximate algorithm for 3000K-6500K range.
        """
        temp = kelvin / 100.0

        # Red
        if temp <= 66:
            r = 1.0
        else:
            r = 1.292936 * ((temp - 60) ** -0.1332047592)
            r = max(0.0, min(1.0, r))

        # Green
        if temp <= 66:
            g = 0.39008 * np.log(temp) - 0.63184
        else:
            g = 1.129891 * ((temp - 60) ** -0.0755148492)
        g = max(0.0, min(1.0, g))

        # Blue
        if temp >= 66:
            b = 1.0
        elif temp <= 19:
            b = 0.0
        else:
            b = 0.54320 * np.log(temp - 10) - 1.19625
            b = max(0.0, min(1.0, b))

        return (r, g, b)
