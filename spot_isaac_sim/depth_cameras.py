"""
Depth Camera Rig for Spot Robot in Isaac Sim
=============================================
Sets up 5 depth cameras on the Spot body and handles:
- Camera creation and attachment to robot prim
- Depth image capture each step
- Realistic noise injection (Gaussian, dropout, edge noise, quantization)

Camera placement (top-down view):

              70°          0°         -70°
        Front-Left    Front-Center   Front-Right
                \\          |          //
                 \\         |         //
            +=====+=========+=========+=====+
            |              SPOT              |
            +=====+===================+=====+
                 //                     \\
                //                       \\
           145°                          -145°
        Rear-Left                     Rear-Right

Coverage: ~330° with overlap in front for safe navigation.
"""

import numpy as np

try:
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.sensor import Camera as IsaacCamera
    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False

from . import config


class DepthCameraRig:
    """
    Manages 5 depth cameras attached to the Spot robot body.

    Usage:
        rig = DepthCameraRig(robot_prim_path="/World/Spot")
        rig.initialize()

        # Each simulation step:
        depth_stack = rig.capture()  # shape: (5, 120, 160)
    """

    def __init__(self, robot_prim_path, camera_config=None, noise_config=None):
        """
        Args:
            robot_prim_path: USD prim path to the robot base link
                             e.g., "/World/Spot/base_link"
            camera_config: Override default CAMERA_RIG config
            noise_config: Override default DEPTH_NOISE config
        """
        self.robot_prim_path = robot_prim_path
        self.cam_cfg = camera_config or config.CAMERA_RIG
        self.noise_cfg = noise_config or config.DEPTH_NOISE

        self.cameras = []
        self.resolution = self.cam_cfg["resolution"]   # [W, H]
        self.n_cameras = len(self.cam_cfg["cameras"])

        # Noise multiplier (randomized per episode by domain randomization)
        self.noise_multiplier = 1.0
        self.dropout_multiplier = 1.0

        self._rng = np.random.default_rng()

    def initialize(self):
        """Create and attach all 5 cameras to the robot in the Isaac Sim stage."""
        if not HAS_ISAAC:
            print("[DepthCameraRig] Isaac Sim not available. "
                  "Using synthetic depth generation for testing.")
            return

        self.cameras = []
        for cam_def in self.cam_cfg["cameras"]:
            cam_prim_path = (
                f"{self.robot_prim_path}/{cam_def['name']}_depth_camera"
            )

            camera = IsaacCamera(
                prim_path=cam_prim_path,
                resolution=(self.resolution[0], self.resolution[1]),
                frequency=self.cam_cfg["update_rate"],
            )

            # Set camera position relative to robot body
            camera.set_local_pose(
                translation=np.array(cam_def["position"]),
                orientation=self._euler_to_quat(cam_def["rotation_deg"]),
            )

            # Configure depth output
            camera.set_clipping_range(
                near_distance=self.cam_cfg["min_range"],
                far_distance=self.cam_cfg["max_range"],
            )

            camera.initialize()
            self.cameras.append(camera)

        print(f"[DepthCameraRig] Initialized {len(self.cameras)} depth cameras")
        for cam_def in self.cam_cfg["cameras"]:
            print(f"  - {cam_def['name']}: {cam_def['description']}")

    def capture(self):
        """
        Capture depth images from all 5 cameras and apply noise.

        Returns:
            np.ndarray: Shape (5, H, W) — stacked depth images in meters.
                        Values range from min_range to max_range.
                        Dropped pixels have value 0.0.
        """
        H, W = self.resolution[1], self.resolution[0]
        depth_stack = np.zeros((self.n_cameras, H, W), dtype=np.float32)

        if HAS_ISAAC and self.cameras:
            for i, camera in enumerate(self.cameras):
                raw_depth = camera.get_depth()
                if raw_depth is not None:
                    depth_stack[i] = raw_depth
        else:
            # Synthetic depth for testing without Isaac Sim
            depth_stack = self._generate_synthetic_depth()

        # Apply noise model
        if self.noise_cfg["enabled"]:
            depth_stack = self._apply_noise(depth_stack)

        # Clip to valid range
        valid_mask = depth_stack > 0
        depth_stack = np.clip(
            depth_stack,
            self.cam_cfg["min_range"],
            self.cam_cfg["max_range"],
        )
        # Restore dropped pixels (keep them as 0)
        depth_stack[~valid_mask] = 0.0

        return depth_stack

    def _apply_noise(self, depth_stack):
        """
        Apply realistic depth camera noise model.

        Noise components:
        1. Gaussian noise (distance-dependent)
        2. Pixel dropout (random missing readings)
        3. Edge noise (noisy at depth discontinuities)
        4. Quantization (finite sensor resolution)
        5. Temporal flicker (random per-frame artifacts)
        """
        noisy = depth_stack.copy()
        n_cams, H, W = noisy.shape

        # 1. Distance-dependent Gaussian noise
        base_std = self.noise_cfg["gaussian_std_base"] * self.noise_multiplier
        scale_std = self.noise_cfg["gaussian_std_scale"] * self.noise_multiplier
        noise_std = base_std + scale_std * np.abs(noisy)
        gaussian_noise = self._rng.normal(0, 1, noisy.shape) * noise_std
        noisy += gaussian_noise.astype(np.float32)

        # 2. Pixel dropout
        dropout_rate = self.noise_cfg["dropout_rate"] * self.dropout_multiplier
        dropout_mask = self._rng.random(noisy.shape) < dropout_rate
        noisy[dropout_mask] = 0.0

        # 3. Edge noise (apply extra noise at depth discontinuities)
        if self.noise_cfg["edge_noise_enabled"]:
            for c in range(n_cams):
                edges = self._detect_depth_edges(noisy[c])
                edge_noise = (
                    self._rng.normal(0, self.noise_cfg["edge_noise_std"],
                                     (H, W)).astype(np.float32)
                )
                noisy[c] += edges * edge_noise

        # 4. Quantization
        q_step = self.noise_cfg["quantization_step"]
        if q_step > 0:
            noisy = np.round(noisy / q_step) * q_step

        # 5. Temporal flicker
        flicker_rate = self.noise_cfg["temporal_flicker_rate"]
        flicker_mask = self._rng.random(noisy.shape) < flicker_rate
        flicker_values = self._rng.uniform(
            self.cam_cfg["min_range"],
            self.cam_cfg["max_range"],
            noisy.shape,
        ).astype(np.float32)
        noisy[flicker_mask] = flicker_values[flicker_mask]

        return noisy

    def _detect_depth_edges(self, depth_image):
        """
        Detect depth discontinuities using simple gradient magnitude.

        Args:
            depth_image: (H, W) depth array

        Returns:
            (H, W) binary edge mask (float 0.0 or 1.0)
        """
        # Sobel-like gradient
        grad_x = np.abs(np.diff(depth_image, axis=1, prepend=depth_image[:, :1]))
        grad_y = np.abs(np.diff(depth_image, axis=0, prepend=depth_image[:1, :]))
        gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Threshold: edges where depth changes by more than 0.3m
        edge_threshold = 0.3
        edges = (gradient_mag > edge_threshold).astype(np.float32)
        return edges

    def _generate_synthetic_depth(self):
        """
        Generate fake depth images for testing without Isaac Sim.
        Creates random obstacles at various depths.
        """
        H, W = self.resolution[1], self.resolution[0]
        depth_stack = np.full(
            (self.n_cameras, H, W),
            self.cam_cfg["max_range"],
            dtype=np.float32,
        )

        for c in range(self.n_cameras):
            # Random number of "obstacles" in each camera view
            n_obs = self._rng.integers(0, 6)
            for _ in range(n_obs):
                # Random rectangle representing an obstacle
                cx = self._rng.integers(0, W)
                cy = self._rng.integers(0, H)
                rw = self._rng.integers(10, W // 3)
                rh = self._rng.integers(10, H // 3)
                depth = self._rng.uniform(0.5, 8.0)

                x1 = max(0, cx - rw // 2)
                x2 = min(W, cx + rw // 2)
                y1 = max(0, cy - rh // 2)
                y2 = min(H, cy + rh // 2)

                # Only overwrite if this obstacle is closer
                depth_stack[c, y1:y2, x1:x2] = np.minimum(
                    depth_stack[c, y1:y2, x1:x2], depth
                )

        return depth_stack

    def set_noise_multipliers(self, noise_mult, dropout_mult):
        """
        Set per-episode noise multipliers for domain randomization.

        Args:
            noise_mult: Multiplier for Gaussian noise (0.5 = half, 2.0 = double)
            dropout_mult: Multiplier for dropout rate
        """
        self.noise_multiplier = noise_mult
        self.dropout_multiplier = dropout_mult

    def reset(self):
        """Reset camera state for new episode."""
        pass  # Cameras persist across episodes, no reset needed

    @staticmethod
    def _euler_to_quat(euler_deg):
        """
        Convert Euler angles (degrees) to quaternion [w, x, y, z].

        Args:
            euler_deg: [roll, pitch, yaw] in degrees

        Returns:
            np.ndarray: [w, x, y, z] quaternion
        """
        roll = np.radians(euler_deg[0])
        pitch = np.radians(euler_deg[1])
        yaw = np.radians(euler_deg[2])

        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z], dtype=np.float64)

    def get_camera_info(self):
        """Return summary of camera configuration for logging."""
        info = []
        for cam_def in self.cam_cfg["cameras"]:
            info.append({
                "name": cam_def["name"],
                "position": cam_def["position"],
                "yaw_deg": cam_def["rotation_deg"][2],
                "resolution": self.resolution,
            })
        return info
