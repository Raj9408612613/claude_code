"""
Physics Backend Interface
=========================
Abstract interface that all physics backends (MJX, Warp, Isaac) must implement.

Any URDF-described robot can be trained through this interface without knowing
which simulator is running underneath.

Usage:
    backend = MJXBackend(urdf_path="models/unitree_g1/g1.urdf", n_envs=4096)
    state = backend.reset(rng)
    state, obs = backend.step(state, action)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


# ── Robot Description (parsed from URDF) ────────────────────────────────────

@dataclass
class JointInfo:
    """Single joint extracted from URDF."""
    name: str
    joint_type: str          # "revolute", "prismatic", "fixed", "continuous"
    parent_link: str
    child_link: str
    lower: float = -3.14
    upper: float = 3.14
    max_effort: float = 100.0
    max_velocity: float = 10.0
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class SensorInfo:
    """Sensor attached to a link."""
    name: str
    sensor_type: str         # "camera_depth", "camera_rgb", "imu", "force_torque", "contact"
    parent_link: str
    resolution: Optional[Tuple[int, int]] = None   # for cameras
    fov: Optional[float] = None                     # for cameras (radians)
    rate_hz: float = 30.0


@dataclass
class RobotDescription:
    """Complete robot description parsed from URDF."""
    name: str
    urdf_path: str
    actuated_joints: List[JointInfo] = field(default_factory=list)
    fixed_joints: List[JointInfo] = field(default_factory=list)
    sensors: List[SensorInfo] = field(default_factory=list)
    base_link: str = "base_link"
    has_floating_base: bool = True     # True for humanoids/mobile robots
    default_joint_positions: Optional[jnp.ndarray] = None
    total_mass: float = 0.0

    @property
    def n_actuated(self) -> int:
        return len(self.actuated_joints)

    @property
    def action_dim(self) -> int:
        return self.n_actuated

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.actuated_joints]

    @property
    def joint_lower(self) -> jnp.ndarray:
        return jnp.array([j.lower for j in self.actuated_joints], dtype=jnp.float32)

    @property
    def joint_upper(self) -> jnp.ndarray:
        return jnp.array([j.upper for j in self.actuated_joints], dtype=jnp.float32)


# ── Observation Space ────────────────────────────────────────────────────────

@dataclass
class ObsSpec:
    """Describes what observations the environment produces."""
    proprio_dim: int             # proprioception vector dimension
    depth_shape: Optional[Tuple[int, ...]] = None   # (n_cams, H, W) or None
    rgb_shape: Optional[Tuple[int, ...]] = None      # (n_cams, H, W, 3) or None
    force_torque_dim: int = 0    # force/torque sensor readings
    contact_dim: int = 0         # contact sensor readings
    custom: Dict[str, Tuple[int, ...]] = field(default_factory=dict)


# ── Abstract Backend ─────────────────────────────────────────────────────────

class PhysicsBackend(abc.ABC):
    """
    Abstract physics backend.

    All backends must implement reset(), step(), and get_obs().
    The interface is designed for batched (vectorized) simulation.
    """

    def __init__(
        self,
        robot: RobotDescription,
        n_envs: int = 4096,
        physics_dt: float = 0.005,       # 200 Hz physics
        control_dt: float = 0.02,        # 50 Hz control
        **kwargs,
    ):
        self.robot = robot
        self.n_envs = n_envs
        self.physics_dt = physics_dt
        self.control_dt = control_dt
        self.physics_substeps = max(1, int(control_dt / physics_dt))

    @abc.abstractmethod
    def reset(self, rng: jax.Array) -> Dict[str, Any]:
        """
        Reset all environments.

        Returns:
            state: dict of environment state (backend-specific internals)
        """
        ...

    @abc.abstractmethod
    def step(self, state: Dict[str, Any], action: jnp.ndarray) -> Tuple[Dict, Dict]:
        """
        Step all environments.

        Args:
            state:  dict from reset() or previous step()
            action: (n_envs, action_dim) normalized actions in [-1, 1]

        Returns:
            new_state: updated state dict
            step_info: dict with keys:
                "obs"        → dict of observations
                "robot_pos"  → (n_envs, 3) world position
                "robot_quat" → (n_envs, 4) orientation (w,x,y,z)
                "joint_pos"  → (n_envs, n_actuated) joint positions
                "joint_vel"  → (n_envs, n_actuated) joint velocities
                "contacts"   → (n_envs,) bool or contact info
        """
        ...

    @abc.abstractmethod
    def get_obs(self, state: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """
        Extract observations from state.

        Returns dict with keys like "proprio", "depth", "rgb", etc.
        """
        ...

    @abc.abstractmethod
    def get_obs_spec(self) -> ObsSpec:
        """Return the observation space specification."""
        ...

    @abc.abstractmethod
    def auto_reset(
        self,
        state: Dict[str, Any],
        terminated: jnp.ndarray,
        rng: jax.Array,
    ) -> Dict[str, Any]:
        """
        Reset only terminated environments.

        Args:
            state: current state
            terminated: (n_envs,) bool mask
            rng: JAX random key

        Returns:
            new_state with terminated envs reset
        """
        ...

    def denormalize_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """Convert normalized [-1, 1] action to joint position targets."""
        action = jnp.clip(action, -1.0, 1.0)
        mid = (self.robot.joint_upper + self.robot.joint_lower) / 2.0
        half_range = (self.robot.joint_upper - self.robot.joint_lower) / 2.0
        return mid + action * half_range
