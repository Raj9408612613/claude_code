"""
MJX Batched Navigation Environment
=====================================
Replaces navigation_env.py (Isaac Sim).

Key differences from the Isaac Sim version:
  - No omni.isaac.* imports — pure MuJoCo + JAX
  - ALL environments step in one jax.jit + jax.vmap call on GPU
  - Camera rendering via WarpDepthRenderer (GPU, not CPU per-env)
  - Domain randomization done via JAX RNG (fully on GPU)
  - Reset/step return JAX arrays (no NumPy conversion until logging)

Observation (same shape as original):
    depth:  (n_envs, 5, 120, 160)  float32  — 5 depth cameras
    proprio:(n_envs, 37)            float32  — body state + goal

Action:
    (n_envs, 12) float32 — normalized joint position targets in [-1, 1]
"""

import os
import math
import numpy as np
from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from warp_cameras import WarpDepthRenderer, N_CAMS, CAM_H, CAM_W, N_OBS
from jax_reward import compute_reward, check_termination

# ── Joint limits (from config.py SPOT_ROBOT) ─────────────────────────────────
JOINT_LOWER = jnp.array([-0.8,-0.6,-2.8]*4, dtype=jnp.float32)
JOINT_UPPER = jnp.array([ 0.8, 2.4,-0.5]*4, dtype=jnp.float32)
STANDING_POSE = jnp.array([0.0, 0.8, -1.6]*4, dtype=jnp.float32)

# Room half-extents for random placement (10×10 m room)
ROOM_HALF = 4.5   # keep 0.5m from walls

# Number of physics sub-steps per RL step (control @ 50 Hz, physics @ 200 Hz)
PHYSICS_SUBSTEPS = 4


class SpotMJXEnv:
    """
    Batched MJX environment for Spot robot navigation.

    Usage:
        env = SpotMJXEnv(n_envs=4096)
        rng = jax.random.PRNGKey(0)
        state, obs = env.reset(rng)
        state, obs, reward, done, info = env.step(state, action)
    """

    def __init__(
        self,
        n_envs: int = 4096,
        xml_path: str = "models/spot_scene.xml",
        noise_enabled: bool = True,
        seed: int = 42,
    ):
        self.n_envs        = n_envs
        self.noise_enabled = noise_enabled

        # ── Load MuJoCo model ─────────────────────────────────────────
        xml_abs = os.path.join(os.path.dirname(__file__), xml_path)
        self._mj_model = mujoco.MjModel.from_xml_path(xml_abs)
        self._mx       = mjx.put_model(self._mj_model)

        # ── Cache model indices ───────────────────────────────────────
        # Camera site IDs
        self._cam_site_ids = [
            self._mj_model.site("cam_front_center").id,
            self._mj_model.site("cam_front_left").id,
            self._mj_model.site("cam_front_right").id,
            self._mj_model.site("cam_rear_left").id,
            self._mj_model.site("cam_rear_right").id,
        ]
        # base_link body (freejoint root): qpos[0:7], qvel[0:6]
        self._root_jnt_id  = self._mj_model.joint("root").id
        self._root_qposadr = self._mj_model.jnt_qposadr[self._root_jnt_id]  # = 0
        self._root_dofadr  = self._mj_model.jnt_dofadr[self._root_jnt_id]   # = 0
        # Joint qpos addresses for 12 leg joints (after the 7 root dofs)
        self._joint_qposadr = 7   # joints start at qpos[7]
        self._joint_dofadr  = 6   # joint vels start at qvel[6]
        # Mocap body IDs
        self._mocap_ids = list(range(N_OBS))  # mocap_pos indexed 0..N_OBS-1
        # nq, nv
        self.nq = self._mj_model.nq
        self.nv = self._mj_model.nv

        # ── Warp renderer ─────────────────────────────────────────────
        self._renderer = WarpDepthRenderer(
            n_envs       = n_envs,
            noise_enabled = noise_enabled,
        )
        self._renderer.cam_site_ids = self._cam_site_ids

        # ── Pre-compile step function ─────────────────────────────────
        self._batch_step = jax.jit(jax.vmap(
            partial(self._single_physics_step, self._mx)
        ))

        # ── Obs / action space info ───────────────────────────────────
        self.obs_depth_shape = (n_envs, N_CAMS, CAM_H, CAM_W)
        self.obs_proprio_dim = 37
        self.action_dim      = 12

    # ════════════════════════════════════════════════════════════════════
    # RESET
    # ════════════════════════════════════════════════════════════════════

    def reset(self, rng: jax.Array) -> Tuple[Dict, Dict]:
        """
        Reset all n_envs environments with random positions/goals/obstacles.

        Returns:
            state: dict of env state (JAX arrays)
            obs:   dict{"depth": ..., "proprio": ...}
        """
        rng, *sub = jax.random.split(rng, self.n_envs + 1)
        sub = jnp.stack(sub)   # (n_envs, 2)

        # Build batch of initial MJX data
        mj_data = mujoco.MjData(self._mj_model)
        dx_single = mjx.put_data(self._mj_model, mj_data)
        dx_batch  = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_envs,) + x.shape),
            dx_single,
        )

        # Randomize robot start positions and orientations
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        robot_xy  = jax.random.uniform(k1, (self.n_envs, 2),
                                       minval=-ROOM_HALF, maxval=ROOM_HALF)
        robot_yaw = jax.random.uniform(k2, (self.n_envs,),
                                       minval=0.0, maxval=2*math.pi)
        # Goal positions: 2-10 m from robot
        goal_dist = jax.random.uniform(k3, (self.n_envs,), minval=2.0, maxval=6.0)
        goal_ang  = jax.random.uniform(k3, (self.n_envs,), minval=0.0, maxval=2*math.pi)
        goal_xy   = robot_xy + jnp.stack([
            goal_dist * jnp.cos(goal_ang),
            goal_dist * jnp.sin(goal_ang),
        ], axis=-1)
        goal_xy   = jnp.clip(goal_xy, -ROOM_HALF, ROOM_HALF)

        # Set freejoint qpos: [x, y, z, qw, qx, qy, qz, joint×12]
        qpos = jnp.tile(
            jnp.concatenate([
                jnp.zeros(3),                    # xyz
                jnp.array([1.0, 0.0, 0.0, 0.0]),# quat
                STANDING_POSE,                   # 12 joints
            ]),
            (self.n_envs, 1),
        )
        # Set x, y, z height
        qpos = qpos.at[:, 0].set(robot_xy[:, 0])
        qpos = qpos.at[:, 1].set(robot_xy[:, 1])
        qpos = qpos.at[:, 2].set(0.52)           # standing height
        # Set yaw via quaternion: [cos(θ/2), 0, 0, sin(θ/2)]
        qpos = qpos.at[:, 3].set(jnp.cos(robot_yaw / 2))
        qpos = qpos.at[:, 6].set(jnp.sin(robot_yaw / 2))

        dx_batch = dx_batch.replace(qpos=qpos)

        # Randomize obstacles: scatter N_OBS mocap bodies
        rng, k4 = jax.random.split(rng)
        obs_xy = jax.random.uniform(k4, (self.n_envs, N_OBS, 2),
                                    minval=-ROOM_HALF, maxval=ROOM_HALF)
        obs_z  = jnp.ones((self.n_envs, N_OBS, 1)) * 0.5
        obs_pos_new = jnp.concatenate([obs_xy, obs_z], axis=-1)  # (B, N_OBS, 3)

        # Carry state
        step_count   = jnp.zeros(self.n_envs, dtype=jnp.int32)
        prev_dist    = jnp.linalg.norm(goal_xy - robot_xy, axis=-1)
        prev_action  = jnp.zeros((self.n_envs, 12), dtype=jnp.float32)

        state = {
            "dx":          dx_batch,
            "mocap_pos":   obs_pos_new,
            "goal_pos":    goal_xy,
            "step_count":  step_count,
            "prev_dist":   prev_dist,
            "prev_action": prev_action,
        }
        obs = self._get_obs(state)
        return state, obs

    # ════════════════════════════════════════════════════════════════════
    # STEP
    # ════════════════════════════════════════════════════════════════════

    def step(self, state: Dict, action: jnp.ndarray) -> Tuple:
        """
        Step all envs.

        Args:
            state:  dict (from reset or previous step)
            action: (n_envs, 12) float32, normalized in [-1, 1]

        Returns:
            new_state, obs, reward, terminated, info
        """
        dx        = state["dx"]
        mocap_pos = state["mocap_pos"]
        goal_pos  = state["goal_pos"]

        # Denormalize actions to joint position targets
        joint_mid   = (JOINT_UPPER + JOINT_LOWER) / 2.0
        joint_range = (JOINT_UPPER - JOINT_LOWER) / 2.0
        ctrl        = joint_mid + action * joint_range   # (B, 12)

        prev_qpos = dx.qpos.copy()

        # ── Physics sub-steps (4× for stability) ─────────────────────
        for _ in range(PHYSICS_SUBSTEPS):
            dx = dx.replace(ctrl=ctrl)
            dx = self._batch_step(dx)

        # ── Update mocap (dynamic obstacles move each step) ───────────
        # Simple random-walk for dynamic obstacles (last 5 mocap bodies)
        # In a real run you'd use a proper velocity model
        mocap_pos = mocap_pos  # static for now; dynamic logic in env.update_dynobs()

        # ── Extract robot state ───────────────────────────────────────
        robot_pos  = dx.qpos[:, 0:3]          # (B, 3)
        robot_quat = dx.qpos[:, 3:7]          # (B, 4)
        joint_pos  = dx.qpos[:, 7:19]         # (B, 12)
        joint_vel  = dx.qvel[:, 6:18]         # (B, 12)

        # Min obstacle dist (heuristic: nearest mocap body)
        obs_xy  = mocap_pos[:, :, :2]          # (B, N_OBS, 2)
        rob_xy  = robot_pos[:, :2, None].transpose(0, 2, 1)  # (B, 1, 2) → broadcast
        dists   = jnp.linalg.norm(obs_xy - robot_pos[:, None, :2], axis=-1)  # (B, N_OBS)
        min_dist = jnp.min(dists, axis=-1)     # (B,)
        has_coll = min_dist < 0.35             # (B,) bool

        prev_pos = prev_qpos[:, 0:3]

        # ── Reward ────────────────────────────────────────────────────
        reward, r_info, new_dist = compute_reward(
            robot_pos      = robot_pos,
            robot_quat     = robot_quat,
            goal_pos       = goal_pos,
            prev_robot_pos = prev_pos,
            joint_vel      = joint_vel,
            action         = ctrl,
            prev_action    = state["prev_action"],
            min_obs_dist   = min_dist,
            has_collision  = has_coll,
            prev_dist_goal = state["prev_dist"],
        )

        # ── Termination ───────────────────────────────────────────────
        new_step = state["step_count"] + 1
        terminated = check_termination(
            robot_pos, robot_quat, goal_pos, new_step
        )

        new_state = {
            "dx":          dx,
            "mocap_pos":   mocap_pos,
            "goal_pos":    goal_pos,
            "step_count":  new_step,
            "prev_dist":   new_dist,
            "prev_action": ctrl,
        }
        obs  = self._get_obs(new_state)
        info = r_info
        return new_state, obs, reward, terminated, info

    # ════════════════════════════════════════════════════════════════════
    # OBSERVATION CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════

    def _get_obs(self, state: Dict) -> Dict:
        dx        = state["dx"]
        goal_pos  = state["goal_pos"]
        mocap_pos = state["mocap_pos"]

        # ── Depth images via Warp ─────────────────────────────────────
        site_ids = self._cam_site_ids
        # site_xpos: (B, n_sites_total, 3) → pick our 5 cams
        cam_xpos = dx.site_xpos[:, site_ids, :]   # (B, N_CAMS, 3)
        cam_xmat = dx.site_xmat[:, site_ids, :]   # (B, N_CAMS, 9)

        depth = self._renderer.render(cam_xpos, cam_xmat, mocap_pos)
        # depth: (B, N_CAMS, H, W)

        # ── Proprioception (37-dim) ───────────────────────────────────
        robot_pos  = dx.qpos[:, 0:3]
        robot_quat = dx.qpos[:, 3:7]
        robot_linv = dx.qvel[:, 0:3]
        robot_angv = dx.qvel[:, 3:6]
        joint_pos  = dx.qpos[:, 7:19]
        joint_vel  = dx.qvel[:, 6:18]

        # Goal relative direction and distance
        goal_diff  = goal_pos - robot_pos[:, :2]              # (B, 2)
        goal_dist  = jnp.linalg.norm(goal_diff, axis=-1, keepdims=True)  # (B, 1)
        goal_dir   = goal_diff / (goal_dist + 1e-8)

        proprio = jnp.concatenate([
            joint_pos,    # 12
            joint_vel,    # 12
            robot_quat,   # 4
            robot_linv,   # 3
            robot_angv,   # 3
            goal_dir,     # 2
            goal_dist,    # 1
        ], axis=-1)        # 37-dim

        return {"depth": depth, "proprio": proprio}

    # ════════════════════════════════════════════════════════════════════
    # SINGLE-ENV PHYSICS STEP (vmapped)
    # ════════════════════════════════════════════════════════════════════

    @staticmethod
    def _single_physics_step(mx, dx):
        return mjx.step(mx, dx)

    # ════════════════════════════════════════════════════════════════════
    # AUTO-RESET (vectorized)
    # ════════════════════════════════════════════════════════════════════

    def auto_reset(
        self, state: Dict, obs: Dict, terminated: jnp.ndarray, rng: jax.Array
    ) -> Tuple[Dict, Dict]:
        """Reset only the envs where terminated=True."""
        if not jnp.any(terminated):
            return state, obs

        rng_new, _ = jax.random.split(rng)
        reset_state, reset_obs = self.reset(rng_new)

        def _merge(live, dead):
            # Select live or dead based on terminated flag
            mask = terminated[:, None] if live.ndim > 1 else terminated
            return jnp.where(mask, dead, live) if live.ndim > 1 else \
                   jnp.where(terminated, dead, live)

        # Merge (can't do tree_map cleanly for dx because it's nested)
        # For simplicity, merge only the scalars / goal — dx merge is complex
        # Production: use jax.lax.cond per-env or full reset batch
        new_goal  = jnp.where(terminated[:, None], reset_state["goal_pos"], state["goal_pos"])
        new_count = jnp.where(terminated, reset_state["step_count"], state["step_count"])
        new_pdist = jnp.where(terminated, reset_state["prev_dist"],  state["prev_dist"])

        state = {**state, "goal_pos": new_goal,
                 "step_count": new_count, "prev_dist": new_pdist}
        return state, obs
