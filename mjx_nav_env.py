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

from warp_cameras import WarpDepthRenderer, N_CAMS, CAM_H, CAM_W, N_OBS, N_STATIC, N_DYNAMIC
from jax_reward import compute_reward, check_termination
from config import HUMANOID_OBSTACLE as HUMANOID_CFG

# ── Health check (JIT-compatible) ────────────────────────────────────────────
@jax.jit
def is_healthy(qpos: jnp.ndarray, proprio: jnp.ndarray) -> jnp.ndarray:
    """
    Returns (B,) bool — True when the env is in a valid state.
    Catches physics explosions (NaN/inf or out-of-range height) before they
    corrupt gradients.
    """
    height    = qpos[:, 2]                              # z position  (B,)
    height_ok = (height > 0.2) & (height < 2.0)
    obs_ok    = jnp.all(jnp.isfinite(proprio), axis=-1) # (B,)
    return height_ok & obs_ok

# ── Joint limits (from config.py SPOT_ROBOT) ─────────────────────────────────
JOINT_LOWER = jnp.array([-0.8,-0.6,-2.8]*4, dtype=jnp.float32)
JOINT_UPPER = jnp.array([ 0.8, 2.4,-0.5]*4, dtype=jnp.float32)
STANDING_POSE = jnp.array([0.0, 0.8, -1.6]*4, dtype=jnp.float32)

# Room half-extents for random placement (10×10 m room)
ROOM_HALF = 4.5   # keep 0.5m from walls

# Number of physics sub-steps per RL step (control @ 50 Hz, physics @ 200 Hz)
PHYSICS_SUBSTEPS = 4

# Index of the humanoid body in mocap_pos (after all static + dynamic obs)
HUMANOID_MOCAP_IDX = N_STATIC + N_DYNAMIC   # = 30


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
        dx_batch  = jax.tree_util.tree_map(
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

        # Randomize static + dynamic obstacles: indices 0 .. N_OBS-2
        rng, k4 = jax.random.split(rng)
        obs_xy = jax.random.uniform(k4, (self.n_envs, N_OBS, 2),
                                    minval=-ROOM_HALF, maxval=ROOM_HALF)
        obs_z  = jnp.ones((self.n_envs, N_OBS, 1)) * 0.5
        obs_pos_new = jnp.concatenate([obs_xy, obs_z], axis=-1)  # (B, N_OBS, 3)

        # ── Place humanoid near the randomised goal ───────────────────
        # Start at goal + patrol_radius along X, clipped to room bounds.
        patrol_r = float(HUMANOID_CFG["patrol_radius"])
        human_x0 = jnp.clip(goal_xy[:, 0] + patrol_r, -ROOM_HALF, ROOM_HALF)
        human_y0 = jnp.clip(goal_xy[:, 1],             -ROOM_HALF, ROOM_HALF)
        human_init_pos = jnp.stack([human_x0, human_y0], axis=-1)  # (B, 2)

        obs_pos_new = (obs_pos_new
                       .at[:, HUMANOID_MOCAP_IDX, 0].set(human_x0)
                       .at[:, HUMANOID_MOCAP_IDX, 1].set(human_y0)
                       .at[:, HUMANOID_MOCAP_IDX, 2].set(HUMANOID_CFG["mocap_z"]))

        # Carry state
        step_count   = jnp.zeros(self.n_envs, dtype=jnp.int32)
        prev_dist    = jnp.linalg.norm(goal_xy - robot_xy, axis=-1)
        prev_action  = jnp.zeros((self.n_envs, 12), dtype=jnp.float32)

        state = {
            "dx":           dx_batch,
            "mocap_pos":    obs_pos_new,
            "goal_pos":     goal_xy,
            "step_count":   step_count,
            "prev_dist":    prev_dist,
            "prev_action":  prev_action,
            # Humanoid-specific carry state
            "human_pos":    human_init_pos,                           # (B, 2)
            "human_wp_idx": jnp.zeros(self.n_envs, dtype=jnp.int32), # (B,)
            "human_t":      jnp.zeros(self.n_envs, dtype=jnp.float32),# (B,)
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

        # ── Guard: sanitize actions before physics ────────────────────
        action = jnp.clip(action, -1.0, 1.0)
        action = jnp.where(jnp.isfinite(action), action, 0.0)

        # Denormalize actions to joint position targets
        joint_mid   = (JOINT_UPPER + JOINT_LOWER) / 2.0
        joint_range = (JOINT_UPPER - JOINT_LOWER) / 2.0
        ctrl        = joint_mid + action * joint_range   # (B, 12)

        prev_qpos = dx.qpos.copy()

        # ── Physics sub-steps (4× for stability) ─────────────────────
        for _ in range(PHYSICS_SUBSTEPS):
            dx = dx.replace(ctrl=ctrl)
            dx = self._batch_step(dx)

        # ── Update humanoid obstacle (patrol near goal) ───────────────
        if HUMANOID_CFG["enabled"]:
            human_pos = state["human_pos"]      # (B, 2)
            human_wp  = state["human_wp_idx"]   # (B,)  int32
            human_t   = state["human_t"]        # (B,)  float32

            step_dt = float(PHYSICS_SUBSTEPS) * float(self._mj_model.opt.timestep)
            patrol_r = float(HUMANOID_CFG["patrol_radius"])

            # Recompute patrol waypoints from the (possibly new) goal pos
            wp0 = jnp.stack([
                jnp.clip(goal_pos[:, 0] + patrol_r, -ROOM_HALF, ROOM_HALF),
                jnp.clip(goal_pos[:, 1],             -ROOM_HALF, ROOM_HALF),
            ], axis=-1)                                               # (B, 2)
            wp1 = jnp.stack([
                jnp.clip(goal_pos[:, 0] - patrol_r, -ROOM_HALF, ROOM_HALF),
                jnp.clip(goal_pos[:, 1],             -ROOM_HALF, ROOM_HALF),
            ], axis=-1)                                               # (B, 2)
            waypoints = jnp.stack([wp0, wp1], axis=1)                # (B, 2, 2)

            # Select the current target waypoint per env
            env_idx = jnp.arange(self.n_envs)
            target  = waypoints[env_idx, human_wp]                   # (B, 2)

            # Move toward target
            diff    = target - human_pos                              # (B, 2)
            dist_h  = jnp.linalg.norm(diff, axis=-1, keepdims=True)  # (B, 1)
            dir_h   = diff / (dist_h + 1e-8)
            new_human_pos = jnp.clip(
                human_pos + dir_h * float(HUMANOID_CFG["speed"]) * step_dt,
                -ROOM_HALF, ROOM_HALF,
            )

            # Toggle waypoint when close enough
            switch_dist = float(HUMANOID_CFG["wp_switch_dist"])
            new_human_wp = jnp.where(dist_h[:, 0] < switch_dist,
                                     1 - human_wp, human_wp)
            new_human_t  = human_t + step_dt

            # Write updated humanoid position into mocap_pos
            mocap_pos = (mocap_pos
                         .at[:, HUMANOID_MOCAP_IDX, 0].set(new_human_pos[:, 0])
                         .at[:, HUMANOID_MOCAP_IDX, 1].set(new_human_pos[:, 1])
                         .at[:, HUMANOID_MOCAP_IDX, 2].set(HUMANOID_CFG["mocap_z"]))
        else:
            new_human_pos = state["human_pos"]
            new_human_wp  = state["human_wp_idx"]
            new_human_t   = state["human_t"]

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
            "dx":           dx,
            "mocap_pos":    mocap_pos,
            "goal_pos":     goal_pos,
            "step_count":   new_step,
            "prev_dist":    new_dist,
            "prev_action":  ctrl,
            "human_pos":    new_human_pos,
            "human_wp_idx": new_human_wp,
            "human_t":      new_human_t,
        }
        obs  = self._get_obs(new_state)

        # Guard: also terminate any env whose state has blown up (NaN/inf)
        terminated = terminated | ~is_healthy(dx.qpos, obs["proprio"])

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

        # Sanitize: replace any NaN/inf with 0 so the network never sees garbage
        proprio = jnp.where(jnp.isfinite(proprio), proprio, 0.0)

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
        """Reset only the envs where terminated=True.

        Merges ALL state fields — including the MJX physics state (dx) —
        so that terminated envs get a clean physics state and don't keep
        feeding corrupted qpos/qvel back into _batch_step.
        """
        if not jnp.any(terminated):
            return state, obs

        rng_new, _ = jax.random.split(rng)
        reset_state, reset_obs = self.reset(rng_new)

        # ── Broadcast helper: (B,) mask → shape matching any leaf ────
        def _pick(fresh, live):
            """Select fresh where terminated=True, live otherwise."""
            if not hasattr(live, 'shape') or live.ndim == 0:
                return live                          # non-array / unbatched scalar
            shape = (terminated.shape[0],) + (1,) * (live.ndim - 1)
            return jnp.where(terminated.reshape(shape), fresh, live)

        # ── Merge MJX physics state (dx) — the critical fix ─────────
        new_dx = jax.tree_util.tree_map(
            _pick, reset_state["dx"], state["dx"]
        )

        # ── Merge mocap_pos and prev_action (also previously skipped) ─
        new_mocap  = _pick(reset_state["mocap_pos"],  state["mocap_pos"])
        new_pact   = _pick(reset_state["prev_action"],state["prev_action"])

        # ── Merge scalar / goal / humanoid state ─────────────────────
        new_goal  = _pick(reset_state["goal_pos"],     state["goal_pos"])
        new_count = _pick(reset_state["step_count"],   state["step_count"])
        new_pdist = _pick(reset_state["prev_dist"],    state["prev_dist"])
        new_hpos  = _pick(reset_state["human_pos"],    state["human_pos"])
        new_hwp   = _pick(reset_state["human_wp_idx"], state["human_wp_idx"])
        new_ht    = _pick(reset_state["human_t"],      state["human_t"])

        new_state = {
            "dx":           new_dx,
            "mocap_pos":    new_mocap,
            "goal_pos":     new_goal,
            "step_count":   new_count,
            "prev_dist":    new_pdist,
            "prev_action":  new_pact,
            "human_pos":    new_hpos,
            "human_wp_idx": new_hwp,
            "human_t":      new_ht,
        }

        # ── Merge observations (reset envs get fresh obs) ────────────
        new_obs = jax.tree_util.tree_map(_pick, reset_obs, obs)

        return new_state, new_obs
