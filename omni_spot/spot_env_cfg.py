"""
Isaac Lab Environment Configuration — Spot Navigation
=======================================================
Defines the full environment config for Isaac Lab's DirectRLEnv.

This replaces mjx_nav_env.py. Physics runs via PhysX 5 (GPU).
Depth cameras use Isaac Sim RTX rendering.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from .config import (
    PHYSICS_DT, CONTROL_DT, PHYSICS_SUBSTEPS,
    JOINT_LOWER, JOINT_UPPER, STANDING_POSE, TARGET_HEIGHT,
    N_CAMS, CAM_H, CAM_W, H_FOV, V_FOV, MIN_DEPTH, MAX_DEPTH,
    N_STATIC, N_DYNAMIC, N_HUMANOID, N_OBS,
    ROOM_HALF, HUMANOID_OBSTACLE,
    ACTION_DIM, PROPRIO_DIM,
)

# NOTE: These imports require Isaac Lab to be installed.
# They will fail in a plain Python env without Omniverse.
# This file serves as the configuration specification.
try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
    from omni.isaac.lab.envs import DirectRLEnvCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.sensors import CameraCfg
    from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
    from omni.isaac.lab.utils import configclass
    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False
    # Provide stub for development without Isaac Lab
    def configclass(cls):
        return cls


# ════════════════════════════════════════════════════════════════════════════
# SIMULATION CONFIG
# ════════════════════════════════════════════════════════════════════════════

if HAS_ISAAC:

    @configclass
    class SpotSimCfg(SimulationCfg):
        """PhysX 5 simulation settings matching MuJoCo physics behavior."""
        dt = PHYSICS_DT                    # 0.005s = 200 Hz
        render_interval = PHYSICS_SUBSTEPS  # render every 4 physics steps
        gravity = (0.0, 0.0, -9.81)

        physx: PhysxCfg = PhysxCfg(
            # GPU-accelerated solver
            use_gpu=True,
            solver_type=1,                 # TGS solver (better for articulations)
            max_position_iteration_count=8,
            max_velocity_iteration_count=1,
            # Contact parameters tuned to approximate MuJoCo behavior
            bounce_threshold_velocity=0.5,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.025,
            # GPU buffer sizes for parallel envs
            gpu_found_lost_pairs_capacity=2 ** 21,
            gpu_found_lost_aggregate_pairs_capacity=2 ** 25,
            gpu_total_aggregate_pairs_capacity=2 ** 21,
            gpu_max_rigid_contact_count=2 ** 23,
            gpu_max_rigid_patch_count=2 ** 23,
            gpu_heap_capacity=2 ** 26,
            gpu_temp_buffer_capacity=2 ** 24,
            gpu_max_num_partitions=8,
        )


    # ════════════════════════════════════════════════════════════════════════
    # SCENE CONFIG
    # ════════════════════════════════════════════════════════════════════════

    @configclass
    class SpotSceneCfg(InteractiveSceneCfg):
        """Scene with Spot robot, ground, and depth cameras."""

        # Ground plane
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
        )

        # Spot robot (imported from MJCF -> USD)
        # The MJCF importer converts spot_scene.xml + OBJ meshes to USD.
        # After conversion, reference the USD path here.
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                # Path to converted USD (user must run MJCF import first)
                usd_path="models/spot_scene.usd",
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, TARGET_HEIGHT),
                joint_pos={
                    # Standing pose for all 12 joints
                    "fl_hx": STANDING_POSE[0],
                    "fl_hy": STANDING_POSE[1],
                    "fl_kn": STANDING_POSE[2],
                    "fr_hx": STANDING_POSE[3],
                    "fr_hy": STANDING_POSE[4],
                    "fr_kn": STANDING_POSE[5],
                    "hl_hx": STANDING_POSE[6],
                    "hl_hy": STANDING_POSE[7],
                    "hl_kn": STANDING_POSE[8],
                    "hr_hx": STANDING_POSE[9],
                    "hr_hy": STANDING_POSE[10],
                    "hr_kn": STANDING_POSE[11],
                },
            ),
            actuators={
                "legs": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=500.0,     # kp matches MuJoCo
                    damping=40.0,        # kv matches MuJoCo
                    effort_limit=1000.0,
                ),
            },
        )

        # ── Depth cameras (5 cameras on Spot body) ──────────────────
        # Each camera matches: 120x160 pixels, 87 deg HFOV, depth only
        cam_front_center = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/cam_front_center",
            update_period=CONTROL_DT,
            height=CAM_H,
            width=CAM_W,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0,
                horizontal_aperture=2.0 * math.tan(math.radians(H_FOV / 2)),
                clipping_range=(MIN_DEPTH, MAX_DEPTH),
            ),
        )
        cam_front_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/cam_front_left",
            update_period=CONTROL_DT,
            height=CAM_H,
            width=CAM_W,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0,
                horizontal_aperture=2.0 * math.tan(math.radians(H_FOV / 2)),
                clipping_range=(MIN_DEPTH, MAX_DEPTH),
            ),
        )
        cam_front_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/cam_front_right",
            update_period=CONTROL_DT,
            height=CAM_H,
            width=CAM_W,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0,
                horizontal_aperture=2.0 * math.tan(math.radians(H_FOV / 2)),
                clipping_range=(MIN_DEPTH, MAX_DEPTH),
            ),
        )
        cam_rear_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/cam_rear_left",
            update_period=CONTROL_DT,
            height=CAM_H,
            width=CAM_W,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0,
                horizontal_aperture=2.0 * math.tan(math.radians(H_FOV / 2)),
                clipping_range=(MIN_DEPTH, MAX_DEPTH),
            ),
        )
        cam_rear_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/cam_rear_right",
            update_period=CONTROL_DT,
            height=CAM_H,
            width=CAM_W,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0,
                horizontal_aperture=2.0 * math.tan(math.radians(H_FOV / 2)),
                clipping_range=(MIN_DEPTH, MAX_DEPTH),
            ),
        )


    # ════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT CONFIG
    # ════════════════════════════════════════════════════════════════════════

    @configclass
    class SpotNavEnvCfg(DirectRLEnvCfg):
        """Full environment config for Spot navigation RL."""

        # Simulation
        sim: SimulationCfg = SpotSimCfg()
        decimation = PHYSICS_SUBSTEPS       # 4 physics steps per RL step

        # Scene
        scene: InteractiveSceneCfg = SpotSceneCfg(
            num_envs=4096,
            env_spacing=5.0,                # 5m between env origins
        )

        # Spaces
        num_observations = PROPRIO_DIM       # 37 (depth handled separately via cameras)
        num_actions = ACTION_DIM             # 12

        # Episode
        episode_length_s = 1000 * CONTROL_DT  # 1000 steps * 0.02s = 20s

else:
    # Stubs when Isaac Lab is not installed (for testing imports)
    class SpotSimCfg:
        pass

    class SpotSceneCfg:
        pass

    class SpotNavEnvCfg:
        pass
