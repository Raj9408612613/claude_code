"""
Isaac Lab Environment Configuration — Spot Navigation
=======================================================
Defines the full environment config for Isaac Lab's DirectRLEnv.

This replaces mjx_nav_env.py. Physics runs via PhysX 5 (GPU).
Depth cameras use Isaac Sim RTX rendering.
"""

from __future__ import annotations

import math
import os
from dataclasses import MISSING

# Absolute path to Spot USD — works whether cwd is /workspace or /isaac-sim
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPOT_USD_PATH = os.path.join(_REPO_ROOT, "models", "spot_omniverse.usd")

from .config import (
    PHYSICS_DT, CONTROL_DT, PHYSICS_SUBSTEPS,
    JOINT_LOWER, JOINT_UPPER, STANDING_POSE, TARGET_HEIGHT,
    N_CAMS, CAM_H, CAM_W, H_FOV, V_FOV, MIN_DEPTH, MAX_DEPTH,
    N_STATIC, N_DYNAMIC, N_HUMANOID, N_OBS, OBS_HALF_SIZES,
    ROOM_HALF, HUMANOID_OBSTACLE,
    ACTION_DIM, PROPRIO_DIM,
)

# NOTE: These imports require Isaac Lab to be installed.
# Isaac Lab 2.0+ uses "isaaclab.*", older versions use "omni.isaac.lab.*".
# Try both to support whatever version is installed in the container.
HAS_ISAAC = False
_ISAAC_IMPORT_ERROR = None

try:
    # Isaac Lab 2.0+ (standalone package)
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, PhysxCfg
    from isaaclab.utils import configclass
    HAS_ISAAC = True
except ImportError:
    try:
        # Isaac Lab 1.x (omniverse extension)
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.actuators import ImplicitActuatorCfg
        from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from omni.isaac.lab.envs import DirectRLEnvCfg
        from omni.isaac.lab.scene import InteractiveSceneCfg
        from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
        from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
        from omni.isaac.lab.utils import configclass
        HAS_ISAAC = True
    except ImportError as _e:
        _ISAAC_IMPORT_ERROR = str(_e)

if not HAS_ISAAC:
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
            # GPU solver is the default in Isaac Lab 0.54+
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

        # Ground plane with tuned friction (compensates for PhysX vs MuJoCo gap)
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                size=(20.0, 20.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.2,     # MuJoCo soft contacts → need more friction
                    dynamic_friction=1.0,
                    restitution=0.0,         # No bounce (MuJoCo has soft contacts)
                ),
            ),
        )

        # Spot robot (imported from MJCF -> USD)
        # The MJCF importer converts spot_scene.xml + OBJ meshes to USD.
        # After conversion, reference the USD path here.
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                # Path to converted USD (user must run MJCF import first)
                usd_path=SPOT_USD_PATH,
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
                "legs": ImplicitActuatorCfg(
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

        # ── Walls (4 axis-aligned boxes enclosing 10x10m room) ────────
        wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Walls/north",
            spawn=sim_utils.CuboidCfg(
                size=(10.0, 0.2, 3.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 5.0, 1.5)),
        )
        wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Walls/south",
            spawn=sim_utils.CuboidCfg(
                size=(10.0, 0.2, 3.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -5.0, 1.5)),
        )
        wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Walls/east",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 10.0, 3.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, 0.0, 1.5)),
        )
        wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Walls/west",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 10.0, 3.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-5.0, 0.0, 1.5)),
        )
        # ── Obstacle rigid bodies (25 static boxes + 5 dynamic + 1 humanoid) ──
        # Each is a kinematic rigid body (position-controlled, not simulated).
        # Positions are set from SpotNavEnv._obs_pos each step.
        # Spawned at off-scene (100, 0, 0.5) — moved into room on reset.
    # Build obstacle configs programmatically from OBS_HALF_SIZES
    def _build_obstacle_cfgs():
        """Generate RigidObjectCfg for each obstacle."""
        cfgs = {}
        for i in range(N_STATIC):
            hs = OBS_HALF_SIZES[i]
            cfgs[f"obs_static_{i:02d}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Obstacles/" + f"static_{i:02d}",
                spawn=sim_utils.CuboidCfg(
                    size=(hs[0] * 2, hs[1] * 2, hs[2] * 2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.4, 0.2),
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(100.0, 0.0, hs[2]),
                ),
            )
        for i in range(N_DYNAMIC):
            idx = N_STATIC + i
            hs = OBS_HALF_SIZES[idx]
            cfgs[f"obs_dynamic_{i:02d}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Obstacles/" + f"dynamic_{i:02d}",
                spawn=sim_utils.CylinderCfg(
                    radius=hs[0],
                    height=hs[2] * 2,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.6, 0.8),
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(100.0, 0.0, hs[2]),
                ),
            )
        # Humanoid (approximated as a tall capsule/box)
        hs = OBS_HALF_SIZES[N_STATIC + N_DYNAMIC]
        cfgs["obs_humanoid"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obstacles/humanoid",
            spawn=sim_utils.CapsuleCfg(
                radius=hs[0],
                height=hs[2] * 2 - hs[0] * 2,  # capsule height excludes end caps
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.3, 0.3),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(100.0, 0.0, HUMANOID_OBSTACLE["mocap_z"]),
            ),
        )
        return cfgs
    # Attach obstacle configs to scene class
    _obs_cfgs = _build_obstacle_cfgs()
    for _name, _cfg in _obs_cfgs.items():
        setattr(SpotSceneCfg, _name, _cfg)
    # ── Contact sensor on robot (detects collisions with obstacles) ────
    SpotSceneCfg.contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,   # every physics step
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacles/.*"],
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
    # Stubs when Isaac Lab is not installed — raise ImportError on use
    # so train.py's except ImportError catches it and shows the real cause
    def _raise():
        raise ImportError(
            f"omni.isaac.lab is not available.\n"
            f"  Root cause: {_ISAAC_IMPORT_ERROR}\n"
            f"  Verify Isaac Lab is installed: "
            f"/isaac-sim/python.sh -c \"import omni.isaac.lab; print('OK')\""
        )

    class SpotSimCfg:
        def __init__(self, *a, **kw): _raise()

    class SpotSceneCfg:
        def __init__(self, *a, **kw): _raise()

    class SpotNavEnvCfg:
        def __init__(self, *a, **kw): _raise()
