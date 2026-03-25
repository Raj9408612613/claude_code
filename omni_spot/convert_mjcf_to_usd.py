"""
MJCF-to-USD Conversion Script for Spot Robot
==============================================
Converts models/spot_scene.xml + 23 OBJ meshes to a USD file
that Isaac Lab can reference via ArticulationCfg.

This script must run inside Isaac Sim's Python environment:
    ~/.local/share/ov/pkg/isaac-sim-*/python.sh convert_mjcf_to_usd.py

It uses omni.importer.mjcf to convert ONLY the Spot robot articulation
(not obstacles/walls — those are handled by SpotSceneCfg as kinematic prims).

After conversion, the USD is written to models/spot_scene.usd, which
spot_env_cfg.py references in the ArticulationCfg.

Usage:
    # From Isaac Sim Python:
    python omni_spot/convert_mjcf_to_usd.py [--output models/spot_scene.usd]
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MJCF_PATH = os.path.join(PROJECT_ROOT, "models", "spot_scene.xml")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "models", "spot_scene.usd")


def convert(mjcf_path: str, output_path: str) -> None:
    """Run the MJCF -> USD conversion inside Isaac Sim."""

    # These imports only work inside Isaac Sim's Python runtime
    try:
        import omni.kit.app  # noqa: F401 — ensures Kit runtime is up
    except ImportError:
        print(
            "ERROR: This script must be run with Isaac Sim's Python:\n"
            "  ~/.local/share/ov/pkg/isaac-sim-*/python.sh "
            "omni_spot/convert_mjcf_to_usd.py",
            file=sys.stderr,
        )
        sys.exit(1)

    from omni.importer.mjcf import MjcfImporter  # type: ignore[import]
    from omni.isaac.core.utils.stage import open_stage, save_stage  # type: ignore[import]
    import omni.isaac.lab.sim as sim_utils  # type: ignore[import]

    print(f"[convert] MJCF source : {mjcf_path}")
    print(f"[convert] USD output  : {output_path}")

    # ── 1. Create a fresh stage ──────────────────────────────────────
    open_stage(output_path)

    # ── 2. Configure the MJCF importer ───────────────────────────────
    importer = MjcfImporter()
    config = importer.get_import_config()

    # Import settings — keep articulation structure intact
    config.set_fix_base(False)                  # Spot has a floating base (freejoint)
    config.set_import_sites(True)               # Keep camera sites
    config.set_import_inertia_tensor(True)      # Preserve real mass properties
    config.set_make_default_prim(True)
    config.set_self_collision(False)            # Matches <contact><exclude> in MJCF
    config.set_density(0.0)                     # Use explicit mass, not density

    # Override joint drive to match our Isaac Lab ImplicitActuatorCfg
    # (kp=500, kv=40 — same as MJCF position actuators)
    config.set_override_joint_dynamics(False)    # Let Isaac Lab cfg set stiffness/damping

    # ── 3. Import ────────────────────────────────────────────────────
    # The importer reads the MJCF, converts meshes (OBJ -> USD mesh prims),
    # and builds an articulation tree under /World/spot_scene/base_link.
    prim_path = "/World/spot_scene"
    result = importer.import_asset(
        asset_path=mjcf_path,
        prim_path=prim_path,
        config=config,
    )

    if not result:
        print("ERROR: MJCF import failed.", file=sys.stderr)
        sys.exit(1)

    print(f"[convert] Articulation imported at {prim_path}")

    # ── 4. Post-process: remove non-robot prims ──────────────────────
    # The MJCF contains obstacles, walls, floor, lights, etc.
    # We only want the robot articulation. Everything else is defined
    # in SpotSceneCfg with Isaac Lab-native prims.
    from pxr import Usd, UsdGeom  # type: ignore[import]

    stage = Usd.Stage.Open(output_path)

    # Prims to KEEP (robot subtree + root)
    keep_prefix = f"{prim_path}/base_link"

    # Remove obstacle, wall, floor, and light prims injected by MJCF import
    prims_to_remove = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        # Skip the robot subtree and the root xform
        if path.startswith(keep_prefix) or path == prim_path or path == "/World":
            continue
        # Skip default prim references
        if path.startswith("/World") and path.count("/") <= 2:
            # Could be /World/spot_scene — keep it
            if path == f"{prim_path}":
                continue
        # Remove obstacle/wall/floor/light prims from the MJCF import
        name = prim.GetName()
        if any(
            tag in name.lower()
            for tag in [
                "obstacle", "dynobs", "human_0", "wall_", "floor",
                "top_light", "front_fill", "back_fill", "tracking",
            ]
        ):
            prims_to_remove.append(path)

    for path in prims_to_remove:
        stage.RemovePrim(path)
        print(f"[convert] Removed non-robot prim: {path}")

    # ── 5. Save ──────────────────────────────────────────────────────
    stage.GetRootLayer().Save()
    print(f"[convert] USD saved to {output_path}")

    # ── 6. Verify articulation joints ────────────────────────────────
    expected_joints = [
        "fl_hx", "fl_hy", "fl_kn",
        "fr_hx", "fr_hy", "fr_kn",
        "hl_hx", "hl_hy", "hl_kn",
        "hr_hx", "hr_hy", "hr_kn",
    ]
    stage = Usd.Stage.Open(output_path)
    found_joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform) or prim.GetTypeName() == "PhysicsRevoluteJoint":
            name = prim.GetName()
            if name in expected_joints:
                found_joints.append(name)

    print(f"[convert] Found {len(found_joints)}/12 expected joints: {found_joints}")
    if len(found_joints) < 12:
        missing = set(expected_joints) - set(found_joints)
        print(f"[convert] WARNING: Missing joints: {missing}")
        print(
            "[convert] Note: Joint prims may use different names after MJCF import. "
            "Check the USD stage manually and update ArticulationCfg joint_pos keys."
        )

    print("[convert] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Spot MJCF to USD for Isaac Lab"
    )
    parser.add_argument(
        "--mjcf", default=MJCF_PATH,
        help=f"Path to MJCF file (default: {MJCF_PATH})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"Output USD path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.mjcf):
        print(f"ERROR: MJCF not found: {args.mjcf}", file=sys.stderr)
        sys.exit(1)

    convert(args.mjcf, args.output)


if __name__ == "__main__":
    main()
