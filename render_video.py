"""
Render a video of the trained Spot agent from a .pkl checkpoint.

Usage:
    python render_video.py --ckpt path/to/checkpoint.pkl --out spot_eval.mp4
    python render_video.py --ckpt best.pkl --episodes 3 --deterministic
"""

import argparse
import os
import pickle
import time

# Enable EGL for headless rendering (must be set before importing mujoco)
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import mediapy as media

from jax_ppo import (
    PPOTrainer, DepthCNNEncoder, PropriEncoder, ACTION_DIM,
    LOG_STD_MIN, LOG_STD_MAX,
)
import flax.linen as nn

# Joint limits (must match mjx_nav_env.py)
JOINT_LOWER = np.array([-0.8, -0.6, -2.8] * 4, dtype=np.float32)
JOINT_UPPER = np.array([0.8, 2.4, -0.5] * 4, dtype=np.float32)
STANDING_POSE = np.array([0.0, 0.8, -1.6] * 4, dtype=np.float32)

ROOM_HALF = 4.5
MAX_STEPS = 1000


def load_checkpoint(ckpt_path):
    """Load params from a .pkl checkpoint."""
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    return jax.device_put(ckpt["params"])


def policy_forward(params, depth, proprio, rng_key, deterministic=False):
    """Run the policy network and return an action."""
    # depth: (1, 5, 120, 160), proprio: (1, 37)
    cnn_feat = DepthCNNEncoder().apply({"params": params["cnn"]}, depth / 10.0)
    pro_feat = PropriEncoder().apply({"params": params["proprio"]}, proprio)
    x = jnp.concatenate([cnn_feat, pro_feat], axis=-1)
    x = nn.Dense(256).apply({"params": params["trunk0"]}, x)
    x = nn.elu(x)
    x = nn.Dense(128).apply({"params": params["trunk1"]}, x)
    x = nn.elu(x)

    mean = nn.Dense(ACTION_DIM).apply({"params": params["actor"]}, x)
    mean = jnp.clip(mean, -2.0, 2.0)

    if deterministic:
        action = jnp.clip(mean, -1.0, 1.0)
    else:
        log_std = jnp.clip(params["log_std"], LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng_key, mean.shape)
        action = jnp.clip(mean + std * noise, -1.0, 1.0)

    return action


def build_proprio(mj_model, mj_data, goal_xy):
    """Build the 37-dim proprioception vector from CPU MuJoCo state."""
    # Robot state
    qpos = mj_data.qpos
    qvel = mj_data.qvel

    joint_pos = qpos[7:19]    # 12
    joint_vel = qvel[6:18]    # 12
    robot_quat = qpos[3:7]   # 4
    robot_linv = qvel[0:3]   # 3
    robot_angv = qvel[3:6]   # 3
    robot_xy = qpos[0:2]     # 2

    # Goal
    goal_diff = goal_xy - robot_xy
    goal_dist = np.linalg.norm(goal_diff)
    goal_dir = goal_diff / (goal_dist + 1e-8)

    proprio = np.concatenate([
        joint_pos / 3.14,     # 12
        joint_vel / 20.0,     # 12
        robot_quat,           # 4
        robot_linv / 5.0,     # 3
        robot_angv / 10.0,    # 3
        goal_dir,             # 2
        [goal_dist / 5.0],    # 1
    ]).astype(np.float32)     # 37-dim

    proprio = np.clip(np.nan_to_num(proprio, 0.0), -10.0, 10.0)
    return proprio


def render_depth_placeholder(n_cams=5, h=120, w=160):
    """Return a zero depth image (placeholder since we can't use Warp for 1 env easily).
    The agent will rely mostly on proprioception for this visualization."""
    return np.zeros((1, n_cams, h, w), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Render Spot agent video from checkpoint")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to .pkl checkpoint file")
    parser.add_argument("--xml", type=str, default="models/spot_scene.xml",
                        help="Path to MuJoCo XML scene")
    parser.add_argument("--out", type=str, default="spot_eval.mp4",
                        help="Output video filename")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to render")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic (mean) actions instead of sampling")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frames per second")
    parser.add_argument("--width", type=int, default=640,
                        help="Video width in pixels")
    parser.add_argument("--height", type=int, default=480,
                        help="Video height in pixels")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--camera", type=str, default=None,
                        help="MuJoCo camera name for rendering (None = free camera)")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    params = load_checkpoint(args.ckpt)

    # Load MuJoCo model (CPU)
    xml_path = os.path.join(os.path.dirname(__file__), args.xml)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Setup renderer
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)

    # Camera setup
    if args.camera:
        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
    else:
        # Third-person tracking camera
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mj_model.body("base_link").id
        cam.distance = 4.0
        cam.azimuth = 135
        cam.elevation = -25

    rng = jax.random.PRNGKey(args.seed)
    all_frames = []
    physics_substeps = 4

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")

        # Reset robot to standing pose
        mujoco.mj_resetData(mj_model, mj_data)

        # Random start position
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        robot_xy = np.array(jax.random.uniform(k1, (2,), minval=-ROOM_HALF, maxval=ROOM_HALF))
        robot_yaw = float(jax.random.uniform(k2, (), minval=0.0, maxval=2 * 3.14159))

        # Set robot position
        mj_data.qpos[0] = robot_xy[0]
        mj_data.qpos[1] = robot_xy[1]
        mj_data.qpos[2] = 0.52  # standing height
        mj_data.qpos[3] = np.cos(robot_yaw / 2)
        mj_data.qpos[4:6] = 0.0
        mj_data.qpos[6] = np.sin(robot_yaw / 2)
        mj_data.qpos[7:19] = STANDING_POSE

        # Random goal
        goal_dist = float(jax.random.uniform(k3, (), minval=2.0, maxval=6.0))
        goal_ang = float(jax.random.uniform(k3, (), minval=0.0, maxval=2 * 3.14159))
        goal_xy = robot_xy + np.array([goal_dist * np.cos(goal_ang),
                                        goal_dist * np.sin(goal_ang)])
        goal_xy = np.clip(goal_xy, -ROOM_HALF, ROOM_HALF)

        # Place a visual marker at the goal (use first mocap body)
        if mj_model.nmocap > 0:
            mj_data.mocap_pos[0] = [goal_xy[0], goal_xy[1], 0.1]

        mujoco.mj_forward(mj_model, mj_data)

        prev_action = np.zeros(12, dtype=np.float32)
        frames = []

        for step in range(MAX_STEPS):
            # Build observation
            proprio = build_proprio(mj_model, mj_data, goal_xy)
            depth = render_depth_placeholder()

            # Get action from policy
            rng, act_key = jax.random.split(rng)
            action = policy_forward(
                params,
                jnp.array(depth),
                jnp.array(proprio)[None],
                act_key,
                deterministic=args.deterministic,
            )
            action_np = np.array(action[0])

            # Denormalize to joint targets
            joint_mid = (JOINT_UPPER + JOINT_LOWER) / 2.0
            joint_range = (JOINT_UPPER - JOINT_LOWER) / 2.0
            ctrl = joint_mid + action_np * joint_range
            mj_data.ctrl[:12] = ctrl

            # Physics substeps
            for _ in range(physics_substeps):
                mujoco.mj_step(mj_model, mj_data)

            # Render frame
            renderer.update_scene(mj_data, cam)
            frame = renderer.render()
            frames.append(frame.copy())

            # Check termination
            height = mj_data.qpos[2]
            robot_pos = mj_data.qpos[0:2]
            dist_to_goal = np.linalg.norm(robot_pos - goal_xy)

            # Compute tilt (how far from upright)
            quat = mj_data.qpos[3:7]
            # z-component of the up vector in body frame
            w, x, y, z = quat
            up_z = 1.0 - 2.0 * (x * x + y * y)
            tilt = np.arccos(np.clip(up_z, -1.0, 1.0))

            if dist_to_goal < 0.5:
                print(f"  Step {step}: GOAL REACHED! dist={dist_to_goal:.2f}")
                # Add a few more frames so the viewer can see
                for _ in range(30):
                    renderer.update_scene(mj_data, cam)
                    frames.append(renderer.render().copy())
                break
            elif height < 0.2:
                print(f"  Step {step}: Fallen. height={height:.2f}")
                break
            elif tilt > 1.05:  # ~60 degrees
                print(f"  Step {step}: Tipped over. tilt={np.degrees(tilt):.1f} deg")
                break

            if step % 100 == 0:
                print(f"  Step {step}: pos=({mj_data.qpos[0]:.1f}, {mj_data.qpos[1]:.1f}), "
                      f"dist_to_goal={dist_to_goal:.2f}, height={height:.2f}")

        print(f"  Episode length: {min(step + 1, MAX_STEPS)} steps, {len(frames)} frames")
        all_frames.extend(frames)

    # Save video
    print(f"\nSaving {len(all_frames)} frames to {args.out} at {args.fps} fps...")
    media.write_video(args.out, all_frames, fps=args.fps)
    print(f"Video saved: {args.out}")
    print(f"Duration: {len(all_frames) / args.fps:.1f}s")


if __name__ == "__main__":
    main()
