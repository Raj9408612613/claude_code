"""
Dual-Camera Visualization for Spot Obstacle Avoidance
Renders side-by-side GIF/MP4: Robot POV (front camera) + Overview (tracking camera)

Usage:
    python test_obstacle_visual.py <model_path>
    python test_obstacle_visual.py <model_path> --output spot_obstacle.gif --fps 25
    python test_obstacle_visual.py --random --episodes 1
    python test_obstacle_visual.py <model_path> --mp4 --output spot_obstacle.mp4
"""

import os
import sys
import argparse
import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIS_WIDTH = 480
VIS_HEIGHT = 480
LABEL_HEIGHT = 30
DEFAULT_FPS = 25
DEFAULT_MAX_STEPS = 500


# ---------------------------------------------------------------------------
# Headless rendering setup
# ---------------------------------------------------------------------------

def setup_headless():
    """Configure MuJoCo for headless rendering (Colab / servers)."""
    if "DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def create_overview_camera(data, distance=8.0, azimuth=90, elevation=-45):
    """Create an MjvCamera that tracks the robot from above-behind."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[0] = data.qpos[0]  # robot x
    cam.lookat[1] = data.qpos[1]  # robot y
    cam.lookat[2] = 0.3            # slightly above ground
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_pov(renderer, data, front_camera_id):
    """Render from the robot's front-mounted camera."""
    renderer.update_scene(data, camera=front_camera_id)
    return renderer.render().copy()


def render_overview(renderer, data, distance=8.0, azimuth=90, elevation=-45):
    """Render from the overview tracking camera."""
    cam = create_overview_camera(data, distance, azimuth, elevation)
    renderer.update_scene(data, cam)
    return renderer.render().copy()


# ---------------------------------------------------------------------------
# Frame compositing
# ---------------------------------------------------------------------------

def _get_font(size=18):
    """Try to load a TTF font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
    return ImageFont.load_default()


def add_label(frame, text):
    """Add a text label bar at the top of a frame."""
    h, w = frame.shape[:2]
    canvas = np.zeros((h + LABEL_HEIGHT, w, 3), dtype=np.uint8)
    # Black label bar already zeros
    canvas[LABEL_HEIGHT:, :, :] = frame

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _get_font(18)
    draw.text((10, 5), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def add_stats_overlay(frame, step, distance, collisions, reward):
    """Add a HUD stats bar at the bottom of a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_font(14)
    h = frame.shape[0]

    # Semi-transparent bar at bottom
    draw.rectangle([(0, h - 28), (frame.shape[1], h)], fill=(0, 0, 0))
    stats = (
        f"Step: {step}  |  Dist: {distance:.2f}m  |  "
        f"Collisions: {collisions}  |  Reward: {reward:.1f}"
    )
    draw.text((8, h - 24), stats, fill=(255, 255, 0), font=font)
    return np.array(img)


def composite_frames(pov_frame, overview_frame):
    """Stack two labelled frames side-by-side."""
    pov_labeled = add_label(pov_frame, "Spot POV (Front Camera)")
    overview_labeled = add_label(overview_frame, "Overview (Tracking)")

    # Ensure same height
    max_h = max(pov_labeled.shape[0], overview_labeled.shape[0])
    if pov_labeled.shape[0] < max_h:
        pad = np.zeros((max_h - pov_labeled.shape[0], pov_labeled.shape[1], 3), dtype=np.uint8)
        pov_labeled = np.concatenate([pov_labeled, pad], axis=0)
    if overview_labeled.shape[0] < max_h:
        pad = np.zeros((max_h - overview_labeled.shape[0], overview_labeled.shape[1], 3), dtype=np.uint8)
        overview_labeled = np.concatenate([overview_labeled, pad], axis=0)

    # 4-pixel black separator
    sep = np.zeros((max_h, 4, 3), dtype=np.uint8)
    return np.concatenate([pov_labeled, sep, overview_labeled], axis=1)


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def run_visualization(
    model_path=None,
    output_path="spot_obstacle_dual.gif",
    n_episodes=1,
    fps=DEFAULT_FPS,
    max_steps=DEFAULT_MAX_STEPS,
    use_random=False,
    use_mp4=False,
    overview_distance=8.0,
    overview_azimuth=90,
    overview_elevation=-45,
):
    """
    Run episodes and render dual-camera composite frames.

    Args:
        model_path: Path to trained PPO model (.zip)
        output_path: Output file path (.gif or .mp4)
        n_episodes: Number of episodes to render
        fps: Frames per second in output
        max_steps: Maximum steps per episode
        use_random: If True, use random actions instead of a model
        use_mp4: If True, save as MP4 (requires imageio-ffmpeg)
        overview_distance: Camera distance for overview
        overview_azimuth: Camera azimuth for overview
        overview_elevation: Camera elevation for overview
    """
    setup_headless()

    # Late imports so headless env vars are set first
    from spot_obstacle_env import SpotObstacleEnv

    # Load model
    model = None
    if not use_random:
        if model_path is None:
            print("Error: model_path required when not using --random")
            sys.exit(1)
        from stable_baselines3 import PPO
        from cnn_policy import SpotCNNExtractor
        print(f"Loading model: {model_path}")
        custom_objects = {
            "policy_kwargs": dict(
                features_extractor_class=SpotCNNExtractor,
                features_extractor_kwargs=dict(
                    cnn_output_dim=128, proprio_hidden_dim=64
                ),
            ),
        }
        model = PPO.load(model_path, custom_objects=custom_objects)

    # Create environment (64x64 for CNN obs, we render vis separately)
    env = SpotObstacleEnv(render_mode=None, camera_width=64, camera_height=64)

    # High-resolution renderer for visualization
    vis_renderer = mujoco.Renderer(env.model, VIS_HEIGHT, VIS_WIDTH)

    # Front camera ID
    front_cam_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera"
    )

    all_frames = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        ep_reward = 0.0

        goal = info["goal_position"]
        print(f"\nEpisode {ep + 1}/{n_episodes}  "
              f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")

        while not done and step_count < max_steps:
            # Action
            if use_random:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step_count += 1

            # Render both views at high resolution
            pov_frame = render_pov(vis_renderer, env.data, front_cam_id)
            overview_frame = render_overview(
                vis_renderer, env.data,
                distance=overview_distance,
                azimuth=overview_azimuth,
                elevation=overview_elevation,
            )

            # Add stats HUD to overview
            overview_frame = add_stats_overlay(
                overview_frame,
                step=step_count,
                distance=info["distance_to_goal"],
                collisions=info["obstacle_collisions"],
                reward=ep_reward,
            )

            # Composite side-by-side
            composite = composite_frames(pov_frame, overview_frame)
            all_frames.append(composite)

            if step_count % 100 == 0:
                print(f"  Step {step_count}: dist={info['distance_to_goal']:.2f}m  "
                      f"reward={ep_reward:.1f}  collisions={info['obstacle_collisions']}")

        # Episode summary
        final_dist = info["distance_to_goal"]
        success = final_dist < 0.5
        status = "SUCCESS" if success else "FAILED"
        print(f"  -> {status}: {step_count} steps, reward={ep_reward:.1f}, "
              f"dist={final_dist:.2f}m, collisions={info['obstacle_collisions']}")

    # Save output
    if len(all_frames) == 0:
        print("No frames to save.")
        vis_renderer.close()
        env.close()
        return

    print(f"\nSaving {len(all_frames)} frames to {output_path} ...")

    if use_mp4 or output_path.endswith(".mp4"):
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                    quality=8)
        for frame in all_frames:
            writer.append_data(frame)
        writer.close()
    else:
        # GIF
        duration_ms = 1000.0 / fps
        imageio.mimsave(output_path, all_frames, duration=duration_ms, loop=0)

    file_size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved! File size: {file_size_mb:.1f} MB")

    vis_renderer.close()
    env.close()

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-camera visualization for Spot obstacle avoidance"
    )
    parser.add_argument(
        "model_path", type=str, nargs="?", default=None,
        help="Path to trained PPO model (.zip)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="spot_obstacle_dual.gif",
        help="Output file path (default: spot_obstacle_dual.gif)"
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to render (default: 1)"
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help=f"Frames per second (default: {DEFAULT_FPS})"
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Use random actions instead of a trained model"
    )
    parser.add_argument(
        "--mp4", action="store_true",
        help="Save as MP4 instead of GIF (requires imageio-ffmpeg)"
    )
    parser.add_argument(
        "--cam-distance", type=float, default=8.0,
        help="Overview camera distance (default: 8.0)"
    )
    parser.add_argument(
        "--cam-azimuth", type=float, default=90,
        help="Overview camera azimuth in degrees (default: 90)"
    )
    parser.add_argument(
        "--cam-elevation", type=float, default=-45,
        help="Overview camera elevation in degrees (default: -45)"
    )

    args = parser.parse_args()

    if not args.random and args.model_path is None:
        parser.error("model_path is required unless --random is specified")

    print("=" * 60)
    print("SPOT OBSTACLE AVOIDANCE - DUAL CAMERA VISUALIZATION")
    print("=" * 60)
    print(f"Model: {args.model_path or 'RANDOM POLICY'}")
    print(f"Output: {args.output}")
    print(f"Episodes: {args.episodes}, FPS: {args.fps}, Max steps: {args.max_steps}")
    print("=" * 60)

    run_visualization(
        model_path=args.model_path,
        output_path=args.output,
        n_episodes=args.episodes,
        fps=args.fps,
        max_steps=args.max_steps,
        use_random=args.random,
        use_mp4=args.mp4,
        overview_distance=args.cam_distance,
        overview_azimuth=args.cam_azimuth,
        overview_elevation=args.cam_elevation,
    )
