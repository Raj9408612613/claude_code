"""Render Spot standing upright on the floor and save as an image."""

import os
os.environ["MUJOCO_GL"] = "osmesa"  # headless software rendering

import numpy as np
import mujoco

# ── Load model ──────────────────────────────────────────────────────
xml_path = os.path.join(os.path.dirname(__file__), "models/spot_scene.xml")
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# ── Set standing pose ───────────────────────────────────────────────
# Freejoint: [x, y, z, qw, qx, qy, qz] + 12 joint angles
STANDING_POSE = np.array([0.0, 1.04, -1.8] * 4, dtype=np.float32)

mj_data.qpos[0] = 0.0    # x
mj_data.qpos[1] = 0.0    # y
mj_data.qpos[2] = 0.46   # z — standing height
mj_data.qpos[3] = 1.0    # qw
mj_data.qpos[4] = 0.0    # qx
mj_data.qpos[5] = 0.0    # qy
mj_data.qpos[6] = 0.0    # qz
mj_data.qpos[7:19] = STANDING_POSE

# Run forward kinematics so the body positions update
mujoco.mj_forward(mj_model, mj_data)

# ── Render ──────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 960
renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)

# Camera: 3/4 view looking at robot
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mj_model.body("base_link").id
cam.distance = 2.5
cam.azimuth = 135
cam.elevation = -20

renderer.update_scene(mj_data, cam)
frame = renderer.render()

# ── Save ────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "spot_standing.png")

try:
    import mediapy as media
    media.write_image(out_path, frame)
except ImportError:
    from PIL import Image
    Image.fromarray(frame).save(out_path)

print(f"Saved {WIDTH}x{HEIGHT} image to {out_path}")
