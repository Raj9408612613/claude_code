"""
GPU Batch Depth Renderer — NVIDIA Warp
=======================================
Replaces depth_cameras.py (Isaac Sim CPU renderer).

How it works:
  1. Scene geometry (floor, walls, obstacles) is stored as Warp arrays on GPU.
  2. Camera poses come from MJX data.site_xpos / data.site_xmat (JAX arrays).
  3. One Warp CUDA kernel casts all rays for all envs × cameras in parallel.
  4. Output shape: (n_envs, N_CAMS, H, W) — pure GPU, no host round-trips.

JAX ↔ Warp bridge uses DLPack (zero-copy when both are on the same CUDA device).

Scene primitives:
  - Floor plane (y=0 up convention)
  - 4 walls (axis-aligned boxes)
  - 25 obstacle boxes      → positions read from data.mocap_pos each step
  - 5  dynamic cylinders  → same
  - 1  humanoid body      → placed near randomised goal at reset; patrols each step

Depth noise (Gaussian + dropout) is applied inside a second Warp kernel.
"""

import math
import numpy as np

try:
    import warp as wp
    import jax
    import jax.numpy as jnp
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    raise ImportError("Install warp-lang and jax: see requirements_mjx.txt")

# ── Camera constants (from config.py) ────────────────────────────────────────
N_CAMS   = 5
CAM_H    = 120   # pixels
CAM_W    = 160
H_FOV    = math.radians(87.0)   # horizontal field of view
V_FOV    = math.radians(58.0)
MIN_D    = 0.1
MAX_D    = 10.0

# Number of obstacle mocap bodies in the XML
N_STATIC   = 25
N_DYNAMIC  = 5
N_HUMANOID = 1   # one capsule-based humanoid mocap body (human_0)
N_OBS      = N_STATIC + N_DYNAMIC + N_HUMANOID   # total mocap bodies = 31


# ════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE RAY DIRECTIONS IN CAMERA LOCAL FRAME
# Shape: (N_CAMS, H, W, 3)  — one direction per pixel per camera
# All cameras share the same local-frame ray grid (orientation handled later)
# ════════════════════════════════════════════════════════════════════════════

def _build_ray_dirs() -> np.ndarray:
    """Local-frame ray directions for one camera (H×W pixels).
    Camera looks along +X; up is +Z. We then rotate to world frame per step.
    Returns (H, W, 3) float32.
    """
    tan_h = math.tan(H_FOV / 2)
    tan_v = math.tan(V_FOV / 2)
    xs = np.linspace(-tan_h, tan_h, CAM_W, dtype=np.float32)
    ys = np.linspace( tan_v, -tan_v, CAM_H, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)               # (H, W)
    gz = np.ones_like(gx)                      # forward = +Z in local cam frame
    dirs = np.stack([gx, gy, gz], axis=-1)     # (H, W, 3)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return (dirs / norms).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# WARP KERNELS
# ════════════════════════════════════════════════════════════════════════════

@wp.kernel
def raycast_kernel(
    # Camera poses — all envs, all cams flattened: (n_envs*N_CAMS,)
    cam_pos:  wp.array(dtype=wp.vec3),      # world positions
    cam_R:    wp.array(dtype=wp.mat33),     # rotation matrices (col = world axis)
    # Local ray directions: (H*W, 3)
    local_dirs: wp.array(dtype=wp.vec3),
    # Scene mesh (static geometry: floor + walls)
    static_mesh: wp.uint64,
    # Obstacle boxes: positions (n_envs*N_OBS,) half-sizes (N_OBS,)
    obs_pos:  wp.array(dtype=wp.vec3),
    obs_half: wp.array(dtype=wp.vec3),
    n_obs:    int,
    n_cams:   int,
    pixels_per_cam: int,
    min_dist: float,
    max_dist: float,
    # Output depth: (n_envs * N_CAMS * H * W,)
    depth_out: wp.array(dtype=float),
):
    """One thread = one pixel across all envs × cams."""
    tid = wp.tid()
    cam_id  = tid // pixels_per_cam
    pix_id  = tid %  pixels_per_cam

    # Determine which env this pixel belongs to (for per-env obstacles)
    env_id  = cam_id // n_cams

    # World-space ray
    local_d = local_dirs[pix_id]
    world_d = cam_R[cam_id] * local_d       # rotate to world frame
    origin  = cam_pos[cam_id]

    best_t = max_dist

    # ── 1. Static mesh (floor + walls) ─────────────────────────────────
    t_mesh = float(0.0)
    u      = float(0.0)
    v      = float(0.0)
    sign   = float(0.0)
    nrm    = wp.vec3(0.0, 0.0, 0.0)
    face   = int(0)
    if wp.mesh_query_ray(static_mesh, origin, world_d, max_dist,
                         t_mesh, u, v, sign, nrm, face):
        if t_mesh < best_t and t_mesh >= min_dist:
            best_t = t_mesh

    # ── 2. Obstacle axis-aligned boxes (slab method) ────────────────────
    #    obs_pos is flattened (n_envs*N_OBS,); index by env_id*n_obs + i
    obs_offset = env_id * n_obs
    for i in range(n_obs):
        bmin = obs_pos[obs_offset + i] - obs_half[i]
        bmax = obs_pos[obs_offset + i] + obs_half[i]

        t_min = float(-1e30)
        t_max = float( 1e30)

        # X slab
        if wp.abs(world_d[0]) < 1e-7:
            if origin[0] < bmin[0] or origin[0] > bmax[0]:
                continue
        else:
            t1 = (bmin[0] - origin[0]) / world_d[0]
            t2 = (bmax[0] - origin[0]) / world_d[0]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = wp.max(t_min, t1)
            t_max = wp.min(t_max, t2)

        # Y slab
        if wp.abs(world_d[1]) < 1e-7:
            if origin[1] < bmin[1] or origin[1] > bmax[1]:
                continue
        else:
            t1 = (bmin[1] - origin[1]) / world_d[1]
            t2 = (bmax[1] - origin[1]) / world_d[1]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = wp.max(t_min, t1)
            t_max = wp.min(t_max, t2)

        # Z slab
        if wp.abs(world_d[2]) < 1e-7:
            if origin[2] < bmin[2] or origin[2] > bmax[2]:
                continue
        else:
            t1 = (bmin[2] - origin[2]) / world_d[2]
            t2 = (bmax[2] - origin[2]) / world_d[2]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = wp.max(t_min, t1)
            t_max = wp.min(t_max, t2)

        if t_max >= t_min and t_min >= min_dist and t_min < best_t:
            best_t = t_min

    depth_out[tid] = best_t


@wp.kernel
def noise_kernel(
    depth:        wp.array(dtype=float),
    noise_std:    float,
    dropout_rate: float,
    quant_step:   float,
    seed:         int,
    depth_out:    wp.array(dtype=float),
):
    """Apply per-pixel depth noise (Gaussian + dropout + quantization)."""
    tid = wp.tid()
    d   = depth[tid]
    rng = wp.rand_init(seed, tid)
    # Dropout
    if wp.randf(rng) < dropout_rate:
        depth_out[tid] = float(0.0)
        return
    # Gaussian noise proportional to distance
    sigma = noise_std * d
    d = d + sigma * wp.randn(rng)
    # Quantize
    if quant_step > 0.0:
        d = wp.round(d / quant_step) * quant_step
    depth_out[tid] = wp.clamp(d, float(0.0), float(10.0))


# ════════════════════════════════════════════════════════════════════════════
# STATIC SCENE MESH BUILDER
# Floor + 4 walls as a triangle mesh for Warp BVH
# ════════════════════════════════════════════════════════════════════════════

def _build_static_mesh() -> wp.Mesh:
    """Return Warp mesh for floor plane + 4 walls (triangulated boxes)."""

    def box_verts_faces(cx, cy, cz, sx, sy, sz, vert_offset):
        """8 verts + 12 tris for one box."""
        vs = np.array([
            [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz],
            [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
            [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz],
            [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz],
        ], dtype=np.float32)
        fs = np.array([
            [0,1,2],[0,2,3], [4,5,6],[4,6,7],
            [0,1,5],[0,5,4], [2,3,7],[2,7,6],
            [0,3,7],[0,7,4], [1,2,6],[1,6,5],
        ], dtype=np.int32) + vert_offset
        return vs, fs

    all_verts, all_faces = [], []
    offset = 0
    # floor (thin box)
    v, f = box_verts_faces(0, 0, -0.05, 15, 15, 0.05, offset); all_verts.append(v); all_faces.append(f); offset += 8
    # walls
    v, f = box_verts_faces( 0,  5, 1.5, 5, 0.1, 1.5, offset); all_verts.append(v); all_faces.append(f); offset += 8
    v, f = box_verts_faces( 0, -5, 1.5, 5, 0.1, 1.5, offset); all_verts.append(v); all_faces.append(f); offset += 8
    v, f = box_verts_faces( 5,  0, 1.5, 0.1, 5, 1.5, offset); all_verts.append(v); all_faces.append(f); offset += 8
    v, f = box_verts_faces(-5,  0, 1.5, 0.1, 5, 1.5, offset); all_verts.append(v); all_faces.append(f); offset += 8

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0).flatten()
    return wp.Mesh(
        points  = wp.array(verts, dtype=wp.vec3),
        indices = wp.array(faces, dtype=int),
    )


# ════════════════════════════════════════════════════════════════════════════
# OBSTACLE HALF-SIZES — fixed per XML geom sizes
# ════════════════════════════════════════════════════════════════════════════

_OBS_HALF_SIZES = np.array([
    [0.30,0.30,0.50],[0.25,0.40,0.40],[0.20,0.20,0.80],[0.40,0.20,0.45],
    [0.35,0.35,0.35],[0.30,0.30,0.50],[0.50,0.20,0.60],[0.20,0.50,0.40],
    [0.40,0.40,0.30],[0.30,0.25,0.70],[0.20,0.30,0.50],[0.35,0.35,0.40],
    [0.45,0.20,0.45],[0.30,0.40,0.30],[0.25,0.25,0.60],[0.30,0.30,0.50],
    [0.40,0.30,0.35],[0.20,0.40,0.55],[0.35,0.20,0.40],[0.30,0.30,0.50],
    [0.40,0.40,0.40],[0.25,0.35,0.45],[0.30,0.25,0.50],[0.45,0.30,0.35],
    [0.20,0.20,0.70],
    # dynamic (cylinders approximated as boxes)
    [0.25,0.25,0.85],[0.25,0.25,0.85],[0.25,0.25,0.85],
    [0.25,0.25,0.85],[0.25,0.25,0.85],
    # humanoid (human_0) — AABB centred at torso mocap origin (z=1.0)
    # covers x: ±0.30, y: ±0.30, z: ±1.00  →  world z: 0.0 – 2.0
    [0.30,0.30,1.00],
], dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════
# MAIN RENDERER CLASS
# ════════════════════════════════════════════════════════════════════════════

class WarpDepthRenderer:
    """
    GPU batch depth renderer for N_CAMS cameras across n_envs environments.

    Usage (called from mjx_nav_env.py each step):
        renderer = WarpDepthRenderer(n_envs=4096)
        # After MJX step:
        depth = renderer.render(
            site_xpos,   # JAX (n_envs, N_CAMS, 3)
            site_xmat,   # JAX (n_envs, N_CAMS, 9)
            mocap_pos,   # JAX (n_envs, N_OBS, 3)
        )
        # depth: JAX (n_envs, N_CAMS, H, W) float32
    """

    def __init__(self, n_envs: int, noise_enabled: bool = True,
                 noise_std: float = 0.005, dropout_rate: float = 0.02,
                 quant_step: float = 0.001):
        wp.init()
        self.n_envs   = n_envs
        self.n_cams   = N_CAMS
        self.noise_en = noise_enabled
        self.noise_std    = noise_std
        self.dropout_rate = dropout_rate
        self.quant_step   = quant_step
        self._step = 0  # used for noise seed

        n_total_cams = n_envs * N_CAMS
        n_total_pix  = n_total_cams * CAM_H * CAM_W

        # ── Static scene mesh (floor + walls) ─────────────────────────
        self._static_mesh = _build_static_mesh()

        # ── Pre-built local ray directions ────────────────────────────
        local_dirs_np = _build_ray_dirs().reshape(-1, 3)   # (H*W, 3)
        self._local_dirs = wp.array(local_dirs_np, dtype=wp.vec3)

        # ── Obstacle half-sizes (fixed) ───────────────────────────────
        self._obs_half = wp.array(_OBS_HALF_SIZES, dtype=wp.vec3)

        # ── Persistent Warp output buffers ────────────────────────────
        self._cam_pos_buf  = wp.zeros(n_total_cams, dtype=wp.vec3)
        self._cam_R_buf    = wp.zeros(n_total_cams, dtype=wp.mat33)
        self._obs_pos_buf  = wp.zeros(n_envs * N_OBS, dtype=wp.vec3)
        self._depth_buf    = wp.zeros(n_total_pix,  dtype=float)
        self._noisy_buf    = wp.zeros(n_total_pix,  dtype=float)

        # ── Camera site indices in MJX (set by env after model load) ──
        self.cam_site_ids: list[int] = []   # filled by mjx_nav_env

    # ------------------------------------------------------------------ #
    def _jax_to_wp_vec3(self, jax_arr) -> wp.array:
        """Zero-copy JAX float32 (N,3) → Warp vec3 array via DLPack."""
        arr = jax_arr.astype(jnp.float32)
        # JAX >=0.7 removed jax.dlpack.to_dlpack; Warp can consume JAX arrays
        # directly via the __dlpack__ protocol.
        try:
            t = wp.from_dlpack(arr)
        except TypeError:
            if not hasattr(jax.dlpack, "to_dlpack"):
                raise
            t = wp.from_dlpack(jax.dlpack.to_dlpack(arr))
        return t.view(wp.vec3)

    def _jax_to_wp_mat33(self, jax_arr) -> wp.array:
        """Zero-copy JAX float32 (N,9) → Warp mat33 array."""
        arr = jax_arr.astype(jnp.float32).reshape(-1, 3, 3)  # mat33 needs (N,3,3) not (N,9)
        try:
            t = wp.from_dlpack(arr)
        except TypeError:
            if not hasattr(jax.dlpack, "to_dlpack"):
                raise
            t = wp.from_dlpack(jax.dlpack.to_dlpack(arr))
        return t.view(wp.mat33)

    # ------------------------------------------------------------------ #
    def render(self, site_xpos, site_xmat, mocap_pos):
        """
        Args:
            site_xpos: JAX (n_envs, N_CAMS, 3) — camera world positions
            site_xmat: JAX (n_envs, N_CAMS, 9) — camera rotation matrices (row-major)
            mocap_pos: JAX (n_envs, N_OBS, 3)  — obstacle positions

        Returns:
            JAX (n_envs, N_CAMS, H, W) float32 depth images
        """
        ne = self.n_envs
        # Flatten: (n_envs, N_CAMS, 3) → (n_envs*N_CAMS, 3)
        cam_pos_flat = site_xpos.reshape(ne * N_CAMS, 3)
        cam_R_flat   = site_xmat.reshape(ne * N_CAMS, 9)  # mat33 in warp is row-major

        # Transfer to Warp (zero-copy via DLPack)
        cam_pos_wp = self._jax_to_wp_vec3(cam_pos_flat)
        cam_R_wp   = self._jax_to_wp_mat33(cam_R_flat)

        # Flatten per-env obstacles: (n_envs, N_OBS, 3) → (n_envs*N_OBS, 3)
        obs_pos_flat = mocap_pos.reshape(ne * N_OBS, 3)
        obs_pos_wp   = self._jax_to_wp_vec3(obs_pos_flat)

        total_pix       = ne * N_CAMS * CAM_H * CAM_W
        pixels_per_cam  = CAM_H * CAM_W

        # ── Raycast kernel ─────────────────────────────────────────────
        wp.launch(
            kernel = raycast_kernel,
            dim    = total_pix,
            inputs = [
                cam_pos_wp, cam_R_wp, self._local_dirs,
                self._static_mesh.id,
                obs_pos_wp, self._obs_half, N_OBS,
                N_CAMS, pixels_per_cam,
                float(MIN_D), float(MAX_D),
                self._depth_buf,
            ],
        )

        out_buf = self._depth_buf

        # ── Noise kernel ───────────────────────────────────────────────
        if self.noise_en:
            wp.launch(
                kernel = noise_kernel,
                dim    = total_pix,
                inputs = [
                    self._depth_buf,
                    float(self.noise_std),
                    float(self.dropout_rate),
                    float(self.quant_step),
                    self._step,
                    self._noisy_buf,
                ],
            )
            out_buf = self._noisy_buf
        self._step += 1

        # ── Back to JAX via DLPack ─────────────────────────────────────
        try:
            depth_jax = jax.dlpack.from_dlpack(out_buf)
        except TypeError:
            depth_jax = jax.dlpack.from_dlpack(wp.to_dlpack(out_buf))
        return depth_jax.reshape(ne, N_CAMS, CAM_H, CAM_W)
