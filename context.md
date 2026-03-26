# MJX Training System — Context File
*Generated: 2026-03-26*
*Branch with all fixes: `claude/debug-spot-training-0Ncu9`*

---

## Problem Summary

Training on branch `claude/sim-2-real-ex2` hung indefinitely at JIT compilation
of `jit__single_physics_step`. The compiled XLA graph was so large that LLVM
never finished optimizing it.

Root causes (by impact):

| # | Cause | Symptom | Fix |
|---|-------|---------|-----|
| 1 | `implicitfast` integrator | XLA encodes the iterative contact solver into the graph | Changed to `Euler` in `spot_scene.xml` |
| 2 | 9 mesh collision geoms | Mesh-mesh contact detection generates enormous XLA graph per pair | Replaced all 9 with primitive boxes/capsules |
| 3 | `compute_gae` Python for-loop | 512 sequential kernel dispatches per rollout step | Replaced with `jax.lax.scan` + `@jax.jit` |
| 4 | `reward_sums` chained additions | 5,621 chained JAX ops materialized at log time | Reverted to buffer-and-stack-once pattern |
| 5 | `auto_reset` on every step | ~30 GPU kernels fired even when no envs terminated | Added short-circuit: skip if `not jnp.any(terminated)` |

---

## Files Changed

### `models/spot_scene.xml`

**1. Integrator change (line 17-18):**
```xml
<!-- BEFORE -->
<option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"
        cone="elliptic" impratio="100"/>

<!-- AFTER — Euler is 5-10x simpler to compile -->
<option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"
        cone="elliptic" impratio="10"/>
```

**2. Contact dimensionality on foot geoms:**
```xml
<!-- BEFORE (condim=6 = torsional+rolling friction, ~2x more contact vars) -->
<default class="foot"><geom ... condim="6"/></default>

<!-- AFTER (condim=3 = tangential only) -->
<default class="foot"><geom ... condim="3"/></default>
```

**3. Mesh collision geoms → primitives (9 geoms total):**
```xml
<!-- BEFORE (causes huge XLA graph from mesh-mesh contact detection): -->
<geom name="body_collision" type="mesh" mesh="body_collision" class="collision"/>
<geom name="fl_upper_leg_collision" type="mesh" mesh="left_upper_leg_collision" class="collision"/>
<geom name="fl_lower_leg_collision" type="mesh" mesh="left_lower_leg_collision" class="collision"/>
<!-- ... repeated for fr, hl, hr legs (9 geoms total) -->

<!-- AFTER (primitive shapes, fast contact detection): -->
<geom type="box" size="0.32 0.1 0.08" pos="0 0 0" class="collision"/>          <!-- body -->
<geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.32" class="collision"/>   <!-- upper leg (per leg) -->
<geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.32" class="collision"/>   <!-- lower leg (per leg) -->
```

**4. Hind-right leg mesh bug fix (line ~318):**
```xml
<!-- BEFORE (wrong mesh — using left side for right side) -->
<geom name="hr_lower_leg_visual" type="mesh" mesh="rear_left_lower_leg"/>

<!-- AFTER -->
<geom name="hr_lower_leg_visual" type="mesh" mesh="rear_right_lower_leg"/>
```

---

### `mjx_nav_env.py`

**1. `render_interval` parameter — depth rendering every N steps:**
```python
def __init__(self, n_envs, render_interval=4, ...):
    self.render_interval = render_interval
    self._step_call_count = 0
    self._cached_depth = None

def step(self, state, action):
    ...
    proprio = self._get_proprio(new_state)
    self._step_call_count += 1
    if self._step_call_count % self.render_interval == 0:
        depth = self._render_depth(new_state)
        self._cached_depth = depth
    else:
        depth = self._cached_depth
    obs = {"proprio": proprio, "depth": depth}
```
Expected speedup: 2-4x rollout throughput (Warp rendering was the bottleneck).

**2. `auto_reset` short-circuit:**
```python
def auto_reset(self, state, obs, terminated, key):
    if not jnp.any(terminated):
        return state, obs   # skip ~30 GPU kernels when nothing terminated
    # ... full reset logic below
```

---

### `jax_ppo.py`

**1. `compute_gae` with `jax.lax.scan`:**
```python
@jax.jit
def compute_gae(rewards, values, dones, last_value=None, gamma=0.99, gae_lambda=0.95):
    """Uses jax.lax.scan (reverse) instead of a Python for-loop."""
    T, B = rewards.shape
    if last_value is None:
        last_value = jnp.zeros(B, dtype=jnp.float32)

    def _scan_fn(carry, t):
        gae, next_val = carry
        r, v, d = rewards[t], values[t], dones[t]
        delta = r + gamma * next_val * (1.0 - d) - v
        gae   = delta + gamma * gae_lambda * (1.0 - d) * gae
        return (gae, v), gae

    init_gae = jnp.zeros(B, dtype=jnp.float32)
    _, gae_rev = jax.lax.scan(_scan_fn, init_gae, jnp.arange(T))
    advantages = gae_rev[::-1]
    returns    = advantages + values
    return advantages, returns
```

**2. Reward buffer (buffer-and-stack-once):**
```python
# BEFORE — chained additions, 5,621 ops at log time:
reward_sums = {k: reward_sums[k] + v for k, v in step_info.items()}

# AFTER — buffer, then stack once per update:
buf_reward_info.append(step_info)
# ...at end of rollout:
for key in buf_reward_info[0]:
    vals = jnp.stack([info[key] for info in buf_reward_info])  # (T, B)
    metrics[key] = float(vals.mean())
```

---

### `mjx_train.py`

**1. XLA memory setting re-enabled:**
```python
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
```

**2. New CLI arguments:**
```
--warmup N          Force JIT compilation upfront with timing (default: 2)
--verbose_jit       Set JAX_LOG_COMPILES=1 (print every trace+compile)
--render_interval N Render depth every N steps (default: 4)
--profile N         Profile first N updates
```

**3. Warmup output format:**
```
[WARMUP] Forcing JIT compilation (this is one-time)...
  [1/5] reset()...                               0.5s
  [2/5] _inference_step (network forward)...     3.2s
  [3/5] env.step (physics + render + reward)...  Xs   ← was hanging here
  [4/5] auto_reset...                            0.3s
  [5/5] compute_gae...                           1.8s
[WARMUP] Complete in Xs. Training will now start.
```

---

### `render_spot_standing.py` (new file)

Script to render Spot in upright standing pose and save `spot_standing.png`.

```python
# Usage in Colab:
# !python render_spot_standing.py
# from IPython.display import Image, display
# display(Image("spot_standing.png"))
```

- Uses OSMesa (software renderer) — no display required
- Disables site group rendering to hide camera mount markers
- Output: 1280×960 PNG

---

## Running Training

### Quick smoke test (Colab T4/L4):
```bash
python mjx_train.py \
    --n_envs 64 \
    --n_steps 128 \
    --total_updates 3 \
    --warmup 2 \
    --render_interval 4
```

### Full training run (L4/A100):
```bash
python mjx_train.py \
    --n_envs 512 \
    --n_steps 1024 \
    --total_updates 500 \
    --render_interval 4
```

### Debug compilation issues:
```bash
python mjx_train.py --n_envs 32 --n_steps 64 --total_updates 1 \
    --warmup 2 --verbose_jit
```

### Expected compilation times after fixes (L4 GPU):
- `[1/5] reset()`: ~0.5s
- `[2/5] inference step`: ~3s
- `[3/5] env.step`: **<30s** (was 5+ minutes with mesh collision geoms)
- `[4/5] auto_reset`: ~0.3s
- `[5/5] compute_gae`: ~2s

---

## Branch History

| Branch | What's on it |
|--------|-------------|
| `claude/sim-2-real-ex2` | Original broken branch (hanging JIT) |
| `claude/debug-spot-training-0Ncu9` | **All fixes** — use this branch |
| `claude/sim-to-real-context-ZqCOq` | Context file only |

### Key commits on `claude/debug-spot-training-0Ncu9`:
```
01af53a  Replace mesh collision geoms with primitives for fast MJX JIT
aa91fe1  Skip Warp depth rendering 3/4 of steps for 2-4x rollout speedup
2f32290  Add JIT warmup profiling: --warmup and --verbose_jit flags
4e2fb89  Add script to render Spot standing upright and save image
a539f28  Fix physics model: Euler integrator, mesh bug, simpler contacts
291b082  Fix training hang: optimize auto_reset, GAE, and memory allocation
```

---

## Architecture Overview

```
mjx_train.py
    └── SpotMJXEnv (mjx_nav_env.py)
            ├── MuJoCo MJX physics   (GPU, batched via jax.vmap)
            ├── WarpDepthRenderer    (GPU, NVIDIA Warp + DLPack)
            ├── jax_reward.py        (GPU, fully jitted)
            └── config.py            (obstacle/humanoid parameters)
    └── PPOTrainer (jax_ppo.py)
            ├── Policy network       (MLP, JAX)
            ├── Value network        (MLP, JAX)
            ├── compute_gae()        (jax.lax.scan)
            └── ppo_update()         (jax.jit)
    └── models/spot_scene.xml        (MuJoCo model)
            ├── Euler integrator     (fast JIT)
            ├── Primitive collision  (fast contact detection)
            └── 5 depth cameras      (sites, rendered by Warp)
```

---

## Observation / Action Spaces

| Space | Shape | Description |
|-------|-------|-------------|
| `depth` | `(n_envs, 5, 120, 160)` | 5 depth cameras, float32, meters |
| `proprio` | `(n_envs, 37)` | Body state + joint pos/vel + goal vector |
| `action` | `(n_envs, 12)` | Normalized joint position targets in `[-1, 1]` |

---

## Pending / Not Yet Verified

1. **Confirm [3/5] compile time** — run with primitive collision geoms and measure whether `env.step` JIT compiles in <30s with 64 envs.
2. **Full training run** — once compilation is fast, test with `--n_envs 512 --n_steps 1024 --total_updates 100` and verify reward improves.
3. **Render-skipping speedup** — confirm `--render_interval 4` gives ~3x rollout steps/second vs `render_interval=1`.
4. **Potential future optimization** — reduce camera resolution to 60×80 during early training for another 2-4x speedup.
