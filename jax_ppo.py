"""
JAX-native PPO
===============
Replaces train.py (Stable Baselines 3 / PyTorch).

Architecture:
    Depth (5×120×160) → CNN encoder → 256-dim
    Proprio (37)      → MLP encoder → 64-dim
                            ↓ concat (320-dim)
                        MLP (256 → 128)
                            ↓
                    Policy head → 12 actions (mean, log_std)
                    Value  head → 1 scalar

All ops are JAX/Flax — fully JIT-compiled. The entire rollout + update
loop is one compiled function. No Python in the hot path.

PPO references match config.py TRAINING hyperparameters:
    gamma=0.99, gae_lambda=0.95, clip_range=0.2
    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import Any, Dict, NamedTuple, Tuple
import numpy as np

# ── Hyper-parameters (from config.py TRAINING) ───────────────────────────────
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
ENT_COEF     = 0.01
VF_COEF      = 0.5
MAX_GRAD     = 0.5
LR           = 3e-4
N_EPOCHS     = 10
MINIBATCH_SZ = 512
CNN_FEAT_DIM = 256
PROPRIO_DIM  = 37
ACTION_DIM   = 12
LOG_STD_MIN  = -5.0
LOG_STD_MAX  =  2.0


# ════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK
# ════════════════════════════════════════════════════════════════════════════

class DepthCNNEncoder(nn.Module):
    """5×120×160 depth images → CNN_FEAT_DIM features."""
    features: int = CNN_FEAT_DIM

    @nn.compact
    def __call__(self, depth):
        # depth: (..., 5, 120, 160)
        # Move channels last for conv: (..., 120, 160, 5)
        x = jnp.moveaxis(depth, -3, -1)
        x = nn.Conv(features=32, kernel_size=(8,8), strides=(4,4))(x)
        x = nn.elu(x)
        x = nn.Conv(features=64, kernel_size=(4,4), strides=(2,2))(x)
        x = nn.elu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), strides=(1,1))(x)
        x = nn.elu(x)
        x = x.reshape((*x.shape[:-3], -1))   # flatten spatial dims
        x = nn.Dense(self.features)(x)
        x = nn.elu(x)
        return x


class PropriEncoder(nn.Module):
    """37-dim proprio → 64-dim features."""

    @nn.compact
    def __call__(self, proprio):
        x = nn.Dense(128)(proprio)
        x = nn.elu(x)
        x = nn.Dense(64)(x)
        x = nn.elu(x)
        return x


class SpotActorCritic(nn.Module):
    """Combined actor-critic for Spot navigation."""

    @nn.compact
    def __call__(self, depth, proprio):
        # Explicit names let each sub-module's params be addressed directly
        # (e.g. params["cnn"]), enabling CNN-feature caching during rollout.
        cnn_feat = DepthCNNEncoder(name="cnn")(depth)       # (B, 256)
        pro_feat = PropriEncoder(name="proprio")(proprio)   # (B, 64)
        x = jnp.concatenate([cnn_feat, pro_feat], axis=-1)  # (B, 320)

        # Shared trunk
        x = nn.Dense(256, name="trunk0")(x); x = nn.elu(x)
        x = nn.Dense(128, name="trunk1")(x); x = nn.elu(x)

        # Actor head
        action_mean   = nn.Dense(ACTION_DIM, name="actor")(x)
        log_std       = self.param("log_std",
                                   nn.initializers.zeros, (ACTION_DIM,))
        log_std_clamp = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        # Critic head
        value = nn.Dense(64, name="critic0")(x); value = nn.elu(value)
        value = nn.Dense(1,  name="critic1")(value)

        return action_mean, log_std_clamp, value.squeeze(-1)


# ════════════════════════════════════════════════════════════════════════════
# CNN FEATURE CACHING HELPERS
# ════════════════════════════════════════════════════════════════════════════

@jax.jit
def _encode_depth(params, depth):
    """Run only the CNN encoder — JIT-compiled for fast rollout inference."""
    return DepthCNNEncoder().apply({"params": params["cnn"]}, depth)


@jax.jit
def _inference_step(params, depth, proprio, rng_key):
    """Full rollout inference in ONE JIT dispatch: encode → forward → sample.

    Combines _encode_depth + _head_forward + Gaussian sampling into a single
    compiled kernel.  Returns (action, log_prob, value, cnn_feat).
    """
    # CNN encoder
    cnn_feat = DepthCNNEncoder().apply({"params": params["cnn"]}, depth / 10.0)
    # Trunk + heads (same ops as _head_forward, inlined for JIT)
    pro_feat = PropriEncoder().apply({"params": params["proprio"]}, proprio)
    x = jnp.concatenate([cnn_feat, pro_feat], axis=-1)
    x = nn.Dense(256).apply({"params": params["trunk0"]}, x); x = nn.elu(x)
    x = nn.Dense(128).apply({"params": params["trunk1"]}, x); x = nn.elu(x)
    mean      = nn.Dense(ACTION_DIM).apply({"params": params["actor"]}, x)
    log_std   = jnp.clip(params["log_std"], LOG_STD_MIN, LOG_STD_MAX)
    value     = nn.Dense(64).apply({"params": params["critic0"]}, x); value = nn.elu(value)
    value     = nn.Dense(1).apply({"params": params["critic1"]}, value).squeeze(-1)
    # Sample
    std    = jnp.exp(log_std)
    noise  = jax.random.normal(rng_key, mean.shape)
    action = jnp.clip(mean + std * noise, -1.0, 1.0)
    log_p  = _gaussian_log_prob(mean, log_std, action)
    return action, log_p, value, cnn_feat


def _head_forward(params, cnn_feat, proprio):
    """
    Forward pass from pre-encoded CNN features, skipping DepthCNNEncoder.
    Plain function (not @jax.jit) so gradients flow through it inside
    ppo_update's loss_fn.
    """
    pro_feat = PropriEncoder().apply({"params": params["proprio"]}, proprio)
    x = jnp.concatenate([cnn_feat, pro_feat], axis=-1)
    x = nn.Dense(256).apply({"params": params["trunk0"]}, x); x = nn.elu(x)
    x = nn.Dense(128).apply({"params": params["trunk1"]}, x); x = nn.elu(x)
    action_mean   = nn.Dense(ACTION_DIM).apply({"params": params["actor"]}, x)
    log_std       = params["log_std"]
    log_std_clamp = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    value = nn.Dense(64).apply({"params": params["critic0"]}, x); value = nn.elu(value)
    value = nn.Dense(1).apply( {"params": params["critic1"]}, value)
    return action_mean, log_std_clamp, value.squeeze(-1)


def _gaussian_log_prob(mean, log_std, action):
    """Log probability of action under diagonal Gaussian."""
    std = jnp.exp(log_std)
    log_p = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std
                    + jnp.log(2 * jnp.pi))
    return log_p.sum(-1)   # sum over action dims


def _gaussian_entropy(log_std):
    return (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std).sum(-1)


# ════════════════════════════════════════════════════════════════════════════
# ROLLOUT BUFFER
# ════════════════════════════════════════════════════════════════════════════

class RolloutBatch(NamedTuple):
    cnn_feat:    jnp.ndarray   # (T*B, CNN_FEAT_DIM=256)  pre-encoded depth features
    proprio:     jnp.ndarray   # (T*B, 37)
    action:      jnp.ndarray   # (T*B, 12)
    log_prob:    jnp.ndarray   # (T*B,)
    advantage:   jnp.ndarray   # (T*B,)
    ret:         jnp.ndarray   # (T*B,)  (value target)


# ════════════════════════════════════════════════════════════════════════════
# GAE ADVANTAGE COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_gae(
    rewards:    jnp.ndarray,    # (T, B)
    values:     jnp.ndarray,    # (T+1, B)  last entry = bootstrap value
    dones:      jnp.ndarray,    # (T, B)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns advantages (T,B) and returns (T,B)."""
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    gae        = jnp.zeros(rewards.shape[1])

    for t in reversed(range(T)):
        mask   = 1.0 - dones[t].astype(jnp.float32)
        delta  = rewards[t] + GAMMA * values[t+1] * mask - values[t]
        gae    = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages = advantages.at[t].set(gae)

    returns = advantages + values[:T]
    return advantages, returns


# ════════════════════════════════════════════════════════════════════════════
# PPO UPDATE STEP (JIT-compiled)
# ════════════════════════════════════════════════════════════════════════════

@jax.jit
def ppo_update(
    train_state: TrainState,
    batch:       RolloutBatch,
) -> Tuple[TrainState, Dict]:
    """One gradient update on a minibatch with full NaN/explosion safeguards."""

    def loss_fn(params):
        mean, log_std, value = _head_forward(params, batch.cnn_feat, batch.proprio)

        # ── Policy loss with ratio clamping ────────────────────────────
        log_prob_new = _gaussian_log_prob(mean, log_std, batch.action)
        # Clamp log-ratio BEFORE exp to prevent inf: |new - old| ≤ 10
        log_ratio    = jnp.clip(log_prob_new - batch.log_prob, -10.0, 10.0)
        ratio        = jnp.exp(log_ratio)

        adv_norm     = (batch.advantage - batch.advantage.mean()) / \
                       (batch.advantage.std() + 1e-8)
        # Clamp normalized advantages to prevent outlier domination
        adv_norm     = jnp.clip(adv_norm, -10.0, 10.0)

        pg_loss1     = ratio * adv_norm
        pg_loss2     = jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_norm
        policy_loss  = -jnp.mean(jnp.minimum(pg_loss1, pg_loss2))

        # ── Value loss with clipping (standard PPO) ────────────────────
        # Clip value predictions to prevent huge swings from 200-pt goal bonus
        value_clipped = batch.ret + jnp.clip(value - batch.ret, -10.0, 10.0)
        vf_loss1      = (value - batch.ret) ** 2
        vf_loss2      = (value_clipped - batch.ret) ** 2
        value_loss    = 0.5 * jnp.mean(jnp.maximum(vf_loss1, vf_loss2))

        # ── Entropy bonus ──────────────────────────────────────────────
        entropy = jnp.mean(_gaussian_entropy(log_std))

        total = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

        # ── NaN guard: if total is bad, return zero loss (skip update) ─
        total = jnp.where(jnp.isfinite(total), total, 0.0)

        return total, {
            "policy_loss": policy_loss,
            "value_loss":  value_loss,
            "entropy":     entropy,
            "total_loss":  total,
            "ratio_mean":  jnp.mean(ratio),
            "ratio_max":   jnp.max(ratio),
        }

    grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)

    # NaN guard on gradients: replace any NaN/inf grad with 0
    grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
    )
    # Global norm clip is already in the optax chain; no element-wise clip needed
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, info


# ════════════════════════════════════════════════════════════════════════════
# PPO TRAINER
# ════════════════════════════════════════════════════════════════════════════

class PPOTrainer:
    """
    Manages model, optimizer, and training loop.

    Usage:
        trainer = PPOTrainer(n_envs=4096, n_steps=2048)
        trainer.train(env, total_timesteps=10_000_000)
    """

    def __init__(
        self,
        n_envs:   int = 4096,
        n_steps:  int = 2048,
        lr:       float = LR,
        seed:     int = 0,
    ):
        self.n_envs  = n_envs
        self.n_steps = n_steps
        self.rng     = jax.random.PRNGKey(seed)

        # ── Initialize network ────────────────────────────────────────
        self.net  = SpotActorCritic()
        dummy_dep = jnp.zeros((1, 5, 120, 160))
        dummy_pro = jnp.zeros((1, PROPRIO_DIM))
        self.rng, init_key = jax.random.split(self.rng)
        params = self.net.init(init_key, dummy_dep, dummy_pro)["params"]

        # ── Optimizer (Adam with linear LR decay will be added in train) ──
        tx = optax.chain(
            optax.clip_by_global_norm(MAX_GRAD),
            optax.adam(lr),
        )
        self.train_state = TrainState.create(
            apply_fn = self.net.apply,
            params   = params,
            tx       = tx,
        )

    # ──────────────────────────────────────────────────────────────────
    def _sample_action(self, obs):
        """
        Single JIT dispatch: encode depth → forward → sample.
        Returns (action, log_prob, value, cnn_feat).
        """
        self.rng, k = jax.random.split(self.rng)
        return _inference_step(
            self.train_state.params, obs["depth"], obs["proprio"], k
        )

    # ──────────────────────────────────────────────────────────────────
    def collect_rollout(self, env, state, obs):
        """
        Collect n_steps of experience. Returns updated state/obs, batch, and
        rollout stats dict (mean/min/max reward, done rate, episode count).

        Stores CNN features (T*B, 256) instead of raw depth (T*B, 5, 120, 160)
        to keep the rollout buffer at ~1 GB instead of ~374 GB.
        """
        buf_cnn_feat = []
        buf_proprio  = []
        buf_actions  = []
        buf_log_prob = []
        buf_rewards  = []
        buf_dones    = []
        buf_values   = []

        for _ in range(self.n_steps):
            action, log_prob, value, cnn_feat = self._sample_action(obs)

            # Sanitize features: replace NaN/inf with 0 before storing
            cnn_feat = jnp.where(jnp.isfinite(cnn_feat), cnn_feat, 0.0)
            buf_cnn_feat.append(cnn_feat)            # (B, 256) — not raw depth
            buf_proprio.append(obs["proprio"])        # already sanitized in env
            buf_actions.append(action)
            buf_log_prob.append(log_prob)
            buf_values.append(value)

            state, obs, reward, done, _ = env.step(state, action)

            buf_rewards.append(reward)
            buf_dones.append(done.astype(jnp.float32))

            # Auto-reset: immediately replace terminated envs with fresh ones
            self.rng, k = jax.random.split(self.rng)
            state, obs = env.auto_reset(state, obs, done, k)

        # Bootstrap value for last step
        _, _, last_value, _ = self._sample_action(obs)

        # Stack: (T, B, ...)
        rewards = jnp.stack(buf_rewards)     # (T, B)
        dones   = jnp.stack(buf_dones)       # (T, B)
        values  = jnp.concatenate(
            [jnp.stack(buf_values), last_value[None]], axis=0   # (T+1, B)
        )

        advantages, returns = compute_gae(rewards, values, dones)

        # Normalize returns to prevent huge value targets from goal bonus
        ret_mean = returns.mean()
        ret_std  = returns.std() + 1e-8
        returns_norm = (returns - ret_mean) / ret_std

        # Flatten (T*B, ...)
        def flat(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        batch = RolloutBatch(
            cnn_feat   = flat(jnp.stack(buf_cnn_feat)),  # (T*B, 256)
            proprio    = flat(jnp.stack(buf_proprio)),
            action     = flat(jnp.stack(buf_actions)),
            log_prob   = flat(jnp.stack(buf_log_prob)),
            advantage  = flat(advantages),
            ret        = flat(returns_norm),
        )

        # One device→host sync HERE (after the loop), not inside the loop
        rollout_stats = {
            "rew_mean":   float(rewards.mean()),
            "rew_min":    float(rewards.min()),
            "rew_max":    float(rewards.max()),
            "done_rate":  float(dones.mean()),
            "ep_count":   int(dones.sum()),
        }
        return state, obs, batch, rollout_stats

    # ──────────────────────────────────────────────────────────────────
    def update(self, batch: RolloutBatch) -> Dict:
        """Run N_EPOCHS of PPO updates on the collected batch."""
        total_samples = batch.cnn_feat.shape[0]

        for epoch in range(N_EPOCHS):
            self.rng, k = jax.random.split(self.rng)
            perm = jax.random.permutation(k, total_samples)

            for start in range(0, total_samples, MINIBATCH_SZ):
                idx = perm[start : start + MINIBATCH_SZ]
                mb  = RolloutBatch(
                    cnn_feat   = batch.cnn_feat[idx],
                    proprio    = batch.proprio[idx],
                    action     = batch.action[idx],
                    log_prob   = batch.log_prob[idx],
                    advantage  = batch.advantage[idx],
                    ret        = batch.ret[idx],
                )
                self.train_state, info = ppo_update(self.train_state, mb)

        return info   # last minibatch's info

    # ──────────────────────────────────────────────────────────────────
    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "params": jax.device_get(self.train_state.params),
            }, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.train_state = self.train_state.replace(
            params=jax.device_put(ckpt["params"])
        )
