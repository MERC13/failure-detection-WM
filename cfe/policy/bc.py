"""BC policy: frozen DINOv2 features + proprioception -> MLP -> action chunk (SPEC.md §4.3).

This policy exists to *generate failures*. SPEC.md §4.3 was amended 2026-07-31 to gate on
failure supply and a mode breakdown rather than a 40-70% success band (see CHANGELOG.md).
Nothing here is tuned toward a gate: whatever the policy measures is reported as-is
(CLAUDE.md rule 3).

Two data facts drive the design, both measured rather than assumed:

- **The gripper dimension is strictly binary.** Dim 6 takes exactly {-1, +1} across the demos,
  so it is a classification target with a BCE loss. CLAUDE.md forbids applying a symmetric L2
  penalty to it as though it were continuous like dims 0-5.
- **The six delta dimensions differ in scale by ~20x** (translation std 0.22-0.49, rotation
  std 0.02-0.05). Without per-dimension target normalisation the rotation dims contribute
  almost nothing to the loss and the policy ignores them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from cfe.data.demos import CAMERA_KEY, proprio_vector
from cfe.encoders.dinov2 import pool

GRIPPER_DIM = 6
N_DELTA = 6
ACTION_DIM = 7


@dataclass(frozen=True)
class PolicyConfig:
    chunk_size: int
    hidden_sizes: tuple[int, ...]
    dropout: float
    include_cls_in_pool: bool
    use_proprio: bool

    @classmethod
    def from_dict(cls, d: dict) -> PolicyConfig:
        return cls(
            chunk_size=int(d["chunk_size"]),
            hidden_sizes=tuple(int(h) for h in d["hidden_sizes"]),
            dropout=float(d["dropout"]),
            include_cls_in_pool=bool(d["include_cls_in_pool"]),
            use_proprio=bool(d["use_proprio"]),
        )


@dataclass
class Normalizer:
    """Standardises inputs and the six delta targets. Dim 6 is never normalised."""

    x_mean: np.ndarray
    x_std: np.ndarray
    d_mean: np.ndarray
    d_std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, deltas: np.ndarray) -> Normalizer:
        flat = deltas.reshape(-1, N_DELTA)
        return cls(
            x_mean=x.mean(0),
            # Guard against a constant feature producing a divide-by-zero.
            x_std=np.maximum(x.std(0), 1e-6),
            d_mean=flat.mean(0),
            d_std=np.maximum(flat.std(0), 1e-6),
        )

    def norm_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def norm_delta(self, d: np.ndarray) -> np.ndarray:
        return (d - self.d_mean) / self.d_std

    def denorm_delta(self, d: np.ndarray) -> np.ndarray:
        return d * self.d_std + self.d_mean

    def state_dict(self) -> dict:
        return {
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "d_mean": self.d_mean,
            "d_std": self.d_std,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> Normalizer:
        return cls(**{k: np.asarray(v) for k, v in d.items()})


class BCPolicy(nn.Module):
    """MLP head over frozen features. Outputs `(B, H, 7)`.

    Channels 0-5 are *normalised* deltas; channel 6 is a gripper **logit**, not an action.
    Converting to an executable action is `to_action()`, kept explicit so the binary
    treatment of the gripper is visible at the call site.
    """

    def __init__(self, input_dim: int, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        dims = [input_dim, *cfg.hidden_sizes]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(cfg.dropout)]
        layers.append(nn.Linear(dims[-1], cfg.chunk_size * ACTION_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.cfg.chunk_size, ACTION_DIM)


def chunk_loss(
    out: torch.Tensor,
    target_delta: torch.Tensor,
    target_grip: torch.Tensor,
    gripper_weight: float = 1.0,
    delta_loss: str = "smooth_l1",
) -> tuple[torch.Tensor, dict]:
    """Split loss: regression on dims 0-5, binary classification on dim 6."""
    pred_delta, pred_logit = out[..., :N_DELTA], out[..., GRIPPER_DIM]
    if delta_loss == "smooth_l1":
        l_delta = F.smooth_l1_loss(pred_delta, target_delta)
    elif delta_loss == "l2":
        l_delta = F.mse_loss(pred_delta, target_delta)
    else:
        raise ValueError(f"unknown delta_loss {delta_loss!r}")
    l_grip = F.binary_cross_entropy_with_logits(pred_logit, target_grip)
    total = l_delta + gripper_weight * l_grip
    return total, {
        "loss": float(total.detach()),
        "delta": float(l_delta.detach()),
        "gripper": float(l_grip.detach()),
    }


def to_action(out: np.ndarray, norm: Normalizer) -> np.ndarray:
    """`(H, 7)` network output -> executable `(H, 7)` actions.

    Deltas are denormalised and clipped to the env's [-1, 1] range; the gripper logit becomes
    a hard -1/+1, matching the only two values the demos ever contain.
    """
    deltas = np.clip(norm.denorm_delta(out[..., :N_DELTA]), -1.0, 1.0)
    grip = np.where(out[..., GRIPPER_DIM] > 0.0, 1.0, -1.0)[..., None]
    return np.concatenate([deltas, grip], axis=-1).astype(np.float64)


# ---------------------------------------------------------------------------
# Dataset assembly from the encoding cache
# ---------------------------------------------------------------------------


def episode_features(cache, episode_id: str, include_cls: bool, use_proprio: bool) -> dict:
    """Pooled latents + proprio + action chunks for one episode.

    `pool()` is called here, at the point of use, rather than inside the encoder — SPEC.md
    §4.1 keeps tokens canonical and CLAUDE.md asks that the pooling choice stay visible.
    """
    tokens = np.asarray(cache.read(episode_id))  # (T+1, N, D) fp16, memory-mapped
    pooled = pool(tokens.astype(np.float32), include_cls=include_cls)  # (T+1, D)
    meta = cache.read_meta(episode_id)
    actions = meta["actions"].astype(np.float32)  # (T, 7)
    proprio = meta["proprio"].astype(np.float32)  # (T+1, P)

    n = len(actions)
    if len(pooled) < n:
        raise ValueError(f"{episode_id}: {len(pooled)} frames for {n} actions")

    feats = pooled[:n]
    if use_proprio:
        feats = np.concatenate([feats, proprio[:n]], axis=1)
    return {"x": feats, "actions": actions}


def build_arrays(cache, episode_ids: list[str], cfg: PolicyConfig) -> dict:
    """Stack per-episode data into flat training arrays.

    Chunks that run past the end of an episode repeat the final action, the usual convention:
    the alternative (dropping the last H-1 steps) throws away the place/release phase, which
    is exactly where failures happen.
    """
    xs, deltas, grips, groups = [], [], [], []
    for j, eid in enumerate(episode_ids):
        ep = episode_features(cache, eid, cfg.include_cls_in_pool, cfg.use_proprio)
        a, n = ep["actions"], len(ep["actions"])
        idx = np.minimum(np.arange(n)[:, None] + np.arange(cfg.chunk_size)[None, :], n - 1)
        chunks = a[idx]  # (n, H, 7)
        xs.append(ep["x"])
        deltas.append(chunks[..., :N_DELTA])
        # {-1,+1} -> {0,1} for BCE.
        grips.append((chunks[..., GRIPPER_DIM] > 0).astype(np.float32))
        groups.append(np.full(n, j, dtype=np.int64))
    return {
        "x": np.concatenate(xs).astype(np.float32),
        "delta": np.concatenate(deltas).astype(np.float32),
        "grip": np.concatenate(grips).astype(np.float32),
        "episode": np.concatenate(groups),
    }


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


#: An object counts as disturbed if it moves this far *horizontally* (metres).
#:
#: Horizontal, not 3D, because objects settle under gravity at reset: measured over 120 idle
#: steps with a null action, peak vertical drift is 0.1062 m while peak horizontal drift is
#: 0.0010 m (both identical across six init states — it is deterministic settling, not the
#: robot). A 3D threshold would mark every episode as "object disturbed" before the arm moved.
#: 0.01 m sits 10x above idle horizontal drift and ~50x below the ~0.5 m an object travels when
#: actually carried to the basket, so it cannot fire spuriously or miss a real manipulation.
CONTACT_EPS = 0.01


def object_position_keys(obs_keys) -> list[str]:
    """Observation keys holding an absolute non-robot body position.

    Excludes `<object>_to_robot0_eef_pos`, which is object-minus-end-effector. Those keys move
    whenever the *arm* moves, so including them reports "the object was disturbed" for every
    episode in which the robot did anything at all.
    """
    return [
        k
        for k in obs_keys
        if k.endswith("_pos") and not k.startswith("robot0") and "_to_robot0" not in k
    ]


def object_positions(obs: dict) -> dict[str, np.ndarray]:
    """Absolute positions of every non-robot body, task-agnostically."""
    return {k: np.asarray(obs[k], dtype=np.float64) for k in object_position_keys(obs)}


def find_target_object_key(obs_keys, language: str) -> str:
    """The position key of the object the task is about, from the task's language string.

    `libero_object` tasks read "pick up the <object> and place it in the basket", and the
    matching observation key is `<object_with_underscores>_1_pos`. Matching on the language is
    a little crude, but the alternative — hard-coding one task's object name — cannot survive
    the ≥3 tasks SPEC.md §4.4 requires.

    Raises rather than guessing. A wrong target silently turns the failure-mode breakdown into
    noise, which is how the first two versions of this metric went wrong.
    """
    lang = language.lower()
    candidates = [
        k
        for k in object_position_keys(obs_keys)
        # "alphabet_soup_1_pos" -> "alphabet soup"
        if k.rsplit("_", 2)[0].replace("_", " ") in lang
    ]
    # "basket" appears in the language too, but it is the goal, not the target.
    candidates = [k for k in candidates if not k.startswith("basket")]
    if len(candidates) != 1:
        raise ValueError(
            f"could not identify a unique target object for {language!r}; got {candidates}"
        )
    return candidates[0]


def classify_failure(target_disp: float, other_disp: float, timed_out: bool) -> str:
    """What kind of failure, in the terms Phase 3 needs to search over.

    SPEC.md §4.3 (amended 2026-07-31) requires a failure-mode breakdown alongside the success
    rate, because the rate alone did not distinguish a policy that misses the object entirely
    from one that grasps and drops it. Those need different edits.

    The target object is separated from everything else on purpose: nudging a distractor while
    never touching the target is an *approach* failure, and reporting it as "an object moved"
    hides precisely the distinction this metric exists to make.
    """
    if target_disp >= CONTACT_EPS:
        return "target_moved_not_placed" if timed_out else "target_moved_then_terminated"
    if other_disp >= CONTACT_EPS:
        return "distractor_only_target_untouched"
    return "no_object_contact"


def rollout(
    env,
    policy: BCPolicy,
    encoder,
    norm: Normalizer,
    init_state: np.ndarray,
    max_steps: int,
    device: torch.device,
    target_key: str,
    capture: bool = False,
) -> dict:
    """Run the policy closed-loop from an exact init state. Success comes from the simulator.

    Observations are encoded one frame at a time, whereas the training cache was built at
    `encoder.cfg.batch_size`. Encoding is not batch-invariant, but the discrepancy is ~1 fp16
    ULP (max 7.8e-3 against a token std of 1.78), far below anything a policy responds to.

    `capture=True` additionally returns `frames`, `proprio` and `actions`, matching
    `replay_episode`'s `T+1`-observations-for-`T`-actions contract so a rollout episode and a
    demo episode are the same shape of thing in the cache. The captured frames are *not* the
    per-step encodings computed above: callers re-encode them in batches of
    `encoder.cfg.batch_size`, which is what the cached demo latents were produced at and what
    the cache fingerprint hashes. Encoding them any other way would put two subtly different
    kinds of latent under one key.
    """
    env.reset()
    obs = env.set_init_state(init_state)
    policy.eval()

    frames = [obs[CAMERA_KEY].copy()] if capture else None
    proprio = [proprio_vector(obs)] if capture else None
    executed: list[np.ndarray] = []

    start_pos = object_positions(obs)
    if target_key not in start_pos:
        raise KeyError(f"target {target_key!r} not among object positions {sorted(start_pos)}")
    target_disp = 0.0
    other_disp = 0.0

    def note_displacement(o: dict) -> None:
        nonlocal target_disp, other_disp
        for k, p0 in start_pos.items():
            d = float(np.linalg.norm(np.asarray(o[k], np.float64)[:2] - p0[:2]))
            if k == target_key:
                target_disp = max(target_disp, d)
            else:
                other_disp = max(other_disp, d)

    def result(steps: int) -> dict:
        success = bool(env.check_success())
        timed_out = steps >= max_steps
        out = {
            "success": success,
            "steps": steps,
            "timed_out": timed_out,
            "target_displacement_xy": round(target_disp, 5),
            "other_object_displacement_xy": round(other_disp, 5),
            "failure_mode": (
                None if success else classify_failure(target_disp, other_disp, timed_out)
            ),
        }
        if capture:
            out |= {
                "frames": np.asarray(frames, dtype=np.uint8),
                "proprio": np.asarray(proprio, dtype=np.float32),
                "actions": np.asarray(executed, dtype=np.float32),
            }
        return out

    steps = 0
    with torch.inference_mode():
        while steps < max_steps:
            tokens = encoder.encode(obs[CAMERA_KEY][None])
            feat = pool(tokens.astype(np.float32), include_cls=policy.cfg.include_cls_in_pool)
            if policy.cfg.use_proprio:
                feat = np.concatenate([feat, proprio_vector(obs)[None].astype(np.float32)], axis=1)
            x = torch.from_numpy(norm.norm_x(feat).astype(np.float32)).to(device)
            chunk = to_action(policy(x)[0].cpu().numpy(), norm)

            for action in chunk:
                if steps >= max_steps:
                    break
                obs, _reward, done, _info = env.step(action)
                steps += 1
                note_displacement(obs)
                if capture:
                    executed.append(np.asarray(action, dtype=np.float32).copy())
                    frames.append(obs[CAMERA_KEY].copy())
                    proprio.append(proprio_vector(obs))
                if done or env.check_success():
                    return result(steps)
    return result(steps)
