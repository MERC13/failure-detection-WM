"""Turning cached episodes into dynamics training windows (SPEC.md §4.2).

Index convention, fixed once here because getting it wrong is silent: an episode has `T+1`
latents `z[0..T]` and `T` actions `a[0..T-1]`, where `a[i]` is the action executed *from* `z[i]`,
producing `z[i+1]`. A window starting at `t` therefore uses context `z[t-k+1 .. t]`, actions
`a[t .. t+nH-1]`, and targets `z[t+1 .. t+nH]`.

Latents are reduced and LayerNormed once, at load, and held in RAM as float16. Reduction first
because full patch tokens for this dataset would be ~13 GB; float16 because even `grid4` is
~860 MB in float32 on a 7 GB machine. The values stored are already unit-variance per token, so
float16's ~1e-3 relative resolution sits far below anything the smooth-L1 loss responds to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfe.dynamics.latents import layer_norm
from cfe.encoders.dinov2 import reduce_latents


@dataclass
class EpisodeLatents:
    episode_id: str
    z: np.ndarray  # (T+1, R, D) float16, reduced and LayerNormed
    actions: np.ndarray  # (T, 7) float32
    success: bool


def load_episode(cache, episode_id: str, reduction: str) -> EpisodeLatents:
    tokens = np.asarray(cache.read(episode_id), dtype=np.float32)  # (T+1, N, D)
    z = layer_norm(reduce_latents(tokens, reduction)).astype(np.float16)
    meta = cache.read_meta(episode_id)
    actions = meta["actions"].astype(np.float32)
    if len(z) != len(actions) + 1:
        raise ValueError(
            f"{episode_id}: {len(z)} latents for {len(actions)} actions; "
            f"expected T+1 latents for T actions"
        )
    return EpisodeLatents(episode_id, z, actions, bool(meta["success"]))


def window_starts(n_latents: int, k: int, h: int, ar_chunks: int) -> np.ndarray:
    """Valid start indices `t` for an episode with `n_latents` frames.

    `t >= k-1` so the context is real rather than padded, and `t + ar_chunks*h <= T` so every
    target frame exists. Padding the context would teach the model that episodes begin with
    repeated frames, which no real rollout does.
    """
    t_max = (n_latents - 1) - ar_chunks * h
    return np.arange(k - 1, t_max + 1) if t_max >= k - 1 else np.empty(0, dtype=np.int64)


class WindowIndex:
    """Flat `(episode, start)` index over a set of episodes. Gathers windows on demand."""

    def __init__(self, episodes: list[EpisodeLatents], k: int, h: int, ar_chunks: int):
        self.episodes = episodes
        self.k, self.h, self.ar_chunks = k, h, ar_chunks
        pairs = [
            (i, int(t))
            for i, ep in enumerate(episodes)
            for t in window_starts(len(ep.z), k, h, ar_chunks)
        ]
        self.pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.pairs)

    def gather(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Context `(B, k, R, D)`, actions `(B, nH, 7)`, targets `(B, nH, R, D)`."""
        k, n = self.k, self.ar_chunks * self.h
        ctx, act, tgt = [], [], []
        for r in np.atleast_1d(rows):
            i, t = self.pairs[r]
            ep = self.episodes[i]
            ctx.append(ep.z[t - k + 1 : t + 1])
            act.append(ep.actions[t : t + n])
            tgt.append(ep.z[t + 1 : t + 1 + n])
        return {
            "context": np.stack(ctx).astype(np.float32),
            "actions": np.stack(act).astype(np.float32),
            "targets": np.stack(tgt).astype(np.float32),
        }


def split_by_episode(
    episode_ids: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Hold out whole episodes, never windows.

    Consecutive windows overlap by `k + nH - 1` frames, so a window-level split would put
    near-duplicates of a validation window in training and report an error that means nothing.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episode_ids))
    n_val = max(1, int(round(len(episode_ids) * val_fraction)))
    val = {episode_ids[i] for i in order[:n_val]}
    return [e for e in episode_ids if e not in val], [e for e in episode_ids if e in val]
