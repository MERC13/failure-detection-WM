"""The latent space the dynamics model works in, and how distance is measured in it.

Deliberately *not* in `cfe/eval/`. That directory holds SPEC.md §6's reported metrics — replay
success rate, edit sparsity, cause accuracy — and is frozen after Phase 2. What lives here is
the search *objective's* substrate: a distance the model optimises against, which CLAUDE.md
rule 2 says is never a result. Keeping the two apart means a change here can never be mistaken
for a change to a reported metric.
"""

from __future__ import annotations

import numpy as np


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm over the feature axis, no affine parameters.

    SPEC.md §4.2 puts the dynamics loss on LayerNormed latents, so the model predicts in this
    space and every distance the search takes must be measured in it too. Applying the loss in
    one space and the objective in another would silently optimise the wrong thing.
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def goal_distance(z_end: np.ndarray, goals: np.ndarray) -> float:
    """Minimum L2 distance from a terminal latent `(R, D)` to any goal in `(G, R, D)`.

    SPEC.md §5 takes the minimum rather than a single reference because "goal states are
    multimodal and a single reference will punish legitimate alternative solutions".
    """
    if goals.ndim != z_end.ndim + 1:
        raise ValueError(f"expected goals (G, R, D) against z_end (R, D); got {goals.shape}")
    diff = goals.reshape(len(goals), -1) - z_end.reshape(1, -1)
    return float(np.linalg.norm(diff, axis=1).min())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, with ties averaged.

    Written out rather than adding scipy for one function. Tie handling is not optional here:
    `b` is a binary success flag, so every value ties with dozens of others and an ordinal
    ranking would fabricate an ordering that does not exist.
    """
    ra, rb = ranks(np.asarray(a, dtype=np.float64)), ranks(np.asarray(b, dtype=np.float64))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def ranks(v: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged. Public because the objective-quality screen ranks candidates
    within an episode before correlating, and a cross-module import of a private name invites
    someone to change it without knowing who depends on it."""
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(len(v), dtype=np.float64)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = np.mean(r[order[i : j + 1]])
        i = j + 1
    return r


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> dict:
    """CI for `mean(a - b)`, resampling the pairs. Used for G2's action-conditioning test.

    Paired because both models are evaluated on the *same* windows: the per-window difference
    removes the variance from some windows simply being harder than others, which is most of
    the spread.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_difference": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_pairs": int(len(d)),
        "n_boot": int(n_boot),
    }
