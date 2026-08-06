"""Stage 1 of SPEC.md §5: choose the edit window.

"Candidate windows of length `W` over the episode. Rank candidates by per-step latent
divergence between predicted and observed rollout (cheap), keep the top `M` for stage 2."

The divergence array is computed **once per episode** and every budget `K` is scored from it,
so a window's rank does not depend on which budget it was ranked for and the expensive part
(one model rollout per step) is not repeated four times.

Divergence here means *the model was surprised*: it rolled forward from the observed context
with the actions that were actually executed, and the world went somewhere else. That is the
cheap proxy for "something went wrong around here". It is a heuristic and is treated as one —
`exhaustive_starts` enumerates every window so the ranking can be checked against the
unranked truth on short episodes, which SPEC.md §5 asks for by name.
"""

from __future__ import annotations

import numpy as np
import torch

from cfe.search.objective import SearchProblem


def per_step_divergence(
    model,
    problem: SearchProblem,
    score_horizon: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Model-vs-observed latent error at every step. `(T,)`, `NaN` where undefined.

    Entry `t` is the mean absolute error over the model's prediction of `z[t+1 .. t+score_horizon]`
    from context `z[t-k+1 .. t]` and the executed actions `a[t .. t+score_horizon-1]`.

    Averaged over the horizon rather than read at one step ahead: one-step error is dominated by
    the copy-last-latent regime, where G2 measured action-conditioning at its weakest
    (rel_improvement 0.035 at h8 against 0.224 at h96). A one-step score would rank windows by
    where the *scene* changes fastest, not by where the *actions* mattered.

    `NaN` rather than 0 for steps without a full context or a full horizon, because 0 is a
    legitimate divergence value and filling with it would rank those steps as maximally
    unsurprising instead of as unscored.
    """
    h = problem.horizon
    k = problem.k_context
    if score_horizon % h:
        raise ValueError(f"score_horizon {score_horizon} is not a multiple of H={h}")

    t_total = problem.n_steps
    out = np.full(t_total, np.nan, dtype=np.float64)
    starts = np.arange(k - 1, t_total - score_horizon + 1)
    if starts.size == 0:
        return out

    device = next(model.parameters()).device
    z = problem.z.to(device)
    actions = torch.as_tensor(problem.actions.astype(np.float32), device=device)

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            chunk = starts[i : i + batch_size]
            ctx = torch.stack([z[s - k + 1 : s + 1] for s in chunk])
            act = torch.stack([actions[s : s + score_horizon] for s in chunk])
            tgt = torch.stack([z[s + 1 : s + 1 + score_horizon] for s in chunk])
            pred = model.rollout(ctx, act)
            err = (pred - tgt).abs().mean(dim=(1, 2, 3))
            out[chunk] = err.cpu().numpy().astype(np.float64)
    return out


def window_scores(divergence: np.ndarray, budget_k: int, valid_starts: np.ndarray) -> np.ndarray:
    """Mean divergence over `[s, s+K)` for each `s` in `valid_starts`. `NaN` if any step is.

    A window containing an unscored step gets `NaN` and is dropped by the rankers rather than
    scored on its scored steps alone — a partially scored window is not comparable to a fully
    scored one.
    """
    d = np.asarray(divergence, dtype=np.float64)
    starts = np.asarray(valid_starts, dtype=np.int64)
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)
    idx = starts[:, None] + np.arange(budget_k)[None, :]
    if idx.max() >= len(d):
        raise ValueError(f"window start {starts.max()} + K={budget_k} runs past {len(d)} steps")
    return d[idx].mean(axis=1)


def rank_windows(
    divergence: np.ndarray,
    problem: SearchProblem,
    budget_k: int,
    stride: int,
    top_m: int,
) -> list[dict]:
    """The top `M` candidate windows of length `K`, most divergent first.

    Ties are broken by earlier start, so the choice is deterministic (I5) rather than dependent
    on sort stability across numpy versions.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    valid = problem.valid_window_starts(budget_k)
    if valid.size == 0:
        return []
    starts = valid[:: int(stride)]
    return _top(starts, window_scores(divergence, budget_k, starts), top_m, budget_k)


def exhaustive_starts(problem: SearchProblem, budget_k: int) -> np.ndarray:
    """Every valid start, no stride. SPEC.md §5's correctness check on the ranking heuristic."""
    return problem.valid_window_starts(budget_k)


def rank_windows_exhaustive(
    divergence: np.ndarray, problem: SearchProblem, budget_k: int, top_m: int
) -> list[dict]:
    """`rank_windows` with stride 1. Run on short episodes to check what the stride skipped."""
    starts = exhaustive_starts(problem, budget_k)
    if starts.size == 0:
        return []
    return _top(starts, window_scores(divergence, budget_k, starts), top_m, budget_k)


def _top(starts: np.ndarray, scores: np.ndarray, top_m: int, budget_k: int) -> list[dict]:
    ok = np.isfinite(scores)
    starts, scores = starts[ok], scores[ok]
    if starts.size == 0:
        return []
    order = np.lexsort((starts, -scores))[: max(1, int(top_m))]
    return [
        {
            "window_start": int(starts[i]),
            "window_end": int(starts[i]) + int(budget_k),
            "divergence": float(scores[i]),
        }
        for i in order
    ]


def window_recall_of_truth(ranked: list[dict], truth: tuple[int, int] | None) -> dict:
    """How well the ranked candidates overlap the ground-truth injected window.

    Diagnostic for stage 1 on its own, separate from the end-to-end RSR: if the search rescues
    episodes while never finding the window that was actually corrupted, that is worth knowing
    and the replay number alone would not show it. Uses the frozen `window_iou` so this agrees
    with what SPEC.md §6's cause-identification metric reports.
    """
    from cfe.eval.metrics import window_iou

    if truth is None:
        return {"best_iou": None, "reason": "natural failure; no ground-truth window"}
    ious = [window_iou((c["window_start"], c["window_end"]), truth) for c in ranked]
    return {
        "best_iou": max(ious) if ious else 0.0,
        "top1_iou": ious[0] if ious else 0.0,
        "n_candidates": len(ranked),
    }
