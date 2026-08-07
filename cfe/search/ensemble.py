"""Ensembled search objective: standardise each member's goal distances, then average.

**Why this exists.** Phase 3's G3 FAILed, and the follow-up measurements said the search
objective is not empty — it is *noisy*. Scoring 168 replayed edits against their real outcomes:

    within-episode pairwise ranking accuracy   67.3%  (35/52, p = 0.0088)
    spearman(p2-dyn-res distance, success)    -0.238
    spearman(p2-dyn     distance, success)    -0.199
    spearman(mean of the two,     success)    -0.306   <- better than either member
    spearman(p2-dyn-res score, p2-dyn score)  +0.209   <- they barely agree with each other

Two networks of identical architecture, trained on identical data, rank the same edits at
ρ=0.21 — yet averaging them tracks reality better than either alone. That is the signature of a
shared signal buried in independent, model-specific noise, and averaging is the direct remedy.

**Why standardise before averaging.** Members put their distances on different absolute scales
(the recorded spread differs by tens of units), so a raw mean is dominated by whichever member
happens to output larger numbers. Each member's distances are z-scored **across the candidate
population** and then averaged, which makes the combination scale-free.

**The cost of that choice, stated rather than buried.** Standardising within a population makes
the objective *population-dependent*: the same action sequence scores differently depending on
what it was scored alongside. That is safe for CEM, which only uses the ranking of `J` to pick
elites, and safe for the ranking diagnostics, which are within-episode. It is **not** a
meaningful absolute quantity, and it must never be compared across populations or reported as a
number in its own right — which CLAUDE.md rule 2 forbids for latent distances anyway.

SPEC.md §5 defines `J(a') = D(ẑ_end, z_goal) + λ·Σ||a'−a||²` against a single model. Making `D`
an ensemble is an amendment to §5, logged in `CHANGELOG.md`.
"""

from __future__ import annotations

import numpy as np
import torch

from cfe.search.objective import (
    SearchProblem,
    action_penalty,
    goal_distance_torch,
    rollout_from_window,
)

ACTION_DIM = 7


def standardise(x: np.ndarray) -> np.ndarray:
    """Z-score, with a zero-variance population mapped to all-zeros rather than to NaN.

    A degenerate population is a real case — every candidate identical, or a model that has
    collapsed — and turning it into NaN would propagate silently through the mean and poison the
    ranking of every other member.
    """
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    if sd <= 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def combine_member_distances(raw: np.ndarray) -> np.ndarray:
    """`(M, P)` per-member distances -> `(P,)` combined score. Standardise each member, average.

    Separated from the rollout so the combination rule can be tested on its own: scale
    invariance is a property of *this* function, and testing it through a model requires a model
    whose outputs are a pure rescaling, which a goal *distance* never is (scaling `z_end` does
    not scale `||z_end - z_goal||`).
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"expected (M, P) per-member distances, got {raw.shape}")
    return np.stack([standardise(row) for row in raw]).mean(axis=0)


def evaluate_candidates_ensemble(
    models,
    problems,
    window_start: int,
    candidates: np.ndarray,
    lam: float,
) -> dict:
    """`J` for a batch of candidates, with `D` averaged over an ensemble of models.

    `models` and `problems` are parallel sequences: member `i` is rolled on `problems[i]`.
    They differ per member because `SearchProblem` carries latents already reduced under that
    model's config — two members with different reductions are not interchangeable. Passing one
    problem for all members is the common case and is accepted as a single object.

    With one member this is **identical** to `cfe.search.objective.evaluate_candidates` in
    ranking, though not in the absolute value of `objective` (standardisation shifts and scales
    it). A test pins the ranking equivalence.
    """
    models = list(models)
    if not models:
        raise ValueError("an ensemble needs at least one member")
    if isinstance(problems, SearchProblem):
        problems = [problems] * len(models)
    problems = list(problems)
    if len(problems) != len(models):
        raise ValueError(f"{len(models)} models but {len(problems)} problems")

    cand_np = np.asarray(candidates, dtype=np.float64)
    if cand_np.ndim != 3 or cand_np.shape[2] != ACTION_DIM:
        raise ValueError(f"candidates must be (P, K, {ACTION_DIM}), got {cand_np.shape}")

    per_member, n_forward = [], 0
    for model, problem in zip(models, problems, strict=True):
        device = next(model.parameters()).device
        cand = torch.as_tensor(cand_np.astype(np.float32), device=device)
        with torch.no_grad():
            z_end, n_chunks = rollout_from_window(model, problem, window_start, cand)
            dist = goal_distance_torch(z_end, problem.z_goal.to(device))
        per_member.append(dist.cpu().numpy().astype(np.float64))
        n_forward += int(cand.shape[0] * n_chunks)

    raw = np.stack(per_member)  # (M, P)
    combined = combine_member_distances(raw)

    # The penalty is model-independent and stays in its own units, computed once.
    ref = problems[0]
    dev = next(models[0].parameters()).device
    original = torch.as_tensor(
        ref.actions[window_start : window_start + cand_np.shape[1]].astype(np.float32), device=dev
    )
    pen = (
        action_penalty(torch.as_tensor(cand_np.astype(np.float32), device=dev), original)
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    return {
        "objective": combined + float(lam) * pen,
        "goal_distance": combined,
        "goal_distance_per_member": raw,
        "penalty": pen,
        "n_forward_passes": n_forward,
        "n_members": len(models),
        "note": "goal_distance is standardised within this candidate population; it is a "
        "ranking quantity, not an absolute distance, and is never a result",
    }


def member_agreement(raw: np.ndarray) -> dict:
    """Rank correlation between ensemble members on the same candidates.

    Low agreement is the whole reason to ensemble, so it is reported rather than assumed: if the
    members agree closely the ensemble buys nothing and the noise is not independent.
    """
    from cfe.dynamics.latents import spearman

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"expected (M, P) per-member distances, got {raw.shape}")
    pairs = [spearman(raw[i], raw[j]) for i in range(len(raw)) for j in range(i + 1, len(raw))]
    return {
        "n_members": int(len(raw)),
        "mean_pairwise_spearman": float(np.mean(pairs)) if pairs else None,
        "min_pairwise_spearman": float(np.min(pairs)) if pairs else None,
        "pairwise": [round(float(p), 4) for p in pairs],
    }
