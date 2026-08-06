"""The counterfactual search objective (SPEC.md §5).

    J(a') = D(z_end(a'), z_goal)  +  lam * sum_t ||a'_t - a_t||^2

**This file computes a search objective, never a result.** CLAUDE.md rule 2 and SPEC.md I2:
latent goal distance is what the search optimises against; the only number that counts is what
the simulator says on replay. Nothing here may be reported as a success measure, and that is
why `cfe/search/` sits outside the frozen `cfe/eval/`.

Three decisions that shape every number the search produces:

- **Distance is measured in LayerNormed reduced-latent space**, the space
  `cfe/dynamics/model.py` predicts in. Measuring the objective in one space while the model was
  trained in another would optimise something the model was never fit to.
- **`z_end` is rolled to the end of the episode**, not to a short horizon. §5 rolls "from the
  context preceding the window", and the window is typically 100+ steps from the end. That is
  also the regime G2's magnitude criterion was moved to h96 to measure: action-conditioning is
  0.035 at h8 and 0.224 at h96, so a short-horizon objective would be reading the model where it
  barely responds to actions at all.
- **No action is ever fabricated to fill a chunk.** `LatentDynamics.rollout` needs a multiple of
  H actions, and the obvious fix is to pad the tail with a repeated last action. Because the
  model predicts a chunk *jointly*, padded actions would influence the prediction at the frame
  we read `z_end` from — a made-up action leaking into the objective. The rollout is truncated
  to the largest whole number of chunks instead, so `z_end` is the model's prediction of a real
  frame at most H-1 steps before the episode's last. The truncation is uniform across every
  candidate for a given window, so it cannot favour one edit over another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ACTION_DIM = 7


@dataclass(frozen=True)
class SearchProblem:
    """One failed episode, in the form the objective needs.

    `z` and `z_goal` are already reduced and LayerNormed (`cfe/dynamics/data.py:load_episode`).
    `z_goal` is `(G, R, D)`: SPEC.md §5 takes the minimum over several successful demos because
    "goal states are multimodal and a single reference will punish legitimate alternative
    solutions".
    """

    episode_id: str
    z: torch.Tensor  # (T+1, R, D)
    actions: np.ndarray  # (T, 7) executed
    z_goal: torch.Tensor  # (G, R, D)
    k_context: int
    horizon: int

    @property
    def n_steps(self) -> int:
        return int(len(self.actions))

    def valid_window_starts(self, budget_k: int) -> np.ndarray:
        """Starts with a real context behind them and at least one whole chunk ahead.

        `s >= k-1` so the context is observed rather than padded — `cfe/dynamics/data.py` makes
        the same choice for training windows, and padding would put the model in a regime no
        real episode produces. `s <= T - H` so there is something to roll; a window with less
        than one chunk after it has no `z_end` to speak of.
        """
        lo = self.k_context - 1
        hi = min(self.n_steps - budget_k, self.n_steps - self.horizon)
        return np.arange(lo, hi + 1) if hi >= lo else np.empty(0, dtype=np.int64)


def goal_distance_torch(z_end: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """Batched twin of `cfe.dynamics.latents.goal_distance`: `(B, R, D)`, `(G, R, D)` -> `(B,)`.

    Minimum L2 to any goal, matching §5. A test asserts this agrees with the numpy original to
    float tolerance rather than trusting that two implementations of "the same" thing agree.
    """
    b = z_end.shape[0]
    g = goals.shape[0]
    diff = z_end.reshape(b, 1, -1) - goals.reshape(1, g, -1)
    return diff.norm(dim=2).min(dim=1).values


def rollout_from_window(
    model,
    problem: SearchProblem,
    window_start: int,
    window_actions: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Predict `z_end` for a batch of candidate windows. Returns `(B, R, D)` and a chunk count.

    The action sequence fed to the model is: the candidate window, then the episode's
    *original* actions after it, truncated to a whole number of chunks. Everything before the
    window is unchanged and enters only through the observed context, which is what makes this
    an edit of bounded budget rather than a re-plan.
    """
    h = problem.horizon
    k = problem.k_context
    b, budget_k, _ = window_actions.shape
    if window_start < k - 1:
        raise ValueError(f"window_start {window_start} has no observed context for k={k}")

    tail = problem.actions[window_start + budget_k :]
    n_total = budget_k + len(tail)
    n_chunks = n_total // h
    if n_chunks == 0:
        raise ValueError(
            f"window [{window_start}, {window_start + budget_k}) leaves {n_total} steps, "
            f"less than one chunk of H={h}; there is no z_end to predict"
        )
    n_used = n_chunks * h

    device = window_actions.device
    tail_t = torch.as_tensor(np.ascontiguousarray(tail), dtype=torch.float32, device=device)
    tail_t = tail_t.unsqueeze(0).expand(b, -1, -1)
    seq = torch.cat([window_actions, tail_t], dim=1)[:, :n_used]

    context = problem.z[window_start - k + 1 : window_start + 1].to(device)
    context = context.unsqueeze(0).expand(b, -1, -1, -1)

    pred = model.rollout(context, seq)  # (B, n_used, R, D)
    return pred[:, -1], n_chunks


def action_penalty(candidates: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """`sum_t ||a'_t - a_t||^2`, over **all seven dimensions**, as SPEC.md §5 writes it.

    The gripper is included here and excluded from the *reported* `edit_magnitude`
    (`cfe/eval/metrics.py`, which explains why). The divergence between objective and metric is
    deliberate; it is flagged at both ends so neither reads as an oversight.
    """
    return ((candidates - original.unsqueeze(0)) ** 2).sum(dim=(1, 2))


def evaluate_candidates(
    model,
    problem: SearchProblem,
    window_start: int,
    candidates: np.ndarray,
    lam: float,
) -> dict:
    """`J` and its two terms for a batch of candidate windows `(P, K, 7)`.

    `n_forward_passes` counts *sequence* forward passes through the dynamics model — population
    times chunks — which is what SPEC.md §6's `search_cost` asks for. Counting calls instead
    would make a batched implementation look free.
    """
    device = next(model.parameters()).device
    cand = torch.as_tensor(np.asarray(candidates, dtype=np.float32), device=device)
    if cand.ndim != 3 or cand.shape[2] != ACTION_DIM:
        raise ValueError(f"candidates must be (P, K, {ACTION_DIM}), got {tuple(cand.shape)}")

    original = torch.as_tensor(
        problem.actions[window_start : window_start + cand.shape[1]].astype(np.float32),
        device=device,
    )
    if original.shape != cand.shape[1:]:
        raise ValueError(
            f"window [{window_start}, {window_start + cand.shape[1]}) runs past the episode's "
            f"{problem.n_steps} actions"
        )

    with torch.no_grad():
        z_end, n_chunks = rollout_from_window(model, problem, window_start, cand)
        dist = goal_distance_torch(z_end, problem.z_goal.to(device))
        pen = action_penalty(cand, original)
        obj = dist + float(lam) * pen

    return {
        "objective": obj.cpu().numpy().astype(np.float64),
        "goal_distance": dist.cpu().numpy().astype(np.float64),
        "penalty": pen.cpu().numpy().astype(np.float64),
        "n_forward_passes": int(cand.shape[0] * n_chunks),
    }
