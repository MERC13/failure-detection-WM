"""A learned critic for SPEC.md §5's `D` slot — CLAUDE.md's "Q as critic" idea, cheap version.

PROGRESS.md 2026-08-10 confirms, across three independent window-quality regimes, that stage 2
— not stage 1 — is the persistent reason the search does not beat `random_perturb`. The
hand-written objective `D(z_end, z_goal) = ||z_end - z_goal||` (`cfe/search/objective.py`) is
one hypothesis for a scoring function; this is another: `D(z_end) = 1 - P(success | z_end)`,
fit by supervised learning on outcomes this project already paid to replay.

**This is the cheap half of the Q-learning idea CLAUDE.md analyses and keeps in scope.** Monte
Carlo, not TD: the label is the episode's real, already-observed replayed outcome, so there is
no bootstrapped target and no Bellman instability to worry about. It is a *critic* — it scores
candidates CEM already generates — not an *actor*; `argmax_a Q` as a candidate generator is the
out-of-scope half (SPEC.md §9 excludes recovery policies, and CLAUDE.md notes the OOD-
exploitation risk is exactly what this project has already measured for the existing objective).

**Additive, not a replacement of the frozen path.** `cfe/search/objective.py`'s
`evaluate_candidates` is untouched; this is a second, parallel scoring function with the
identical interface, selected by passing `evaluate_fn=evaluate_candidates_critic` to
`cem_search` (`cfe/search/cem.py`). Nothing that omits it is affected.

**Still a search diagnostic, never a result** (CLAUDE.md rule 2). Replacing a hand-written
distance with a learned one does not change what counts as ground truth: only simulator replay
does.

`train_critic` is a deliberate near-duplicate of `scripts/diagnose_fault_localization.py`'s
`train_probe` rather than a shared import: that script's ceiling-test numbers are already cited
in PROGRESS.md, and this module must not be able to change its behaviour as a side effect of an
unrelated change here. Keep the two in sync by hand if the training recipe changes in either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cfe.search.objective import action_penalty, rollout_from_window

ACTION_DIM = 7


@dataclass(frozen=True)
class CriticConfig:
    d_in: int
    hidden: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> CriticConfig:
        return cls(d_in=int(d["d_in"]), hidden=int(d.get("hidden", 0)))


class Critic(nn.Module):
    """`pool(z_end) -> logit(P(success))`. Plain logistic regression by default (`hidden=0`).

    The boring first thing to try, per `diagnose_fault_localization.py`'s same choice for stage
    1: `critic-screen-v1` (PROGRESS.md 2026-08-10) already cleared the objective's ranking
    accuracy at `hidden=0`, so there is no evidence yet that an MLP is needed.
    """

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(cfg.d_in, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1)
            )
        else:
            self.net = nn.Linear(cfg.d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_critic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device,
    seed: int,
) -> Critic:
    """Full-batch, class-weighted logistic regression (or a 1-hidden-layer MLP).

    Class weighting matters: successes and failures are not balanced in the replayed banks
    (`critic-screen-v1`: 402 success / 606 fail), so an unweighted fit can under-predict the
    minority class and still look fine on raw accuracy.
    """
    torch.manual_seed(seed)
    model = Critic(CriticConfig(d_in=x_train.shape[1], hidden=hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    return model.eval()


def save_critic(
    path: str | Path, model: Critic, feature_mean: np.ndarray, feature_std: np.ndarray, extra: dict
) -> None:
    """Mirrors `train_dynamics.py`'s checkpoint convention: state dict, config, and everything
    needed to reproduce the exact preprocessing, in one `torch.save`."""
    torch.save(
        {
            "critic": model.state_dict(),
            "critic_config": {"d_in": model.cfg.d_in, "hidden": model.cfg.hidden},
            "feature_mean": np.asarray(feature_mean, dtype=np.float32).tolist(),
            "feature_std": np.asarray(feature_std, dtype=np.float32).tolist(),
            **extra,
        },
        path,
    )


def load_critic(path: str | Path, device) -> tuple[Critic, np.ndarray, np.ndarray, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = CriticConfig.from_dict(ck["critic_config"])
    model = Critic(cfg).to(device)
    model.load_state_dict(ck["critic"])
    model.eval()
    mean = np.asarray(ck["feature_mean"], dtype=np.float32)
    std = np.asarray(ck["feature_std"], dtype=np.float32)
    return model, mean, std, ck


@dataclass(frozen=True)
class DynamicsAndCritic:
    """Bundle passed as `cem_search`'s opaque `model` when using `evaluate_candidates_critic`.

    `cem_search` never inspects `model` itself — it only forwards it to `evaluate_fn` — so
    bundling two models here needs no change to `cfe/search/cem.py` beyond the `evaluate_fn`
    parameter it already gained.
    """

    dynamics: object  # LatentDynamics
    critic: Critic
    feature_mean: torch.Tensor  # (D,), same device as the models
    feature_std: torch.Tensor  # (D,)


def evaluate_candidates_critic(
    bundle: DynamicsAndCritic, problem, window_start: int, candidates: np.ndarray, lam: float
) -> dict:
    """Same interface as `cfe.search.objective.evaluate_candidates`; `D` replaced by the critic.

    SPEC.md §5's `J = D(z_end, z_goal) + lam * penalty` becomes `J = (1 - P(success | z_end)) +
    lam * penalty` — minimising `J` still means maximising predicted success, so CEM's elite
    selection (argmin `J`) needs no other change. **The returned `"goal_distance"` key holds
    `1 - P(success)` here, not a latent distance** — kept under the same key so it drops into
    `CEMResult` and every row schema unchanged, flagged here the way the objective/metric
    divergence over the gripper dimension already is (`cfe/eval/metrics.py`). Still a search
    diagnostic, never a result (CLAUDE.md rule 2): a learned `D` is still a `D`.
    """
    device = next(bundle.dynamics.parameters()).device
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
        z_end, n_chunks = rollout_from_window(bundle.dynamics, problem, window_start, cand)
        feat = (z_end.reshape(z_end.shape[0], -1) - bundle.feature_mean) / bundle.feature_std
        prob_success = torch.sigmoid(bundle.critic(feat))
        dist = 1.0 - prob_success
        pen = action_penalty(cand, original)
        obj = dist + float(lam) * pen

    return {
        "objective": obj.cpu().numpy().astype(np.float64),
        "goal_distance": dist.cpu().numpy().astype(np.float64),
        "penalty": pen.cpu().numpy().astype(np.float64),
        "n_forward_passes": int(cand.shape[0] * n_chunks),
    }


# ---------------------------------------------------------------------------
# Training-data assembly — shared by the screen (diagnose_critic_quality.py) and the trainer
# (train_critic.py), so both use exactly the same features the deployed critic will see.
# ---------------------------------------------------------------------------


def load_labelled_candidates(repo_root: str | Path, banks: list[str]) -> list[dict]:
    """Every (window, candidate actions, real outcome) triple across `banks`, `variant=primary`.

    Arm-agnostic on purpose — the critic learns "does THIS action sequence at THIS window
    succeed", not which method proposed it, so `no_edit` and `expert_oracle` rows are valid
    training signal too. Join key `(failure_id, arm, lam, window_source)` matches
    `diagnose_objective_quality.py`'s.
    """
    root = Path(repo_root)
    out = []
    for bank in banks:
        mpath = root / "results" / "runs" / bank / "metrics.jsonl"
        epath = root / "results" / "runs" / bank / "edits.jsonl"
        if not mpath.exists() or not epath.exists():
            raise FileNotFoundError(f"{bank} needs both metrics.jsonl and edits.jsonl")
        outcomes = {}
        for line in mpath.read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") != "replay":
                continue
            key = (r["failure_id"], r["arm"], r["lam"], r.get("window_source"))
            outcomes[key] = bool(r["success"])
        for line in epath.read_text().splitlines():
            e = json.loads(line)
            if e.get("variant") != "primary" or "actions" not in e or e.get("not_applicable"):
                continue
            key = (e["failure_id"], e["arm"], e["lam"], e.get("window_source"))
            if key not in outcomes:
                continue
            out.append(
                {
                    "bank": bank,
                    "failure_id": e["failure_id"],
                    "cause": e.get("cause"),
                    "arm": e["arm"],
                    "budget_k": int(e["budget_k"]),
                    "window_start": int(e["window_start"]),
                    # The candidate-variant tag ("cem_best" / "cem_elite_mean" / "cem_iter1_best"
                    # / "identity" / "random_seed{0,1}" for bank-mode rows; otherwise a window
                    # source like "stage1_top1"). Needed to tell apart candidates that otherwise
                    # share (failure_id, arm, window_start) — every bank-mode "search" row does.
                    "window_source": e.get("window_source"),
                    "actions": e["actions"],
                    "success": outcomes[key],
                }
            )
    # An empty result here is a legitimate outcome of the filters above (e.g. a bank that is
    # entirely open_loop rows), not necessarily a misconfiguration — so this library function
    # returns it rather than deciding that's an error. Callers that treat zero candidates as a
    # fatal `--banks` typo (as the two scripts using this do) check for it themselves.
    return out


def rollout_features(
    candidates: list[dict], problems: dict, model, device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll every candidate to `z_end`, flattened (unnormalised). `(N, R*D)`, labels, episode ids.

    `problems` must already hold a `SearchProblem` per `candidates[i]["failure_id"]` — built
    once per episode (needs the encoding cache), not once per candidate.
    """
    feats, labels, episodes = [], [], []
    with torch.no_grad():
        for c in candidates:
            problem = problems[c["failure_id"]]
            actions = torch.as_tensor(
                np.asarray(c["actions"], dtype=np.float32), device=device
            ).unsqueeze(0)
            z_end, _ = rollout_from_window(model, problem, c["window_start"], actions)
            feats.append(z_end[0].reshape(-1).cpu().numpy())
            labels.append(float(c["success"]))
            episodes.append(c["failure_id"])
    x = np.stack(feats).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return x, y, np.asarray(episodes)
