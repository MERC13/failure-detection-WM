"""The learned critic (SPEC.md §5's `D` slot, CLAUDE.md's "Q as critic").

Two kinds of test, same split SPEC.md §8 draws for the goal-distance objective:

- **CEM-against-a-known-critic, on the toy dynamics** (`test_search_cem.py`'s `ToyDynamics`):
  the critic's weights are hand-set so the optimum is analytic, isolating "does CEM correctly
  optimise against whatever `evaluate_fn` it is given" from "is the critic any good" — the same
  reason SPEC.md §8 names an analytic CEM test for the original objective.
- **Everything else**: save/load fidelity, the training recipe on a separable synthetic set,
  and the data-loading join `train_critic.py` / `diagnose_critic_quality.py` both depend on.
"""

import json

import numpy as np
import pytest
import torch

from cfe.search.cem import cem_search
from cfe.search.critic import (
    Critic,
    CriticConfig,
    DynamicsAndCritic,
    evaluate_candidates_critic,
    load_critic,
    load_labelled_candidates,
    rollout_features,
    save_critic,
    train_critic,
)
from tests.test_search_cem import R, ToyDynamics, _cfg, _problem

D = 3  # matches tests.test_search_cem's toy dynamics (R=1, D=3)


# ---------------------------------------------------------------------------
# CEM against a known critic — the analytic recovery test
# ---------------------------------------------------------------------------


def _all_ones_critic() -> Critic:
    """`logit = sum(z_end)`. Maximised, under no penalty and box-bounded actions, by pushing
    every edited dimension to its upper bound — a fully-determined analytic optimum."""
    critic = Critic(CriticConfig(d_in=R * D, hidden=0))
    with torch.no_grad():
        critic.net.weight.copy_(torch.ones(1, R * D))
        critic.net.bias.zero_()
    return critic.eval()


def test_cem_recovers_the_analytic_optimum_against_a_known_critic():
    """K=1: the one edited action must saturate every dim at +1 to maximise `sum(z_end)`.

    `ToyDynamics` is `z[t+1] = z[t] + a[t, :3]`, so `z_end` under the edit is a fixed constant
    (context plus the unedited tail) plus the edited action itself — maximising `sum(z_end)` is
    therefore maximising `sum(a')`, uniquely solved at `a' = [1, 1, 1]` on `[-1, 1]^3`.
    """
    problem, _ = _problem()
    s = 8
    bundle = DynamicsAndCritic(
        dynamics=ToyDynamics(),
        critic=_all_ones_critic(),
        feature_mean=torch.zeros(R * D),
        feature_std=torch.ones(R * D),
    )

    res = cem_search(
        bundle,
        problem,
        s,
        budget_k=1,
        lam=0.0,
        cfg=_cfg(),
        rng=np.random.default_rng(0),
        evaluate_fn=evaluate_candidates_critic,
    )

    assert res.actions[0, :3] == pytest.approx([1.0, 1.0, 1.0], abs=0.08)
    # 1 - P(success) at the true optimum: a large positive logit, so this should be near zero,
    # and far below the do-nothing baseline (0.53) — CEM need not fully converge in n_iters.
    assert res.goal_distance < 0.15
    assert res.goal_distance < res.baseline_objective / 2


def test_evaluate_candidates_critic_matches_the_objective_interface():
    """Same keys, same shapes as `evaluate_candidates` — this is what lets `cem_search` treat
    the two interchangeably via `evaluate_fn`."""
    problem, actions = _problem()
    s = 8
    bundle = DynamicsAndCritic(
        dynamics=ToyDynamics(),
        critic=_all_ones_critic(),
        feature_mean=torch.zeros(R * D),
        feature_std=torch.ones(R * D),
    )
    cand = np.stack([actions[s : s + 2], actions[s : s + 2]]).astype(np.float32)
    out = evaluate_candidates_critic(bundle, problem, s, cand, lam=0.1)
    assert set(out) == {"objective", "goal_distance", "penalty", "n_forward_passes"}
    assert out["objective"].shape == (2,)
    assert out["goal_distance"].shape == (2,)
    assert out["penalty"].shape == (2,)
    # 1 - P(success) is a probability complement: always in [0, 1], unlike an unbounded distance.
    assert np.all(out["goal_distance"] >= 0.0) and np.all(out["goal_distance"] <= 1.0)


def test_evaluate_candidates_critic_rejects_a_window_past_the_episode():
    problem, _ = _problem(n_steps=24)
    bundle = DynamicsAndCritic(
        dynamics=ToyDynamics(),
        critic=_all_ones_critic(),
        feature_mean=torch.zeros(R * D),
        feature_std=torch.ones(R * D),
    )
    cand = np.zeros((1, 4, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="runs past"):
        evaluate_candidates_critic(bundle, problem, 22, cand, lam=0.0)


# ---------------------------------------------------------------------------
# The critic model itself: architecture, training, save/load
# ---------------------------------------------------------------------------


def test_critic_hidden_zero_is_plain_logistic_regression():
    critic = Critic(CriticConfig(d_in=5, hidden=0))
    assert isinstance(critic.net, torch.nn.Linear)


def test_critic_forward_shape():
    critic = Critic(CriticConfig(d_in=5, hidden=8))
    out = critic(torch.randn(3, 5))
    assert out.shape == (3,)  # squeezed, one logit per row


def test_train_critic_separates_a_linearly_separable_dataset():
    """Not a claim about real data — just that the training loop converges to something useful
    on a problem it should trivially solve."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)  # separable on one axis alone
    critic = train_critic(
        x, y, hidden=0, epochs=300, lr=0.1, weight_decay=0.0, device="cpu", seed=0
    )
    with torch.no_grad():
        pred = torch.sigmoid(critic(torch.from_numpy(x))).numpy() > 0.5
    assert (pred == y.astype(bool)).mean() > 0.9


def test_train_critic_is_deterministic_under_the_same_seed():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3)).astype(np.float32)
    y = (rng.random(50) > 0.5).astype(np.float32)
    kw = dict(hidden=0, epochs=20, lr=0.1, weight_decay=0.0, device="cpu", seed=3)
    a = train_critic(x, y, **kw)
    b = train_critic(x, y, **kw)
    with torch.no_grad():
        pa = torch.sigmoid(a(torch.from_numpy(x))).numpy()
        pb = torch.sigmoid(b(torch.from_numpy(x))).numpy()
    assert np.array_equal(pa, pb)


def test_save_and_load_critic_round_trips_predictions(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)
    critic = train_critic(
        x, y, hidden=4, epochs=50, lr=0.05, weight_decay=1e-3, device="cpu", seed=0
    )
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6

    path = tmp_path / "critic.pt"
    save_critic(path, critic, mean, std, extra={"banks": ["fake-bank"], "seed": 0})
    loaded, loaded_mean, loaded_std, ck = load_critic(path, device="cpu")

    with torch.no_grad():
        want = torch.sigmoid(critic(torch.from_numpy(x))).numpy()
        got = torch.sigmoid(loaded(torch.from_numpy(x))).numpy()
    assert got == pytest.approx(want, abs=1e-6)
    assert loaded_mean == pytest.approx(mean, abs=1e-6)
    assert loaded_std == pytest.approx(std, abs=1e-6)
    assert ck["banks"] == ["fake-bank"]


# ---------------------------------------------------------------------------
# Data assembly: the join `train_critic.py` and `diagnose_critic_quality.py` share
# ---------------------------------------------------------------------------


def _write_bank(root, name, edits, replays):
    run_dir = root / "results" / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "edits.jsonl").write_text("\n".join(json.dumps(r) for r in edits))
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in replays))


def _edit_row(**over):
    row = {
        "failure_id": "ep1",
        "cause": "grasp_offset",
        "arm": "search",
        "budget_k": 8,
        "lam": 0.01,
        "window_source": "stage1_top1",
        "variant": "primary",
        "window_start": 10,
        "actions": [[0.0] * 7] * 8,
    }
    row.update(over)
    return row


def _replay_row(**over):
    row = {
        "kind": "replay",
        "failure_id": "ep1",
        "arm": "search",
        "lam": 0.01,
        "window_source": "stage1_top1",
        "success": True,
    }
    row.update(over)
    return row


def test_load_labelled_candidates_joins_edits_to_outcomes(tmp_path):
    _write_bank(tmp_path, "bank1", [_edit_row()], [_replay_row()])
    out = load_labelled_candidates(tmp_path, ["bank1"])
    assert len(out) == 1
    assert out[0]["success"] is True
    assert out[0]["failure_id"] == "ep1"


def test_load_labelled_candidates_preserves_the_window_source_tag(tmp_path):
    """Needed to tell apart bank-mode candidates that share (failure_id, arm, window_start) —
    e.g. cem_best vs elite_mean vs iter1_best, all arm="search" at the same forced window."""
    _write_bank(
        tmp_path,
        "bank1",
        [_edit_row(window_source="cem_elite_mean")],
        [_replay_row(window_source="cem_elite_mean")],
    )
    out = load_labelled_candidates(tmp_path, ["bank1"])
    assert out[0]["window_source"] == "cem_elite_mean"


def test_load_labelled_candidates_drops_open_loop_variant(tmp_path):
    """The critic learns "does this edit at this window succeed", which open-loop rows (the
    whole original sequence replayed, no window semantics) don't describe."""
    _write_bank(
        tmp_path,
        "bank1",
        [_edit_row(variant="open_loop")],
        [_replay_row(success=False)],
    )
    assert load_labelled_candidates(tmp_path, ["bank1"]) == []


def test_load_labelled_candidates_drops_not_applicable_rows(tmp_path):
    """`expert_oracle` on a natural failure has no actions; must not be silently included."""
    row = _edit_row(arm="expert_oracle")
    row.pop("actions")
    row["not_applicable"] = "no expert trajectory"
    _write_bank(tmp_path, "bank1", [row], [_replay_row(arm="expert_oracle")])
    assert load_labelled_candidates(tmp_path, ["bank1"]) == []


def test_load_labelled_candidates_requires_a_matching_outcome(tmp_path):
    """An edit with no replay row (e.g. a still-running replay job) is skipped, not fabricated."""
    _write_bank(tmp_path, "bank1", [_edit_row()], [])
    assert load_labelled_candidates(tmp_path, ["bank1"]) == []


def test_load_labelled_candidates_raises_on_a_missing_bank(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labelled_candidates(tmp_path, ["does-not-exist"])


def test_rollout_features_shapes_and_labels():
    problem, actions = _problem()
    s = 8
    window = actions[s : s + 1].tolist()
    candidates = [
        {"failure_id": "toy", "window_start": s, "actions": window, "success": True},
        {"failure_id": "toy", "window_start": s, "actions": window, "success": False},
    ]
    x, y, episodes = rollout_features(candidates, {"toy": problem}, ToyDynamics(), "cpu")
    assert x.shape == (2, R * D)
    assert y.tolist() == [1.0, 0.0]
    assert list(episodes) == ["toy", "toy"]
