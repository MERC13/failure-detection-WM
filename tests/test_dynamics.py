"""Latent reductions, the search objective's substrate, and the dynamics model (SPEC.md §4.2).

The reduction and the distance are what the counterfactual search will optimise against, so a
quiet bug here would not crash anything — it would just make every later number meaningless.
That is the same shape of failure the Phase 1 failure-mode metric had, so these are checked
against hand-computed answers rather than against themselves.
"""

import numpy as np
import pytest
import torch

from cfe.dynamics.data import (
    EpisodeLatents,
    WindowIndex,
    load_episode,
    split_by_episode,
    window_starts,
)
from cfe.dynamics.latents import goal_distance, layer_norm, paired_bootstrap_ci, spearman
from cfe.dynamics.model import (
    DynamicsConfig,
    LatentDynamics,
    chunk_loss,
    rollout_loss,
    tokens_per_frame,
)
from cfe.encoders.dinov2 import REDUCTIONS, grid_pool, pool, reduce_latents

# --- reductions -------------------------------------------------------------


def _tokens(t: int = 3, d: int = 5) -> np.ndarray:
    """`(T, 257, D)` with a distinguishable CLS row."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((t, 257, d)).astype(np.float32)
    x[:, 0, :] = 100.0  # CLS, far from any patch value
    return x


@pytest.mark.parametrize("mode,expected_r", [("cls", 1), ("mean", 1), ("grid2", 5), ("grid4", 17)])
def test_reduction_shapes(mode, expected_r):
    out = reduce_latents(_tokens(), mode)
    assert out.shape == (3, expected_r, 5)


def test_full_reduction_is_the_identity():
    x = _tokens()
    assert reduce_latents(x, "full") is x


def test_every_named_reduction_is_implemented():
    for mode in REDUCTIONS:
        assert reduce_latents(_tokens(), mode).ndim == 3


def test_unknown_reduction_raises():
    with pytest.raises(ValueError, match="unknown reduction"):
        reduce_latents(_tokens(), "grid3x")


def test_mean_reduction_matches_pool_and_excludes_cls():
    x = _tokens()
    got = reduce_latents(x, "mean")[:, 0, :]
    assert np.allclose(got, pool(x, include_cls=False))
    # CLS is 100.0; including it would drag the mean far from the patch mean.
    assert np.abs(got).max() < 10.0


def test_grid_pool_averages_the_right_cells():
    """Hand-computed: patch p is given value p, so a 2x2 block's mean is checkable by eye."""
    d = 1
    x = np.zeros((1, 257, d), dtype=np.float32)
    x[0, 0, 0] = -1.0  # CLS
    x[0, 1:, 0] = np.arange(256, dtype=np.float32)  # row-major 16x16

    out = grid_pool(x, 8)  # 8x8 blocks of 2x2 patches
    assert out.shape == (1, 65, d)
    assert out[0, 0, 0] == -1.0, "CLS must survive as token 0"
    # Top-left block covers patches 0, 1, 16, 17.
    assert out[0, 1, 0] == pytest.approx(np.mean([0, 1, 16, 17]))
    # Bottom-right block covers 238, 239, 254, 255.
    assert out[0, -1, 0] == pytest.approx(np.mean([238, 239, 254, 255]))


def test_grid_pool_preserves_the_overall_mean():
    x = _tokens(t=2, d=4)
    for grid in (2, 4, 8):
        assert np.allclose(
            grid_pool(x, grid)[:, 1:, :].mean(axis=1), pool(x, include_cls=False), atol=1e-5
        )


def test_grid_pool_rejects_an_indivisible_grid():
    with pytest.raises(ValueError, match="not divisible"):
        grid_pool(_tokens(), 5)  # 16 is not divisible by 5


def test_grid_pool_rejects_a_non_square_patch_count():
    with pytest.raises(ValueError, match="square patch grid"):
        grid_pool(np.zeros((1, 100, 3), dtype=np.float32), 2)


def test_grid_pool_actually_keeps_location():
    """The point of a grid over a mean: two scenes with the same mean must stay distinct."""
    d = 1
    a = np.zeros((1, 257, d), dtype=np.float32)
    b = np.zeros((1, 257, d), dtype=np.float32)
    a[0, 1:17, 0] = 1.0  # top row lit
    b[0, 241:257, 0] = 1.0  # bottom row lit
    assert np.allclose(pool(a), pool(b)), "constructed to have identical means"
    assert not np.allclose(grid_pool(a, 4)[:, 1:, :], grid_pool(b, 4)[:, 1:, :])


# --- the space distances are measured in ------------------------------------


def test_layer_norm_standardises_the_feature_axis():
    x = np.random.default_rng(0).standard_normal((4, 6, 32)) * 7.0 + 3.0
    z = layer_norm(x)
    assert np.allclose(z.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=-1), 1.0, atol=1e-3)


def test_layer_norm_normalises_each_token_independently():
    """Per-token, not per-frame: token 1 must not be affected by token 0's scale."""
    x = np.stack([np.arange(4.0), np.arange(4.0) * 1000.0])[None]  # (1, 2, 4)
    z = layer_norm(x)
    assert np.allclose(z[0, 0], z[0, 1], atol=1e-4)


def test_goal_distance_takes_the_minimum_over_goals():
    z = np.zeros((2, 3))
    goals = np.stack([np.full((2, 3), 10.0), np.full((2, 3), 1.0), np.full((2, 3), 5.0)])
    # Nearest goal is all-ones: sqrt(6 * 1^2).
    assert goal_distance(z, goals) == pytest.approx(np.sqrt(6.0))


def test_goal_distance_is_zero_against_itself():
    z = np.random.default_rng(1).standard_normal((5, 8))
    assert goal_distance(z, z[None]) == pytest.approx(0.0, abs=1e-6)


def test_goal_distance_rejects_mismatched_ranks():
    with pytest.raises(ValueError, match=r"\(G, R, D\)"):
        goal_distance(np.zeros((2, 3)), np.zeros((2, 3)))


# --- statistics used by the G2 gate -----------------------------------------


def test_spearman_is_one_for_a_monotone_relationship():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -(x**3)) == pytest.approx(-1.0)


def test_spearman_averages_ties():
    """A binary success flag is nothing but ties; getting this wrong invents an ordering."""
    d = np.array([1.0, 2.0, 3.0, 4.0])
    success = np.array([0.0, 0.0, 1.0, 1.0])
    assert spearman(d, success) == pytest.approx(0.8944, abs=1e-3)
    assert spearman(d, 1 - success) == pytest.approx(-0.8944, abs=1e-3)


def test_spearman_is_zero_when_one_side_is_constant():
    assert spearman(np.arange(5.0), np.ones(5)) == 0.0


def test_bootstrap_ci_detects_a_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 200)
    b = rng.normal(0.0, 0.1, 200)
    out = paired_bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert out["excludes_zero"]
    assert out["ci_low"] > 0


def test_bootstrap_ci_does_not_detect_a_null_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 200)
    out = paired_bootstrap_ci(a, a.copy(), n_boot=2000, seed=0)
    assert not out["excludes_zero"]
    assert out["mean_difference"] == pytest.approx(0.0)


def test_bootstrap_ci_is_deterministic_under_a_seed():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=80), rng.normal(size=80)
    assert paired_bootstrap_ci(a, b, 500, seed=7) == paired_bootstrap_ci(a, b, 500, seed=7)


# --- window assembly --------------------------------------------------------


def _episode(t_actions: int, r: int = 3, d: int = 4, eid: str = "ep") -> EpisodeLatents:
    """Latents carry their frame index in every element, so misalignment is visible."""
    z = np.arange(t_actions + 1, dtype=np.float16)[:, None, None] * np.ones((1, r, d), np.float16)
    a = np.arange(t_actions, dtype=np.float32)[:, None] * np.ones((1, 7), np.float32)
    return EpisodeLatents(eid, z, a, True)


def test_window_starts_respects_context_and_horizon():
    # 31 latents (T=30 actions), k=4, h=8, 2 chunks -> t in [3, 30-16] = [3, 14]
    starts = window_starts(31, k=4, h=8, ar_chunks=2)
    assert starts[0] == 3 and starts[-1] == 14
    assert len(starts) == 12


def test_window_starts_is_empty_for_a_short_episode():
    assert len(window_starts(10, k=4, h=8, ar_chunks=2)) == 0


def test_gather_aligns_actions_to_the_frames_they_act_from():
    """a[i] takes z[i] -> z[i+1]. An off-by-one here trains the model on the wrong pairing."""
    idx = WindowIndex([_episode(30)], k=4, h=8, ar_chunks=2)
    t = int(idx.pairs[0][1])
    b = idx.gather(np.array([0]))
    assert np.allclose(b["context"][0, :, 0, 0], np.arange(t - 3, t + 1))
    assert np.allclose(b["actions"][0, :, 0], np.arange(t, t + 16))
    assert np.allclose(b["targets"][0, :, 0, 0], np.arange(t + 1, t + 17))


def test_windows_never_cross_an_episode_boundary():
    eps = [_episode(30, eid="a"), _episode(40, eid="b")]
    idx = WindowIndex(eps, k=4, h=8, ar_chunks=2)
    for row in range(len(idx)):
        i, t = idx.pairs[row]
        assert t - 3 >= 0
        assert t + 16 <= len(eps[i].z) - 1


def test_split_holds_out_whole_episodes():
    ids = [f"ep_{i}" for i in range(10)]
    train, val = split_by_episode(ids, val_fraction=0.2, seed=0)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == sorted(ids)
    assert len(val) == 2
    assert split_by_episode(ids, 0.2, seed=0) == (train, val), "must be seed-deterministic"


def test_split_always_holds_out_at_least_one_episode():
    train, val = split_by_episode([f"ep_{i}" for i in range(3)], val_fraction=0.01, seed=0)
    assert len(val) >= 1 and len(train) >= 1


def test_load_episode_rejects_a_latent_action_count_mismatch(tmp_path):
    class _Cache:
        def read(self, _):
            return np.zeros((5, 257, 8), np.float16)  # 5 latents

        def read_meta(self, _):
            return {"actions": np.zeros((9, 7), np.float32), "success": np.array(True)}  # 9 actions

    with pytest.raises(ValueError, match="expected T\\+1 latents"):
        load_episode(_Cache(), "ep", "mean")


# --- the model --------------------------------------------------------------


def _cfg(**over) -> DynamicsConfig:
    base = dict(
        k_context=4,
        horizon=8,
        reduction="grid2",
        d_model=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        ffn_mult=2,
        latent_dim=6,
    )
    return DynamicsConfig.from_dict(base | over)


def test_tokens_per_frame_matches_the_reductions():
    assert tokens_per_frame("cls") == 1
    assert tokens_per_frame("mean") == 1
    assert tokens_per_frame("grid2") == 5
    assert tokens_per_frame("grid4") == 17
    assert tokens_per_frame("full") == 257


def test_forward_shape():
    cfg = _cfg()
    m = LatentDynamics(cfg)
    ctx = torch.zeros(2, cfg.k_context, cfg.r, cfg.latent_dim)
    act = torch.zeros(2, cfg.horizon, 7)
    assert m(ctx, act).shape == (2, cfg.horizon, cfg.r, cfg.latent_dim)


def test_forward_rejects_wrong_shapes():
    cfg = _cfg()
    m = LatentDynamics(cfg)
    with pytest.raises(ValueError, match="expected context"):
        m(torch.zeros(2, 3, cfg.r, cfg.latent_dim), torch.zeros(2, cfg.horizon, 7))
    with pytest.raises(ValueError, match="expected actions"):
        m(torch.zeros(2, cfg.k_context, cfg.r, cfg.latent_dim), torch.zeros(2, 3, 7))


def test_actions_change_the_prediction():
    """If they do not, the G2 comparison is measuring nothing."""
    torch.manual_seed(0)
    cfg = _cfg()
    m = LatentDynamics(cfg).eval()
    ctx = torch.randn(2, cfg.k_context, cfg.r, cfg.latent_dim)
    a1, a2 = torch.randn(2, cfg.horizon, 7), torch.randn(2, cfg.horizon, 7)
    with torch.no_grad():
        assert not torch.allclose(m(ctx, a1), m(ctx, a2))


def test_the_action_ignoring_twin_really_ignores_actions():
    """The control must be blind at eval too, not just during training."""
    torch.manual_seed(0)
    cfg = _cfg()
    m = LatentDynamics(cfg, ignore_actions=True).eval()
    ctx = torch.randn(2, cfg.k_context, cfg.r, cfg.latent_dim)
    with torch.no_grad():
        a = m(ctx, torch.randn(2, cfg.horizon, 7))
        b = m(ctx, torch.randn(2, cfg.horizon, 7) * 100)
        c = m(ctx, torch.zeros(2, cfg.horizon, 7))
    assert torch.equal(a, b) and torch.equal(a, c)


def test_residual_defaults_off_so_old_run_configs_still_describe_their_architecture():
    """`p2-dyn` was trained before this field existed; its snapshot must not be reinterpreted."""
    assert _cfg().residual is False
    assert _cfg(residual=True).residual is True


def test_residual_adds_the_last_context_frame():
    """Not just "output changes" — it must be exactly the non-residual output plus z[t]."""
    torch.manual_seed(0)
    ctx = torch.randn(2, 4, 5, 6)
    act = torch.randn(2, 8, 7)

    torch.manual_seed(1)
    plain = LatentDynamics(_cfg(residual=False)).eval()
    torch.manual_seed(1)
    res = LatentDynamics(_cfg(residual=True)).eval()

    with torch.no_grad():
        assert torch.allclose(res(ctx, act), plain(ctx, act) + ctx[:, -1:], atol=1e-6)


def test_residual_twin_still_ignores_actions():
    """The residual path must not smuggle a second route for actions into the control."""
    torch.manual_seed(0)
    cfg = _cfg(residual=True)
    m = LatentDynamics(cfg, ignore_actions=True).eval()
    ctx = torch.randn(2, cfg.k_context, cfg.r, cfg.latent_dim)
    with torch.no_grad():
        a = m(ctx, torch.randn(2, cfg.horizon, 7))
        b = m(ctx, torch.randn(2, cfg.horizon, 7) * 100)
    assert torch.equal(a, b)


def test_residual_rollout_compounds_off_its_own_predictions():
    """Chunk 2's residual base must be the model's own output, not the original context."""
    torch.manual_seed(0)
    cfg = _cfg(residual=True)
    m = LatentDynamics(cfg).eval()
    ctx = torch.randn(1, cfg.k_context, cfg.r, cfg.latent_dim)
    act = torch.randn(1, cfg.horizon * 2, 7)
    with torch.no_grad():
        out = m.rollout(ctx, act)
        chunk1 = m(ctx, act[:, : cfg.horizon])
        chunk2 = m(chunk1[:, -cfg.k_context :], act[:, cfg.horizon :])
    assert torch.allclose(out[:, cfg.horizon :], chunk2, atol=1e-6)


def test_rollout_chains_chunks():
    cfg = _cfg()
    m = LatentDynamics(cfg).eval()
    ctx = torch.randn(1, cfg.k_context, cfg.r, cfg.latent_dim)
    with torch.no_grad():
        out = m.rollout(ctx, torch.randn(1, cfg.horizon * 3, 7))
    assert out.shape == (1, cfg.horizon * 3, cfg.r, cfg.latent_dim)


def test_rollout_rejects_a_ragged_action_length():
    cfg = _cfg()
    m = LatentDynamics(cfg)
    with pytest.raises(ValueError, match="not a multiple"):
        m.rollout(torch.zeros(1, cfg.k_context, cfg.r, cfg.latent_dim), torch.zeros(1, 9, 7))


def test_rollout_requires_horizon_at_least_context():
    """Otherwise the second chunk's context cannot be filled from predictions alone."""
    cfg = _cfg(k_context=8, horizon=4)
    m = LatentDynamics(cfg)
    with pytest.raises(ValueError, match="H >= k"):
        m.rollout(torch.zeros(1, 8, cfg.r, cfg.latent_dim), torch.zeros(1, 8, 7))


def test_rollout_loss_reports_one_value_per_chunk():
    torch.manual_seed(0)
    cfg = _cfg()
    m = LatentDynamics(cfg)
    ctx = torch.randn(2, cfg.k_context, cfg.r, cfg.latent_dim)
    act = torch.randn(2, cfg.horizon * 2, 7)
    tgt = torch.randn(2, cfg.horizon * 2, cfg.r, cfg.latent_dim)
    total, per_chunk = rollout_loss(m, ctx, act, tgt, ar_chunks=2)
    assert len(per_chunk) == 2
    assert float(total) == pytest.approx(sum(per_chunk) / 2, rel=1e-5)


def test_chunk_loss_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown loss"):
        chunk_loss(torch.zeros(1), torch.zeros(1), "huber")


def test_a_training_step_is_deterministic_under_a_seed():
    def one_step() -> float:
        torch.manual_seed(0)
        cfg = _cfg()
        m = LatentDynamics(cfg)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(1)
        ctx = torch.randn(4, cfg.k_context, cfg.r, cfg.latent_dim, generator=g)
        act = torch.randn(4, cfg.horizon * 2, 7, generator=g)
        tgt = torch.randn(4, cfg.horizon * 2, cfg.r, cfg.latent_dim, generator=g)
        loss, _ = rollout_loss(m, ctx, act, tgt, ar_chunks=2)
        loss.backward()
        opt.step()
        return float(loss)

    assert one_step() == one_step()


# --- gradient accumulation (Phase 3.5) --------------------------------------


def test_micro_batch_accumulation_reproduces_the_full_batch_gradient():
    """`rollout_loss` chains ar_chunks forward passes into one graph, so a long-horizon run does
    not fit at the configured batch. Accumulating micro-batches is only a valid substitute if the
    accumulated gradient equals the full-batch gradient — otherwise the horizon-matched model is
    trained differently from its control and the comparison means nothing."""
    torch.manual_seed(0)
    cfg = DynamicsConfig(
        k_context=2,
        horizon=4,
        reduction="grid2",
        d_model=32,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        ffn_mult=2,
        latent_dim=8,
    )
    model = LatentDynamics(cfg)
    b, ar = 8, 2
    ctx = torch.randn(b, cfg.k_context, cfg.r, cfg.latent_dim)
    act = torch.randn(b, ar * cfg.horizon, 7)
    tgt = torch.randn(b, ar * cfg.horizon, cfg.r, cfg.latent_dim)

    model.zero_grad(set_to_none=True)
    rollout_loss(model, ctx, act, tgt, ar)[0].backward()
    full = [p.grad.clone() for p in model.parameters()]

    model.zero_grad(set_to_none=True)
    for lo in (0, 4):  # two micro-batches of 4, each weighted by its share
        sl = slice(lo, lo + 4)
        loss, _ = rollout_loss(model, ctx[sl], act[sl], tgt[sl], ar)
        (loss * (4 / b)).backward()
    accum = [p.grad.clone() for p in model.parameters()]

    for f, a in zip(full, accum, strict=True):
        assert torch.allclose(f, a, atol=1e-6), (f - a).abs().max()
