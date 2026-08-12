"""Ceiling test (Phase 3.5, post-G3): can these latents localize a fault at all?

G3 FAILed and PROGRESS.md's diagnosis traced it to two compounding weaknesses: stage 1's
window selection burns about half the available headroom because per-step latent divergence
between the trained model's prediction and the observed rollout peaks ~50 steps *after* a
fault (once consequences are visible, not when the fault happens), and stage 2's objective is
weak even at the right window. This asks the question underneath both: does the *encoder's*
representation even carry the information needed to localize a fault, independent of whether a
trained dynamics model can extract it?

Two parts, both scored the same way — a candidate window of length `W` against the
ground-truth `edit_window`, via the frozen `window_iou` — so they are directly comparable to
each other and to the trained-model heuristic's recorded number (mean IoU 0.035, PROGRESS.md
2026-08-05).

**Part 1 — oracle pairwise divergence.** For each of the ~56 injected failures, its matched
clean source demo shares the identical initial state (`FailureRecord.expert_episode_id`). The
per-step distance between the two episodes' *observed* latents is the divergence signal stage 1
approximates with a trained predictor, computed here directly against the true counterfactual
instead — no rollout, no model, just two recorded trajectories from the encoding cache. If this
also peaks late, the late-peak problem is a property of the representation or the physics (a
fault's *consequences* simply take that long to become visible), not of the trained model's
prediction error, and Part 2 is unlikely to do better.

**Part 2 — leave-one-episode-out classifier.** A small probe trained on `pool(z_t)` from the
corrupted episode alone (never its clean counterpart) to predict whether `t` falls in the
ground-truth edit window, evaluated with grouped leave-one-episode-out CV so no episode's own
timesteps leak into its own held-out prediction. This is the "ceiling" proper: it asks whether a
method that only ever sees a failed episode (as any deployed method would) could in principle
learn to recognise a fault's latent signature, decoupled from whether the current CEM/divergence
machinery does.

**Part 3 — position-only control, not in the original diagnosis, added because Part 2's first
run demanded it.** `libero_object` demos close the gripper ~t=42 and release ~t=128 in nearly
every episode (CLAUDE.md), and every injected cause's window sits at a fixed phase relative to
those events — so a probe could localize by memorising *when* faults happen, never reading `z_t`
at all, and Part 2 would not be able to tell the difference. Part 3 is the identical probe and
the identical LOEO procedure, given only `(t, t/T)` and nothing latent. Part 2's result only
means what it claims if it clears Part 3 by a wide margin; see `content_vs_position` in the
output.

**Oracle-assisted and not deployable.** Part 1 uses the ground-truth clean counterpart, which a
deployed method never has. Part 2's *labels* come from the ground-truth injection window at
train time; only the *architecture* (a probe on cached latents) would be available at
deployment, and building a deployable version of it is a separate follow-up, not this script.
Neither part runs the replay protocol or steps the simulator — `open_cache` below only opens the
on-disk encoding cache (the same call every sibling diagnostic makes to read its fingerprint),
no episode is rolled out or replayed. Every row is stamped `diagnostic: true` with the caveat
below, matching `diagnose_window_oracle.py` / `diagnose_objective_quality.py`'s convention.
CLAUDE.md rule 2: nothing here may be reported as a replay success rate or as a working
window-selection method.

Usage:
    python scripts/diagnose_fault_localization.py --config configs/search.yaml --seed 0 \
        --run-id ceiling-test-v1
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

NOT_A_RESULT = (
    "oracle-assisted: Part 1 uses the ground-truth clean counterpart episode and Part 2's "
    "training labels come from the ground-truth injection window, neither of which a deployed "
    "method has. No replay, no simulator step, no success/fail outcome anywhere in this script. "
    "Diagnostic only; never reportable as an RSR result for C1 or as a working window-selection "
    "method."
)

#: Reference point this run is judged against: the trained model's stage-1 heuristic, measured
#: on the full G3 grid (PROGRESS.md 2026-08-05). If the oracle (Part 1) does not clear this,
#: that is itself suspicious and worth re-checking before trusting either number (CLAUDE.md:
#: "distrust a new metric until something independent agrees with it").
TRAINED_MODEL_STAGE1_MEAN_IOU = 0.035


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ceiling test: can cached latents localize a fault at all? No simulator."
    )
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument(
        "--window-length",
        type=int,
        default=8,
        help="candidate window length W; matched to K=8, which had the largest measured "
        "headroom (PROGRESS.md 2026-08-05)",
    )
    p.add_argument(
        "--stride", type=int, default=None, help="override search.yaml:search.windows.stride"
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=0,
        help="probe hidden width; 0 = plain logistic regression (the boring default — try an "
        "MLP only if the linear probe is near chance)",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="seed-ensemble the LOEO probe (average predicted probabilities before windowing) "
        "-- mirrors the seed-ensembling that measurably improved the dynamics-model objective's "
        "ranking accuracy (PROGRESS.md 2026-08-06: 0.621->0.668, McNemar p=0.023). Default 1 "
        "(no ensembling) reproduces every prior ceiling-test run's exact behaviour.",
    )
    p.add_argument(
        "--top-m",
        type=int,
        default=1,
        help="also record the top-M candidate windows per episode (SPEC.md §5's stage 1 design "
        "-- 'keep the top M for stage 2' -- which the ceiling test simplified to M=1 when first "
        "built). Default 1 adds only a found_windows field alongside the unchanged found_window.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pure helpers — no torch, no env. Unit-tested in tests/test_diagnose_fault_localization.py.
# ---------------------------------------------------------------------------


def top_window(scores: np.ndarray, w: int, stride: int) -> tuple[int, int] | None:
    """The length-`w` window with the highest mean score, `None` if the episode is shorter.

    Ties break on the earlier start: `starts` is ascending and `np.argmax` returns the first
    maximum, matching `cfe/search/windows.py:_top`'s tie-break so results stay comparable.
    """
    t_total = len(scores)
    if t_total < w:
        return None
    starts = np.arange(0, t_total - w + 1, max(1, int(stride)))
    means = np.array([scores[s : s + w].mean() for s in starts])
    best = int(starts[int(np.argmax(means))])
    return (best, best + w)


def top_m_windows(scores: np.ndarray, w: int, stride: int, m: int) -> list[tuple[int, int]]:
    """The `m` highest-mean-scoring length-`w` windows, best first, `[]` if too short.

    SPEC.md §5 always specified stage 1 as "rank candidates... keep the top `M` for stage 2" —
    `top_window` (this file's original, m=1 only) was a simplification made when the ceiling
    test was first built, not a deliberate departure. This restores the top-M version so a
    deployable pipeline gets the same "try several plausible windows" chance the divergence
    heuristic already has (`configs/search.yaml:search.windows.top_m`). Ranking and tie-break
    (earlier start wins) match `cfe/search/windows.py:_top` exactly; `top_window` is untouched
    and this does not change its behaviour or any existing caller's.
    """
    t_total = len(scores)
    if t_total < w:
        return []
    starts = np.arange(0, t_total - w + 1, max(1, int(stride)))
    means = np.array([scores[s : s + w].mean() for s in starts])
    order = np.lexsort((starts, -means))[: max(1, int(m))]
    return [(int(starts[i]), int(starts[i]) + w) for i in order]


def shared_length(len_c: int, len_e: int) -> tuple[int, bool]:
    """Shared latent-frame count between a corrupted and its matched clean episode, and whether
    either was cut to reach it. The two are simulated forward independently past the injection
    point, so their recorded lengths can differ even though they share an initial state."""
    n = min(len_c, len_e)
    return n, n < max(len_c, len_e)


def labels_from_window(t_total: int, edit_window: tuple[int, int]) -> np.ndarray:
    """Per-action-step binary label, 1 inside the half-open `edit_window`, else 0. `(t_total,)`."""
    s, e = edit_window
    y = np.zeros(t_total, dtype=np.float32)
    y[max(0, s) : min(t_total, e)] = 1.0
    return y


def position_features(t_total: int) -> np.ndarray:
    """`(t_total, 2)`: raw step index and fraction through the episode. No latent content.

    The content-blind control for Part 2 — the same role `action_ignoring_dynamics` plays for
    the trained search: isolate whether the thing being tested (there, action-conditioning;
    here, latent CONTENT) is doing any work, versus a structural shortcut. `libero_object`
    demos have a strikingly consistent phase structure (CLAUDE.md: gripper closes ~t=42,
    releases ~t=128 in nearly every demo), and every injected cause's window sits at a fixed
    phase (`grasp_offset`/`mistimed_close` before the close, `early_release` before the
    release) — so a probe could score well by memorising *when* faults happen without ever
    reading `z_t`. If this control matches Part 2, Part 2 is not measuring what it claims to.
    """
    t = np.arange(t_total, dtype=np.float32)
    frac = t / max(t_total - 1, 1)
    return np.stack([t, frac], axis=1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """AUROC via the rank-sum formula. `None` if only one class is present in `labels`.

    `cfe.dynamics.latents.ranks` is 0-indexed (`0..n-1`), so the usual 1-indexed rank-sum
    formula `(R1 - n1(n1+1)/2) / (n1 n2)` is shifted by `n1` here to match.
    """
    from cfe.dynamics.latents import ranks

    y = np.asarray(labels, dtype=np.float64)
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = ranks(np.asarray(scores, dtype=np.float64))
    return (float(r[y == 1].sum()) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)


def summarise_iou(rows: list[dict]) -> dict:
    """Mean/median IoU per cause and pooled, matching the existing per-cause breakdown table."""
    by_cause = defaultdict(list)
    for r in rows:
        by_cause[r["cause"]].append(r["iou"])
    out = {
        cause: {
            "mean_iou": float(np.mean(v)),
            "median_iou": float(np.median(v)),
            "n": len(v),
        }
        for cause, v in sorted(by_cause.items())
    }
    all_ious = [r["iou"] for r in rows]
    out["pooled"] = {
        "mean_iou": float(np.mean(all_ious)) if all_ious else None,
        "median_iou": float(np.median(all_ious)) if all_ious else None,
        "n": len(all_ious),
    }
    return out


# ---------------------------------------------------------------------------
# The probe (Part 2)
# ---------------------------------------------------------------------------


def build_probe(d_in: int, hidden: int):
    import torch.nn as nn

    if hidden > 0:
        return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    return nn.Linear(d_in, 1)


def train_probe(x_train, y_train, *, d_in, hidden, epochs, lr, weight_decay, device, seed):
    """Full-batch logistic regression (or a 1-hidden-layer MLP), class-weighted.

    Class weighting matters here: the window is a small fraction of the episode, so an
    unweighted fit can minimise loss by predicting "clean" everywhere and still score well on
    accuracy while being useless for localisation.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = build_probe(d_in, hidden).to(device)
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
        loss = loss_fn(model(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    return model.eval()


def loeo_probe(
    feats_by_episode: list[np.ndarray], episodes: list[dict], *, w, stride, args, device, label
) -> tuple[list[dict], float | None]:
    """Grouped leave-one-episode-out probe: fit on all-but-one episode, score the held-out one.

    Shared by Part 2 (latent features) and Part 3 (position-only control) so both are scored by
    the identical procedure and differ only in what the probe is allowed to see — the same
    reason `action_ignoring_dynamics` reuses the trained search's own code path with only the
    actions zeroed, rather than being a separately-written baseline.
    """
    import torch

    from cfe.eval.metrics import window_iou

    labs, groups = [], []
    for gi, e in enumerate(episodes):
        t = e["t_actions"]
        labs.append(labels_from_window(t, e["rec"].edit_window))
        groups.append(np.full(t, gi, dtype=np.int64))
    x_all = np.concatenate(feats_by_episode, axis=0)
    y_all = np.concatenate(labs, axis=0)
    g_all = np.concatenate(groups, axis=0)
    d_in = x_all.shape[1]

    n_seeds = max(1, int(getattr(args, "n_seeds", 1)))
    rows, held_scores, held_labels = [], [], []
    for gi, e in enumerate(episodes):
        rec = e["rec"]
        train_mask, held_mask = g_all != gi, g_all == gi
        mu = x_all[train_mask].mean(axis=0)
        sd = x_all[train_mask].std(axis=0) + 1e-6
        x_held = torch.as_tensor((x_all[held_mask] - mu) / sd, dtype=torch.float32).to(device)
        # Seed-ensemble: average PROBABILITIES across independently-trained probes on the same
        # fold, simple mean rather than the ensemble.py standardise-then-average dance -- unlike
        # raw goal-distances, probabilities from different seeds already share the same [0, 1]
        # scale, so there is no cross-model scale mismatch to correct for.
        seed_probs = []
        for s in range(n_seeds):
            model = train_probe(
                (x_all[train_mask] - mu) / sd,
                y_all[train_mask],
                d_in=d_in,
                hidden=args.hidden,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                # s=0 uses exactly args.seed, so --n-seeds 1 (the default) is bit-identical to
                # every prior ceiling-test run; s>0 offsets by a large prime for a distinct seed.
                seed=args.seed if s == 0 else args.seed + s * 7919,
            )
            with torch.no_grad():
                seed_probs.append(torch.sigmoid(model(x_held).squeeze(-1)).cpu().numpy())
        probs = np.mean(seed_probs, axis=0)
        found = top_window(probs, w, stride)
        iou = window_iou(found, rec.edit_window) if found is not None else 0.0
        top_m = getattr(args, "top_m", 1)
        found_m = top_m_windows(probs, w, stride, top_m) if top_m > 1 else []
        held_scores.append(probs)
        held_labels.append(y_all[held_mask])
        rows.append(
            {
                "failure_id": rec.failure_id,
                "cause": rec.cause,
                "found_window": list(found) if found else None,
                "found_windows": [list(fw) for fw in found_m] if found_m else None,
                "truth_window": list(rec.edit_window),
                "iou": iou,
            }
        )
        if (gi + 1) % 10 == 0 or gi + 1 == len(episodes):
            print(f"  [{gi + 1:2d}/{len(episodes)}] {label}: fold {gi + 1}/{len(episodes)}")

    pooled = auroc(np.concatenate(held_scores), np.concatenate(held_labels))
    return rows, pooled


# ---------------------------------------------------------------------------


def main() -> int:  # noqa: PLR0915 — a linear diagnostic pipeline reads better than split-up
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    dyn_path = REPO_ROOT / cfg["dynamics_config"]
    env_cfg = yaml.safe_load(env_path.read_text())
    reduction = yaml.safe_load(dyn_path.read_text())["dynamics"]["reduction"]
    stride = int(args.stride if args.stride is not None else cfg["search"]["windows"]["stride"])
    run_id = args.run_id or f"ceiling-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    w = int(args.window_length)

    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    import torch

    from cfe.data.failure_set import INJECTED, failure_set_dir, load_failure_set
    from cfe.dynamics.data import load_episode
    from cfe.eval.metrics import window_iou
    from scripts.train_dynamics import open_cache

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"diagnose_fault_localization  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"task_id={args.task_id}  window_length={w}  stride={stride}  reduction={reduction}")
    print(f"!! {NOT_A_RESULT}")
    print("-" * 78)

    cache, renderer, _, _ = open_cache({"encoder_config": cfg["encoder_config"], **cfg}, args.seed)
    records = load_failure_set(failure_set_dir(REPO_ROOT), source=INJECTED, task_id=args.task_id)
    if not records:
        raise RuntimeError(f"no injected failures for task {args.task_id}")

    # -- load every matched pair once; latents are memory-mapped, so this is cheap --
    episodes, skipped = [], []
    for rec in records:
        if rec.expert_episode_id is None:
            skipped.append({"failure_id": rec.failure_id, "reason": "no matched expert episode"})
            continue
        ep_c = load_episode(cache, rec.episode_id, reduction)
        ep_e = load_episode(cache, rec.expert_episode_id, reduction)
        n, truncated = shared_length(len(ep_c.z), len(ep_e.z))
        t_actions = n - 1
        if t_actions < w:
            skipped.append(
                {
                    "failure_id": rec.failure_id,
                    "reason": f"only {t_actions} shared action steps, shorter than window {w}",
                }
            )
            continue
        episodes.append(
            {
                "rec": rec,
                "z_c": ep_c.z[:n].reshape(n, -1).astype(np.float32),
                "z_e": ep_e.z[:n].reshape(n, -1).astype(np.float32),
                "t_actions": t_actions,
                "truncated": truncated,
            }
        )
    n_truncated = sum(1 for e in episodes if e["truncated"])
    print(f"matched pairs usable: {len(episodes)}/{len(records)}  skipped: {len(skipped)}")
    print(f"truncated (corrupted/clean episode lengths differed): {n_truncated}/{len(episodes)}")

    # -- Part 1: oracle pairwise divergence --------------------------------------------------
    part1_rows = []
    for e in episodes:
        rec = e["rec"]
        # Frames 1..n-1 (post-action), aligned to action indices 0..t_actions-1 — the same
        # convention `cfe/search/windows.py:per_step_divergence` uses for its entry `t`.
        div = np.linalg.norm(e["z_c"][1:] - e["z_e"][1:], axis=1)
        found = top_window(div, w, stride)
        iou = window_iou(found, rec.edit_window) if found is not None else 0.0
        part1_rows.append(
            {
                "failure_id": rec.failure_id,
                "cause": rec.cause,
                "truncated": e["truncated"],
                "found_window": list(found) if found else None,
                "truth_window": list(rec.edit_window),
                "iou": iou,
            }
        )
    print(f"Part 1 (oracle pairwise divergence): {len(part1_rows)} episodes scored")

    # -- Part 2: leave-one-episode-out classifier on latent content --------------------------
    latent_feats = [e["z_c"][1 : 1 + e["t_actions"]] for e in episodes]
    part2_rows, part2_auroc = loeo_probe(
        latent_feats, episodes, w=w, stride=stride, args=args, device=device, label="part2/latent"
    )
    print(f"Part 2 (LOEO classifier, latent content): {len(part2_rows)} folds")

    # -- Part 3: the same probe, given ONLY timestep position, never the latent --------------
    # Content-blind control for Part 2 — see position_features()'s docstring for why this is
    # necessary here specifically (libero_object's very regular phase timing).
    position_feats = [position_features(e["t_actions"]) for e in episodes]
    part3_rows, part3_auroc = loeo_probe(
        position_feats,
        episodes,
        w=w,
        stride=stride,
        args=args,
        device=device,
        label="part3/position-only",
    )
    print(f"Part 3 (LOEO classifier, position-only control): {len(part3_rows)} folds")

    # -- summarise, sanity-check, write ------------------------------------------------------
    part1_summary = summarise_iou(part1_rows)
    part2_summary = summarise_iou(part2_rows)
    part3_summary = summarise_iou(part3_rows)
    part1_pooled_mean = part1_summary["pooled"]["mean_iou"] or 0.0
    part2_pooled_mean = part2_summary["pooled"]["mean_iou"] or 0.0
    part3_pooled_mean = part3_summary["pooled"]["mean_iou"] or 0.0
    sanity_warning = None
    if part1_pooled_mean < TRAINED_MODEL_STAGE1_MEAN_IOU:
        sanity_warning = (
            f"Part 1's oracle mean IoU ({part1_pooled_mean:.4f}) did not clear the trained "
            f"model's stage-1 heuristic ({TRAINED_MODEL_STAGE1_MEAN_IOU}), despite using the "
            "true counterfactual latent. That is suspicious — recheck before trusting either "
            "number (CLAUDE.md: distrust a new metric until something independent agrees)."
        )
        print(f"!! SANITY CHECK FAILED: {sanity_warning}")

    content_vs_position = {
        "part2_minus_part3_pooled_mean_iou": part2_pooled_mean - part3_pooled_mean,
        "part2_minus_part3_pooled_auroc": (
            None if part2_auroc is None or part3_auroc is None else part2_auroc - part3_auroc
        ),
        "reading": "part2 (latent content) must clear part3(position-only) by a wide margin "
        "for part2's IoU/AUROC to mean the CONTENT of z_t localises the fault. If the two are "
        "close, part2 is most likely reading libero_object's consistent phase timing (gripper "
        "closes ~t42, releases ~t128 in nearly every demo) rather than anything encoded about "
        "the fault itself — the same confound that would let a position-only probe do just as "
        "well without ever looking at a latent.",
    }
    if part2_pooled_mean - part3_pooled_mean < 0.05:
        print(
            "!! CONFOUND WARNING: part2 (latent) barely beats part3 (position-only); its "
            "localization signal may be a timing prior, not latent content. See "
            "content_vs_position in the summary."
        )

    summary = {
        "run_id": run_id,
        "kind": "summary",
        "diagnostic": True,
        "not_a_result": NOT_A_RESULT,
        "seed": args.seed,
        "task_id": args.task_id,
        "window_length": w,
        "stride": stride,
        "reduction": reduction,
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "n_records": len(records),
        "n_used": len(episodes),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_truncated": n_truncated,
        "part1_oracle_pairwise_divergence": part1_summary,
        "part2_loeo_classifier": {**part2_summary, "pooled_auroc": part2_auroc},
        "part3_position_only_control": {**part3_summary, "pooled_auroc": part3_auroc},
        "content_vs_position": content_vs_position,
        "reference_trained_model_stage1_mean_iou": TRAINED_MODEL_STAGE1_MEAN_IOU,
        "sanity_warning": sanity_warning,
    }

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "search.yaml")
    shutil.copyfile(enc_path, out_dir / "encoder.yaml")
    shutil.copyfile(env_path, out_dir / "env.yaml")
    shutil.copyfile(dyn_path, out_dir / "dynamics.yaml")
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for r in part1_rows:
            row = {"run_id": run_id, "kind": "part1_episode", **r}
            fh.write(json.dumps(row, default=str) + "\n")
        for r in part2_rows:
            row = {"run_id": run_id, "kind": "part2_episode", **r}
            fh.write(json.dumps(row, default=str) + "\n")
        for r in part3_rows:
            row = {"run_id": run_id, "kind": "part3_episode", **r}
            fh.write(json.dumps(row, default=str) + "\n")
        fh.write(json.dumps(summary, default=str) + "\n")

    print("-" * 78)
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"}, indent=2, default=str))
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
