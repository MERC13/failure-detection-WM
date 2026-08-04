"""G2 gate (SPEC.md §7 and §4.2).

The check that matters is `beats_action_ignoring`. SPEC.md §4.2: "If the model does not clearly
beat the action-ignoring variant, it has not learned action-conditioning and the counterfactual
search is meaningless. This is the most important diagnostic in the repo."

What counts as "clearly" was fixed in `configs/dynamics.yaml:gate` **before the model was
trained**, and is read from there rather than written here. Pre-registration is the point:
picking the bar after seeing the number is precisely the failure CLAUDE.md rule 3 describes.

Reuses `scripts/verify_env.py`'s `Result` so G0, G1 and G2 read alike and a check that raises
is a FAIL rather than a skip.

Usage:
    python scripts/verify_g2.py --dyn-run p2-dyn --probe-run p2-probe
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_env import Result  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G2 gate")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "dynamics.yaml")
    p.add_argument("--dyn-run", required=True, help="train_dynamics.py run_id")
    p.add_argument("--probe-run", default=None, help="probe_representation.py run_id for rho")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Open-loop error
# ---------------------------------------------------------------------------


def per_window_errors(model, index, horizons: list[int], batch_size: int, device) -> dict:
    """Mean absolute latent error per window at each horizon, in LayerNormed space.

    Per-window rather than aggregated, because G2's confidence interval is *paired*: the same
    windows are scored for both models and the difference is taken window by window.
    """
    import torch

    n_steps = max(horizons)
    model.eval()
    out = {h: [] for h in horizons}
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = np.arange(start, min(start + batch_size, len(index)))
            b = index.gather(rows)
            pred = model.rollout(
                torch.from_numpy(b["context"]).to(device),
                torch.from_numpy(b["actions"][:, :n_steps]).to(device),
            )
            tgt = torch.from_numpy(b["targets"][:, :n_steps]).to(device)
            err = (pred - tgt).abs().mean(dim=(2, 3))  # (B, n_steps)
            for h in horizons:
                out[h].append(err[:, h - 1].cpu().numpy())
    return {h: np.concatenate(v) for h, v in out.items()}


def copy_last_latent_errors(index, horizons: list[int], batch_size: int) -> dict:
    """Baseline: predict that nothing changes. Strong at short horizons, useless at long ones."""
    out = {h: [] for h in horizons}
    for start in range(0, len(index), batch_size):
        rows = np.arange(start, min(start + batch_size, len(index)))
        b = index.gather(rows)
        last = b["context"][:, -1]  # (B, R, D)
        for h in horizons:
            err = np.abs(b["targets"][:, h - 1] - last).mean(axis=(1, 2))
            out[h].append(err)
    return {h: np.concatenate(v) for h, v in out.items()}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_beats_copy_last(model_err: dict, copy_err: dict, horizons: list[int]) -> Result:
    detail, ok = {}, True
    for h in horizons:
        m, c = float(model_err[h].mean()), float(copy_err[h].mean())
        detail[f"h{h}"] = {
            "model": round(m, 6),
            "copy_last_latent": round(c, 6),
            "rel_improvement": round((c - m) / c, 4) if c else None,
        }
        ok = ok and m < c
    return Result("beats_copy_last_latent", ok, detail)


def check_beats_action_ignoring(
    model_err: dict,
    twin_err: dict,
    rel_model_err: dict,
    rel_twin_err: dict,
    gate: dict,
    n_boot: int,
    seed: int,
) -> Result:
    """The gate. Criteria are read from the config, where they were pre-registered.

    Two criteria on two different window sets, which is why the errors arrive in two pairs:

    - the **CI criterion** runs on the short gated horizons over every window (`model_err`);
    - the **magnitude criterion** runs at `gate["rel_improvement_horizon"]` (`rel_model_err`),
      which only longer episodes can supply, so it necessarily scores fewer windows.

    Scoring them on one set is not possible: a window set that can supply 96 future frames
    excludes a third of the episodes, and applying that restriction to the CI criterion would
    silently change a criterion nobody decided to change.
    """
    from cfe.dynamics.latents import paired_bootstrap_ci

    ci_horizons = [int(h) for h in gate["ci_horizons"]]
    min_rel = float(gate["min_rel_improvement"])
    rel_h = int(gate["rel_improvement_horizon"])

    detail = {
        "criteria": {
            "ci_excludes_zero_at": ci_horizons,
            "min_rel_improvement": min_rel,
            "rel_improvement_horizon": rel_h,
        }
    }
    ci_ok = True
    for h in sorted(set(ci_horizons) | set(model_err)):
        m, t = float(model_err[h].mean()), float(twin_err[h].mean())
        # (twin - model): positive means the action-conditioned model is better.
        ci = paired_bootstrap_ci(twin_err[h], model_err[h], n_boot, seed)
        rel = (t - m) / t if t else 0.0
        detail[f"h{h}"] = {
            "model": round(m, 6),
            "action_ignoring": round(t, 6),
            "rel_improvement": round(rel, 4),
            "ci_low": round(ci["ci_low"], 6),
            "ci_high": round(ci["ci_high"], 6),
            "excludes_zero": ci["excludes_zero"],
            "n_pairs": ci["n_pairs"],
        }
        if h in ci_horizons:
            ci_ok = ci_ok and ci["excludes_zero"] and ci["ci_low"] > 0

    rm, rt = float(rel_model_err[rel_h].mean()), float(rel_twin_err[rel_h].mean())
    rel = (rt - rm) / rt if rt else 0.0
    rel_ci = paired_bootstrap_ci(rel_twin_err[rel_h], rel_model_err[rel_h], n_boot, seed)
    detail[f"h{rel_h}"] = {
        "model": round(rm, 6),
        "action_ignoring": round(rt, 6),
        "rel_improvement": round(rel, 4),
        "ci_low": round(rel_ci["ci_low"], 6),
        "ci_high": round(rel_ci["ci_high"], 6),
        "excludes_zero": rel_ci["excludes_zero"],
        "n_pairs": rel_ci["n_pairs"],
        "is_magnitude_criterion": True,
    }
    detail["ci_criterion_met"] = ci_ok
    detail["rel_criterion_met"] = rel >= min_rel
    return Result("beats_action_ignoring", ci_ok and rel >= min_rel, detail)


def check_goal_distance_correlates(probe_run: str | None, gate: dict, reduction: str) -> Result:
    """SPEC.md §4.2's Spearman rho, read from a completed probe run.

    Gates on the rho for the reduction the model **actually uses**, in LayerNormed space, not on
    whichever row of the sweep scored best. Those can differ, and reporting the best row while
    the model runs on another would be a number that describes something nobody is using.

    This measures the *representation*, not the dynamics model: it uses true terminal latents,
    so no trained weights enter it. That is deliberate — §4.2's stated worry is that "the search
    objective is not a proxy for anything", and the search objective is a distance in this
    latent space. A model-predicted variant would confound representation quality with
    long-horizon rollout error, which is reported separately.
    """
    name = "goal_distance_correlates"
    min_abs = float(gate["min_abs_spearman"])
    if probe_run is None:
        return Result(name, False, {"reason": "no --probe-run given"})
    metrics = REPO_ROOT / "results" / "runs" / probe_run / "metrics.jsonl"
    if not metrics.exists():
        return Result(name, False, {"reason": f"{metrics} not found"})

    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    summary = next((r for r in rows if r.get("kind") == "summary"), None)
    # LayerNormed, because that is the space the dynamics model trains and predicts in.
    row = next(
        (
            r
            for r in rows
            if r.get("kind") == "reduction"
            and r["reduction"] == reduction
            and r["space"] == "layernorm"
        ),
        None,
    )
    if row is None:
        return Result(
            name,
            False,
            {"reason": f"probe run has no layernorm row for reduction {reduction!r}"},
        )

    best = (summary or {}).get("best") or {}
    return Result(
        name,
        bool(row["sign_correct"]) and abs(float(row["spearman_rho"])) >= min_abs,
        {
            "probe_run": probe_run,
            "reduction": reduction,
            "space": "layernorm",
            "spearman_rho": row["spearman_rho"],
            "sign_correct": row["sign_correct"],
            "min_abs_spearman": min_abs,
            "mean_dist_success": row["mean_dist_success"],
            "mean_dist_failure": row["mean_dist_failure"],
            "n_episodes": (summary or {}).get("n_rollout_episodes"),
            "best_in_sweep": {k: best.get(k) for k in ("reduction", "space", "spearman_rho")},
            "measures": "representation (true terminal latents), not model rollout",
        },
    )


def check_eval_is_deterministic(model, index, horizons, batch_size, device) -> Result:
    """The reported numbers must not move between two evaluations of the same weights."""
    a = per_window_errors(model, index, horizons, batch_size, device)
    b = per_window_errors(model, index, horizons, batch_size, device)
    identical = all(np.array_equal(a[h], b[h]) for h in horizons)
    return Result(
        "eval_deterministic",
        identical,
        {
            "horizons": horizons,
            "comparison": "np.array_equal on per-window errors, bit-exact",
            "n_windows": int(len(next(iter(a.values())))),
        },
    )


def check_config_snapshot(run_dir: Path) -> Result:
    """CLAUDE.md rule 4: a number without its config is an orphan."""
    wanted = ["dynamics.yaml", "encoder.yaml", "env.yaml", "metrics.jsonl", "dynamics.pt"]
    present = {f: (run_dir / f).exists() for f in wanted}
    return Result(
        "config_snapshot_present", all(present.values()), {"run_dir": str(run_dir)} | present
    )


# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run_id = args.run_id or f"g2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = REPO_ROOT / "results" / "runs" / args.dyn_run

    env_cfg = yaml.safe_load((REPO_ROOT / cfg["env_config"]).read_text())
    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    import torch

    from cfe.dynamics.data import WindowIndex, load_episode
    from cfe.dynamics.model import DynamicsConfig, LatentDynamics
    from scripts.train_dynamics import open_cache

    ckpt_path = run_dir / "dynamics.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found; run scripts/train_dynamics.py first")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dyn_cfg = DynamicsConfig.from_dict(ck["dynamics_config"])

    ev, gate = cfg["eval"], cfg["gate"]
    horizons = [int(h) for h in ev["horizons"]]
    diag_horizons = [int(h) for h in ev["diagnostic_horizons"]]
    batch_size = int(cfg["train"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"G2 gate  run_id={run_id}  dyn_run={args.dyn_run}  device={device}")
    print(f"pre-registered criteria: {json.dumps(gate)}")
    print("-" * 78)

    cache, renderer, _, _ = open_cache(cfg, args.seed)
    val_ids = ck["val_episode_ids"]
    val_eps = [load_episode(cache, e, dyn_cfg.reduction) for e in val_ids]
    print(f"held-out episodes: {len(val_eps)}")

    models = {}
    for tag, ignore in (("dynamics", False), ("action_ignoring", True)):
        m = LatentDynamics(dyn_cfg, ignore_actions=ignore).to(device)
        m.load_state_dict(ck[tag])
        models[tag] = m.eval()

    # Gated horizons use every window with H future frames; the long diagnostic horizons need
    # 64, which only longer episodes can supply, so they get their own (smaller) window set.
    gated_idx = WindowIndex(val_eps, dyn_cfg.k_context, dyn_cfg.horizon, 1)
    diag_chunks = max(diag_horizons) // dyn_cfg.horizon
    diag_idx = WindowIndex(val_eps, dyn_cfg.k_context, dyn_cfg.horizon, diag_chunks)
    # The magnitude criterion moved to a long horizon on 2026-08-04 (CHANGELOG.md), and only
    # longer episodes can supply that many future frames, so it gets its own window set.
    rel_h = int(gate["rel_improvement_horizon"])
    rel_idx = WindowIndex(val_eps, dyn_cfg.k_context, dyn_cfg.horizon, rel_h // dyn_cfg.horizon)
    print(
        f"windows: {len(gated_idx)} gated, {len(diag_idx)} for horizons up to "
        f"{max(diag_horizons)}, {len(rel_idx)} for the h{rel_h} magnitude criterion"
    )
    print("-" * 78)

    model_err = per_window_errors(models["dynamics"], gated_idx, horizons, batch_size, device)
    twin_err = per_window_errors(models["action_ignoring"], gated_idx, horizons, batch_size, device)
    copy_err = copy_last_latent_errors(gated_idx, horizons, batch_size)
    rel_model_err = per_window_errors(models["dynamics"], rel_idx, [rel_h], batch_size, device)
    rel_twin_err = per_window_errors(
        models["action_ignoring"], rel_idx, [rel_h], batch_size, device
    )

    results = [
        check_beats_copy_last(model_err, copy_err, horizons),
        check_beats_action_ignoring(
            model_err,
            twin_err,
            rel_model_err,
            rel_twin_err,
            gate,
            int(ev["n_bootstrap"]),
            args.seed,
        ),
        check_goal_distance_correlates(args.probe_run, gate, dyn_cfg.reduction),
        check_eval_is_deterministic(models["dynamics"], gated_idx, horizons, batch_size, device),
        check_config_snapshot(run_dir),
    ]

    # Non-gating: SPEC.md §5's search rolls far past horizon 8 and nothing above would show
    # that the error has exploded by then.
    diagnostics = {}
    if len(diag_idx):
        dm = per_window_errors(models["dynamics"], diag_idx, diag_horizons, batch_size, device)
        dt = per_window_errors(
            models["action_ignoring"], diag_idx, diag_horizons, batch_size, device
        )
        dc = copy_last_latent_errors(diag_idx, diag_horizons, batch_size)
        diagnostics = {
            f"h{h}": {
                "model": round(float(dm[h].mean()), 6),
                "action_ignoring": round(float(dt[h].mean()), 6),
                "copy_last_latent": round(float(dc[h].mean()), 6),
                "rel_improvement_vs_action_ignoring": round(
                    float((dt[h].mean() - dm[h].mean()) / dt[h].mean()), 4
                ),
            }
            for h in diag_horizons
        }

    for r in results:
        print(r.line())
    if diagnostics:
        print("-" * 78)
        print("extended horizons (NOT gated; SPEC.md §5 rolls far past h=8):")
        for h, d in diagnostics.items():
            print(
                f"  {h:>4s}  model={d['model']:.5f}  action_ignoring={d['action_ignoring']:.5f}  "
                f"copy_last={d['copy_last_latent']:.5f}  "
                f"rel_improvement={d['rel_improvement_vs_action_ignoring']:+.3f}"
            )

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "dynamics.yaml")
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "gate": "G2",
                        "check": r.name,
                        "passed": r.passed,
                        "seed": args.seed,
                        "dyn_run": args.dyn_run,
                        "detail": r.detail,
                    },
                    default=str,
                )
                + "\n"
            )
        fh.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "kind": "diagnostic",
                    "extended_horizons": diagnostics,
                    "renderer": renderer,
                    "n_val_episodes": len(val_eps),
                    "n_gated_windows": len(gated_idx),
                    "n_diagnostic_windows": len(diag_idx),
                    "n_rel_criterion_windows": len(rel_idx),
                },
                default=str,
            )
            + "\n"
        )

    failed = [r.name for r in results if not r.passed]
    print("-" * 78)
    print(
        f"G2: {'PASS' if not failed else 'FAIL'} "
        f"({len(results) - len(failed)}/{len(results)} checks"
        + (f"; failed: {', '.join(failed)}" if failed else "")
        + ")"
    )
    print(f"artifacts: {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
