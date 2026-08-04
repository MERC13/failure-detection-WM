"""Does the model discriminate *small* action edits? (diagnostic, NOT a gate)

SPEC.md 5's search perturbs actions by small amounts and asks the world model what changes. No
gate currently tests that capability. `beats_action_ignoring` only asks whether actions help *on
average*, which a model could satisfy by predicting coarse trajectory drift while being blind to
the fine edits the search actually makes.

That distinction decides whether G2's h8 FAIL is a scale artifact or a real defect, so it is
worth measuring before anyone argues about the gated horizon.

**The measurement, and why it needs no simulator.** Ground truth is the episode's real future.
Feed the model the true actions, then feed it deliberately wrong ones, and the error must rise.
How much it rises as a function of edit size *is* edit discrimination:

  - error climbs smoothly from the smallest edits -> the model resolves fine action differences,
    and the search has a gradient to follow;
  - error flat until the edit is huge -> the model only sees coarse motion, and a search over
    small edits is optimising noise no matter how good `beats_action_ignoring` looks.

The action-ignoring twin is the control. It zeroes actions inside `forward`, so its error MUST
be identical under every perturbation. If it moves, this harness is broken and no row below
means anything.

**Only the 6 end-effector delta dimensions are perturbed.** CLAUDE.md: the gripper dimension is
not continuous in the same sense as the others, so adding Gaussian noise scaled by its standard
deviation would not be the same operation at all. Gripper edits are a separate question and are
deliberately not mixed into this one.

Diagnostic only. Gates nothing, changes nothing, cannot make G2 pass.

Usage:
    python scripts/diagnose_edit_discrimination.py --dyn-run p2-dyn --seed 0
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EE_DIMS = 6  # first 6 of the 7 action dims; index 6 is the gripper and is left alone
HORIZONS = [8, 64]
# Edit sizes as multiples of each EE dimension's own standard deviation over the dataset.
EPSILONS = [0.05, 0.10, 0.25, 0.50, 1.00]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="small-action-edit discrimination (diagnostic)")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "dynamics.yaml")
    p.add_argument("--dyn-run", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def errors_under(model, index, action_fn, horizons, batch_size, device, seed):
    """Per-window error at each horizon, with `action_fn` applied to the action chunk first."""
    n_steps = max(horizons)
    rng = np.random.default_rng(seed)
    out = {h: [] for h in horizons}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = np.arange(start, min(start + batch_size, len(index)))
            b = index.gather(rows)
            acts = action_fn(b["actions"][:, :n_steps].copy(), rng)
            pred = model.rollout(
                torch.from_numpy(b["context"]).to(device),
                torch.from_numpy(acts).to(device),
            )
            tgt = torch.from_numpy(b["targets"][:, :n_steps]).to(device)
            err = (pred - tgt).abs().mean(dim=(2, 3))
            for h in horizons:
                out[h].append(err[:, h - 1].cpu().numpy())
    return {h: np.concatenate(v) for h, v in out.items()}


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"diag-edit-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    cfg = yaml.safe_load(args.config.read_text())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from cfe.dynamics.data import WindowIndex, load_episode
    from cfe.dynamics.latents import paired_bootstrap_ci
    from cfe.dynamics.model import DynamicsConfig, LatentDynamics
    from scripts.train_dynamics import open_cache

    run_dir = REPO_ROOT / "results" / "runs" / args.dyn_run
    ck = torch.load(run_dir / "dynamics.pt", map_location="cpu", weights_only=False)
    dyn_cfg = DynamicsConfig.from_dict(ck["dynamics_config"])
    batch_size = int(cfg["train"]["batch_size"])
    n_boot = int(cfg["eval"]["n_bootstrap"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"DIAGNOSTIC (not a gate)  run_id={run_id}  dyn_run={args.dyn_run}  device={device}")
    print(f"residual={dyn_cfg.residual}  perturbing action dims 0..{EE_DIMS - 1} (gripper held)")
    print("-" * 78)

    cache, _, _, _ = open_cache(cfg, args.seed)
    val_eps = [load_episode(cache, e, dyn_cfg.reduction) for e in ck["val_episode_ids"]]
    idx = WindowIndex(val_eps, dyn_cfg.k_context, dyn_cfg.horizon, max(HORIZONS) // dyn_cfg.horizon)
    print(f"held-out episodes: {len(val_eps)}   windows: {len(idx)}")

    models = {}
    for tag, ignore in (("dynamics", False), ("action_ignoring", True)):
        m = LatentDynamics(dyn_cfg, ignore_actions=ignore).to(device)
        m.load_state_dict(ck[tag])
        models[tag] = m.eval()

    # Per-dimension action scale over the held-out windows, so an edit of "0.1 std" means the
    # same thing in every dimension regardless of its units.
    all_acts = np.concatenate([e.actions for e in val_eps], axis=0)
    std = all_acts[:, :EE_DIMS].std(axis=0)
    print(f"EE action std per dim: {np.round(std, 4).tolist()}")
    print("-" * 78)

    def perturb(eps):
        def fn(acts, rng):
            noise = rng.standard_normal(size=acts[:, :, :EE_DIMS].shape) * (std * eps)
            acts[:, :, :EE_DIMS] += noise.astype(acts.dtype)
            return acts

        return fn

    def shuffle_fn(acts, rng):
        """Whole chunk replaced by another window's actions -- the largest possible edit."""
        return acts[rng.permutation(len(acts))]

    rows = []
    for tag, model in models.items():
        base = errors_under(model, idx, lambda a, r: a, HORIZONS, batch_size, device, args.seed)
        print(
            f"\n[{tag}]  baseline error: "
            + "  ".join(f"h{h}={base[h].mean():.6f}" for h in HORIZONS)
        )
        print(
            f"{'edit':>10}  "
            + "  ".join(
                f"{'h' + str(h) + ' delta':>12}  {'rel':>8}  {'CI excl 0':>10}" for h in HORIZONS
            )
        )
        for label, fn in [(f"{e:.2f} std", perturb(e)) for e in EPSILONS] + [
            ("shuffled", shuffle_fn)
        ]:
            pe = errors_under(model, idx, fn, HORIZONS, batch_size, device, args.seed)
            cells, rec = [], {"model": tag, "edit": label}
            for h in HORIZONS:
                d = float(pe[h].mean() - base[h].mean())
                rel = d / float(base[h].mean())
                ci = paired_bootstrap_ci(pe[h], base[h], n_boot, args.seed)
                cells.append(f"{d:>+12.6f}  {rel:>+8.2%}  {str(ci['excludes_zero']):>10}")
                rec[f"h{h}"] = {
                    "baseline": round(float(base[h].mean()), 6),
                    "perturbed": round(float(pe[h].mean()), 6),
                    "delta": round(d, 6),
                    "rel_delta": round(rel, 4),
                    "ci_low": round(ci["ci_low"], 6),
                    "ci_high": round(ci["ci_high"], 6),
                    "excludes_zero": ci["excludes_zero"],
                }
            rows.append(rec)
            print(f"{label:>10}  " + "  ".join(cells))

    print("\n" + "-" * 78)
    twin = [r for r in rows if r["model"] == "action_ignoring"]
    if all(abs(r[f"h{h}"]["delta"]) < 1e-9 for r in twin for h in HORIZONS):
        print("CONTROL OK: the action-ignoring twin is bit-identical under every perturbation.")
    else:
        print("!! CONTROL FAILED: the twin's error moved under perturbation. It is supposed to")
        print("!! ignore actions entirely, so this harness is wrong and no row above is usable.")
    print("This is a DIAGNOSTIC. G2's FAIL at h8 is unaffected and no threshold has moved.")

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dynamics.yaml", "encoder.yaml", "env.yaml"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "kind": "edit_discrimination",
                        "dyn_run": args.dyn_run,
                        "seed": args.seed,
                        **r,
                    }
                )
                + "\n"
            )
        fh.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "kind": "summary",
                    "dyn_run": args.dyn_run,
                    "seed": args.seed,
                    "n_windows": len(idx),
                    "perturbed_dims": EE_DIMS,
                    "is_gate": False,
                }
            )
            + "\n"
        )
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
