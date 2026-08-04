"""How far does action-conditioning benefit grow with rollout horizon? (diagnostic, NOT a gate)

G2 gates `beats_action_ignoring` at h=8 and it fails there: 0.0350 relative improvement for
`p2-dyn-res`, 0.0499 for `p2-dyn`, against a pre-registered 0.20. But the same quantity measured
at the non-gated diagnostic horizons rises monotonically -- +6.3% at h16, +11.6% at h32, +17.3%
at h64 -- and SPEC.md 5's counterfactual search rolls forward from the edit window to episode
end, which is far past 8 steps.

So the gated number may be measured in a regime the project never operates in. This script
extends the same measurement to h96 and h128 to find out where, if anywhere, the curve crosses
0.20.

**This changes nothing and gates nothing.** It does not touch `configs/dynamics.yaml`, does not
re-run G2, and cannot make a failing gate pass. G2's FAIL at h8 stands regardless of what this
prints. Its only purpose is to put a measurement in front of the user so *they* can decide
whether what G2 measures is the right thing to measure -- a decision that belongs in
CHANGELOG.md if it is ever taken, per CLAUDE.md rule 3.

Measurement functions are imported from scripts/verify_g2.py rather than reimplemented, so these
numbers come from the identical code path as the gated ones and are directly comparable.

Usage:
    python scripts/diagnose_action_horizon.py --dyn-run p2-dyn-res --seed 0
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

from scripts.verify_g2 import copy_last_latent_errors, per_window_errors  # noqa: E402

# Includes the gated h8 so the diagnostic reproduces the gate's own number as a cross-check:
# if h8 here disagrees with verify_g2.py, this script is wrong and nothing below it is usable.
HORIZONS = [8, 16, 32, 64, 96, 128]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="action-conditioning benefit vs horizon (diagnostic)")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "dynamics.yaml")
    p.add_argument("--dyn-run", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"diag-horizon-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    print(f"residual={dyn_cfg.residual}  reduction={dyn_cfg.reduction}  H={dyn_cfg.horizon}")
    print("-" * 78)

    cache, _, _, _ = open_cache(cfg, args.seed)
    val_eps = [load_episode(cache, e, dyn_cfg.reduction) for e in ck["val_episode_ids"]]

    models = {}
    for tag, ignore in (("dynamics", False), ("action_ignoring", True)):
        m = LatentDynamics(dyn_cfg, ignore_actions=ignore).to(device)
        m.load_state_dict(ck[tag])
        models[tag] = m.eval()

    # Every horizon must be scored on the SAME window set, or a rising curve could just be
    # survivorship: long horizons only exist in long episodes, which may be the easy ones.
    chunks = max(HORIZONS) // dyn_cfg.horizon
    idx = WindowIndex(val_eps, dyn_cfg.k_context, dyn_cfg.horizon, chunks)
    print(f"held-out episodes: {len(val_eps)}")
    print(f"windows supplying {max(HORIZONS)} future frames: {len(idx)}")
    if len(idx) < 100:
        print(f"\n!! only {len(idx)} windows -- too few for a trustworthy CI at h{max(HORIZONS)}.")
        print("!! Reporting anyway, but treat the long-horizon rows as indicative, not decided.")
    print("-" * 78)

    model_err = per_window_errors(models["dynamics"], idx, HORIZONS, batch_size, device)
    twin_err = per_window_errors(models["action_ignoring"], idx, HORIZONS, batch_size, device)
    copy_err = copy_last_latent_errors(idx, HORIZONS, batch_size)

    rows = []
    print(
        f"{'h':>5}  {'model':>9}  {'twin':>9}  {'copy_last':>9}  {'rel_impr':>9}  "
        f"{'CI_low':>9}  {'CI_high':>9}  {'>=0.20':>7}"
    )
    for h in HORIZONS:
        m, t, c = float(model_err[h].mean()), float(twin_err[h].mean()), float(copy_err[h].mean())
        rel = (t - m) / t if t else 0.0
        # (twin - model): positive means the action-conditioned model is better. Same argument
        # order as verify_g2.py's gated call, so the h8 row is directly comparable.
        ci = paired_bootstrap_ci(twin_err[h], model_err[h], n_boot, args.seed)
        rows.append(
            {
                "horizon": h,
                "model": round(m, 6),
                "action_ignoring": round(t, 6),
                "copy_last_latent": round(c, 6),
                "rel_improvement": round(rel, 4),
                "ci_low": round(ci["ci_low"], 6),
                "ci_high": round(ci["ci_high"], 6),
                "excludes_zero": ci["excludes_zero"],
                "meets_0.20": bool(rel >= 0.20),
                "n_windows": int(len(model_err[h])),
            }
        )
        print(
            f"{h:>5}  {m:>9.6f}  {t:>9.6f}  {c:>9.6f}  {rel:>+9.4f}  "
            f"{ci['ci_low']:>9.6f}  {ci['ci_high']:>9.6f}  "
            f"{'YES' if rel >= 0.20 else 'no':>7}"
        )

    print("-" * 78)
    crossing = next((r["horizon"] for r in rows if r["meets_0.20"]), None)
    if crossing is None:
        print(f"rel_improvement does NOT reach 0.20 at any horizon up to {max(HORIZONS)}.")
    else:
        print(f"rel_improvement first reaches 0.20 at h={crossing}.")
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
                        "kind": "horizon_scaling",
                        "dyn_run": args.dyn_run,
                        "seed": args.seed,
                        "gated": r["horizon"] in cfg["eval"]["horizons"],
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
                    "residual": dyn_cfg.residual,
                    "n_windows": len(idx),
                    "first_horizon_meeting_0.20": crossing,
                    "is_gate": False,
                }
            )
            + "\n"
        )
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
