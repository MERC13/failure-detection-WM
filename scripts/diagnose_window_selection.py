"""Is the rising action-conditioning benefit horizon-driven or episode-selection-driven?

`scripts/diagnose_action_horizon.py` found `rel_improvement` rising from +3.9% at h8 to +28.9%
at h96 -- but on only the 742 windows long enough to supply 128 future frames. Scored on the
gate's full window set the same model reads +3.5% at h8, and on the 1702-window h64 set it reads
+17.3% at h64 versus +25.6% on the 742-window set. Same model, same horizon, different windows,
materially different answer. Long horizons are only measurable in long episodes, so horizon and
episode length are confounded and neither number can be trusted on its own.

This separates them. `rel_improvement` is computed for every (horizon, window-set) pair, where a
window set is defined by how many future frames it requires:

    reading DOWN a column  = pure horizon effect   (window set held fixed)
    reading ACROSS a row   = pure selection effect (horizon held fixed)

If the h8 row is flat across window sets, episode length does not matter and the horizon trend
is real. If the h8 row climbs, longer episodes simply have more action-conditioning signal and
the horizon story is at least partly selection.

Diagnostic only. Gates nothing, changes nothing, cannot make G2 pass.

Usage:
    python scripts/diagnose_window_selection.py --dyn-run p2-dyn-res --seed 0
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

from scripts.verify_g2 import per_window_errors  # noqa: E402

HORIZONS = [8, 16, 32, 64, 96]
# Minimum future frames a window must supply to enter the set. 64 is the gate's own diagnostic
# set; 128 is the set diagnose_action_horizon.py used.
MIN_FRAMES = [8, 16, 32, 64, 96, 128]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="horizon vs episode-selection (diagnostic)")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "dynamics.yaml")
    p.add_argument("--dyn-run", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"diag-window-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    cfg = yaml.safe_load(args.config.read_text())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from cfe.dynamics.data import WindowIndex, load_episode
    from cfe.dynamics.model import DynamicsConfig, LatentDynamics
    from scripts.train_dynamics import open_cache

    run_dir = REPO_ROOT / "results" / "runs" / args.dyn_run
    ck = torch.load(run_dir / "dynamics.pt", map_location="cpu", weights_only=False)
    dyn_cfg = DynamicsConfig.from_dict(ck["dynamics_config"])
    batch_size = int(cfg["train"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"DIAGNOSTIC (not a gate)  run_id={run_id}  dyn_run={args.dyn_run}  device={device}")
    print("-" * 78)

    cache, _, _, _ = open_cache(cfg, args.seed)
    val_eps = [load_episode(cache, e, dyn_cfg.reduction) for e in ck["val_episode_ids"]]

    models = {}
    for tag, ignore in (("dynamics", False), ("action_ignoring", True)):
        m = LatentDynamics(dyn_cfg, ignore_actions=ignore).to(device)
        m.load_state_dict(ck[tag])
        models[tag] = m.eval()

    table: dict[int, dict[int, float]] = {h: {} for h in HORIZONS}
    counts: dict[int, int] = {}
    rows = []

    for mf in MIN_FRAMES:
        idx = WindowIndex(val_eps, dyn_cfg.k_context, dyn_cfg.horizon, mf // dyn_cfg.horizon)
        counts[mf] = len(idx)
        if len(idx) == 0:
            continue
        # Only horizons this set can actually score.
        hs = [h for h in HORIZONS if h <= mf]
        me = per_window_errors(models["dynamics"], idx, hs, batch_size, device)
        te = per_window_errors(models["action_ignoring"], idx, hs, batch_size, device)
        for h in hs:
            m, t = float(me[h].mean()), float(te[h].mean())
            rel = (t - m) / t if t else 0.0
            table[h][mf] = rel
            rows.append(
                {
                    "horizon": h,
                    "min_future_frames": mf,
                    "n_windows": len(idx),
                    "model": round(m, 6),
                    "action_ignoring": round(t, 6),
                    "rel_improvement": round(rel, 4),
                }
            )

    hdr = "  ".join(f"{mf:>8}" for mf in MIN_FRAMES)
    print("rel_improvement by horizon (rows) x min future frames required (cols)\n")
    print(f"{'':>5}  {hdr}")
    print(f"{'n=':>5}  " + "  ".join(f"{counts.get(mf, 0):>8}" for mf in MIN_FRAMES))
    print("-" * 78)
    for h in HORIZONS:
        cells = "  ".join(
            f"{table[h][mf]:>+8.4f}" if mf in table[h] else f"{'-':>8}" for mf in MIN_FRAMES
        )
        print(f"h{h:<4}  {cells}")
    print("-" * 78)

    # The decisive read: how much does rel_improvement move at a FIXED horizon as the window set
    # is restricted to longer episodes?
    for h in HORIZONS:
        vals = table[h]
        if len(vals) >= 2:
            lo_mf, hi_mf = min(vals), max(vals)
            lo, hi = vals[lo_mf], vals[hi_mf]
            fold = (hi / lo) if lo else float("nan")
            print(
                f"h{h}: {lo:+.4f} (n={counts[lo_mf]}, >={lo_mf}f) -> "
                f"{hi:+.4f} (n={counts[hi_mf]}, >={hi_mf}f)   {fold:.2f}x from selection alone"
            )
    print("-" * 78)
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
                        "kind": "window_selection",
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
                    "window_counts": counts,
                    "is_gate": False,
                }
            )
            + "\n"
        )
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
