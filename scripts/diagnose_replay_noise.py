"""How much does RSR move for reasons that are not the method? Run before G3, not after.

CLAUDE.md: "Distrust a new metric until something independent agrees with it. ... When you add
a measurement, measure its noise floor first and check it against a signal you already trust
before reporting anything from it." RSR is a new measurement. A margin smaller than the spread
measured here is not a margin, and that has to be known *before* the number it qualifies exists,
or the threshold gets chosen against the answer.

Three things, in increasing order of how much they should worry us:

1. **Replay determinism (I5).** The same edit replayed twice must give a byte-identical
   `ReplayResult`. If this fails, nothing else in the file means anything and G3 cannot be run.
2. **`random_perturb` re-seed spread.** The same episodes, the same budget, the same matched
   magnitude, different RNG seeds. This is the sampling floor of §6's decisive baseline: if
   re-seeding alone moves RSR by 0.2, a 0.2 margin over it is noise.
3. **Single-step sensitivity.** Change exactly one action by a small amount and see how often
   the episode's outcome flips. The first smoke run showed a K=1 `expert_oracle` failing where
   `no_edit` at the same window succeeded, which suggests outcomes are near-chaotic at small
   budgets. Quantified here rather than inferred from one anecdote.

Usage:
    python scripts/diagnose_replay_noise.py --run-id p3-noise --n-episodes 5
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise floor of the replay success rate")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--n-reseeds", type=int, default=3)
    p.add_argument("--budget-k", type=int, default=8)
    p.add_argument("--magnitude", type=float, default=0.3, help="matched per-step L2, dims 0-5")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument(
        "--placement",
        choices=("after", "before"),
        default="after",
        help="place the edit window at the injection ('after') or ahead of it ('before')",
    )
    return p.parse_args()


def main() -> int:  # noqa: PLR0915 — a linear pipeline reads better than split-up stages
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    enc_yaml = yaml.safe_load(enc_path.read_text())
    env_cfg = yaml.safe_load(env_path.read_text())
    run_id = args.run_id or f"p3-noise-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    import torch

    from cfe.data.demos import make_replay_env
    from cfe.data.failure_set import INJECTED, failure_set_dir, load_failure_set
    from cfe.eval.metrics import edit_magnitude, no_edit, random_perturb, replay_success_rate
    from cfe.eval.replay_protocol import Edit, replay_primary
    from scripts.run_replay import make_policy_fn
    from scripts.verify_env import load_suite

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = load_failure_set(failure_set_dir(REPO_ROOT), source=INJECTED, task_id=args.task_id)
    if not records:
        raise RuntimeError("no injected failures; run scripts/build_failure_set.py first")
    # Stratified across causes. Taking the first N sorted by failure_id picked five
    # `early_release` episodes on the first run — "early_release" sorts first — and every
    # quantity below then described one cause rather than the failure set.
    by_cause: dict[str, list] = {}
    for r in records:
        by_cause.setdefault(r.cause, []).append(r)
    chosen, i = [], 0
    while len(chosen) < args.n_episodes and any(len(v) > i for v in by_cause.values()):
        for c in sorted(by_cause):
            if len(chosen) >= args.n_episodes:
                break
            if i < len(by_cause[c]):
                chosen.append(by_cause[c][i])
        i += 1

    suite = load_suite({"suite": enc_yaml["data"]["suite"]})
    task = suite.get_task(args.task_id)
    env = make_replay_env(env_cfg, task, args.seed)
    policy_fn, _ = make_policy_fn(cfg["policy_runs"][args.task_id], env_cfg, enc_yaml, device)

    # The window is placed relative to the INJECTION, not at a fixed fraction of the episode.
    #
    # SPEC.md §6's primary variant replaces everything after `window_end` with the policy, so an
    # edit window that starts *before* the injected fault means the fault is never executed at
    # all: the replay becomes "original prefix, then the BC policy", and it succeeds at the
    # policy's own rate. The first version of this script used a mid-episode window and, on
    # `early_release` episodes injected at step ~90-140, placed every window at ~t=77 — before
    # the fault. Every arm came out at RSR 1.000 and the two noise quantities measured a
    # ceiling instead of a floor.
    #
    # `--placement after` starts the window at the injection, which is the only placement where
    # the primary variant actually replays the failure. `before` reproduces the bypass, so the
    # size of the effect is measured rather than asserted.
    def window_for(rec) -> int:
        lo, _hi = rec.edit_window
        start = lo if args.placement == "after" else max(0, lo - 3 * args.budget_k)
        return int(min(max(start, 4), rec.n_steps - args.budget_k))

    print(f"diagnose_replay_noise  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"episodes={len(chosen)}  K={args.budget_k}  reseeds={args.n_reseeds}")
    print("-" * 78)

    rows = []
    t_start = time.time()
    try:
        # -- 1. determinism -------------------------------------------------
        det = []
        for rec in chosen[:3]:
            w = window_for(rec)
            edit = no_edit(rec.actions, w, args.budget_k)
            outs = [
                replay_primary(env, rec.init_state, rec.actions, edit, policy_fn) for _ in range(2)
            ]
            identical = outs[0] == outs[1]
            det.append(
                {
                    "failure_id": rec.failure_id,
                    "identical": bool(identical),
                    "success": [bool(o.success) for o in outs],
                    "steps": [o.steps_executed for o in outs],
                }
            )
            print(
                f"  determinism {rec.failure_id[-30:]}: identical={identical} "
                f"success={[o.success for o in outs]} steps={[o.steps_executed for o in outs]}"
            )
        rows.append(
            {"kind": "determinism", "cases": det, "all_identical": all(d["identical"] for d in det)}
        )

        # -- 2. random_perturb re-seed spread -------------------------------
        print("-" * 78)
        per_seed = []
        for s in range(args.n_reseeds):
            results, mags = [], []
            # One generator per seed, drawn from across the whole episode set: constructing it
            # inside the loop would hand every episode the same random direction and measure
            # one draw instead of the spread of the arm.
            rng = np.random.default_rng(1000 + s)
            for rec in chosen:
                w = window_for(rec)
                edit = random_perturb(
                    rec.actions, w, args.budget_k, rng, magnitude_scale=args.magnitude
                )
                out = replay_primary(env, rec.init_state, rec.actions, edit, policy_fn)
                results.append(out)
                mags.append(
                    edit_magnitude(rec.actions[w : w + args.budget_k], edit.actions)["mean_l2_ee"]
                )
            rsr = replay_success_rate(results)
            per_seed.append(
                {
                    "rng_seed": 1000 + s,
                    "replay_success_rate": rsr["replay_success_rate"],
                    "n_success": rsr["n_success"],
                    "n": rsr["n"],
                    "mean_realised_l2_ee": round(float(np.mean(mags)), 6),
                }
            )
            print(
                f"  random_perturb seed={1000 + s}: RSR={rsr['replay_success_rate']:.3f} "
                f"({rsr['n_success']}/{rsr['n']})  realised_l2={np.mean(mags):.4f}"
            )
        spread = max(p["replay_success_rate"] for p in per_seed) - min(
            p["replay_success_rate"] for p in per_seed
        )
        rows.append(
            {
                "kind": "random_perturb_reseed",
                "per_seed": per_seed,
                "rsr_spread": spread,
                "requested_magnitude": args.magnitude,
            }
        )

        # -- 3. single-step sensitivity -------------------------------------
        print("-" * 78)
        sens = []
        for rec in chosen:
            w = window_for(rec)
            base = replay_primary(
                env, rec.init_state, rec.actions, no_edit(rec.actions, w, 1), policy_fn
            )
            flips = 0
            for eps in (0.01, 0.05, 0.20):
                nudged = rec.actions[w : w + 1].copy()
                nudged[0, 0] = float(np.clip(nudged[0, 0] + eps, -1.0, 1.0))
                out = replay_primary(
                    env,
                    rec.init_state,
                    rec.actions,
                    Edit(window_start=w, actions=nudged),
                    policy_fn,
                )
                flipped = out.success != base.success
                flips += int(flipped)
                sens.append(
                    {
                        "failure_id": rec.failure_id,
                        "eps": eps,
                        "baseline_success": bool(base.success),
                        "nudged_success": bool(out.success),
                        "flipped": bool(flipped),
                    }
                )
            print(
                f"  sensitivity {rec.failure_id[-30:]}: baseline={base.success} "
                f"outcome flipped by {flips}/3 single-step nudges"
            )
        n_flip = sum(1 for s in sens if s["flipped"])
        rows.append(
            {
                "kind": "single_step_sensitivity",
                "cases": sens,
                "n_flipped": n_flip,
                "n_total": len(sens),
                "flip_rate": round(n_flip / len(sens), 4) if sens else None,
            }
        )
    finally:
        env.close()

    elapsed = time.time() - t_start
    summary = {
        "run_id": run_id,
        "kind": "summary",
        "seed": args.seed,
        "task_id": args.task_id,
        "n_episodes": len(chosen),
        "failure_ids": [r.failure_id for r in chosen],
        "budget_k": args.budget_k,
        "placement": args.placement,
        "causes": sorted({r.cause for r in chosen}),
        "device": device,
        "elapsed_s": round(elapsed, 1),
        "reading": (
            "rsr_spread is the floor below which a margin over random_perturb is not a margin. "
            "flip_rate says how often a single small action change flips the outcome; a high "
            "value means small-K results are near-chaotic and should be read as such."
        ),
    }

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "search.yaml")
    shutil.copyfile(enc_path, out_dir / "encoder.yaml")
    shutil.copyfile(env_path, out_dir / "env.yaml")
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({"run_id": run_id, **r}, default=str) + "\n")
        fh.write(json.dumps(summary, default=str) + "\n")

    print("-" * 78)
    print(json.dumps({**summary, "results": rows}, indent=2, default=str))
    print(f"artifacts: {out_dir}")

    determinism_ok = rows[0]["all_identical"]
    if not determinism_ok:
        print(
            "\nReplay is NOT deterministic. I5 is violated and no RSR number can be trusted "
            "until this is understood. Stop here."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
