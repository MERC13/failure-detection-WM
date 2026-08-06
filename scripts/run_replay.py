"""Replay every edit in a `run_search.py` run through the frozen protocol (SPEC.md §6).

**This is where the project's only real numbers come from** (SPEC.md I2). Everything upstream
proposes edits; this executes them in the simulator and records what happened. It calls
`cfe/eval/replay_protocol.py` and does not reimplement a single step of it — `no_edit` is §6's
control for replay bugs and a control that runs different code from the thing it controls
detects nothing.

**Resumable, per replay.** One JSON line is appended and flushed after every episode, and a
restart skips cells already present. The machine this runs on has hard-powered-off six or more
times with no bugcheck (SESSION_HANDOFF), and a ~5 h run that has to start over on each of
those never finishes. A crash costs one replay.

The BC policy supplies step 4 of the primary variant. It is loaded here and handed to the
protocol as a plain callable, because `cfe/eval/` deliberately does not import
`cfe/policy/bc.py`: importing it would leave a hole in the freeze through which an edit to an
unfrozen file could change every number.

Usage:
    python scripts/run_replay.py --search-run p3-search-R --run-id p3-replay-R
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
    p = argparse.ArgumentParser(description="Replay proposed edits; emits the only real numbers")
    # One or more. Passing both experiments to a single replay job puts every arm in one
    # metrics.jsonl, so the gate reports the gated table and the secondary comparison side by
    # side instead of from two files that could drift apart. `cell_key` includes `experiment`,
    # so rows from different searches cannot collide.
    p.add_argument(
        "--search-run",
        required=True,
        nargs="+",
        help="one or more run_search.py run_ids holding edits.jsonl",
    )
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--limit", type=int, default=None, help="stop after N replays, for smoke runs")
    p.add_argument("--restart", action="store_true", help="discard prior results and replay all")
    return p.parse_args()


def cell_key(row: dict) -> str:
    """Identity of one replay, for resumption. Must name everything that changes the outcome."""
    return "|".join(
        str(row.get(f))
        for f in (
            "experiment",
            "failure_id",
            "arm",
            "budget_k",
            "lam",
            "variant",
            "window_source",
        )
    )


def load_done(path: Path) -> set[str]:
    """Cells already replayed. Malformed trailing lines are dropped, not fatal.

    A hard power-off mid-write can leave a truncated final line; refusing to start because of it
    would make the crash cost the whole run rather than one replay.
    """
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "replay":
            done.add(cell_key(rec))
    return done


def make_policy_fn(
    policy_run: str, env_cfg: dict, enc_yaml: dict, device, task_id: int | None = None
):
    """The BC policy as a plain `obs -> (C, 7)` callable for step 4 of the primary variant.

    Encoding is per frame here while the training cache was built at `encoder.cfg.batch_size`.
    That discrepancy was measured in Phase 1 at ~1 fp16 ULP (7.8e-3 against a token std of
    1.78), far below anything a policy responds to, and it is the same regime the 72 % BC rate
    was measured in — so RSR stays comparable to it.
    """
    import torch

    from cfe.data.demos import CAMERA_KEY, proprio_vector
    from cfe.encoders import Dinov2Encoder, EncoderConfig
    from cfe.encoders.dinov2 import pool
    from cfe.policy.bc import BCPolicy, Normalizer, PolicyConfig, to_action

    ckpt = REPO_ROOT / "results" / "runs" / policy_run / "policy.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found; run scripts/train_policy.py first")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # A BC policy is single-task and its input is a pooled DINOv2 feature, so deploying one on
    # the wrong task produces plausible actions and a wrong number rather than an error.
    # Checkpoints written before Phase 3 carry no task id; those are not second-guessed here,
    # only ones that state a task and disagree with the episode.
    ck_task = ck.get("task_id")
    if task_id is not None and ck_task is not None and int(ck_task) != int(task_id):
        raise RuntimeError(
            f"policy run {policy_run!r} was trained on task {ck_task} but is mapped to task "
            f"{task_id} in configs/search.yaml:policy_runs"
        )
    pol_cfg = PolicyConfig.from_dict(ck["policy_config"])

    encoder = Dinov2Encoder(EncoderConfig.from_dict(enc_yaml["encoder"]), device=device)
    policy = BCPolicy(ck["input_dim"], pol_cfg).to(encoder.device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    norm = Normalizer.from_state_dict(ck["normalizer"])

    def policy_fn(obs: dict) -> np.ndarray:
        with torch.inference_mode():
            tokens = encoder.encode(obs[CAMERA_KEY][None])
            feat = pool(tokens.astype(np.float32), include_cls=pol_cfg.include_cls_in_pool)
            if pol_cfg.use_proprio:
                feat = np.concatenate([feat, proprio_vector(obs)[None].astype(np.float32)], axis=1)
            x = torch.from_numpy(norm.norm_x(feat).astype(np.float32)).to(encoder.device)
            return to_action(policy(x)[0].cpu().numpy(), norm)

    return policy_fn, encoder


def main() -> int:  # noqa: PLR0915 — a linear pipeline reads better than split-up stages
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    enc_yaml = yaml.safe_load(enc_path.read_text())
    env_cfg = yaml.safe_load(env_path.read_text())
    run_id = args.run_id or f"p3-replay-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    search_dirs = [REPO_ROOT / "results" / "runs" / r for r in args.search_run]
    edits_paths = [d / "edits.jsonl" for d in search_dirs]
    for path in edits_paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/run_search.py first")

    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    from cfe.data.demos import make_replay_env
    from cfe.data.failure_set import failure_set_dir, load_failure_set
    from cfe.eval.replay_protocol import (
        OPEN_LOOP,
        PRIMARY,
        Edit,
        replay_open_loop,
        replay_primary,
    )
    from scripts.verify_env import load_suite

    rows = [
        json.loads(line)
        for path in edits_paths
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    replayable = [r for r in rows if r.get("not_applicable") is None and "actions" in r]
    skipped_na = [r for r in rows if r.get("not_applicable") is not None]

    records = {r.failure_id: r for r in load_failure_set(failure_set_dir(REPO_ROOT))}
    missing = {r["failure_id"] for r in replayable} - set(records)
    if missing:
        raise RuntimeError(f"edits reference failures not in the failure set: {sorted(missing)}")

    task_ids = sorted({r["task_id"] for r in replayable})
    policy_runs = {int(k): v for k, v in cfg["policy_runs"].items()}
    unmapped = [t for t in task_ids if t not in policy_runs]
    if unmapped:
        raise RuntimeError(
            f"no BC policy configured for task(s) {unmapped}; the primary variant hands control "
            f"to the policy and there is nothing to hand it to. Add them to policy_runs."
        )

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if args.restart and metrics_path.exists():
        metrics_path.unlink()
    done = load_done(metrics_path)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    suite_name = enc_yaml["data"]["suite"]
    suite = load_suite({"suite": suite_name})

    todo = [r for r in replayable if cell_key(r) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"run_replay  run_id={run_id}  search_runs={args.search_run}  seed={args.seed}")
    print(f"edits: {len(rows)}  replayable: {len(replayable)}  not_applicable: {len(skipped_na)}")
    print(f"already done: {len(done)}  to replay now: {len(todo)}")
    print(f"tasks: {task_ids}  policies: {policy_runs}")
    print("-" * 78)
    if not todo:
        print("nothing to do; every cell is already replayed")

    # One env and one policy per task, reused across replays: constructing an env costs seconds
    # and there are hundreds of replays.
    envs, policies, encoders = {}, {}, {}
    n_done = 0
    t_start = time.time()

    try:
        with metrics_path.open("a") as fh:
            for i, row in enumerate(todo):
                tid = int(row["task_id"])
                if tid not in envs:
                    task = suite.get_task(tid)
                    envs[tid] = make_replay_env(env_cfg, task, args.seed)
                    policies[tid], encoders[tid] = make_policy_fn(
                        policy_runs[tid], env_cfg, enc_yaml, device, task_id=tid
                    )
                env = envs[tid]
                rec = records[row["failure_id"]]
                edit = Edit(
                    window_start=int(row["window_start"]),
                    actions=np.asarray(row["actions"], dtype=np.float64),
                )

                t0 = time.time()
                if row["variant"] == PRIMARY:
                    result = replay_primary(env, rec.init_state, rec.actions, edit, policies[tid])
                elif row["variant"] == OPEN_LOOP:
                    result = replay_open_loop(env, rec.init_state, rec.actions, edit)
                else:
                    raise ValueError(f"unknown variant {row['variant']!r}")
                dt = time.time() - t0

                out = {
                    "run_id": run_id,
                    "kind": "replay",
                    "search_run": row.get("run_id"),
                    "seed": args.seed,
                    **{
                        f: row.get(f)
                        for f in (
                            "experiment",
                            "dyn_run",
                            "failure_id",
                            "episode_id",
                            "task_id",
                            "cause",
                            "truth_window",
                            "arm",
                            "budget_k",
                            "lam",
                            "variant",
                            "window_source",
                            "realised_mean_l2_ee",
                            "realised_max_l2_ee",
                            "gripper_flips",
                            "sparsity",
                            "magnitude_scale_requested",
                            "window_iou_vs_truth",
                            "search",
                        )
                    },
                    # The simulator's verdict, and the only one that counts.
                    "success": bool(result.success),
                    "steps_executed": result.steps_executed,
                    "scripted_steps": result.scripted_steps,
                    "policy_steps": result.policy_steps,
                    "timed_out": result.timed_out,
                    "max_steps_total": result.max_steps_total,
                    "policy_never_ran": result.policy_never_ran,
                    "window_start": result.window_start,
                    "window_end": result.window_end,
                    "replay_s": round(dt, 2),
                }
                fh.write(json.dumps(out, default=str) + "\n")
                # Flushed and fsynced every line: an unflushed buffer is exactly what a hard
                # power-off destroys, and it would cost every replay since the last flush.
                fh.flush()
                os.fsync(fh.fileno())
                n_done += 1

                elapsed = time.time() - t_start
                eta = (len(todo) - n_done) * elapsed / max(n_done, 1)
                print(
                    f"[{i + 1:4d}/{len(todo)}] {row['arm']:24s} K={row['budget_k']} "
                    f"lam={str(row['lam']):5s} {row['variant']:9s} {row['window_source']:12s} "
                    f"{row['failure_id'][-28:]}: success={result.success!s:5s} "
                    f"steps={result.steps_executed:3d} {dt:5.1f}s  eta={eta / 60:5.1f}m"
                )
    finally:
        for env in envs.values():
            env.close()

    elapsed = time.time() - t_start
    summary = {
        "run_id": run_id,
        "kind": "summary",
        "search_runs": list(args.search_run),
        "experiments": sorted({r.get("experiment") for r in rows}),
        "seed": args.seed,
        "n_edits_total": len(rows),
        "n_replayable": len(replayable),
        "n_not_applicable": len(skipped_na),
        "not_applicable": [
            {k: r.get(k) for k in ("failure_id", "arm", "budget_k", "not_applicable")}
            for r in skipped_na
        ],
        "n_replayed_this_invocation": n_done,
        "n_done_total": len(done) + n_done,
        "complete": len(done) + n_done >= len(replayable),
        "elapsed_s": round(elapsed, 1),
        "s_per_replay": round(elapsed / n_done, 2) if n_done else None,
        "device": device,
    }
    shutil.copyfile(args.config, out_dir / "search.yaml")
    shutil.copyfile(enc_path, out_dir / "encoder.yaml")
    shutil.copyfile(env_path, out_dir / "env.yaml")
    for d in search_dirs:
        if (d / "dynamics.yaml").exists():
            shutil.copyfile(d / "dynamics.yaml", out_dir / "dynamics.yaml")
    # One concatenated edits file beside the results, so the run dir is self-contained: the gate
    # reads it to check that every §6 baseline was proposed, including any that raised
    # NotApplicable and therefore has no replay row of its own.
    with (out_dir / "edits.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    with metrics_path.open("a") as fh:
        fh.write(json.dumps(summary, default=str) + "\n")

    print("-" * 78)
    print(json.dumps(summary, indent=2, default=str))
    print(f"artifacts: {out_dir}")
    if summary["complete"]:
        print(f"next: make g3-check RUN={run_id}")
    else:
        joined = " ".join(args.search_run)
        print(f"INCOMPLETE — re-run to resume: --search-run {joined} --run-id {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
