"""Build the failure set of SPEC.md §4.4: injected failures with causes, plus natural ones.

Injected candidates are generated from the cached demos, replayed in our simulator from the
demo's exact `states[0]`, and **kept only if they actually fail**. A corruption that the
controller absorbs is not a failure, and counting it as one would put successful episodes into
the failure set. Candidates that still succeed are counted and reported.

Natural failures are registered from the BC rollouts already in the cache. They carry
`cause = None` and no expert trajectory: Phase 1 verified that no demo shares a benchmark init
state, so there is no uncorrupted counterpart and none may be manufactured.

Two independent checks ride along, because CLAUDE.md asks that a new measurement be checked
against something that did not produce it:

- the action-to-metres calibration is fit per task from that task's own demos, and its
  correlation is reported so a bad fit is visible rather than silently scaling every offset;
- for the two offset causes, the *realised* end-effector separation at the grasp is read from
  the replayed proprioception and compared against what the calibration predicted.

Usage:
    python scripts/build_failure_set.py --config configs/search.yaml --seed 0 \
        --run-id p3-failures-t0
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
    p = argparse.ArgumentParser(description="Build SPEC.md §4.4's failure set")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--task-id", type=int, default=None, help="override encoder.yaml data.task_id")
    p.add_argument("--limit-demos", type=int, default=None)
    p.add_argument("--per-cause", type=int, default=None, help="override inject.per_cause_per_task")
    p.add_argument("--causes", nargs="*", default=None, help="subset of causes, for smoke runs")
    p.add_argument("--skip-natural", action="store_true")
    p.add_argument("--skip-injected", action="store_true")
    p.add_argument("--force", action="store_true", help="rebuild records that already exist")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def candidate_params(cause: str, inject_cfg: dict) -> list[dict]:
    """The pre-registered parameter grid for one cause, in a fixed order.

    Order matters: candidates are tried in this order and the first that fail are kept, so a
    reshuffle would change which episodes make up the failure set for the same config.
    """
    c = inject_cfg[cause]
    if cause == "grasp_offset":
        return [
            {"delta_m": float(d), "direction": [float(x) for x in v]}
            for d in c["delta_m"]
            for v in c["directions"]
        ]
    if cause == "wrong_target":
        return [
            {"fraction": float(f), "distractor_rank": int(i)}
            for f in c["fractions"]
            for i in range(int(c["n_distractors"]))
        ]
    if cause == "mistimed_close":
        return [{"shift": int(s)} for s in c["shifts"]]
    if cause == "early_release":
        return [{"steps": int(s)} for s in c["steps"]]
    raise ValueError(f"no parameter grid for cause {cause!r}")


def pair_demos_with_params(demo_ids: list[str], params: list[dict]) -> list[tuple[str, int]]:
    """Every (demo, param) pair, ordered so the first N use N *distinct* demos.

    Demo-major order would spend the whole quota on the first few demos and their init states;
    param-major would spend it on one parameter value. Pairing demo `j` with param `(j + r) % P`
    for successive `r` walks the grid diagonally, so the first pass covers every demo with a
    different parameter and later passes fill in without ever repeating a pair.
    """
    n_p = len(params)
    return [(demo_ids[j], (j + r) % n_p) for r in range(n_p) for j in range(len(demo_ids))]


def build_injection(cause: str, param: dict, actions, phases, scene: dict, window_steps: int):
    """Dispatch one candidate to its injection function."""
    from cfe.data.inject import early_release, grasp_offset, mistimed_close, wrong_target

    if cause == "grasp_offset":
        delta = np.asarray(param["direction"], dtype=np.float64) * param["delta_m"]
        return grasp_offset(actions, phases, delta, window_steps, scene["m_per_unit"])
    if cause == "wrong_target":
        distractors = scene["distractor_xy"]
        if not distractors:
            return None
        d = distractors[param["distractor_rank"] % len(distractors)]
        return wrong_target(
            actions,
            phases,
            scene["target_xy"],
            d,
            param["fraction"],
            window_steps,
            scene["m_per_unit"],
        )
    if cause == "mistimed_close":
        return mistimed_close(actions, phases, param["shift"])
    if cause == "early_release":
        return early_release(actions, phases, param["steps"])
    raise ValueError(f"unknown cause {cause!r}")


# ---------------------------------------------------------------------------


def scene_geometry(env, init_state, task_language: str) -> dict:
    """Target and distractor positions at the episode's own initial state.

    Read per episode rather than once per task: LIBERO's init states move the objects, so a
    distractor offset computed from one state would point at empty table in another.
    `object_position_keys` filters `<object>_to_robot0_eef_pos`, which is object *minus*
    end-effector and therefore moves when the arm does.
    """
    from cfe.policy.bc import find_target_object_key, object_positions

    env.reset()
    obs = env.set_init_state(init_state)
    positions = object_positions(obs)
    target_key = find_target_object_key(obs.keys(), task_language)
    distractors = {
        k: v for k, v in positions.items() if k != target_key and not k.startswith("basket")
    }
    return {
        "target_key": target_key,
        "target_xy": positions[target_key][:2],
        # Sorted so "distractor_rank 0" names the same object on every run.
        "distractor_xy": [distractors[k][:2] for k in sorted(distractors)],
        "distractor_keys": sorted(distractors),
    }


def calibrate(cache, demo_cids: list[str]) -> dict:
    """Fit metres-per-unit-action over several demos and report the spread across them.

    Per demo rather than pooled, so the *disagreement* between demos is visible. A calibration
    that varies wildly between episodes is not a property of the controller and should not be
    used to specify an offset in metres.
    """
    from cfe.data.inject import metres_per_unit_action

    fits = []
    for cid in demo_cids:
        meta = cache.read_meta(cid)
        fits.append(metres_per_unit_action(meta["actions"], meta["proprio"][:, :3]))
    slopes = np.stack([f["m_per_unit"] for f in fits])
    corrs = np.stack([f["corr"] for f in fits])
    return {
        "m_per_unit": slopes.mean(axis=0),
        "m_per_unit_std": slopes.std(axis=0),
        "corr_mean": corrs.mean(axis=0),
        "corr_min": corrs.min(axis=0),
        "n_demos": len(fits),
    }


def main() -> int:  # noqa: PLR0915 — a linear pipeline reads better than split-up stages
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    enc_yaml = yaml.safe_load(enc_path.read_text())
    env_cfg = yaml.safe_load(env_path.read_text())
    run_id = args.run_id or f"p3-failures-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    from cfe.data.demos import (
        demo_file_path,
        gl_renderer,
        make_replay_env,
        read_episode,
        replay_episode,
    )
    from cfe.data.failure_set import (
        INJECTED,
        NATURAL,
        FailureRecord,
        failure_set_dir,
        load_failure_set,
        summarise,
        write_record,
    )
    from cfe.data.inject import CAUSES, gripper_phases, realised_offset_from_proprio
    from cfe.encoders import Dinov2Encoder, EncoderConfig
    from cfe.encoders.cache import EncodingCache, Fingerprint
    from scripts.verify_env import load_suite

    data_cfg, runtime_cfg = enc_yaml["data"], enc_yaml["runtime"]
    task_id = int(args.task_id if args.task_id is not None else data_cfg["task_id"])
    suite_name = data_cfg["suite"]
    inject_cfg = cfg["inject"]
    per_cause = int(args.per_cause or inject_cfg["per_cause_per_task"])
    window_steps = int(inject_cfg["approach_window_steps"])
    causes = tuple(args.causes) if args.causes else CAUSES

    np.random.seed(args.seed)

    suite = load_suite({"suite": suite_name})
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    demo_path = demo_file_path(data_cfg["dataset_dir"], suite_name, task)

    print(f"build_failure_set  run_id={run_id}  seed={args.seed}  task={task_id} ({task.name})")
    print(f"causes={list(causes)}  per_cause={per_cause}")
    print("-" * 78)

    env = make_replay_env(env_cfg, task, args.seed)
    renderer = gl_renderer()
    enc_cfg = EncoderConfig.from_dict(enc_yaml["encoder"])
    encoder = Dinov2Encoder(enc_cfg, device=runtime_cfg.get("device", "auto"))
    fingerprint = Fingerprint.build(
        enc_cfg.hash_payload(), encoder.device, renderer=renderer, obs_source="replay"
    )
    cache = EncodingCache(runtime_cfg["cache_dir"], fingerprint)
    out_dir = failure_set_dir(REPO_ROOT)

    cached = set(cache.episode_ids())
    demo_cids = sorted(
        (c for c in cached if c.startswith(f"{suite_name}_t{task_id}_demo_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    if args.limit_demos:
        demo_cids = demo_cids[: args.limit_demos]
    if not demo_cids:
        raise RuntimeError(
            f"no cached demos for {suite_name} task {task_id}; run scripts/collect_demos.py first"
        )

    calib = calibrate(cache, demo_cids)
    print(f"renderer: {renderer}   device: {encoder.device}")
    print(f"cache:    {cache.root}")
    print(f"demos:    {len(demo_cids)}")
    print(
        f"calibration m/unit = {np.round(calib['m_per_unit'], 5).tolist()}  "
        f"(std {np.round(calib['m_per_unit_std'], 5).tolist()}, "
        f"min corr {np.round(calib['corr_min'], 3).tolist()})"
    )
    print("-" * 78)

    existing = {r.failure_id for r in load_failure_set(out_dir, task_id=task_id)}
    records, attempts = [], []
    t_start = time.time()
    n_frames = 0

    try:
        # -- injected -------------------------------------------------------
        if not args.skip_injected:
            for cause in causes:
                params = candidate_params(cause, inject_cfg)
                pairs = pair_demos_with_params(demo_cids, params)
                kept = 0
                for demo_cid, p_idx in pairs:
                    if kept >= per_cause:
                        break
                    demo_local = demo_cid.split(f"{suite_name}_t{task_id}_", 1)[1]
                    # `src<N>`, not `demo_<N>`: the id must not contain the substring `_demo_`
                    # or consumers that classify episodes by substring would load an injected
                    # failure as expert data. See cfe/data/demos.py:episode_kind.
                    src = demo_local.replace("demo_", "src")
                    fid = f"{suite_name}_t{task_id}_inject_{cause}_{src}_p{p_idx}"
                    if fid in existing and not args.force:
                        kept += 1
                        continue

                    demo = read_episode(demo_path, demo_local)
                    clean_actions = np.asarray(demo["actions"], dtype=np.float64)
                    init_state = demo["states"][0]
                    try:
                        phases = gripper_phases(clean_actions)
                    except ValueError as exc:
                        attempts.append(
                            {
                                "failure_id": fid,
                                "outcome": "no_clean_gripper_phases",
                                "why": str(exc),
                            }
                        )
                        continue

                    scene = scene_geometry(env, init_state, task.language) | {
                        "m_per_unit": calib["m_per_unit"]
                    }
                    param = params[p_idx]
                    try:
                        inj = build_injection(
                            cause, param, clean_actions, phases, scene, window_steps
                        )
                    except ValueError as exc:
                        attempts.append(
                            {"failure_id": fid, "outcome": "injection_refused", "why": str(exc)}
                        )
                        continue
                    if inj is None:
                        attempts.append({"failure_id": fid, "outcome": "no_distractor_available"})
                        continue

                    t0 = time.time()
                    out = replay_episode(env, inj.actions, init_state)
                    t_replay = time.time() - t0

                    if out["success"]:
                        # Not a failure. Reported, never filed as one.
                        attempts.append(
                            {
                                "failure_id": fid,
                                "cause": cause,
                                "params": param,
                                "outcome": "still_succeeded",
                                "replay_s": round(t_replay, 2),
                            }
                        )
                        print(f"  {fid}: injection absorbed, still succeeds — not a failure")
                        continue

                    # Independent check on the offset causes: what the arm actually did.
                    check = None
                    if cause in ("grasp_offset", "wrong_target"):
                        clean_eef = cache.read_meta(demo_cid)["proprio"][:, :3]
                        check = realised_offset_from_proprio(
                            clean_eef, out["proprio"][:, :3], phases.close_idx
                        )

                    t0 = time.time()
                    tokens = encoder.encode(out["frames"])
                    t_encode = time.time() - t0
                    cache.write(fid, tokens)
                    cache.write_meta(
                        fid,
                        actions=out["actions"],
                        proprio=out["proprio"],
                        success=np.array(False),
                    )
                    n_frames += len(out["frames"])

                    rec = FailureRecord(
                        failure_id=fid,
                        source=INJECTED,
                        suite=suite_name,
                        task_id=task_id,
                        task_name=task.name,
                        episode_id=fid,
                        init_state=np.asarray(init_state, dtype=np.float64),
                        actions=inj.actions,
                        success=False,
                        cause=cause,
                        edit_window=inj.window,
                        expert_episode_id=demo_cid,
                        params={
                            "grid_param": param,
                            "grid_index": p_idx,
                            "injection": inj.params,
                            "close_idx": int(phases.close_idx),
                            "release_idx": int(phases.release_idx),
                            "regrasped": bool(phases.regrasped),
                            "target_object": scene["target_key"],
                            "realised_offset_from_proprio": check,
                        },
                    )
                    write_record(out_dir, rec)
                    records.append(rec)
                    kept += 1
                    attempts.append(
                        {
                            "failure_id": fid,
                            "cause": cause,
                            "params": param,
                            "outcome": "failed_as_intended",
                            "edit_window": list(inj.window),
                            "replay_s": round(t_replay, 2),
                            "encode_s": round(t_encode, 2),
                            "realised_offset_from_proprio": check,
                        }
                    )
                    sep = f"  sep={check['separation_norm_m']:.4f}m" if check else ""
                    print(
                        f"  [{kept:2d}/{per_cause}] {fid}: FAILED as intended  "
                        f"window={inj.window}  replay={t_replay:5.1f}s{sep}"
                    )
                print(f"{cause}: kept {kept}/{per_cause}")

        # -- natural --------------------------------------------------------
        if not args.skip_natural:
            roll_cids = sorted(
                (c for c in cached if c.startswith(f"{suite_name}_t{task_id}_rollout_")),
                key=lambda c: int(c.rsplit("_", 1)[1]),
            )
            n_nat = 0
            for cid in roll_cids:
                meta = cache.read_meta(cid)
                if bool(meta["success"]):
                    continue
                if cid in existing and not args.force:
                    n_nat += 1
                    continue
                idx = int(meta["init_state_index"])
                rec = FailureRecord(
                    failure_id=cid,
                    source=NATURAL,
                    suite=suite_name,
                    task_id=task_id,
                    task_name=task.name,
                    episode_id=cid,
                    init_state=np.asarray(init_states[idx], dtype=np.float64),
                    actions=np.asarray(meta["actions"], dtype=np.float64),
                    success=False,
                    # No cause and no expert: nothing here is inferred.
                    cause=None,
                    edit_window=None,
                    expert_episode_id=None,
                    params={
                        "init_state_index": idx,
                        "target_displacement_xy": float(meta["target_displacement_xy"]),
                        "other_object_displacement_xy": float(meta["other_object_displacement_xy"]),
                    },
                )
                write_record(out_dir, rec)
                records.append(rec)
                n_nat += 1
            print(f"natural: registered {n_nat} from {len(roll_cids)} cached rollouts")
    finally:
        env.close()

    elapsed = time.time() - t_start
    everything = load_failure_set(out_dir)
    still_succeeded = [a for a in attempts if a["outcome"] == "still_succeeded"]
    summary = {
        "run_id": run_id,
        "seed": args.seed,
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "encoder_cfg_hash": cache.hash,
        "renderer": renderer,
        "device": str(encoder.device),
        "causes": list(causes),
        "per_cause_target": per_cause,
        "calibration": {
            "m_per_unit": calib["m_per_unit"].tolist(),
            "m_per_unit_std": calib["m_per_unit_std"].tolist(),
            "corr_mean": calib["corr_mean"].tolist(),
            "corr_min": calib["corr_min"].tolist(),
            "n_demos": calib["n_demos"],
        },
        "n_written_this_run": len(records),
        "n_candidates_attempted": len(attempts),
        "n_still_succeeded": len(still_succeeded),
        "frames_encoded": n_frames,
        "failure_set_now": summarise(everything),
        "failure_set_this_task": summarise([r for r in everything if r.task_id == task_id]),
        "elapsed_s": round(elapsed, 1),
    }

    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "search.yaml")
    shutil.copyfile(enc_path, run_dir / "encoder.yaml")
    shutil.copyfile(env_path, run_dir / "env.yaml")
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for a in attempts:
            fh.write(json.dumps({"run_id": run_id, "kind": "candidate", **a}, default=str) + "\n")
        fh.write(json.dumps({"run_id": run_id, "kind": "summary", **summary}, default=str) + "\n")

    print("-" * 78)
    print(json.dumps(summary, indent=2, default=str))
    print(f"artifacts: {run_dir}")
    print(f"records:   {out_dir}")

    if still_succeeded:
        print(
            f"\n{len(still_succeeded)} injection(s) were absorbed and still succeeded. They are "
            f"recorded as attempts and excluded from the failure set."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
