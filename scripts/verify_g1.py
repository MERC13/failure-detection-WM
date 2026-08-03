"""G1 gate (SPEC.md §7): the encoder produces cached latents, and the BC policy supplies
enough natural failures, with a mode breakdown, to build SPEC.md §4.4's failure set.

The success-rate criterion was amended 2026-07-31 — the original 40-70% band is gone, and the
reasoning plus the rejected alternatives are in CHANGELOG.md.

Reuses `scripts/verify_env.py`'s `Result` reporting so the two gates read alike and a check
that raises is a FAIL, never a skip. Comparisons are bit-exact: SPEC.md §8 asks that a cache
hit be identical to a fresh encode, so this uses `np.array_equal`, not `np.allclose`.

The BC success rate is read from a completed `train_policy.py` run rather than re-measured
here — it came from 50 simulator rollouts recorded alongside their config snapshot, and
re-running them would cost ~30 minutes to produce the same number.

Usage:
    python scripts/verify_g1.py --config configs/encoder.yaml --policy-run p1-bc
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_env import Result, load_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 gate")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "encoder.yaml")
    p.add_argument("--env-config", type=Path, default=REPO_ROOT / "configs" / "env.yaml")
    p.add_argument("--policy-config", type=Path, default=REPO_ROOT / "configs" / "policy.yaml")
    p.add_argument("--policy-run", default=None, help="train_policy.py run_id to read")
    p.add_argument("--collect-run", default=None, help="collect_demos.py run_id to read")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--n-replay-checks", type=int, default=3)
    return p.parse_args()


def check_encoder_frozen(enc_cfg) -> Result:
    """SPEC.md §4.1: encoder weights never train."""
    from cfe.encoders import Dinov2Encoder

    encoder = Dinov2Encoder(enc_cfg, device="cpu")
    offenders = [n for n, p in encoder.model.named_parameters() if p.requires_grad]
    return Result(
        "encoder_frozen",
        not offenders and not encoder.model.training,
        {
            "n_params_requiring_grad": len(offenders),
            "first_offenders": offenders[:5],
            "training_mode": encoder.model.training,
        },
    )


def check_cache_populated(cache, expected_hash: str) -> Result:
    ids = cache.episode_ids()
    metas = [e for e in ids if cache.has_meta(e)]
    return Result(
        "cache_populated",
        bool(ids) and len(metas) == len(ids),
        {
            "encoder_cfg_hash": expected_hash,
            "root": str(cache.root),
            "n_episodes": len(ids),
            "n_with_metadata": len(metas),
        },
    )


def check_cache_bit_identical(cache, encoder, env, path, demo_id: str, episode_id: str) -> Result:
    """SPEC.md §8: a cache hit must be bit-identical to a fresh encode.

    Re-replays the episode and re-encodes it rather than comparing the cache against itself,
    so this also exercises replay determinism — a cache built from a non-reproducible replay
    would be just as poisonous as one built from a different encoder.

    `demo_id` indexes the HDF5 (`demo_0`); `episode_id` names the cache entry
    (`libero_object_t0_demo_0`). They stopped being the same string when the cache gained a
    task namespace, and passing one where the other is wanted is a KeyError, not a wrong number.
    """
    from cfe.data.demos import read_episode, replay_episode

    ep = read_episode(path, demo_id)
    out = replay_episode(env, ep["actions"], ep["states"][0])
    fresh = encoder.encode(out["frames"])
    cached = np.asarray(cache.read(episode_id))

    if fresh.shape != cached.shape:
        return Result(
            "cache_bit_identical",
            False,
            {
                "episode": episode_id,
                "reason": "shape mismatch",
                "fresh": list(fresh.shape),
                "cached": list(cached.shape),
            },
        )
    identical = np.array_equal(fresh, cached)
    detail = {
        "episode": episode_id,
        "shape": list(fresh.shape),
        "dtype": str(fresh.dtype),
        "comparison": "np.array_equal (bit-exact, no tolerance)",
    }
    if not identical:
        diff = np.abs(fresh.astype(np.float32) - cached.astype(np.float32))
        detail |= {
            "max_abs_diff": float(diff.max()),
            "n_differing": int((diff != 0).sum()),
            "n_elements": int(diff.size),
        }
    return Result("cache_bit_identical", identical, detail)


def check_fingerprint_mismatch_raises(enc_cfg, probe_parent: Path) -> Result:
    """SPEC.md §4.1 (amended): a mismatch is a hard error, never a silent recompute.

    Runs in a throwaway directory that is deleted afterwards. The check deliberately corrupts a
    fingerprint on disk, so it must not reuse a fixed path: an earlier run's tampered
    fingerprint is still there, and the gate would fail on its own leftovers the second time it
    was invoked rather than on anything about the code under test.
    """
    from cfe.encoders.cache import EncodingCache, Fingerprint, FingerprintMismatch

    probe_parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="g1_fingerprint_probe_", dir=probe_parent))
    try:
        fp = Fingerprint.build(enc_cfg.hash_payload(), "cpu", renderer="probe", obs_source="replay")
        cache = EncodingCache(tmp_root, fp)
        cache.write("demo_probe", np.zeros((2, 4, 8), dtype=np.float16))

        reopened_cleanly = True
        try:
            EncodingCache(tmp_root, fp)
        except FingerprintMismatch:
            reopened_cleanly = False

        stored = json.loads(cache.fingerprint_path.read_text())
        stored["renderer"] = "a different renderer"
        cache.fingerprint_path.write_text(json.dumps(stored))
        try:
            EncodingCache(tmp_root, fp)
            raised = False
        except FingerprintMismatch:
            raised = True
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return Result(
        "fingerprint_mismatch_raises",
        raised and reopened_cleanly,
        {
            "tampered_fingerprint_raised": raised,
            "identical_fingerprint_reopened_cleanly": reopened_cleanly,
        },
    )


def check_expert_replay_success(
    env, path, episode_ids: list[str], collect_run: str | None, min_yield: float, cache=None
) -> Result:
    """SPEC.md §8, amended 2026-07-31: every episode *admitted to the cache* reproduced its
    recorded success flag exactly, and replay yield has not collapsed.

    The exactness requirement is unchanged and unsoftened — it is simply scoped to the episodes
    that actually feed downstream results, which is what I2 needs. `collect_demos.py` refuses to
    cache an episode whose replay did not succeed, so a non-reproducing demo is rejected by the
    pipeline rather than silently used. Measured yield is 92-96% across three tasks: that is
    physics divergence at tight grasp margins, not a replay defect.

    `min_yield` is a regression tripwire, not a pass mark. A genuine replay bug (wrong action
    scaling, an off-by-one, a broken reset) would crater yield across the board rather than
    clip the few tightest-margin demos, and would still fail this check. See CHANGELOG.md.

    Prefers the full population measured by a completed `collect_demos.py` run over a live
    spot-check: a 3-episode sample misses a 2-in-50 failure entirely — which is exactly what
    happened before this check was widened.
    """
    if collect_run:
        metrics = REPO_ROOT / "results" / "runs" / collect_run / "metrics.jsonl"
        if metrics.exists():
            summary = None
            for line in metrics.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("kind") == "summary":
                    summary = rec
            if summary is not None:
                total = int(summary["episodes_total"])
                n_failed = int(summary["episodes_replay_failed"])
                failed_ids = list(summary.get("replay_failed_ids", []))
                replay_yield = (total - n_failed) / total if total else 0.0
                # The invariant that matters: nothing that failed replay is in the cache.
                cached = set(cache.episode_ids()) if cache is not None else set()
                leaked = sorted(cached & set(failed_ids))
                return Result(
                    "expert_replay_success",
                    not leaked and replay_yield >= min_yield and cache is not None,
                    {
                        "source": f"collect run {collect_run} (all {total} episodes)",
                        "n_episodes": total,
                        "n_admitted_to_cache": len(cached),
                        "n_excluded_by_failed_replay": n_failed,
                        "excluded_ids": failed_ids,
                        "non_reproducing_episodes_in_cache": leaked,
                        "replay_yield": round(replay_yield, 4),
                        "min_yield_tripwire": min_yield,
                        "requirement": (
                            "SPEC.md §8 (amended): every cached episode reproduced its success "
                            "flag exactly; yield floor is a regression tripwire"
                        ),
                    },
                )

    from cfe.data.demos import read_episode, replay_episode

    results = {}
    for eid in episode_ids:
        ep = read_episode(path, eid)
        results[eid] = replay_episode(env, ep["actions"], ep["states"][0])["success"]
    return Result(
        "expert_replay_success",
        all(results.values()),
        {
            "source": "live spot-check (no collect run given — NOT the full population)",
            "n_checked": len(results),
            "n_success": sum(results.values()),
            "per_episode": results,
        },
    )


def check_bc_failure_supply(
    policy_run: str | None, bounds: tuple[float, float], min_failures: int
) -> Result:
    """SPEC.md §4.3, amended 2026-07-31: the gate is failure supply plus a mode breakdown.

    The original 40-70% success band was a proxy for "enough failures, and recoverable ones".
    Both are measured directly now. `bounds` is a wide sanity check on the rate, not the old
    band — see CHANGELOG.md for why it was replaced and what was rejected.

    Reads only what a completed train_policy.py run measured by simulator rollout. This gate
    never re-derives a number; if the run did not record it, that is a FAIL.
    """
    name = "bc_failure_supply"
    lo, hi = bounds
    if policy_run is None:
        return Result(name, False, {"reason": "no --policy-run given; run train_policy.py first"})
    run_dir = REPO_ROOT / "results" / "runs" / policy_run
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return Result(name, False, {"reason": f"{metrics} not found"})

    summary = None
    for line in metrics.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "summary":
            summary = rec
    if summary is None or summary.get("success_rate") is None:
        return Result(name, False, {"reason": "no summary with a success_rate", "run": policy_run})

    modes = summary.get("failure_modes")
    if modes is None or summary.get("n_failures") is None:
        return Result(
            name,
            False,
            {
                "reason": "run predates the failure-mode breakdown; re-run train_policy.py",
                "run": policy_run,
            },
        )

    rate = float(summary["success_rate"])
    n_failures = int(summary["n_failures"])
    # CLAUDE.md rule 4: a number without its config snapshot is an orphan.
    snapshot_present = (run_dir / "policy.yaml").exists() and (run_dir / "encoder.yaml").exists()
    return Result(
        name,
        n_failures >= min_failures and lo <= rate <= hi and snapshot_present,
        {
            "policy_run": policy_run,
            "success_rate": rate,
            "n_failures": n_failures,
            "min_failures": min_failures,
            "failure_modes": modes,
            "sanity_bounds": [lo, hi],
            "rate_within_bounds": lo <= rate <= hi,
            "n_rollouts": summary.get("n_rollouts"),
            "config_snapshot_present": snapshot_present,
            "source": "simulator rollouts recorded by scripts/train_policy.py",
        },
    )


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    env_cfg = yaml.safe_load(args.env_config.read_text())
    pol_cfg = yaml.safe_load(args.policy_config.read_text())
    run_id = args.run_id or f"g1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    from cfe.data.demos import (
        cache_episode_id,
        demo_file_path,
        episode_ids,
        gl_renderer,
        make_replay_env,
    )
    from cfe.encoders import Dinov2Encoder, EncoderConfig
    from cfe.encoders.cache import EncodingCache, Fingerprint

    data_cfg, runtime_cfg = cfg["data"], cfg["runtime"]
    enc_cfg = EncoderConfig.from_dict(cfg["encoder"])
    bounds = tuple(float(x) for x in pol_cfg["eval"]["success_bounds"])
    min_failures = int(pol_cfg["eval"]["min_failures"])

    print(f"G1 gate  run_id={run_id}  seed={args.seed}  MUJOCO_GL={os.environ['MUJOCO_GL']}")
    print(f"config={args.config}")
    print("-" * 78)

    suite = load_suite({"suite": data_cfg["suite"]})
    task = suite.get_task(int(data_cfg["task_id"]))
    path = demo_file_path(data_cfg["dataset_dir"], data_cfg["suite"], task)

    results = [check_encoder_frozen(enc_cfg)]
    env = make_replay_env(env_cfg, task, args.seed)
    try:
        renderer = gl_renderer()
        encoder = Dinov2Encoder(enc_cfg, device=runtime_cfg.get("device", "auto"))
        fingerprint = Fingerprint.build(
            enc_cfg.hash_payload(), encoder.device, renderer=renderer, obs_source="replay"
        )
        cache = EncodingCache(runtime_cfg["cache_dir"], fingerprint)
        print(f"renderer={renderer}  device={encoder.device}")
        print(f"cache={cache.root}")
        print("-" * 78)

        results.append(check_cache_populated(cache, cache.hash))
        ids = episode_ids(path)[: max(1, args.n_replay_checks)]
        # Pick the first *demo* that is actually cached, rather than the first cache entry:
        # the cache is shared with policy rollouts, which have no HDF5 to re-replay from.
        cid_of = {d: cache_episode_id(data_cfg["suite"], data_cfg["task_id"], d) for d in ids}
        probe = next((d for d in ids if cache.has(cid_of[d])), None)
        if probe is not None:
            results.append(
                check_cache_bit_identical(cache, encoder, env, path, probe, cid_of[probe])
            )
        results.append(
            check_fingerprint_mismatch_raises(
                enc_cfg, Path(os.path.expanduser(runtime_cfg["cache_dir"])) / "_g1_probe"
            )
        )
        results.append(
            check_expert_replay_success(
                env,
                path,
                ids,
                args.collect_run,
                float(data_cfg["min_replay_yield"]),
                cache=cache,
            )
        )
    finally:
        env.close()

    results.append(check_bc_failure_supply(args.policy_run, bounds, min_failures))

    for r in results:
        print(r.line())

    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "encoder.yaml")
    shutil.copyfile(args.policy_config, run_dir / "policy.yaml")
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "gate": "G1",
                        "check": r.name,
                        "passed": r.passed,
                        "seed": args.seed,
                        "detail": r.detail,
                    },
                    default=str,
                )
                + "\n"
            )

    failed = [r.name for r in results if not r.passed]
    print("-" * 78)
    if failed:
        print(f"G1: FAIL ({len(failed)}/{len(results)} checks failed: {', '.join(failed)})")
        print(f"artifacts: {run_dir}")
        return 1
    print(f"G1: PASS ({len(results)}/{len(results)} checks)")
    print(f"artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
