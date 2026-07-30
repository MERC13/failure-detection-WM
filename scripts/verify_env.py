"""G0 gate (SPEC.md §7): prove the environment is real before any modeling code exists.

The load-bearing check is invariant I1 (SPEC.md §1): resetting the simulator to a demo's
recorded initial state must reproduce a *bit-identical* observation, and stepping an identical
action sequence from that state must stay bit-identical. Replay is the only ground truth in
this project (I2), so if I1 does not hold, no number downstream means anything.

This script never softens a comparison to make a check pass. Tolerances are not used: I1 says
byte-identical, so the comparisons are np.array_equal, not np.allclose.

Usage:
    python scripts/verify_env.py --config configs/env.yaml [--seed N] [--run-id ID]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G0 environment gate")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=None, help="overrides config seed")
    p.add_argument("--run-id", default=None, help="results/runs/<run_id>/; default g0-<utc>")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Small reporting helpers. Each check appends one Result; the gate fails if any
# result is not passed. A check that raises is a FAIL, never a skip.
# ---------------------------------------------------------------------------


class Result:
    def __init__(self, name: str, passed: bool, detail: dict):
        self.name = name
        self.passed = passed
        self.detail = detail

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {json.dumps(self.detail, default=str)}"


def compare_obs(a: dict, b: dict) -> dict:
    """Bit-exact comparison of two observation dicts. Returns {} when identical."""
    keys_a, keys_b = set(a), set(b)
    if keys_a != keys_b:
        return {
            "reason": "observation keys differ",
            "only_in_first": sorted(keys_a - keys_b),
            "only_in_second": sorted(keys_b - keys_a),
        }
    for key in sorted(keys_a):
        va, vb = a[key], b[key]
        if isinstance(va, np.ndarray):
            if va.shape != vb.shape:
                return {"reason": "shape mismatch", "key": key, "shapes": [va.shape, vb.shape]}
            if not np.array_equal(va, vb):
                diff = np.abs(va.astype(np.float64) - vb.astype(np.float64))
                return {
                    "reason": "values differ",
                    "key": key,
                    "max_abs_diff": float(diff.max()),
                    "n_differing_elements": int((diff != 0).sum()),
                    "n_elements": int(diff.size),
                }
        elif va != vb:
            return {"reason": "scalar differs", "key": key, "values": [va, vb]}
    return {}


def hash_arrays(arrays: list) -> str:
    """Stable digest of a rollout, used to check determinism across processes."""
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def numeric_obs(obs: dict) -> dict:
    """Observation entries that are arrays or scalars, sorted for stable hashing."""
    return {k: v for k, v in sorted(obs.items()) if isinstance(v, (np.ndarray, int, float))}


# ---------------------------------------------------------------------------
# Environment construction. MUJOCO_GL must be set before MuJoCo is imported,
# which is why libero is imported lazily inside these helpers rather than at
# module scope.
# ---------------------------------------------------------------------------


def load_suite(cfg: dict):
    from libero.libero import benchmark

    suite_name = cfg["suite"]
    registry = benchmark.get_benchmark_dict()
    if suite_name not in registry:
        raise KeyError(f"suite {suite_name!r} not in registry; available: {sorted(registry)}")
    return registry[suite_name]()


def bddl_path_for(task) -> str:
    from libero.libero import get_libero_path

    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


def make_env(cfg: dict, task, seed: int):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path_for(task),
        camera_heights=cfg["camera_height"],
        camera_widths=cfg["camera_width"],
    )
    env.seed(seed)
    return env


def sim_state(env) -> np.ndarray:
    """Full MuJoCo state vector.

    Resolved against libero==0.1.1: `get_sim_state()` lives on the ControlEnv wrapper
    (`libero/libero/envs/env_wrapper.py`), not on the inner robosuite env. It returns
    `sim.get_state().flatten()`, the same 110-wide layout as the recorded init states,
    which is why set_init_state / get_sim_state round-trip.

    Deliberately not a fallback chain: if this attribute ever moves, the correct response
    is a loud AttributeError, not silently reading a different state vector.
    """
    return np.asarray(env.get_sim_state()).copy()


def reset_to(env, init_state) -> dict:
    """Reset to an exact recorded init state and return the resulting observation."""
    env.reset()
    obs = env.set_init_state(init_state)
    if obs is None:
        raise RuntimeError(
            "set_init_state returned None; cannot verify I1 from its return value. "
            "Inspect the installed LIBERO source and update this helper."
        )
    return obs


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_versions() -> Result:
    import importlib.metadata as md

    versions = {}
    missing = []
    for pkg in ("libero", "robosuite", "mujoco", "bddl", "robomimic", "h5py", "numpy", "torch"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            missing.append(pkg)
    versions["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "<unset>")
    versions["python"] = sys.version.split()[0]
    if missing:
        return Result("versions", False, {"missing": missing, **versions})
    return Result("versions", True, versions)


def check_render(cfg: dict, task, seed: int) -> Result:
    env = make_env(cfg, task, seed)
    try:
        obs = env.reset()
        key = "agentview_image"
        if key not in obs:
            return Result(
                "headless_render", False, {"reason": f"{key} absent", "obs_keys": sorted(obs)}
            )
        img = obs[key]
        expected = (cfg["camera_height"], cfg["camera_width"], 3)
        # A broken offscreen renderer commonly returns a uniform frame, which a
        # shape-only assertion would happily pass.
        n_unique = int(np.unique(img).size)
        ok = img.shape == expected and img.dtype == np.uint8 and n_unique > 1
        return Result(
            "headless_render",
            ok,
            {
                "shape": img.shape,
                "expected_shape": expected,
                "dtype": str(img.dtype),
                "n_unique_values": n_unique,
                "mean": round(float(img.mean()), 3),
            },
        )
    finally:
        env.close()


def check_exact_reset(cfg: dict, task, init_states, seed: int) -> Result:
    """I1: two independently constructed envs, same init state, bit-identical observations."""
    n = min(int(cfg["i1_num_init_states"]), len(init_states))
    env_a = make_env(cfg, task, seed)
    env_b = make_env(cfg, task, seed)
    try:
        for i in range(n):
            obs_a = reset_to(env_a, init_states[i])
            obs_b = reset_to(env_b, init_states[i])
            mismatch = compare_obs(numeric_obs(obs_a), numeric_obs(obs_b))
            if mismatch:
                return Result(
                    "i1_exact_reset", False, {"init_state_index": i, "observation": mismatch}
                )
            state_a, state_b = sim_state(env_a), sim_state(env_b)
            if not np.array_equal(state_a, state_b):
                diff = np.abs(state_a - state_b)
                return Result(
                    "i1_exact_reset",
                    False,
                    {
                        "init_state_index": i,
                        "reason": "sim state vectors differ",
                        "max_abs_diff": float(diff.max()),
                        "n_differing": int((diff != 0).sum()),
                    },
                )
        return Result(
            "i1_exact_reset",
            True,
            {
                "init_states_compared": n,
                "cross_env_instances": True,
                "sim_state_dim": int(sim_state(env_a).size),
                "comparison": "np.array_equal (bit-exact, no tolerance)",
            },
        )
    finally:
        env_a.close()
        env_b.close()


def check_reset_under_stepping(cfg: dict, task, init_states, seed: int) -> Result:
    """I1 under stepping: identical actions from an identical init state must not diverge.

    A reset that matches at t=0 but drifts once stepped would silently poison every replay
    number in the project, so this is checked per-step rather than only at the end.
    """
    steps = int(cfg["i1_rollout_steps"])
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-0.2, 0.2, size=(steps, 7))

    env_a = make_env(cfg, task, seed)
    env_b = make_env(cfg, task, seed)
    try:
        reset_to(env_a, init_states[0])
        reset_to(env_b, init_states[0])
        for t, action in enumerate(actions):
            obs_a = env_a.step(action.copy())[0]
            obs_b = env_b.step(action.copy())[0]
            mismatch = compare_obs(numeric_obs(obs_a), numeric_obs(obs_b))
            if mismatch:
                return Result(
                    "i1_reset_under_stepping",
                    False,
                    {"diverged_at_step": t, "of_steps": steps, "observation": mismatch},
                )
            if not np.array_equal(sim_state(env_a), sim_state(env_b)):
                return Result(
                    "i1_reset_under_stepping",
                    False,
                    {"diverged_at_step": t, "reason": "sim state vectors differ"},
                )
        return Result(
            "i1_reset_under_stepping",
            True,
            {"steps": steps, "action_dim": int(actions.shape[1]), "diverged": False},
        )
    finally:
        env_a.close()
        env_b.close()


def check_demo_hdf5(cfg: dict, task, suite_name: str) -> Result:
    """A demo HDF5 loads, and we record its *actual* layout rather than assuming SPEC.md's."""
    import h5py
    from libero.libero import get_libero_path

    candidates = []
    cfg_dir = Path(os.path.expanduser(cfg["dataset_dir"]))
    candidates += sorted(cfg_dir.rglob("*.hdf5"))
    try:
        candidates += sorted(Path(get_libero_path("datasets")).rglob("*.hdf5"))
    except Exception:  # noqa: BLE001 - absent dataset path is reported, not raised
        pass
    preferred = [p for p in candidates if suite_name in str(p)] or candidates
    if not preferred:
        return Result(
            "demo_hdf5",
            False,
            {"reason": "no .hdf5 found", "searched": [str(cfg_dir), "libero datasets path"]},
        )

    path = preferred[0]
    with h5py.File(path, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys())
        demo = data[demo_keys[0]]
        fields = sorted(demo.keys())
        actions = demo["actions"][()]
        obs_keys = sorted(demo["obs"].keys()) if "obs" in demo else []
        detail = {
            "file": str(path),
            "n_demos": len(demo_keys),
            "episode_fields": fields,
            "actions_shape": list(actions.shape),
            "actions_dtype": str(actions.dtype),
            "obs_keys": obs_keys,
            "has_states": "states" in fields,
            "states_shape": list(demo["states"].shape) if "states" in fields else None,
            "root_attrs": sorted(f["data"].attrs.keys()),
        }
    # SPEC.md §2/§4.4: 7-DOF actions, and a per-episode state blob is what I1 replays from.
    ok = actions.ndim == 2 and actions.shape[1] == 7 and detail["has_states"]
    if not ok:
        detail["reason"] = "expected actions (T,7) and a per-episode 'states' dataset"
    return Result("demo_hdf5", ok, detail)


def check_seed_determinism(cfg: dict, task, init_states, seed: int) -> Result:
    """I5: the same seed and config produce an identical rollout digest.

    The digest is printed so that running this script twice in separate processes verifies
    determinism across processes, not merely within one.
    """
    steps = int(cfg["i1_rollout_steps"])

    def rollout() -> str:
        rng = np.random.default_rng(seed)
        actions = rng.uniform(-0.2, 0.2, size=(steps, 7))
        env = make_env(cfg, task, seed)
        try:
            obs = reset_to(env, init_states[0])
            frames = list(numeric_obs(obs).values())
            for action in actions:
                obs = env.step(action.copy())[0]
                frames += list(numeric_obs(obs).values())
            return hash_arrays([np.asarray(f) for f in frames])
        finally:
            env.close()

    first, second = rollout(), rollout()
    return Result(
        "i5_seed_determinism",
        first == second,
        {"seed": seed, "rollout_digest": first, "second_run_digest": second, "steps": steps},
    )


# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    run_id = args.run_id or f"g0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    # Must precede any MuJoCo import.
    os.environ.setdefault("MUJOCO_GL", cfg["render_backend"])

    print(f"G0 gate  run_id={run_id}  seed={seed}  MUJOCO_GL={os.environ['MUJOCO_GL']}")
    print(f"config={args.config}")
    print("-" * 78)

    results = [check_versions()]
    if results[0].passed:
        suite = load_suite(cfg)
        task = suite.get_task(int(cfg["task_id"]))
        init_states = suite.get_task_init_states(int(cfg["task_id"]))
        print(f"suite={cfg['suite']}  task={task.name}  n_init_states={len(init_states)}")
        print("-" * 78)
        results.append(check_render(cfg, task, seed))
        results.append(check_exact_reset(cfg, task, init_states, seed))
        results.append(check_reset_under_stepping(cfg, task, init_states, seed))
        results.append(check_demo_hdf5(cfg, task, cfg["suite"]))
        results.append(check_seed_determinism(cfg, task, init_states, seed))

    for r in results:
        print(r.line())

    # CLAUDE.md rule 4: no orphan numbers in the terminal.
    run_dir = Path("results/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.yaml")
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "gate": "G0",
                        "check": r.name,
                        "passed": r.passed,
                        "seed": seed,
                        "detail": r.detail,
                    },
                    default=str,
                )
                + "\n"
            )

    failed = [r.name for r in results if not r.passed]
    print("-" * 78)
    if failed:
        print(f"G0: FAIL ({len(failed)}/{len(results)} checks failed: {', '.join(failed)})")
        print(f"artifacts: {run_dir}")
        return 1
    print(f"G0: PASS ({len(results)}/{len(results)} checks)")
    print(f"artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
