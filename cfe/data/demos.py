"""LIBERO demo loading and replay materialisation.

**Observations come from replaying demos in our simulator, not from the HDF5.** Measured on
2026-07-30: resetting to a demo's recorded `states[0]` is bit-exact, yet our render of that
identical state differs substantially from the recorded frame (demo mean 139.1 vs live 103.3,
corr 0.80) — a lighting/asset difference in the recording setup. Training on recorded frames
and evaluating on live ones would put a domain gap between the BC policy's train and deploy
distributions. Replaying costs ~3 min/task and removes the gap by construction.

**Recorded `states[t]` are not ground truth.** Replaying recorded `actions[t]` from `states[0]`
does not reproduce recorded `states[t]` — drift exceeds 1e-3 by the first step and peaks
around 4.5-10.6. Task success is still reproduced (5/5 on task 0), so the demos remain valid
expert *action* sequences, but their recorded state trajectory must never be used as a target
or as ground truth. See PROGRESS.md.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import h5py
import numpy as np

# Live-env proprioception fed to the BC policy alongside the pooled latent. These are the
# LIVE env's key names; the demo HDF5 uses different ones (ee_pos / ee_ori / gripper_states)
# and the two are NOT interchangeable — SPEC.md §2 flags this as a silent-wrong-answer trap.
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
PROPRIO_DIM = 3 + 4 + 2

CAMERA_KEY = "agentview_image"


def gl_renderer() -> str:
    """Identify the GL renderer backing the current context, for the cache fingerprint.

    Must be called while a GL context exists (i.e. after an env has been constructed).
    Returns e.g. ``"egl:llvmpipe (LLVM 20.1.2, 256 bits)"``. Two machines that both software-
    rasterise with different mesa drivers render the same state to different pixels, so this
    string is part of what identifies a cached latent.
    """
    backend = os.environ.get("MUJOCO_GL", "<unset>")
    try:
        gl = ctypes.CDLL("libGL.so.1")
        gl.glGetString.restype = ctypes.c_char_p
        name = gl.glGetString(0x1F01)  # GL_RENDERER
    except OSError:
        name = None
    return f"{backend}:{name.decode() if name else 'unknown'}"


def make_replay_env(env_cfg: dict, task, seed: int):
    """Env for replay and policy rollouts, rendering only the camera we actually use.

    LIBERO's default is `camera_names=["agentview", "robot0_eye_in_hand"]`, and rendering
    goes through mesa's llvmpipe on the CPU here. Dropping the unused wrist camera measured
    1.56x faster (5.8 -> 9.1 steps/s), which is ~14 min instead of ~28 for one task.

    Deliberately separate from `scripts/verify_env.py:make_env`, which is the G0 gate's env
    and is left exactly as Phase 0 signed it off. Collection and rollout must both come
    through here so the policy is deployed in the same env it was trained from.
    """
    from libero.libero.envs import OffScreenRenderEnv

    from scripts.verify_env import bddl_path_for

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path_for(task),
        camera_heights=env_cfg["camera_height"],
        camera_widths=env_cfg["camera_width"],
        camera_names=["agentview"],
    )
    env.seed(seed)
    return env


def demo_file_path(dataset_dir: str | Path, suite: str, task) -> Path:
    """The HDF5 for one task. LIBERO names it after the task's bddl file."""
    root = Path(os.path.expanduser(str(dataset_dir))) / suite
    stem = Path(task.bddl_file).stem
    path = root / f"{stem}_demo.hdf5"
    if not path.exists():
        candidates = sorted(p.name for p in root.glob("*.hdf5"))
        raise FileNotFoundError(f"{path} not found; {suite} contains: {candidates}")
    return path


def episode_ids(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as f:
        # demo_0, demo_1, ... — sort numerically so ordering is stable and human-sensible.
        return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))


def read_episode(path: str | Path, episode_id: str) -> dict:
    """Recorded `actions (T,7)` and `states (T,110)`.

    `states` is returned for its first row only — the exact reset state that makes I1 hold.
    The rest of the trajectory is deliberately not treated as ground truth (see module docs).
    """
    with h5py.File(path, "r") as f:
        d = f["data"][episode_id]
        return {"actions": d["actions"][()], "states": d["states"][()]}


def proprio_vector(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=np.float64).ravel() for k in PROPRIO_KEYS])


def replay_episode(env, actions: np.ndarray, init_state: np.ndarray) -> dict:
    """Replay `actions` from the exact `init_state`, capturing what our simulator shows.

    Returns `T+1` observations for `T` actions: `frames[t]` is the observation the policy
    would act on when emitting `actions[t]`, and `frames[T]` is the terminal frame (used as
    `z_goal` in SPEC.md §5).
    """
    env.reset()
    obs = env.set_init_state(init_state)
    if obs is None:
        raise RuntimeError("set_init_state returned None; cannot materialise observations")

    frames = [obs[CAMERA_KEY].copy()]
    proprio = [proprio_vector(obs)]
    reward_sum = 0.0
    for a in actions:
        obs, reward, _done, _info = env.step(np.asarray(a, dtype=np.float64).copy())
        frames.append(obs[CAMERA_KEY].copy())
        proprio.append(proprio_vector(obs))
        reward_sum += float(reward)

    return {
        "frames": np.stack(frames),
        "proprio": np.stack(proprio).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        # Canonical success flag: what the simulator says at the end of the episode.
        "success": bool(env.check_success()),
        "reward_sum": reward_sum,
    }
