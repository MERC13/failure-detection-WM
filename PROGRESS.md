# PROGRESS

Append-only. One entry per session (CLAUDE.md working style).

---

## 2026-07-30 — Phase 0: environment + G0 gate

**Status: G0 PASS (6/6). I1 verified bit-exact. I5 verified across processes.**

Run twice, identical results both times; artifacts in `results/runs/g0-20260730T195925Z/`.

```
[PASS] versions: {"libero":"0.1.1","robosuite":"1.4.0","mujoco":"3.3.7","bddl":"1.0.1",
                  "robomimic":"0.2.0","h5py":"3.16.0","numpy":"2.2.6","torch":"2.13.0+cpu",
                  "MUJOCO_GL":"egl","python":"3.10.12"}
[PASS] headless_render: {"shape":[128,128,3],"dtype":"uint8","n_unique_values":249,"mean":104.291}
[PASS] i1_exact_reset: {"init_states_compared":3,"cross_env_instances":true,
                        "sim_state_dim":110,"comparison":"np.array_equal (bit-exact, no tolerance)"}
[PASS] i1_reset_under_stepping: {"steps":10,"action_dim":7,"diverged":false}
[PASS] demo_hdf5: {"n_demos":50,"actions_shape":[148,7],"states_shape":[148,110],
                   "episode_fields":["actions","dones","obs","rewards","robot_states","states"]}
[PASS] i5_seed_determinism: {"rollout_digest":"4742366b72d3a5e1...","second_run_digest":"4742366b72d3a5e1..."}
G0: PASS (6/6 checks)
```

`make test-slow` → **6 passed**. `make lint` → clean, 8 files.

**I1 holds.** Exact-state reset reproduces byte-identical observations *and* byte-identical
110-wide sim state across two independently constructed env instances, and does not diverge
over 10 stepped actions. The design assumption in SPEC.md §2 is sound; no approximation was
substituted and no tolerance was introduced.

### Session premise

`SESSION_HANDOFF.md` and `PHASE0_PLAN.md` were requested but do not exist — not in this repo,
not in the git tree, not anywhere under `/mnt/c/Users/jonah` (searched to depth 6). Phase 0 was
reconstructed from SPEC.md §7 (G0) and WORKFLOW.md's P0 prompt, with the user's approval.

### What landed

- `git init` — this is now a dedicated repo on branch `main` (it had been an untracked
  subdirectory of the unrelated `Deep learning` repo). No commit made yet.
- venv at `~/.venvs/failure-detection-WM`, deliberately outside the repo: the repo is on a
  `/mnt/c` 9p mount where installing torch/mujoco is far slower than on native ext4.
- `requirements.txt` — resolved pins with the load-bearing ones commented.
- `scripts/verify_env.py` — the G0 gate, six checks, PASS/FAIL each, nonzero exit on any FAIL.
- `configs/env.yaml`, `Makefile` (`env-check`/`test`/`test-slow`/`lint`), `pyproject.toml`,
  `.gitignore` (`results/` ignored except `results/figures/`, per SPEC.md §3).
- `tests/test_env_reset.py` (I1), `tests/test_determinism.py` (I5), `tests/conftest.py`.
  Tests call the gate's own check functions so a test cannot drift from the invariant it guards.
- CPU-only torch 2.13.0+cpu installed.
- `requirements.lock.txt` — full transitive `pip freeze` (109 packages) of the environment that
  passed G0. `requirements.txt` stays the human-readable "why" for the direct deps.
- `~/.libero/config.yaml` provisioned non-interactively; `libero_object` demos (7.0 GB) at
  `~/data/libero/libero_object/`.

### Verified by actually running it

- `ruff check` + `ruff format --check`: clean, 7 files.
- Fast tests: **3 passed, 3 deselected** (the deselected three are `slow`, needing the sim).
  The passing three cover the comparison logic itself — including a test asserting that a
  single changed byte is reported, which guards against the I1 comparison being loosened to a
  tolerance later.

### The blocker that was cleared

`pip install libero` initially failed building `egl_probe` and `hf-egl-probe` (both C
extensions): `FileNotFoundError: 'cmake'`. This WSL2 instance had no toolchain at all — no
`gcc`, `g++`, `cmake`, `make`, and no EGL headers. Neither package publishes a wheel
(`pip download --only-binary=:all: egl_probe` → `from versions: none`), so compiling was
mandatory. Resolved by the user running:

```
sudo apt update && sudo apt install -y build-essential cmake libegl1-mesa-dev \
    libgl1-mesa-dev libosmesa6 python3.10-venv
```

Deliberately **not** worked around with `--no-deps`, which would have disabled robosuite's EGL
device probing — the exact headless path G0 exists to verify — trading a loud install failure
for a silent rendering defect.

### Corrections to SPEC.md §2 (guess vs. installed reality)

SPEC.md §2 has been rewritten with verified code; this is the diff in prose.

| SPEC.md §2 said | Reality in `libero==0.1.1` |
|---|---|
| "benchmark/task registry" | `benchmark.get_benchmark_dict()["libero_object"]()` → 10 tasks. `.get_task(i)`, `.get_task_bddl_file_path(i)`, `.get_task_init_states(i)` |
| "offscreen-render env constructor" | `from libero.libero.envs import OffScreenRenderEnv`, kwargs `bddl_file_name=`, `camera_heights=`, `camera_widths=` |
| "`env.set_init_state(state)` or equivalent" | Exact match, and it **returns the observation dict**. Init states are `(50, 110) float64` from `get_task_init_states(task_id)` — a method SPEC.md did not mention |
| "7-DOF: 6 eef delta + 1 gripper" | Confirmed |
| `get_sim_state()` | **Exists, on the `ControlEnv` wrapper — not on `env.env`.** Returns `(110,) float64`. `verify_env.py:sim_state()` is now pinned to it rather than probing at runtime |
| "`states`/`init_state`" in demos | It is `states`, `(T, 110)` — same width as init states and live sim state |

### What surprised me

1. **Two LIBERO distributions now exist.** Besides the original repo (Python 3.8.13 + torch
   1.11.0+cu113, reportedly 2-4h of dependency resolution), there is a Hugging Face maintained
   fork on PyPI as `libero==0.1.1` (Nov 2025) that keeps the same API. We chose the fork. Worth
   noting in any writeup that results derive from the fork, not the canonical repo.
2. **The robosuite pin is a real trap, not bookkeeping.** robosuite 1.5 removed `SingleArmEnv`,
   which LIBERO imports; a fresh unpinned install breaks `import libero`. pip's resolver picked
   1.4.0 correctly on its own, but nothing would stop a later `pip install -U` from breaking it.
3. **PyPI's `libero` is legitimate** — it is the HF fork, not a name-squat, which is not the
   assumption to make by default.
4. **This host has no GPU** (`nvidia-smi` absent) and no compiler. Fine for Phase 0, which is
   CPU-only, but Phase 1+ needs the rented A100 from SPEC.md §2.
5. **Rendering works without a GPU, on software rasterization.** EGL logs
   `failed to open /dev/dri/renderD128: Permission denied` then
   `NEEDS EXTENSION: falling back to kms_swrast`. It renders correctly (249 unique values,
   mean 104.3), but this is CPU rasterization, not the WSLg d3d12 path I predicted — the user
   is not in the `render` group. Fine for G0; **rendering throughput will be poor**, which
   matters for Phase 1 encoding of many frames. Worth revisiting on the A100 box.
6. **`mujoco` was the version trap, not `robosuite`.** I had flagged robosuite 1.4.x as the
   load-bearing pin. It is — but robosuite *also* declares `mujoco` with no upper bound, so
   pip happily installed mujoco 3.11.0, which builds the scene fine and then dies on the first
   controller update: `AttributeError: 'MjData' object has no attribute 'qM'`
   (`robosuite/controllers/base_controller.py:156`). `MjData.qM` became `MjData.M` in mujoco
   3.4.0. Bisected: 3.3.7 is the newest working version, now pinned with that reason in
   `requirements.txt`. A `pip install -U` would silently reintroduce this.
7. **Importing `libero` runs `input()`.** With no `~/.libero/config.yaml` it interactively asks
   "Do you want to specify a custom path for the dataset folder?" and blocks — under a
   non-interactive runner it dies with `EOFError`. Provisioned by writing that file directly.
   This is machine-global state living *outside* the repo, so it is a reproducibility hole: a
   fresh checkout does not carry it. Documented in `requirements.txt` and SPEC.md §2.
8. **Demo image keys differ from live env image keys.** Demos store `agentview_rgb` /
   `eye_in_hand_rgb`; the live env returns `agentview_image` / `robot0_eye_in_hand_image`.
   Any code reading both must translate. This is a silent-wrong-answer trap for Phase 1.
9. **`libero_object` alone is 7.0 GB** (10 tasks × ~750 MB, 50 demos each). Downloading all
   130 tasks would have been ~90 GB. `allow_patterns='libero_object/*'` restricts it.

### Gate state

**G0: PASS.** I1 verified (bit-exact, cross-instance, stable under stepping). I5 verified
(identical digest across two separate processes). No modeling code written, per WORKFLOW.md P0.
Phase 1 is unblocked.

### Open questions for the next session

- **Which mujoco did the demos record under?** The `libero_object` demos ship recorded `states`
  `(T, 110)`. We replay under mujoco 3.3.7, which is almost certainly *not* the version they
  were collected with (LIBERO originally targeted 2.3.x). Physics changes across versions mean
  replaying recorded actions may not reproduce the recorded states. G0 did not test this — it
  only tested that *our* sim is self-consistent, which is what I1 asks. Before trusting demo
  replay in Phase 1, add a check that replaying `actions[t]` from `states[0]` reproduces
  `states[t]`, and report the drift. If it drifts, that is a finding, not something to tune away.
- Rendering is on CPU software rasterization (see surprise 5); measure frames/sec before
  committing to a Phase 1 encoding budget.

### Next

1. Decide the demo-replay-fidelity question above.
2. Phase 1 per WORKFLOW.md.

---

## 2026-07-30 — Phase 0 addendum: machine move prepared

Phase 0 signed off and committed as `8b8a80c`. This addendum covers the handoff to the GPU
machine, done *before* Phase 1 rather than after. No modeling code written.

### Why move now rather than after Phase 1

Measured on the WSL2 box: DINOv2-base at **9.9 img/s** (224px) and **29.6 img/s** (128px);
simulator at **53 steps/s**. Phase 1 is feasible here (~13 min to encode a task's demos, ~16
min for a 50-rollout policy eval) but Phase 2 dynamics training is not, so the move was
inevitable. The deciding argument was not speed:

**The encoding cache key identifies neither machine nor device.** CPU and CUDA latents differ
in the low bits under the same `(episode_id, frame_idx, encoder_cfg_hash)`. Moving after
Phase 1 would silently poison every cached latent — the exact failure CLAUDE.md warns about,
in a form the hash cannot catch. Nothing is cached yet, so moving today costs nothing.

Secondary: I1 and I5 were proven on *this* box only, and it renders through software
rasterization while a GPU box uses hardware EGL. That is a different code path, so G0 must be
re-run there regardless.

### What landed

- `scripts/setup_env.sh` — idempotent bootstrap: toolchain check (exits with the exact apt
  command if missing), venv with a `virtualenv` fallback for absent `ensurepip`,
  hardware-matched torch (CPU index only when no GPU is present), pinned deps,
  `~/.libero/config.yaml`, single-suite demo download, then runs G0.
  **Verified by running it here end-to-end**: every step correctly detected already-done work
  and G0 re-passed 6/6 with the identical rollout digest `4742366b…`.
- `SESSION_HANDOFF.md` — bootstrap doc for the new session: gate state, first commands, the
  five traps, the two open questions, and the rules easiest to violate at a phase boundary.
- SPEC.md §4.1 amended on two counts (below).

### Corrections to SPEC.md §4.1

1. **"Keeping tokens costs nothing now" was wrong.** At 224px DINOv2-base emits 257×768 —
   786 KB/frame fp32, so 6.1 GB for one task's 7,808 frames, and §4.4 wants ≥3 tasks. Tokens
   remain canonical, but `output_dtype` is now an explicit config field defaulting to fp16
   (≈3 GB/task). The encoder *returns* that dtype rather than storing a narrowed copy, so the
   G1 invariant "cache hit is bit-identical to a fresh encode" stays literally true.
2. **The cache key was under-specified.** It now hashes all encoder config fields plus model
   id, revision, output dtype, `torch.__version__`, and device kind; and the cache writes a
   `fingerprint.json` whose mismatch is a hard error rather than a silent recompute.

### What surprised me

10. **`libero_object` demo actions have a strictly binary gripper dimension.** Dim 6 takes
    exactly `{-1, +1}` across the demos — nothing in between. CLAUDE.md warned not to apply
    symmetric L2 to the gripper; the data is stronger than that, and it means the BC head
    should treat dim 6 as a classification target (BCE), not a regression one. Recorded here
    so the Phase 1 choice is visible rather than buried in a loss function.
11. **Demo episodes are short and uniform**: 50 demos/task, lengths 136–196 (mean 156),
    7,808 frames per task, images 128×128 native. Encoding at DINOv2's 224 upsamples them.

### Gate state

**G0: PASS** on the WSL2 CPU box, run three times with identical results. **Not yet run
anywhere else** — that is the first task on the GPU machine.

### Next

1. On the GPU box: `bash scripts/setup_env.sh`, then `make test-slow` and `make lint`.
   **If I1 fails there, stop and report.**
2. Answer the demo-replay-fidelity question (SPEC.md §8) before training on the demos.
3. Phase 1: encoder + cache + data + BC policy + G1, per the approved plan.
