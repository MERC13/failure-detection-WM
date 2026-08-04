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

---

## 2026-07-30 — Phase 1: encoder + cache, demo materialisation, BC policy

**Status: G1 FAIL (4/6). Both failures are findings about the data and the task, not defects
in the Phase 1 code.** Artifacts in `results/runs/{p1-demos,p1-bc,g1-phase1}/`.

```
[PASS] encoder_frozen               0 params require grad
[PASS] cache_populated              48 episodes, all with metadata
[PASS] cache_bit_identical          fresh replay+encode == cache, np.array_equal
[PASS] fingerprint_mismatch_raises  tamper raises; identical fingerprint reopens cleanly
[FAIL] expert_replay_success        48/50 (demo_27, demo_31 do not reproduce success)
[FAIL] bc_success_band              72.0% vs the 40-70% band
G1: FAIL (2/6 checks failed)
```

`make test-slow` → 33 passed (24 fast + 9 slow). `make lint` → clean, 20 files.

### The machine changed; G0 was re-run here first

This is not the Phase 0 box. Python 3.12.3 (not 3.10.12), torch 2.13.0+cu130 with CUDA
available, RTX 4070 Laptop 8.59 GB. **G0 re-passed 6/6 on this machine** before any Phase 1
work started; I1 holds bit-exactly here. The pinned requirements resolved on 3.12 without
conflict.

One setup obstacle worth recording: every tracked file arrived with **CRLF line endings**
(the repo lives on `/mnt/c`), which would have made `bash scripts/setup_env.sh` die on
`$'\r': command not found`. Normalised all 16 files to LF and verified content byte-identical
to HEAD via `git hash-object`. A `.gitattributes` with `* text=auto eol=lf` would stop this
recurring; not added yet.

### The two G1 failures

**1. `expert_replay_success` — 48/50.** `demo_27` and `demo_31` replay to `reward_sum = 0.0`
and `success = False`, deterministically (identical across two runs). Not near-misses: the
task never registers success at any timestep. SPEC.md §8 requires the success flag be
reproduced *exactly*, so this check fails rather than being softened to a rate threshold.
Both episodes are excluded from the cache — `collect_demos.py` refuses to cache an episode
whose replay did not succeed. **Consequence for later phases: `expert_oracle` (SPEC.md §6),
described as the upper bound, is capped near 96% on this task rather than 100%.**

**2. `bc_success_band` — 72.0%, band 40-70%.** Measured over 50 simulator rollouts (36
success / 14 failure). Above the band by 2 points. **Not tuned in either direction.** Tuning
it down to land in the band would be as much a manipulation of the measurement as tuning it
up, and which way to go is a call for the project owner, not for me. Note the phase still
yields failures at this rate (28%), which is what the band exists to guarantee.

### The demo-replay-fidelity question is answered (it was blocking Phase 3)

Measured directly, and the answer has three parts that must not be conflated:

| Property | Result |
|---|---|
| `set_init_state(states[0])` reproduces the state | **Bit-exact**, `max\|Δ\| = 0.0`. I1 is solid |
| Replaying `actions[t]` reproduces `states[t]` | **No.** Drift > 1e-3 by the first step, peaks 4.5-10.6 |
| Replaying `actions[t]` reproduces *success* | **Mostly.** 48/50 |

So the demos are valid expert **action** sequences, but their recorded **state trajectories
are not ground truth and must never be used as targets**. This carries into Phase 2 (dynamics
training targets must come from replay, not from recorded states) and Phase 3.
`tests/test_demo_replay.py` pins all three facts, including a test that fails loudly if
recorded states ever *start* reproducing — that would mean the physics changed and this
guidance is stale.

### What surprised me

12. **The renderer is software, on a machine with a GPU.** `MUJOCO_GL=egl` resolves through
    robosuite's own `EGLGLContext` to **`llvmpipe` (Mesa 25.2.8)** — mesa is the only
    registered EGL vendor and `libEGL_nvidia` is absent. The GPU serves CUDA (DINOv2 encodes
    at 144 img/s vs 9.9 on the Phase 0 CPU box) but not MuJoCo. **Rendering, not encoding, is
    now the bottleneck**: ~9 steps/s, so 50 rollouts took 23 minutes and demo collection 18.
    Phase 2 should budget against rendering, not against the GPU.
13. **The i5 digest differs across machines and that is not an I5 violation.** `8cfe0513…`
    here vs `4742366b…` on the Phase 0 box. Cause: both boxes software-rasterise but with
    *different mesa drivers*, so pixels differ (`n_unique` 249→253, mean 104.291→104.049).
    Physics is unaffected — `sim_state` and physics-observation digests are byte-identical
    across `egl` and `osmesa` here. I5 as the gate tests it (determinism within a machine)
    holds; cross-machine pixel reproducibility was never tested and does not hold.
14. **MuJoCo frames are upside down.** Verified visually: the as-returned `agentview_image`
    has the robot hanging from the bottom of the frame. Recorded demo frames share the same
    orientation, so the two are consistent — but DINOv2 is pretrained on upright photographs,
    and feeding it inverted scenes is a silent quality loss, not a crash. The encoder now
    flips before normalising (`flip_vertical`, hashed).
15. **Recorded demo frames ≠ our render of the identical state.** With the sim state applied
    bit-exactly, demo frames still differ from ours (mean 139.1 vs 103.3, corr 0.80) — a
    lighting/asset difference in the recording setup. Training on recorded frames and
    deploying on live ones would have put a domain gap between train and deploy. Observations
    are therefore regenerated by replay (~20 s/episode). This is the single decision most
    likely to have quietly capped the BC policy had it gone the other way.
16. **Encoding is not batch-invariant on GPU.** `encode(x[:1])` differs from `encode(x)[:1]`
    by up to 7.8e-3 (one fp16 ULP) against a token std of 1.78. `batch_size` is therefore
    hashed into the cache key. It also means rollout features (encoded one frame at a time)
    differ from training features (batch 64) at ULP level — noted in `rollout()`, and it
    cannot affect a reported result because results come from simulator replay (I2).
17. **The benchmark init states are disjoint from the demo init states.** Zero exact matches;
    every demo initial state sits ~0.98 from its nearest `get_task_init_states()` entry. So
    the 72% is a genuinely held-out evaluation, not a replay of training initial conditions.
18. **A 3-episode spot check missed a 2-in-50 failure.** My first replay-fidelity probe was
    5/5 and I nearly took it as settled. The G1 check now reads the full 50-episode population
    from the collection run rather than sampling.

### Corrections to SPEC.md

- **§4.1 amended** (Phase 1): the fingerprint also covers the **GL renderer string** and
  **`obs_source`**, because a `cpu`/`cuda` tag does not separate two machines that both
  software-rasterise with different mesa drivers. `batch_size` is hashed for the reason in
  surprise 16.
- **§4.3 amended** (Phase 1): observations are regenerated by replay rather than read from the
  HDF5; policy input and the split gripper/delta loss are specified.

### Deviation from SPEC.md §3

Added **`cfe/data/`**, which §3's layout does not list. Demo loading is shared by the policy
now and the dynamics model in Phase 2, so filing it under `cfe/policy/` would have misplaced
it. Flagged rather than done silently.

### Gate state

**G0: PASS** (re-verified on this machine). **G1: FAIL (4/6)** — encoder and cache checks all
pass; the two failures are the expert-replay rate and the BC success rate. Phase 2 is
**blocked pending a decision on both** (see below).

### Next

1. **Decision needed on the BC success rate.** 72% vs a 40-70% band. Options: accept and
   record the deviation; deliberately weaken the policy (fewer epochs / smaller head) as an
   explicit logged decision; or widen the band in SPEC.md. I have not chosen — each of these
   changes what the failure set looks like in Phase 3/4.
2. **Decision needed on `demo_27` / `demo_31`.** They are already excluded from training.
   The open question is whether a 96% `expert_oracle` ceiling is acceptable for SPEC.md §6 or
   whether the demo set needs auditing across the other tasks too.
3. Consider `.gitattributes` (`* text=auto eol=lf`) so the CRLF problem does not recur.
4. Phase 2 per WORKFLOW.md P2, once 1 and 2 are settled.

### Addendum — what the 14 natural failures actually are

Characterised from `results/runs/p1-bc/metrics.jsonl` before deciding anything, because the
answer changes what decision 1 is worth:

- **All 14 failures are `max_steps` timeouts.** Every one ran the full 300 steps. Not one is a
  terminal failure (grasped and dropped, knocked the object away, released early). The natural
  arm of the failure set currently has **exactly one failure mode: the policy stalls.**
- Successes are bimodal: median 150 steps, but 5 of 36 took >200 and one took 286. The 300-step
  cap therefore sits close to genuine slow successes.
- Success rate 0.72, Wilson 95% CI **[0.583, 0.825]**. The lower bound is above 0.70, so this is
  not a small-sample artefact of the band — the policy is genuinely above it.
- **Failure supply is not the constraint.** SPEC.md §4.4 needs ~40 natural failures across ≥3
  tasks (~13/task). At a 28% failure rate that is 50 rollouts/task; at 40% it would be 35.
  Both are cheap. The band's stated purpose in §4.3 ("one that always fails gives no
  recoverable ones") is not under threat at 72%.

**A knob that must not be turned.** Because every failure is a timeout, *lowering* `max_steps`
would move the success rate down into the band without touching the policy. That is tuning the
measurement to pass a gate — CLAUDE.md rule 3. Recording it here so it is not reached for
absent-mindedly later. Raising `max_steps` moves the rate up, and is equally off-limits as a
gate-directed edit.

**The real cost of 72% is diversity, not count.** Whatever is decided about the band, a natural
failure set that is 100% timeouts gives Phase 3's search a single mode to work on. Injected
failures (§4.4's four cause tags) carry the diversity; the natural arm may need a second task
or a lower-capacity policy to produce terminal failures rather than stalls.

---

## 2026-07-31 — Phase 1 decisions resolved

Both open decisions from the previous entry are settled. Diagnostics ran first; the two
decisions turned out to share one root cause.

### Diagnostic: what the failures actually are

Traced object and gripper positions through 6 of the 14 failed BC rollouts, plus `demo_27`
and `demo_31`, with two successful controls each.

| | min gripper→object | object → basket | object lift |
|---|---|---|---|
| 6 failed rollouts | 0.040 – 0.120 m | 0.509 → 0.509 m (never moves) | +0.043 m |
| successful rollout | 0.034 m | → 0.019 m | +0.252 m |
| `demo_27` / `demo_31` | 0.049 / 0.023 m | 0.516 → 0.517, 0.526 → 0.540 m | +0.005 / +0.042 m |
| successful demos | 0.007 – 0.011 m | → 0.007 – 0.016 m | +0.235 – 0.268 m |

**Every BC failure is an approach failure** — the target is never grasped or transported. The
`+0.043 m` lift is identical to three decimals across all six failures *including* the one where
the gripper never came within 12 cm, so it is a reset-settle artefact of the object's own
geometry, not the robot.

**Corrected 2026-07-31:** an earlier version of this entry said the object is *never touched*.
That was inferred from distance-to-basket being constant, which is the wrong quantity — an
object can be brushed sideways without getting closer to the basket. Measured directly, the
target is displaced 1.3-3.8 cm in 7 of the 14 failures. It is never *transported*: max target
displacement across all failures is 0.0381 m against a minimum of 0.4843 m across all
successes, a 13x gap.

**`demo_27` and `demo_31` fail the same way** — the recorded actions replayed in our simulator
put the gripper 2–5 cm from the object instead of on it. This is the Phase 1 physics drift
(peaks 4.5–10.6) biting at the one moment with no tolerance: contact. Those two demos had the
tightest grasp margins; the other 48 had slack.

### Decision 1 — the 72% BC rate: SPEC.md §4.3 and §7 G1 amended

The 40–70% band is replaced by what it was proxying for: **≥13 natural failures per task per
50 rollouts**, a **reported failure-mode breakdown**, and a wide **[0.20, 0.85]** sanity bound
on the rate. Full reasoning, the rejected alternatives, and the rule-3 caveat are in the new
`CHANGELOG.md` — this entry does not restate them.

The decisive evidence was that **weakening the policy would not buy failure diversity**: the
failure mode is approach imprecision, so a weaker policy produces more of the same mode. The
option was rejected on measurement, not preference.

Recorded so it is not reached for later: because every failure is a `max_steps` timeout,
**lowering `max_steps` would slide the rate into the old band without touching the policy.**
That is measurement-tuning and is off-limits in either direction. Noted in `configs/policy.yaml`
next to the parameter itself.

### Decision 2 — `demo_27` / `demo_31`: closed

The earlier framing of a "96% `expert_oracle` ceiling" was wrong. §6 defines `expert_oracle`
per-episode as *"replay the successful demo's actions from the window"*, and §4.4 builds
injected failures by corrupting **successful** demos. The two bad demos never enter the source
pool, so `expert_oracle` is ~100% on the injected set by construction. Supply is comfortable:
48 usable demos/task × 3 tasks = 144 sources against §4.4's ≥40 per cause.

### Open: `expert_oracle` has no referent for natural failures

§6 lists `expert_oracle` as mandatory for every table, but natural failures start from
benchmark init states, which share **no exact match** with demo init states (verified in
Phase 1). There is no demo whose actions could serve as the oracle. Logged now, to be resolved
in Phase 3 when `build_failure_set.py` forces the issue. The likely resolution is that
`expert_oracle` applies only to the injected arm, and natural-failure tables report the other
three baselines — but that is a §6 change and is not being made unilaterally.

### Consequence carried into Phase 3

The natural failure set is **100% approach failures**, which nearly duplicates the
`grasp_offset` injected cause. Failure-set diversity now rests entirely on the injected arm.

### Demo replay yield across tasks (decision 2 follow-up)

Replay-only sweep, no encoding, on two further `libero_object` tasks:

| Task | Name | Demos | Failed | Yield |
|---|---|---|---|---|
| 0 | `pick_up_the_alphabet_soup...` | 50 | 2 (`demo_27`, `demo_31`) | **0.96** |
| 1 | `pick_up_the_cream_cheese...` | 50 | 4 (`demo_8, 28, 29, 40`) | **0.92** |
| 2 | `pick_up_the_salad_dressing...` | 50 | 3 (`demo_10, 35, 46`) | **0.94** |

Yield is 92–96%, so task 0 was not unusually forgiving and the grasp-margin explanation is a
general property of replaying these demos in our simulator, not a task-0 quirk. The Phase 4
source pool across these three tasks is **141 usable demos** against §4.4's ≥40 per injected
cause. No constraint, and no surprise waiting in Phase 4.

### Correction: the first failure-mode breakdown was wrong

The first run of the new breakdown reported `object_moved_not_placed: 14`, contradicting the
diagnostic above, which showed the object is never touched. The diagnostic was right and the
classifier was wrong, for two independent reasons — both found by measurement, not inspection:

1. **`object_positions` matched `<object>_to_robot0_eef_pos` keys.** Those are object-minus-
   end-effector *relative* positions, so they change whenever the **arm** moves. Every episode
   in which the robot did anything at all was marked "object disturbed."
2. **Objects settle under gravity at reset.** Over 120 idle steps with a null action, peak
   vertical drift is **0.1062 m** and peak horizontal drift is **0.0010 m** — both identical
   across six init states, so it is deterministic settling, not the robot. A 3-D displacement
   threshold fires before the arm has moved.

Fixed by excluding relative keys and thresholding on **horizontal** displacement.
`CONTACT_EPS = 0.01 m` now sits 10× above the measured idle drift and ~50× below the ~0.5 m an
object travels when actually carried to the basket. Two regression tests pin both facts, and
the threshold's margins are asserted against the measured numbers rather than hard-coded.

`results/runs/p1-bc2/` carries the bad breakdown and should not be cited; its success rate and
rollouts are valid. Superseded by `p1-bc3`.

### I5 re-verified

`p1-bc` and `p1-bc2` (same seed, same config, separate processes, one run under CPU contention
from the yield sweep) produced **identical rollout outcomes across all 50 rollouts** and
identical losses to 5 decimals (`train 0.12041`, `val 0.36531`).

### Second correction: the breakdown had to separate target from distractors

`p1-bc3` reported `no_object_contact: 3, object_moved_not_placed: 11`. The 11 were the robot
**nudging a distractor** — this scene holds five distractors plus the basket — while never
touching the alphabet soup. That collapsed the one distinction the amended §4.3 asks the
breakdown to make: a missed approach needs a different Phase 3 edit than a grasp-and-drop.

`classify_failure` now tracks the task's target object separately. The target is derived from
`task.language` ("pick up the alphabet soup..." → `alphabet_soup_1_pos`) rather than
hard-coded, since §4.4 needs ≥3 tasks, and `find_target_object_key` **raises** if it cannot
identify exactly one target — a silently wrong target turns the metric into noise, which is
how the first two versions failed.

`results/runs/p1-bc3/` carries the pre-fix breakdown and is superseded by `p1-bc4`.

### Final Phase 1 failure-mode breakdown (`p1-bc4`)

50 rollouts, seed 0, success rate 0.72 (36/50), 14 failures:

| Mode | n | Target displacement | Other-object displacement |
|---|---|---|---|
| `target_moved_not_placed` | 7 | 0.0134 – 0.0381 m | ≤ 0.0118 m |
| `distractor_only_target_untouched` | 4 | 0.0010 m (idle floor) | 0.0165 – 0.1024 m |
| `no_object_contact` | 3 | 0.0010 – 0.0036 m | ≤ 0.0016 m |

Successes displace the target 0.4843 – 0.5074 m. **No failure exceeds 0.0381 m**, so the two
populations are separated by 13× with nothing in between: every failure ends before a grasp is
achieved, and none is a drop-in-transit. The untouched cases sit exactly at the 0.0010 m idle
drift floor measured above, which is the intended behaviour of the threshold.

Three attempts were needed to make this metric measure what it claims (relative-position keys,
reset settle, then distractor conflation). The success rate was bit-identical across all four
runs; only the breakdown was wrong.

### Gate bug: `fingerprint_mismatch_raises` was not re-runnable

The check deliberately corrupts a fingerprint on disk, and it did so in a fixed directory
(`<cache_dir>/_g1_probe`) that it never cleaned up. The first G1 run left the tampered
fingerprint behind, so the *second* invocation crashed on its own leftovers before reaching
any check — a gate that can only be run once is not a gate. Now runs in a `tempfile.mkdtemp`
directory removed in a `finally`.

### G1 gate: 5/6 (`g1-20260731T191027Z`, policy run `p1-bc4`)

```
[PASS] encoder_frozen               0 params require grad
[PASS] cache_populated              48 episodes, all with metadata
[PASS] cache_bit_identical          fresh replay+encode == cache, np.array_equal
[PASS] fingerprint_mismatch_raises  tamper raises; identical fingerprint reopens cleanly
[FAIL] expert_replay_success        48/50 (demo_27, demo_31), rate 0.96
[PASS] bc_failure_supply            72.0%, 14 failures >= 13, modes reported
```

`make lint` clean (21 files). `make test-slow` **41 passed** (32 fast + 9 slow).

**`expert_replay_success` is the one open blocker, and it cannot pass as written.** SPEC.md §8
requires the success flag to be reproduced *exactly*, over all 50 demos in the HDF5. Measured
yield is 92–96% across three tasks. This is not a defect in our replay code as far as the
evidence goes — reset is bit-exact (max diff 0.0), 48/50 reproduce, and the two failures show
a coherent physical story (gripper 2–5 cm short, tightest grasp margins in the set) — but it
means the check is permanently red on this data. **Not amended: one spec amendment was already
made this session, and a second one, on the invariant that protects against replay bugs, is
not a call to make unilaterally.** Options are in "Next" below.

### G1: PASS (6/6) — Phase 1 complete

`expert_replay_success` resolved by amending SPEC.md §8 to scope the exactness requirement to
episodes admitted to the cache, with a yield tripwire (`data.min_replay_yield = 0.85`).
Reasoning in `CHANGELOG.md`. The requirement itself is unchanged: 48/48 cached episodes
reproduce their success flag exactly, no tolerance.

```
[PASS] encoder_frozen               0 params require grad
[PASS] cache_populated              48 episodes, all with metadata
[PASS] cache_bit_identical          fresh replay+encode == cache, np.array_equal
[PASS] fingerprint_mismatch_raises  tamper raises; identical fingerprint reopens cleanly
[PASS] expert_replay_success        48/48 cached exact; yield 0.96 >= 0.85 tripwire
[PASS] bc_failure_supply            72.0%, 14 failures >= 13, modes reported
G1: PASS (6/6 checks)
```

`make test-slow` **41 passed** (32 fast + 9 slow). `make lint` clean (21 files).

Confirms in passing that the `data` block of `configs/encoder.yaml` is not hashed: adding
`min_replay_yield` left `encoder_cfg_hash` at `a005df8c057942c3` and the 2.8 GB cache opened
without a fingerprint mismatch.

### Gate state

**G0: PASS. G1: PASS.** Phase 2 is unblocked.

### Next

1. Phase 2 per WORKFLOW.md P2 — latent dynamics model, G2 gate. Note G2's hard stop: if the
   action-ignoring model is competitive, fix the dynamics model before writing search code.
2. Carry into Phase 3: the natural failure set is 100% pre-grasp approach failures (7 target
   brushed, 4 distractor-only, 3 no contact), which nearly duplicates the `grasp_offset`
   injected cause. Diversity rests on the injected arm.
3. Open §6 question, deliberately deferred: `expert_oracle` has no referent for natural
   failures, whose init states share no exact match with any demo init state. Resolve when
   `build_failure_set.py` forces it.
4. Nothing in this session is committed.

## 2026-08-01 — Phase 2 Part A: latent dynamics, G2 **FAIL**

Scope was Part A only (WORKFLOW P2). `cfe/eval/` was deliberately not written this session.

### What landed

- Episode IDs namespaced to `libero_object_t0_<id>` (96 cache files renamed, no re-encoding).
  G1 re-verified **PASS 6/6** after the rename before anything was built on top.
- `scripts/collect_rollouts.py` — 50 BC rollouts (34 success / 13 failure / 3 already cached),
  9520 frames at 6.71 frames/s, **zero determinism mismatches vs `p1-bc4`**. Failures are
  cached here on purpose; they were correctly excluded as BC *targets* but are training data
  for dynamics, and they double as Phase 3's natural failure set.
- `cfe/dynamics/{latents,model,data}.py`, `configs/dynamics.yaml`, `scripts/probe_representation.py`,
  `scripts/train_dynamics.py`, `scripts/verify_g2.py`, `tests/test_dynamics.py`.
- `make lint` clean (32 files). `make test` **74 passed** (32 existing + 42 new).

### Representation probe (`p2-probe`) — run before any model existed

Distance from each rollout's terminal latent to the goal set, against the simulator success
flag. Deliberately measured before a model could bias the choice of reduction.

| reduction | R | rho (raw) | rho (layernorm) |
|---|---|---|---|
| cls | 1 | -0.7315 | -0.7285 |
| mean | 1 | -0.7315 | -0.7408 |
| grid2 | 5 | -0.7439 | -0.7562 |
| **grid4** | **17** | **-0.7778** | **-0.7778** |
| full | 257 | -0.7778 | -0.7778 |

All ten clear the pre-registered `|rho| >= 0.30` with the correct sign. `grid4` ties `full`
exactly at 1/15 the tokens, so `grid4` was chosen. Mean distance success 23.83 vs failure
49.04 (grid4/layernorm).

### SPEC.md §5 reading, recorded here as a clarification

§5 wants "a successful demo for the same task *and initial state*". Phase 1 established that no
demo init state matches any benchmark init state, so that is unimplementable for the natural
arm. Used §5's own fallback: **`z_goal` = minimum L2 over the 48 successful demo terminal
latents**. Same root cause as the still-open `expert_oracle` gap in §6.

### SPEC.md §4.2 reading, recorded here as a clarification

"teacher-forced one-step loss plus a short autoregressive rollout loss (2 steps)" is read as
***chunk*-level**: chunk 1 teacher-forced, chunk 2 conditioned on the model's own output
(`ar_chunks=2`). The single-frame reading trains nothing about compounding error across a
chunk, which is the regime §5's search actually runs in. Chunks weighted equally.

### G2: **FAIL (3/5)** — `g2-20260801T014804Z`, dyn run `p2-dyn`

15 held-out episodes, 2542 gated windows. Criteria were pre-registered in `configs/dynamics.yaml`
before any number existed.

```
[FAIL] beats_copy_last_latent   h1 0.1058 vs 0.0793 (-33.4%);  h2 0.1058 vs 0.0976 (-8.3%)
                                h4 0.1063 vs 0.1200 (+11.4%);  h8 0.1095 vs 0.1546 (+29.2%)
[FAIL] beats_action_ignoring    h8 rel_improvement 0.0499 vs pre-registered 0.20
                                CI excludes zero at ALL horizons (h8 [0.005375, 0.006158], n=2542)
                                ci_criterion_met=true, rel_criterion_met=false
[PASS] goal_distance_correlates grid4/layernorm rho=-0.7778
[PASS] eval_deterministic       bit-exact per-window errors
[PASS] config_snapshot_present
```

Non-gated: h16 +9.2%, h32 +14.2%, h64 +20.3% vs the twin. Training: dynamics val 0.011649,
twin val 0.013349; 2546 s / 2719 s; 7,773,312 params; 13,361 train / 2,422 val windows.

**The result is reported as-is. Neither the threshold nor the model was adjusted.** Action
conditioning helps *reliably* (CI excludes zero everywhere, and the margin grows monotonically
with horizon) but *weakly* — 5% at h8 against a pre-registered 20%.

### Why it failed — diagnosis, incomplete

The FAIL only means "action-conditioning does not help enough" if the model is sound. It is not.

- **Not a wiring bug.** Predictions differentiate across the horizon (pred std 0.0373 vs target
  0.0972; pred[h1] vs pred[h8] 0.0941 vs target 0.1604) — under-predicts change, but the query
  tokens are doing distinct work.
- **Not output collapse.** Pred overall std 1.0025 vs target 1.0000.
- **The model cannot overfit 64 windows.** 600 steps on a fixed batch: 0.5476 -> 0.0479 ->
  0.0257 -> 0.0202 -> 0.0168, final `mean|err|` **0.1173** against copy-last **0.1261** on the
  very batch it is memorising. Full-run val loss was 0.0116. **This is an architectural floor,
  not a data or epoch limit** — so G2's failure does *not* establish that action-conditioning
  is useless.

Two structural causes were proposed:

- **H1 — `d_model=384 < latent_dim=768` is a rank bottleneck. REFUTED.** Tested as linear
  algebra rather than by training: the best rank-r reconstruction of the targets (SVD,
  Eckart-Young) is a lower bound on any rank-r linear head. rank 64 -> 0.0808, rank 128 ->
  0.0566, rank 256 -> 0.0354, **rank 384 -> 0.0236**, capturing 99.79% of target variance. A
  384-wide head can represent the targets 5x better than the observed 0.117 floor. The head
  is not the constraint.
- **H2 — no residual path from the last context frame**, so the model must synthesise the whole
  latent instead of a delta off `z[t]`. **Supported, with a caveat that matters more than the
  result.** Tested at reduced scale on CPU (d=128, L=2, 622K params) because the box was
  crashing; both arms identical in config, seed, and batch, differing only in the residual path.

  | | mean\|err\| | vs copy-last |
  |---|---|---|
  | copy-last | 0.126149 | — |
  | small as-shipped | 0.151249 | -19.9% |
  | **small RESIDUAL** | **0.117384** | **+6.9%** |
  | full-scale as-shipped (7.77M) | 0.117304 | +7.0% |

  The residual path is worth **+22.4%** at matched scale, and a 622K residual model reaches
  what the 7.77M shipped model reaches — 12x fewer parameters, same number.

  That equivalence — 0.117384 vs 0.117304 across a 12x parameter gap — looked suspicious, and
  chasing it produced the most important correction of the session.

### Correction: "the model cannot overfit 64 windows" was a step-budget artifact

The 600-step overfit test was the basis for calling the floor architectural. It does not survive.
Running the residual arm 4x longer (2400 steps, CPU):

```
copy-last                    0.126149
full-scale as-shipped (600)  0.117304
small residual (600 steps)   0.117384
small residual (2400 steps)  0.088903      +24.3%,  loss 0.008476 and still descending
```

**~0.117 is neither an information floor nor an architecture floor — it is simply where this
setup lands after 600 steps.** The overfit test was undertrained, so the inference built on it
("architectural floor, therefore G2's FAIL establishes nothing") was wrong.

### And the full run, unlike the overfit batch, *was* converged

Checked independently from `p2-dyn/metrics.jsonl` rather than re-deriving it:

```
dynamics        val, last 5 epochs: 0.012029 0.011862 0.011901 0.011619 0.011649  (delta -0.00038)
                train 0.011316 -> 0.010582                      (train-val gap opening)
action_ignoring val, last 5 epochs: 0.013458 0.013108 0.013116 0.013220 0.013349  (rising)
```

The twin is past its best at ~epoch 26 and overfitting. So the undertraining found on a fixed
64-window batch **does not transfer to the full run**, and more epochs will not rescue G2.

**Revised reading: G2's FAIL is probably real.** Action-conditioning helps reliably — the paired
CI excludes zero at every horizon and the margin grows monotonically to +20.3% at h64 — but
weakly, ~5% at h8 against a pre-registered 20%. On the current evidence that is a true property
of this model on this data, not a defect introduced in the implementation.

### H2 confirmed against its matched control

Running both arms to 2400 steps, identical in everything but the residual path:

| | 600 steps | 2400 steps |
|---|---|---|
| as-shipped | 0.151249 | 0.103996 |
| **residual** | **0.117384** | **0.088903** |
| residual advantage | +22.4% | **+14.5%** |

The advantage survives 4x the budget, so it is not merely faster convergence. It does narrow
(22.4% -> 14.5%), so part of it is optimisation speed. Both arms still descending at 2400.

### Why residual may nevertheless move G2, for a reason worth stating in advance

Both gate failures have a common cause: **the model spends most of its capacity reconstructing
`z[t]`**, which is the bulk of `z[t+1]`.

- `beats_copy_last_latent` fails hardest at h1 (-33.4%), exactly where copy-last is strongest.
  A residual model *starts* at copy-last by construction, so this check should flip.
- `beats_action_ignoring` compares two models that both carry that same large shared
  reconstruction term. It is common to both and dilutes the relative gap. Removing it means the
  measured ratio is taken over the *delta* — which is precisely where action information lives.
  Absolute gap at h8 is 0.1153 - 0.1095 = 0.0058; if residual cuts both errors substantially
  while preserving that gap, `rel_improvement` rises toward the pre-registered 0.20.

**This is a prediction, not a result, and it is recorded here before the run so it cannot be
retrofitted.** It could also fail: residual helps the twin too, and if it helps the twin more,
the gap shrinks and G2 fails harder. Either outcome is informative; neither changes the gate.

### What surprised me

- The gate's two failures point in opposite directions and both matter. `beats_copy_last_latent`
  fails at *short* horizons (h1 -33%) because copy-last is extremely strong when latents barely
  move; `beats_action_ignoring` fails at *long* horizons only in magnitude, never in sign.
  A model that beat copy-last at h1 would likely clear both.
- `grid4` matching `full` to four decimals on rho was not expected, and it saved 15x on tokens.

### Hardware: four unexpected shutdowns, and why it is not OOM

The box hard-rebooted four times today (13:55, 14:11, 14:13, 14:59) while running GPU work.
**Windows Kernel-Power Event 41 each time, no bugcheck code, no BSOD since May 7**, paired with
Event 125. Linux logged no OOM kills and system RAM peaked well under the 7.7 GB WSL allocation.
A CUDA OOM raises a Python exception; it does not reboot the host.

`nvidia-smi -q -d POWER` reports **Current Power Limit 105.65 W against a Default of 100 W** on
an RTX 4070 Laptop. This is a power-delivery trip under sustained 100% GPU, and it is a hard
constraint on this box for the rest of the project — Phase 3's search budget must assume it.

Mitigation is `nvidia-smi -pl 85` from an **elevated Windows** shell; WSL cannot set clocks
("current user does not have permission"). Resets on reboot.

Note for future diagnostics: H1 was the variant driving VRAM to 7.7/8.2 GB, and refuting it
analytically on CPU cost seconds and zero crash risk. Prefer that ordering.

### Gate state

**G0: PASS. G1: PASS. G2: FAIL (3/5).**

Per WORKFLOW P2's hard stop, no counterfactual search code is written until G2 passes.

### Next

1. **Decision for the user, not for me: add the residual connection and retrain.** It is a
   one-line change to `LatentDynamics.forward`, but it is an architecture change to a gated
   model, so it is logged and decided rather than slipped in. Cost is ~87 min of sustained GPU
   for model + twin, plus the G2 re-run.
2. **Cap GPU power before that run** (`nvidia-smi -pl 85`, elevated Windows shell). Today's
   trips came from runs of 150 s and up; losing an 87-minute run to a reboot costs the run and
   the uptime.
3. If residual does not move `beats_action_ignoring`, the next candidates in order:
   proprioception (§4.2 specifies latents + actions only; this would be a documented deviation),
   a shorter horizon, a finer reduction than `grid4`, and a larger `k_context`.
4. `cfe/eval/` (WORKFLOW P2 Part B) is still unwritten and is frozen the moment it is.
3. `cfe/eval/` (WORKFLOW P2 Part B) is still unwritten and is frozen the moment it is.
4. Nothing from Phase 2 is committed.

## 2026-08-03 — Phase 2 Part A: residual retrain, G2 **FAIL** again (4/5)

### What landed

- Phase 2 Part A committed: `b5861f2` on branch `phase2-dynamics` (branched off `main` first).
  Lint clean, 78 fast tests green. Nothing from `results/` leaked past the ignore rules.
- `p2-dyn-res` trained: residual model 2899 s, action-ignoring twin 2919 s, 5849 s total.
- G2 re-run `g2-20260804T000955Z`: **FAIL, 4/5.** `beats_action_ignoring` is the sole failure.

### Both halves of the pre-registered prediction came true at once

The prediction recorded at PROGRESS.md:764-779 *before* this run said residual should flip
`beats_copy_last_latent`, and warned it "could also fail: residual helps the twin too, and if it
helps the twin more, the gap shrinks and G2 fails harder." **Both happened.**

```
                        p2-dyn (first run)      p2-dyn-res (this run)
beats_copy_last_latent  FAIL                    PASS
  h1                    -33.4%                  +2.07%   (0.077650 vs 0.079292)
  h2                     -8.3%                  +7.56%   (0.090253 vs 0.097631)
  h4                    +11.4%                 +16.98%   (0.099599 vs 0.119971)
  h8                    +29.2%                 +29.47%   (0.109058 vs 0.154620)

beats_action_ignoring   FAIL                    FAIL (worse)
  h8 rel_improvement    0.0499                  0.0350   vs pre-registered 0.20
  CI excludes zero      all horizons            h2/h4/h8 (not h1); criterion met
extended, not gated
  h16 / h32 / h64       +9.2 / +14.2 / +20.3%   +6.3 / +11.6 / +17.3%
```

The prediction's arithmetic is where it broke. It required residual to "cut both errors
substantially **while preserving that gap**." Errors fell as expected, but the absolute h8 gap
went **0.0058 → 0.003955, a 32% shrink**, so the ratio fell instead of rising. The model got
better at every horizon and simultaneously less action-dependent.

Mechanism, consistent with the numbers: initialising at copy-last-latent makes it easier to
score well *without using actions*, so the twin banks most of the gain. Residual bought absolute
accuracy and paid for it in action-sensitivity.

### Correction: "not merely faster convergence" was wrong

The 2026-08-01 entry, the `configs/dynamics.yaml` comment, and `cfe/dynamics/model.py:148-154`
all assert residual is worth +22.4% at 600 steps and +14.5% at 2400, and that surviving 4x the
budget proves it is not just faster convergence. **At full scale on the real dataset that does
not hold.** Val loss, residual minus non-residual:

```
epoch     0       5      10      15      20      25      29
delta   -54%   +0.6%   +1.2%   +4.9%   +2.4%   +2.8%   +3.0%
```

The head start is gone by epoch 5 and residual converges ~3% *worse* (final val 0.012002 vs
0.011649). The train-val gap also inverts — non-residual 0.010582/0.011649 (10% gap), residual
0.011725/0.012002 (2.4% gap) — so residual is **underfitting**: the delta constraint costs
capacity the model was using. The reduced-scale diagnostic misled because on a 64-window overfit
batch there is not enough data for lost capacity to bite.

### What surprised me

- **I over-read epoch-wise val separation three times in one session** ("widening each
  checkpoint", "roughly halved", and the reduced-scale diagnostic that started this). `p2-dyn`'s
  *own* model-twin separation swings 8.5%-11.8% across epochs with no trend. That +-3 point band
  is the noise floor, and it is larger than the effect I kept forecasting from. CLAUDE.md says
  to measure a metric's noise floor before reporting from it; I did not do that here, and it
  cost three wrong calls in a row. Epoch-wise val separation cannot discriminate these runs.
- Two independent architectures now agree that action-conditioning is worth 3.5-5% at h8. That
  is the independent confirmation the repo asks for, and it means the weak result is not
  plausibly a training or architecture artifact.

### Hardware: measured, and not thermal

Instrumented ramps replaced speculation. Both logs are in `~/runlogs/` and `~/*_ramp.csv`
(deliberately in $HOME — WSL clears /tmp on every boot, which destroyed several earlier rounds).

- **GPU ramp**, staged to 100% duty for 180 s: survived. Settles at 73-74°C / 55 W / 1560 MHz
  after derating from a 110 W transient. No crash.
- **CPU ramp**: survived ~9 min of full saturation. Throughput trend within the constant-config
  final stage was **+2.3%** — flat, so the host CPU is not thermally throttling. Chassis proxy
  (idle GPU temp) soaked +10°C and plateaued at 61°C.
- **The OOM hypothesis is refuted for this workload.** Across the 97-minute training run, WSL
  MemAvailable never fell below 3671 MiB and child RSS was flat at ~2985 MiB. GPU memory pinned
  at 6869/8188 MiB with zero growth — worth remembering, since that is 84% of VRAM and Phase 3
  runs rollouts alongside a loaded model.
- **Correction:** the 2026-08-01 entry's advice to cap power with `nvidia-smi -pl 85` does not
  work — ASUS OEM firmware refuses it. Silent mode was also wrong advice: it lowers fan RPM
  along with TGP, the wrong direction if dissipation is the constraint.
- **Correction:** `nvidia-smi`'s `SW/HW Thermal Slowdown` *counters* read 0 us on this die and
  are unreliable — the `SW Power Capping` counter reads 0.06 s against 218 s of observed
  capping. Per-second sampling of `clocks_throttle_reasons.active` is the trustworthy channel,
  and it shows SW Thermal Slowdown (0x20) active in 119 samples at a mean of 75.0°C. The GPU
  does throttle thermally, at a low trip point, which suggests degraded dissipation worth
  cleaning — but it throttles *gracefully* and did not crash in 10 minutes of load.

The machine did not crash once during ~2 hours of instrumented load today, including the full
97-minute training run. The shutdown cause remains unidentified; no load hypothesis reproduced
it, and the crash 91 s after a boot with nothing running is unexplained by any of them.

### Gate state

**G0: PASS. G1: PASS. G2: FAIL (4/5).** Thresholds pre-registered 2026-07-31 are untouched.
Per WORKFLOW P2's hard stop, no counterfactual search code is written until G2 passes.

### Next

1. **Decision for the user: is `residual: true` kept?** It is strictly better on
   `beats_copy_last_latent` and strictly worse on the gated check and on val loss. Keeping it
   means the h1/h2 pathology is fixed; reverting restores 0.0499 at h8. Neither passes G2. The
   comments in `configs/dynamics.yaml:29-37` and `cfe/dynamics/model.py:148-154` still assert
   the refuted "not merely faster convergence" claim and must be corrected either way.
2. **The real question is whether the 0.20 bar is reachable at all in this latent space.** Two
   architectures give 3.5% and 5.0%. Remaining candidates, in order: proprioception (§4.2
   specifies latents + actions only, so this is a documented deviation), a finer reduction than
   `grid4`, a larger `k_context`, and a shorter horizon. If none moves it, the honest conclusion
   is that the pre-registered bar was miscalibrated for this representation — **which is a
   measurement to bring to the user, not a threshold for me to lower.**
3. `cfe/eval/` (WORKFLOW P2 Part B) is still unwritten and is frozen the moment it is.

## 2026-08-04 — residual reverted; three diagnostics; the horizon question put to a decision

### What landed

- **`residual` reverted to `false`.** Non-residual is better on the metric the project depends
  on at every horizon (`rel_improvement` h8 0.0499 vs 0.0350, h64 0.203 vs 0.173). Residual won
  only `beats_copy_last_latent`, and won it by making the model less action-sensitive. The code
  path and its four tests are kept: a reproducible negative result beats a deleted branch.
  Numbers are recorded at `configs/dynamics.yaml` and `cfe/dynamics/model.py` so the default
  reads as a result rather than a preference. **No gate threshold moved.**
- Three diagnostics, none of which gate anything: `diagnose_action_horizon.py`,
  `diagnose_window_selection.py`, `diagnose_edit_discrimination.py`.

### Action-conditioning benefit is horizon-driven, not episode-selection-driven

The first sweep to h96/h128 looked decisive (+28.9% at h96) until its own built-in cross-check
disagreed with the gate at h8 — 0.0393 vs 0.0350 — which exposed the confound: long horizons are
only measurable in long episodes. `diagnose_window_selection.py` separates them by computing
every (horizon, window-set) pair.

```
              8        16        32        64        96       128   <- min future frames
   n=      2542      2422      2182      1702      1222       742
h8      +0.0350   +0.0347   +0.0347   +0.0380   +0.0369   +0.0393
h16           -   +0.0586   +0.0572   +0.0630   +0.0635   +0.0757
h32           -         -   +0.1005   +0.1158   +0.1291   +0.1456
h64           -         -         -   +0.1729   +0.2129   +0.2555
h96           -         -         -         -   +0.2238   +0.2892
```

Down a column is pure horizon; across a row is pure selection. **Horizon 4.5x-6.1x, selection
1.12x-1.48x**, and the h8 row is nearly flat across window sets differing 3.4x in size. The
trend is real. Correction to an intermediate claim made here: the curve does **not** cross 0.20
at h64 — that used the selection-inflated 0.2555. On the largest set that can measure h64 it is
0.1729. The unconfounded crossing is between h64 and h96 (h96 = 0.2238 on 1222 windows).

### The model does resolve small action edits

`beats_action_ignoring` only asks whether actions help *on average*. A model could pass it by
predicting coarse drift while being blind to the fine edits §5's search makes, and **no gate
tests that**. Ground truth needs no simulator: feed the model wrong actions and the error must
rise, with the shape of the rise as a function of edit size being the answer.

```
      edit      h8 delta       rel     h64 delta       rel     (p2-dyn, 1702 windows)
  0.05 std     +0.000019    +0.02%     +0.000075    +0.05%     all CIs exclude zero
  0.10 std     +0.000070    +0.07%     +0.000344    +0.25%
  0.25 std     +0.000435    +0.41%     +0.002225    +1.60%
  0.50 std     +0.001823    +1.71%     +0.007939    +5.70%
  1.00 std     +0.007140    +6.70%     +0.022491   +16.15%
  shuffled     +0.050657   +47.54%     +0.115286   +82.79%
```

Response is **quadratic** — each 2x in edit size gives ~4x in error change — which is the
signature of a smooth local gradient, not coarse-motion-only behaviour. Detectable down to 0.05
std. The action-ignoring twin is bit-identical under every perturbation, which validates the
harness; without that control none of the rows would mean anything.

Only the 6 EE-delta dims are perturbed. Scaling Gaussian noise by the gripper dimension's std
would not be the same operation, per CLAUDE.md; gripper edits are a separate question.

**Caveat worth carrying into Phase 3:** at search-scale edits the effect is small relative to
the model's own error — a full 1.0-std edit moves h8 error by 6.7%, so model inaccuracy is ~15x
the edit effect (~6x at h64). Systematic error may partly cancel when *comparing* edits, which
is what the search does, so this is a concern rather than a verdict. It does argue the search
needs longer horizons or larger edits to accumulate usable signal.

### What surprised me

- A diagnostic's own sanity cross-check caught a confound that would otherwise have produced a
  confident, wrong recommendation to move the gate. Recomputing a number the gate already knows
  cost nothing and was the most valuable line in the script.
- Quadratic edit response was not something I expected to be so clean.

### Gate state

**G0: PASS. G1: PASS. G2: FAIL (3/5 with residual reverted; `p2-dyn` is the current model).**
Thresholds pre-registered 2026-07-31 are untouched. WORKFLOW P2's hard stop still applies.

### Next — an open decision for the user, not for me

**Is h8 the right horizon for `beats_action_ignoring`?** The evidence for re-examining it is now
independent of "it would pass": the model has a genuine smooth action gradient, the benefit is
~6x stronger at the horizons §5's search actually runs at, and that is horizon-driven rather
than selection-driven. The evidence against changing it: 0.20-at-h8 was pre-registered, and
picking a horizon after seeing which one passes is the exact shape of the failure CLAUDE.md
rule 3 exists to prevent.

**I am not making this call.** If it is taken it goes in `CHANGELOG.md` with reasoning and
rejected alternatives. If it is not taken, the remaining levers are proprioception (a documented
§4.2 deviation), a finer reduction than `grid4`, and a larger `k_context`.

`cfe/eval/` (WORKFLOW P2 Part B) is still unwritten and is frozen the moment it is.

## 2026-08-04 (later) — gate amended by the user; **G2 PASS 5/5**; residual revert reversed

### What landed

- **G2's magnitude criterion moved from h8 to h96**, decided by the user after both diagnostics
  were presented. Threshold unchanged at 0.20; only the horizon moved. `ci_horizons: [4, 8]` and
  `min_abs_spearman: 0.30` untouched. Logged in `CHANGELOG.md` with reasoning and four rejected
  alternatives. `scripts/verify_g2.py` scores the magnitude criterion on its own window set
  (1222 of 2542, since only longer episodes supply 96 future frames) and reports
  `n_rel_criterion_windows` so the narrowing is visible in every run.
- **`residual` restored to `true`.** The revert earlier the same day was wrong.
- **G2: PASS 5/5** on `p2-dyn-res` (`g2-20260804T185522Z`).

### The revert was wrong, and the way it was wrong is the lesson

```
                    beats_copy_last   beats_action_ignoring @h96   G2
p2-dyn (plain)          FAIL              0.2678 PASS            4/5 FAIL
p2-dyn-res              PASS              0.2238 PASS            5/5 PASS
```

`beats_copy_last_latent` never moved, and `p2-dyn` had been failing it since 2026-08-01. So on
criteria that never changed the standings were **always p2-dyn 3/5 vs p2-dyn-res 4/5** —
`p2-dyn-res` was a full check ahead the whole time. The gate amendment fixed the one criterion
both were failing, so the model already ahead came out on top. Predictable in hindsight; not
predicted.

The revert recommended the model scoring *lower* on the pre-registered gate, on the argument
that action-sensitivity matters more than the gate says. That is substituting judgement for the
gate. The argument was not wrong on the facts — residual **is** less action-sensitive at every
horizon — it was wrong about whose call it was.

**The trade-off it identified still stands and is carried forward:** `p2-dyn-res` clears 0.20 by
12% where `p2-dyn` clears it by 34%, and it converges ~3% worse with an inverted train-val gap
showing it underfits. If Phase 3's search is starved of action signal, `residual` is the first
flag to revisit and `p2-dyn` is already trained and on disk.

**Worth stating plainly:** two models and two gate configurations were tried, and something
passes. That pattern deserves suspicion. What defends it is that `beats_copy_last_latent` was
fixed *before* any gate change and its result never moved, so no newly-discovered fact was
reached for to justify the switch. A reader who finds that insufficient has every number needed
in `CHANGELOG.md`.

### Gate state

**G0: PASS. G1: PASS. G2: PASS (5/5).** Model of record: `p2-dyn-res`, gate run
`g2-20260804T185522Z`. WORKFLOW P2's hard stop is cleared.

### Next

`cfe/eval/` (WORKFLOW P2 Part B): `metrics.py` and `replay_protocol.py`. **Frozen the moment it
is written** (CLAUDE.md rule 1), so it gets its own session and its own review. Every experiment
built on it reports `random_perturb` (same edit budget) and `no_edit` alongside — the first says
whether the world model does anything, the second catches replay bugs.
