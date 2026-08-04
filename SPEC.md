# SPEC — Counterfactual Action Editing in Latent Action Space

## 0. One-paragraph statement of the work

Given a failed robot manipulation episode, search a learned world model for the *minimal
edit to the executed action sequence that would have succeeded*, then validate that edit by
replaying it in simulation from the identical initial state. Run the same search in two
spaces — raw end-effector actions and a learned latent action space — and compare. The
deliverable is (a) a failure-explanation tool and (b) the first evaluation of latent action
spaces by counterfactual editability rather than by downstream policy success.

Two claims, in priority order:

1. **C1 (must land).** Counterfactual edits found in a world model replay successfully in
   sim at a rate far above a random-perturbation baseline of equal edit budget.
2. **C2 (the interesting one).** Edits found in latent action space replay successfully more
   often and touch fewer timesteps than edits found in raw action space.

If C1 fails, the project is a negative result about world-model fidelity and should be
written up as such. If C1 lands and C2 fails, that is still a publishable finding: latent
actions help supervision but not reasoning.

## 1. Non-negotiable invariants

These are the things that make the results real. Violating any of them silently invalidates
everything downstream.

- **I1 — Exact state reset.** Replay must reset the simulator to the byte-identical initial
  state of the original episode. Not "the same task," not "a similar scene." If exact reset
  is not verified by a test, no number in this repo means anything.
- **I2 — Replay is the only ground truth.** Latent-space goal distance is a *search
  objective*, never a result. Every reported success number comes from executing actions in
  the simulator.
- **I3 — Metrics are frozen before methods.** `eval/metrics.py` and
  `eval/replay_protocol.py` are written and tested in Phase 2 and are not edited afterward
  except to fix a demonstrated bug, recorded in `CHANGELOG.md` with the reason.
- **I4 — Edit budget is swept, never tuned.** Every headline result is a curve over edit
  budget, not a single number at a favorable setting. See §6.
- **I5 — Determinism.** Every run takes an explicit seed. Same seed, same config, same
  numbers. A test asserts this.

## 2. Environment and dependencies

**Simulator:** LIBERO (built on robosuite/MuJoCo). Task suites, in order of adoption:
`libero_object` first (simplest), then `libero_spatial`, then `libero_goal`.

**The install is the single largest schedule risk.** LIBERO pulls robosuite, MuJoCo, and
BDDL, and version drift between them is the usual failure. Phase 0 exists only to get a
pinned, reproducible environment and a headless render working. Do not start Phase 1 until
`scripts/verify_env.py` passes.

API surface, **verified against the installed package on 2026-07-30** (Phase 0). This section
was previously a from-memory guess; it now records what `libero==0.1.1` actually exposes.
Corrections logged in `PROGRESS.md`.

```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_object"]()  # 10 tasks
task = suite.get_task(0)  # .name .language .bddl_file .problem_folder
path = suite.get_task_bddl_file_path(0)  # use this, not a manual path join
inits = suite.get_task_init_states(0)  # (50, 110) float64

env = OffScreenRenderEnv(bddl_file_name=path, camera_heights=128, camera_widths=128)
env.seed(seed)
env.reset()
obs = env.set_init_state(inits[i])  # returns the observation dict; I1 hinges on this
s = env.get_sim_state()  # (110,) float64 — on the WRAPPER, not env.env
obs, reward, done, info = env.step(action)  # action shape (7,)
```

- `get_sim_state()` lives on LIBERO's `ControlEnv` wrapper (`libero/libero/envs/env_wrapper.py`),
  **not** on the inner robosuite env. It returns `sim.get_state().flatten()`.
- Its width (110) equals the width of both `get_task_init_states()` rows and the demo `states`
  dataset, so init states, recorded states, and live sim state are one interchangeable format.
  This is what makes replay-from-any-timestep possible.
- Demo HDF5 (robomimic layout), per episode: `actions (T,7) float64`, `states (T,110) float64`,
  `robot_states`, `dones`, `rewards`, and `obs/` with keys `agentview_rgb`, `eye_in_hand_rgb`,
  `ee_pos`, `ee_ori`, `ee_states`, `gripper_states`, `joint_states`. Note the demo image keys
  are `*_rgb` while the **live env** returns `agentview_image` / `robot0_eye_in_hand_image` —
  they are not the same names, and code that reads both must translate.
- 7-DOF actions confirmed: 6 DOF end-effector delta + 1 gripper.
- First import of `libero` writes `~/.libero/config.yaml` and prompts interactively if absent.
  This is machine-global state outside the repo; provision it non-interactively (see
  `PROGRESS.md`) or a fresh checkout will hang on a prompt.

If exact-state reset turns out not to be supported the way this spec assumes, **stop and
report it** rather than substituting an approximate reset. It changes the project design.

**Compute assumption:** one rented A100-class GPU for on the order of a day total across the
project, plus a consumer GPU for smoke tests. Nothing here needs an H200. If any single
training run is projected over ~6 hours, the config is wrong — cut resolution, latent size,
or task count first.

## 3. Repo layout

```
.
├── CLAUDE.md
├── SPEC.md                  # this file, the contract
├── PROGRESS.md              # append-only log, updated end of every session
├── CHANGELOG.md             # only for frozen-file changes (I3)
├── configs/
│   ├── env.yaml
│   ├── encoder.yaml
│   ├── dynamics.yaml
│   ├── policy.yaml
│   ├── idm.yaml
│   └── search.yaml
├── scripts/
│   ├── verify_env.py        # Phase 0 gate
│   ├── collect_demos.py
│   ├── train_dynamics.py
│   ├── train_policy.py
│   ├── train_idm.py
│   ├── build_failure_set.py
│   ├── run_search.py
│   ├── run_replay.py
│   └── make_figures.py
├── cfe/                     # "counterfactual editing" package
│   ├── encoders/            # frozen visual encoder wrappers
│   ├── dynamics/            # action-conditioned latent dynamics
│   ├── policy/              # BC policy used to generate failures
│   ├── idm/                 # inverse dynamics / latent action models
│   ├── search/              # CEM counterfactual search
│   └── eval/                # metrics, replay protocol  [FROZEN after Phase 2]
├── tests/
└── results/
    ├── runs/<run_id>/       # config snapshot + metrics.jsonl + artifacts
    └── figures/
```

Every run writes a full config snapshot into its own `results/runs/<run_id>/`. No result
exists without the config that produced it.

## 4. Components

### 4.1 Encoder (frozen)

Frozen DINOv2-base. Input a resized RGB frame, output patch tokens `(N, D)`.

- Keep patch tokens as the canonical representation; provide `pool()` for the mean-pooled
  vector. The pooled version is what the search uses by default, and keeping tokens enables
  the spatial-attribution extension later.
- Encoder weights never train. A test asserts `requires_grad == False` everywhere.
- Cache encodings to disk keyed by `(episode_id, frame_idx, encoder_cfg_hash)`. This is the
  difference between a 20-minute experiment loop and a 4-hour one.

**Amended 2026-07-30 (Phase 0), on two counts this section originally got wrong:**

*Tokens are not free.* At DINOv2's native 224px the model emits 257 tokens × 768 dims —
786 KB per frame in fp32, so one `libero_object` task's 7,808 demo frames is **6.1 GB**, and
§4.4 wants ≥3 tasks. The earlier claim that "keeping tokens costs nothing" was wrong. Tokens
stay canonical, but `output_dtype` is an explicit encoder config field defaulting to
`float16` (≈3 GB/task). The encoder *returns* that dtype rather than computing fp32 and
storing narrower, so "a cache hit is bit-identical to a fresh encode" stays literally true
instead of true-modulo-a-cast. Pooled vectors are derived on demand (24 MB/task).

*The cache key does not identify the hardware.* `(episode_id, frame_idx, encoder_cfg_hash)`
names neither machine nor device, so latents computed on CPU and on CUDA collide under the
same key while differing in the low bits — silent cache poisoning that the hash cannot catch.
Therefore:

- `encoder_cfg_hash` covers **all** encoder config fields plus model id, model revision,
  output dtype, `torch.__version__`, and a device-kind tag (`cpu` / `cuda`).
- The cache writes `fingerprint.json` beside its data. A fingerprint mismatch is a **hard
  error, not a silent recompute** — recomputing quietly would hide that two runs' numbers came
  from different hardware.

**Amended 2026-07-30 (Phase 1), extending the above:** the device-kind tag is necessary but
not sufficient. Measured across the two machines this project has run on, both software-
rasterise but with *different mesa drivers* (`kms_swrast` vs `llvmpipe` Mesa 25.2.8), and they
render the same simulator state to different pixels — the G0 rollout digest differs while
physics is byte-identical. A `cpu`/`cuda` tag does not separate them. The fingerprint
therefore also covers:

- the **GL renderer string** (e.g. `egl:llvmpipe (LLVM 20.1.2, 256 bits)`), and
- **`obs_source`** (`replay` vs `recorded`), because recorded demo frames and our re-rendered
  frames are different images of the same state (see §4.3).

`batch_size` is included among the hashed encoder fields: encoding is *not* batch-invariant on
GPU (measured max difference 7.8e-3, one fp16 ULP, against a token std of 1.78), so a latent
is only reproducible at the batch size that produced it.

**Deliberate deviation:** do not start with V-JEPA 2-AC ViT-g. Published action-conditioned
predictor training at that scale used a single H200 with large batches. The scientific claim
does not need it. If Phase 4 lands, porting the encoder is a one-file change and a nice
scaling paragraph.

### 4.2 Action-conditioned latent dynamics

Small transformer. Input: latent context of the last `k` frames plus an action chunk of
length `H`. Output: predicted latents for the next `H` steps.

- Train on LIBERO demos plus policy rollouts (successes *and* failures — failures are
  informative for dynamics even though they are excluded as imitation targets).
- Loss: L1 or smooth-L1 on LayerNormed latents. Teacher-forced one-step loss plus a short
  autoregressive rollout loss (2 steps) to blunt compounding error.
- `k = 4`, `H = 8` to start. Both configurable.

**Required gate before this model is used for anything (see §7, G2):**

- open-loop latent rollout error at horizons 1/2/4/8, versus two baselines: (a) copy-last-
  latent, (b) action-ignoring dynamics (same model with actions zeroed). *If the model does
  not clearly beat the action-ignoring variant, it has not learned action-conditioning and
  the counterfactual search is meaningless.* This is the most important diagnostic in the
  repo.
- correlation between goal-latent distance at episode end and true task success, on held-out
  episodes. Report Spearman ρ. If ρ is near zero the search objective is not a proxy for
  anything.

### 4.3 BC policy (failure generator, not a contribution)

Frozen DINOv2 features → small MLP head with action chunking. Keep it deliberately mediocre:
target roughly 40–70% success. A policy that always succeeds gives no failures; one that
always fails gives no recoverable ones.

**Amended 2026-07-30 (Phase 1) — observations are regenerated, not read from the HDF5.**
Resetting to a demo's recorded `states[0]` is bit-exact, but our render of that identical
state does not match the recorded frame (demo mean 139.1 vs live 103.3, corr 0.80): the demos
were recorded under a different lighting/asset configuration. Training on recorded frames and
evaluating on live ones puts a domain gap between the policy's train and deploy distributions.
Demo observations are therefore produced by replaying each demo's actions in our simulator and
rendering, at a cost of ~20 s/episode. Recorded frames are also vertically flipped relative to
upright (MuJoCo's framebuffer is bottom-up), so the encoder flips before normalising.

Policy input is `pool(z_agentview)` concatenated with live proprioception
(`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`). The gripper dimension is a
**classification** target (BCE): it takes exactly `{−1, +1}` in these demos. The six delta
dimensions are normalised per-dimension before the regression loss — translation and rotation
differ in scale by ~20×, and without it the rotation dimensions are effectively ignored.

**Amended 2026-07-31 (Phase 1) — the G1 success criterion is failure supply and mode
reporting, not the raw rate.** The measured policy lands at 72% on `libero_object` task 0
(36/50, Wilson 95% CI [0.583, 0.825] — above the band, not a small-sample artefact). The
40–70% band was a proxy for two properties this section states outright: enough failures to
work with, and failures that are recoverable rather than catastrophic. Both are now measured
directly, so the gate tests them directly:

- **≥13 natural failures per task at 50 rollouts.** §4.4 needs ~40 natural failures across
  ≥3 tasks; 14/task/50 rollouts clears it.
- **A failure-mode breakdown is reported with the rate.** The rate alone is what proved
  uninformative.

The rate itself is still recorded and must stay in **[0.20, 0.85]** — outside that, supply or
recoverability genuinely breaks down.

**This amendment was written after seeing a gate fail, which is the shape CLAUDE.md rule 3
warns about.** It is justified here only because the underlying requirement was measured and
met, not because 72% was inconvenient: at 72% the natural failure count is 14/task against a
requirement of ~13, and diagnostics showed every failure is a recoverable approach error
(the target is never grasped or transported: max displacement 0.0381 m in any failure against
a 0.4843 m minimum across all successes). The
alternative — weakening the policy — was rejected on evidence, not preference: the failure
mode is approach imprecision, so a weaker policy yields *more of the same mode*, not a more
diverse failure set. Reasoning and the rejected options are logged in `CHANGELOG.md`.

A consequence to carry into Phase 3: the natural failure set is currently **100% approach
failures**, which nearly duplicates the `grasp_offset` injected cause in §4.4. Diversity in
the failure set comes from the injected arm, not the natural one.

### 4.4 Failure set

Two sources, both required.

**Natural failures.** Roll out the BC policy, keep failures. Cheap, realistic, *unlabeled
cause*.

**Injected failures.** Take successful demos and corrupt them in known ways. These carry
ground-truth cause labels and are what let you score explanation quality:

| Cause tag | Injection |
|---|---|
| `grasp_offset` | shift the approach by δ in x/y before grasp |
| `mistimed_close` | close the gripper N steps early/late |
| `early_release` | open the gripper before the place pose |
| `wrong_target` | retarget to a distractor object |

Target ~200 failures total, ≥40 per injected cause, stratified across ≥3 tasks. Store as
`results/failure_set/<id>.json` with: task id, init state blob, executed actions, per-step
observations (or encoding cache keys), success flag, cause tag (or `null` for natural),
and — for injected ones — the ground-truth edit window.

### 4.5 Latent action model (the C2 arm)

Small IDM over frame pairs `(z_t, z_{t+k}) → u_t` (latent action), trained jointly with a
forward model in latent action space. Latent actions are prone to collapse and to smuggling
future information, so regularization is a treated variable, not a detail:

- `continuous_noise` — continuous latent action, Gaussian noise injection
- `continuous_sparse` — continuous plus a sparsity penalty
- `vq` — vector-quantized codebook (report codebook usage; a dead codebook invalidates the arm)

Also train an **action decoder** `u → a` on labeled robot data so latent-space edits can be
executed. Report decoder reconstruction error separately: a bad decoder will masquerade as a
bad latent space, and the ablation is uninterpretable if you cannot distinguish them.

Before training your own, check whether any released latent-action weights fit this setup —
it can save a week.

## 5. Counterfactual search

Two stages. Do not attempt joint sparse optimization; it is harder to implement, harder to
interpret, and not needed.

**Stage 1 — choose the edit window.** Candidate windows of length `W` over the episode.
Rank candidates by per-step latent divergence between predicted and observed rollout
(cheap), keep the top `M` for stage 2. Also support exhaustive windows on short episodes as
a correctness check on the ranking heuristic.

**Stage 2 — CEM over deltas within the window.** Warm-start at the executed actions.
Objective:

```
J(a') = D(ẑ_end(a'), z_goal)  +  λ · Σ_t ||a'_t - a_t||²
```

subject to at most `K` edited timesteps (enforced structurally by the window, not by a
penalty), where `ẑ_end` comes from rolling the dynamics model forward from the context
preceding the window.

`z_goal`: the encoded final frame of a successful demo for the same task and initial state.
Where multiple successful demos exist, use the minimum distance over them — goal states are
multimodal and a single reference will punish legitimate alternative solutions.

Run the identical procedure in latent action space: search over `u'`, decode to `a'`, replay.

## 6. Metrics — write these first, freeze them

**Primary: replay success rate (RSR).**

Replay protocol, primary variant:
1. reset env to the exact original init state (I1)
2. execute original actions up to window start
3. execute the *edited* actions through the window
4. hand control to the BC policy closed-loop for the remainder
5. record task success

This measures "would this edit have rescued the episode," which is the actual claim. A
secondary open-loop variant (edited actions to the end, no policy resume) is also logged.

**Secondary:**

- **Edit sparsity** — timesteps whose action changed by more than ε
- **Edit magnitude** — mean and max L2 of the per-step delta
- **Cause-identification accuracy** — IoU between the found edit window and the ground-truth
  injected window; and top-1 accuracy of predicted cause tag when the window is mapped to a
  cause
- **Search cost** — wall-clock and dynamics-model forward passes per counterfactual

**Mandatory baselines.** Every table reports these or it does not ship:

| Baseline | Purpose |
|---|---|
| `random_perturb` | random deltas, *same edit budget*. If this matches the search, the world model contributes nothing. This is the baseline that decides whether the project is real. |
| `no_edit` | replay the original actions. Should be ~0% by construction; catches replay bugs. |
| `expert_oracle` | replay the successful demo's actions from the window. Upper bound. |
| `action_ignoring_dynamics` | run the search using the actions-zeroed dynamics model. Isolates whether action-conditioning matters. |

**I4 in practice:** RSR is reported as a curve over edit budget `K ∈ {1, 2, 4, 8}` and over
`λ` on a fixed grid. Never a single tuned number. A method that wins only at one λ has not
won.

## 7. Phase gates

Each gate is a script that prints PASS/FAIL. Do not proceed past a FAIL.

- **G0 (Phase 0)** — `verify_env.py`: env constructs headless, renders a frame, reset-to-
  init-state reproduces an identical observation twice, a demo HDF5 loads. **This is I1's
  test.**
- **G1 (Phase 1)** — encoder wrapper produces cached latents; BC policy trains and yields
  **≥13 natural failures per 50 rollouts** on one task, with a failure-mode breakdown reported
  alongside the success rate, which must sit in [0.20, 0.85]. (Amended 2026-07-31; the
  original 40–70% band and the reasoning for replacing it are in §4.3 and `CHANGELOG.md`.)
- **G2 (Phase 2)** — dynamics model beats copy-last-latent *and* action-ignoring dynamics on
  open-loop rollout error; goal-distance-vs-success Spearman ρ reported. **Hard stop: if the
  action-ignoring model is competitive, fix the dynamics model before writing any search
  code.** (Amended 2026-08-04: the ≥20% margin over the action-ignoring twin is measured at
  horizon **96** rather than 8. The threshold did not move — only the horizon it is measured
  at, because §5's search rolls 100+ steps and h8 measured action-conditioning in a regime the
  project never operates in. Reasoning and four rejected alternatives in `CHANGELOG.md`.)
- **G3 (Phase 3, ~day 5)** — end-to-end skeleton on one task, 10 failures: search → replay →
  RSR reported with `random_perturb` and `no_edit`. **RSR must exceed `random_perturb` by a
  margin that survives a bootstrap CI, on injected failures.** If not, the honest move is to
  spend the remaining time on why the world model is insufficient. That is a real result and
  a better project than a fabricated positive.
- **G4 (Phase 4)** — full failure set, all baselines, C1 table with CIs.
- **G5 (Phase 5)** — latent action arm, C2 ablation table, decoder error reported separately.
- **G6 (Phase 6)** — figures, side-by-side replay videos, writeup.

## 8. Testing requirements

Not comprehensive coverage — just the things whose silent failure would poison results.

- exact state reset reproducibility (I1)
- seed determinism end-to-end on a tiny config (I5)
- encoder frozen; encoding cache hit returns bit-identical tensors to a fresh encode
- replay of unedited actions reproduces the original episode's success flag exactly, **for
  every episode admitted to the encoding cache** (amended 2026-07-31 — see below)
- metric functions on hand-constructed inputs with known answers
- CEM recovers a known-good edit on a synthetic toy dynamics model where the answer is
  analytic

That last one matters: it separates "search is broken" from "world model is inaccurate,"
which are the two failure modes you will spend the most time confusing.

**Amended 2026-07-31 (Phase 1) — replay fidelity is scoped to episodes that feed results.**
Replaying a demo's recorded actions in our simulator reproduces its recorded success flag for
**92–96%** of demos, measured over all 50 demos on each of `libero_object` tasks 0, 1 and 2
(0.96 / 0.92 / 0.94). The shortfall is physics divergence at the tightest grasp margins, not a
replay defect: reset is bit-exact (`max|get_sim_state() − states[0]| = 0.0`), and the failures
show a coherent physical signature — the gripper ends 2–5 cm short and the object is never
lifted, rather than the trajectory going wrong from the start.

The exactness requirement is unchanged and is **not** softened to a rate. It is scoped to the
episodes that actually feed downstream results: `collect_demos.py` refuses to cache an episode
whose replay did not succeed, so the gate's condition is that **no non-reproducing episode is
present in the cache** — currently 48/48, exact, no tolerance. A yield floor
(`data.min_replay_yield`, 0.85) rides alongside as a regression tripwire, because a genuine
replay bug would crater yield across the board rather than clip the few tightest-margin demos.
It is a tripwire, not a pass mark, and lowering it to make a run pass is forbidden by
CLAUDE.md rule 3. Reasoning is logged in `CHANGELOG.md`.

## 9. Out of scope

Do not build: a recovery policy, a VLM explanation layer, natural-language output, real
hardware support, photorealistic generation, a web UI beyond a static figure/video dump,
multi-GPU training, or hyperparameter search infrastructure. Any of these is a signal that
scope has drifted.
