# CLAUDE.md

Research codebase. The contract is `@SPEC.md` — read it before writing code. Where this file
and SPEC.md conflict, SPEC.md wins and you should tell me about the conflict.

Read this file first, then `SPEC.md`, then `PROGRESS.md` for the full history. `CHANGELOG.md`
records every change to a metric, threshold, gate, or the replay protocol. `WORKFLOW.md` is the
prompting playbook and holds the unrun P4-P6 phase prompts.

## What this project is

Search a learned world model for the minimal edit to a failed robot episode's actions that
would have succeeded, then validate by replaying in simulation. Compare raw action space vs
learned latent action space.

This is research code with one job: produce numbers that are true. Correctness of the
measurement matters more than elegance, coverage, or speed.

## Commands

```bash
make env-check       # G0 gate — must pass before anything else
make test            # pytest, fast tests only
make test-slow       # includes sim-dependent tests
make lint            # ruff + format

make g1-check POLICY_RUN=<train_policy run_id> COLLECT_RUN=<collect_demos run_id>
make g2-check DYN_RUN=<train_dynamics run_id> PROBE_RUN=<probe_representation run_id>
make g3-check RUN=<run_replay run_id>
```

The Phase 3 pipeline, in order. The split between `run_search` and `run_replay` is deliberate:
search is GPU-cheap and re-runnable in minutes, replay is CPU simulation and takes hours.

```bash
scripts/build_failure_set.py    # inject failures, keep only those that ACTUALLY fail
scripts/run_search.py           # GPU: emits edits.jsonl for every arm x cell
scripts/run_replay.py           # CPU sim: the ONLY source of real numbers. Resumable.
scripts/verify_g3.py            # the gate; criteria read from configs/search.yaml
```

Diagnostics (none of these produce a reportable result):

```bash
scripts/diagnose_replay_noise.py       # determinism, re-seed spread, single-step sensitivity
scripts/diagnose_window_oracle.py      # stage 2 with the window forced to the true fault
scripts/diagnose_objective_quality.py  # score an objective against replayed edits, NO simulator
```

Training and experiment entrypoints are the scripts in `scripts/`, each taking
`--config configs/<name>.yaml --seed N --run-id <id>`.

The venv lives **outside** the repo at `~/.venvs/failure-detection-WM` (the repo is on a slow
`/mnt/c` 9p mount). `make` handles this; running a script directly needs
`PYTHONPATH=$PWD ~/.venvs/failure-detection-WM/bin/python …`.

Gates read numbers from completed runs rather than recomputing them, so a gate can fail simply
because the run predates a metric. Re-run the script; do not make the gate re-derive the value.

## The five rules that matter

1. **Never edit `cfe/eval/` after Phase 2 is signed off.** Those files hold the metrics and
   the replay protocol. If you believe there is a bug in them, stop and ask — do not fix it
   inline. A change there rewrites every prior result. If I approve a change, log it in
   `CHANGELOG.md` with the reason. `CHANGELOG.md` is the log for **any** change to a metric,
   threshold, gate, or the replay protocol, wherever it lives — not only `cfe/eval/`.
2. **A number is only real if it came from simulator replay.** Latent-space goal distance is
   a search objective, never a result. Never report, log, or summarize latent distance as
   though it measured success.
3. **Never weaken a metric, threshold, or gate to make a run pass.** If a gate fails, say it
   failed and tell me why you think it failed. A failing gate is information. Silently
   relaxing it destroys the project. This is the single most damaging thing you could do here.
   This extends to *evaluation protocol* knobs, not just thresholds: `eval.max_steps` moves the
   BC success rate as surely as retraining does, because every failure is a timeout. If you
   believe a criterion is genuinely miscalibrated, bring me the measurement that shows it and
   let me decide — do not amend it yourself. Every such change goes in `CHANGELOG.md` with its
   reasoning and its rejected alternatives.
4. **Every result writes a config snapshot** to `results/runs/<run_id>/` alongside
   `metrics.jsonl`. No orphan numbers in the terminal.
5. **Every entrypoint takes an explicit seed and is deterministic.** Same seed + same config
   must give identical output.

## Verify against the installed package, don't trust the spec

SPEC.md describes the expected LIBERO / robosuite API from memory, and these names have moved
between releases. Before using any of them, check the installed source and correct the spec
if it's wrong — then note the correction in `PROGRESS.md`.

If exact-state reset (`env.set_init_state` or equivalent) does not work as SPEC.md assumes,
**stop and tell me**. Do not substitute an approximate reset. It invalidates the whole design
and I need to know immediately, not in week three.

## Baselines are not optional

Any experiment that reports a success rate also reports `random_perturb` (same edit budget)
and `no_edit`. The first tells us whether the world model does anything; the second catches
replay bugs. If a table is missing them it is not finished, even if I forgot to ask.

## Conventions this repo uses that you wouldn't guess

- Latents are patch tokens `(N, D)`, not pooled vectors. `pool()` exists; call it explicitly
  at the point of use so the pooling choice is always visible in the code.
- All encodings are disk-cached keyed by `(episode_id, frame_idx, encoder_cfg_hash)`.
  Add to the hash inputs whenever you add an encoder config field, or stale cache entries
  will silently poison runs.
- Actions are 7-DOF: 6 end-effector delta + 1 gripper. The gripper dimension is not
  continuous in the same sense as the others — do not apply symmetric L2 penalties to it
  without flagging that choice.
- `results/` is gitignored except `results/figures/`. Ignore patterns for `data/` and `cache/`
  are **root-anchored** (`/data/`) on purpose: unanchored, they also match `cfe/data/` and
  silently drop source code from commits.
- Only the `encoder:` block of `configs/encoder.yaml` is hashed into the cache key. `data:` and
  `runtime:` are not, so adding fields there will not invalidate a 2.8 GB cache.
- Failure episodes carry a `cause` tag that is `null` for naturally occurring failures and a
  string for injected ones. Code that assumes non-null will crash on half the dataset.
- **Observation keys lie in two ways that have already cost a full re-run each.**
  `<object>_to_robot0_eef_pos` is object *minus* end-effector, so it moves when the arm moves,
  not the object — filter it out when you want absolute positions. And objects settle under
  gravity at reset: ~0.10 m vertically, ~0.001 m horizontally, before the robot does anything.
  Any "did the robot touch it" test must be horizontal and must clear that floor.
- Rendering runs on the CPU through mesa's `llvmpipe` (~9 sim steps/s) even though CUDA is
  available for the encoder. Rollouts, not encoding, are the bottleneck: budget ~30 min for
  50 rollouts of 300 steps.
- **A replay costs ~16 s.** That is the unit of every cost estimate in Phase 3+: the G3 grid was
  1060 replays ≈ 4.6 h. Running a GPU job alongside sim roughly doubles it (measured 31 s), and
  two sim jobs at once is worse. Sequence sim work; overlap GPU with sim only when you must.
- **The encoding cache is flat and now holds THREE kinds of episode** — `demo`, `rollout`,
  `inject`. Select with `cfe/data/demos.py:episode_kind()` and `episode_task_id()`, **never with
  a substring**. An injected id is named after the demo it corrupts, so `"_demo_" in cid` was
  true for injected episodes and would have loaded corrupted data as expert data. Both filters
  are now explicit in `train_policy.py` (demos, one task), `train_dynamics.py` (demos +
  rollouts) and `probe_representation.py` (demos as the goal set).
- **`train_dynamics.py --tasks` pins the training corpus.** Without it a retrain silently widens
  its training set as new tasks land in the shared cache, and any comparison against an earlier
  checkpoint is confounded. Every model in a comparison must use the same value.
- **`rollout_loss` chains `ar_chunks` forward passes into one autograd graph**, so activation
  memory scales with `ar_chunks x batch_size`. `ar_chunks=8` OOMs at the configured batch of 128
  on an 8 GB card. Use `--micro-batch` (gradient accumulation) rather than shrinking the batch —
  shrinking it confounds the horizon against the batch size.
- **SPEC §6's primary replay variant replaces everything after `window_end` with the BC policy.**
  So an edit window that starts *before* the fault means the fault is **never executed**: the
  replay becomes "original prefix, then policy" and succeeds at the policy's own rate. This is
  not a bug, it is what §6 specifies — but it makes window position dominate the number, and it
  invalidated the first noise-floor run.
- **`no_edit` in the primary variant is NOT the replay-bug control.** Only the open-loop variant
  replays the original action sequence, and only it must read 0%. Primary-variant `no_edit`
  measures "would the policy have rescued this anyway" — informative, and at the fault window it
  is ~0.48, which is a very hard baseline to beat.
- **Screen an objective before you simulate it.** `diagnose_objective_quality.py` scores any
  objective against already-replayed edits with no simulator, using within-episode pairwise
  ranking accuracy. A variant that does not move that number does not deserve hours of sim.
  Pool nothing across episodes: pooled correlations are confounded by episode difficulty and
  have flipped sign between runs on this data.
- **Ensembled goal distance is population-relative.** `cfe/search/ensemble.py` standardises each
  member within the candidate population before averaging, so the value is a *ranking* quantity
  and is meaningless across populations. Never compare or report it as a distance.
- **Oracle-assisted runs are tagged `experiment: ORACLE`** and are deliberately absent from
  `configs/search.yaml:experiments`, so `verify_g3.py:gated_rows` structurally cannot pull them
  into a gate. They use the ground-truth fault window, which the method does not have at
  deployment, and can never be reported as an RSR result.

## Working style

- Use plan mode for anything spanning more than two files. Show me the plan first.
- One phase per session. At the end of a session, append to `PROGRESS.md`: what landed, what
  gate state changed, what surprised you, what's next.
- Prefer boring, readable implementations. No abstraction layers added for hypothetical
  future needs. No Hydra, no plugin registries, no dependency injection.
- When something doesn't work, report the actual failure. Do not narrow the test, catch the
  exception, or add a fallback path to make output appear. I would rather see a stack trace
  than a plausible number.
- If you find yourself about to write "this should work" without running it, run it.
- **Distrust a new metric until something independent agrees with it.** The Phase 1
  failure-mode breakdown took four runs to become correct, and its first two versions produced
  confident, plausible, wrong numbers that were caught only because they contradicted a
  separate diagnostic. When you add a measurement, measure its noise floor first and check it
  against a signal you already trust before reporting anything from it.

## What is not in the repo

`results/` is gitignored except `results/figures/` (SPEC.md §3). `setup_env.sh` recreates the
rest. Run directories under `results/runs/` are the provenance for every number in `PROGRESS.md`
and `CHANGELOG.md` — **do not delete them to save disk.** CHANGELOG.md 2026-08-04 says of two
failed G2 runs "both stand exactly as recorded"; deleting them destroys exactly that.

| What | Where | Size |
|---|---|---|
| `libero_object` demos | `~/data/libero/` | 7.0 GB |
| Encoding cache | `~/data/cfe_cache/a005df8c057942c3/` | ~10 GB, 154+ episodes |
| Failure set records | `results/failure_set/` | small, gitignored |
| LIBERO simulation assets | `~/.cache/libero/assets` | 409 MB |
| DINOv2 weights | `~/.cache/huggingface` | 334 MB |
| LIBERO global config | `~/.libero/config.yaml` | tiny, but see trap 3 |
| venv | `~/.venvs/failure-detection-WM` | — |

Models of record: BC policy `p1-bc4`; dynamics `p2-dyn-res` (G2 run `g2-20260804T185522Z`);
G3 run `g3-phase3` reading `p3-replay-RD`.

## Five install traps that cost real time

1. **`mujoco` must stay `<= 3.3.7`.** robosuite declares it with *no upper bound*, so a fresh
   resolve grabs the latest. `MjData.qM` became `MjData.M` in 3.4.0 and the OSC controller dies
   on the first env construction. A plain `pip install -U` reintroduces this.
2. **`robosuite` must stay `1.4.x`.** 1.5 removed `SingleArmEnv`, which LIBERO imports, so
   `import libero` breaks outright.
3. **Importing `libero` runs `input()`** if `~/.libero/config.yaml` is absent — it blocks
   interactively and dies with `EOFError` under any non-interactive runner.
4. **Demo image keys != live env image keys.** Demos store `agentview_rgb`; the live env returns
   `agentview_image`. A silent-wrong-answer trap, not a crash.
5. **`configs/encoder.yaml` stores `cache_dir` with a literal `~`.** Python's `glob` does not
   expand it; use `os.path.expanduser`. This silently returns zero files.

## Unresolved: the machine hard power-offs

The box has hard-rebooted seven or more times — Windows Kernel-Power Event 41, **no bugcheck**, no
WHEA, no TDR, on AC. **Not thermal and not OOM**, both measured: 10 min of sustained GPU load
held 73-74 C; across a 97-minute training run WSL `MemAvailable` never fell below 3671 MiB. The
signature points at power delivery; the battery is 15.5% worn and the GPU transient-spikes to
110 W against a 90 W cap. Unconfirmed.

Consequences for how you run things: long jobs must be **resumable** (`run_replay.py` fsyncs
after every replay and skips completed cells) and should be launched with `setsid` — a session
teardown killed one training run outright. Logs go to `~/runlogs/`, **not** `/tmp`, which WSL
clears on every boot — and that applies to **driver scripts too**, not just logs; a queue script
written to `/tmp` was lost to the 2026-08-07 reboot. Keep them in `$HOME`.
`~/run_with_telemetry.py` wraps a command with crash-survivable logging.
`nvidia-smi`'s thermal-slowdown *counters* read 0 us on this die and are unreliable; sample
`clocks_throttle_reasons.active` instead.

**`train_dynamics.py` does not checkpoint** — artifacts are written only at the end, so a crash
costs the whole run (~1 h at `ar_chunks=2`, ~3.9 h at 8). A resume path was considered and
rejected: staying bit-reproducible across a resume means persisting optimizer state plus both
RNG streams, and rule 5 makes a subtly wrong resume worse than a re-run. **Do not overlap a
dynamics training run with simulation.** Serialising is nearly free anyway — a GPU job alongside
sim roughly doubles each replay — and it halves what a single crash destroys.
`~/p35_queue.sh` is the serialised driver (ar8, then the §4.4 expansion for tasks 1 and 2).

## Where the project actually stands (read before proposing a fix)

**G3 FAILed and the diagnosis is done.** Do not re-derive it. Full numbers in `PROGRESS.md`
2026-08-05/06; the short version:

- The search does not beat magnitude-matched `random_perturb`, **even when the edit window is
  forced to the true fault** (n=56, K=8: +0.125 / −0.018 / −0.018 across λ, no CI excluding
  zero). So this is not merely a window-selection failure.
- The objective is **noisy, not empty**: it ranks edits correctly 62–67% of the time
  (chance 50%). Real signal, too weak to win.
- **Optimising harder makes it worse.** One CEM iteration beat convergence (0.589 vs 0.536) —
  the search finds the model's errors, not good edits. Any proposal that leans harder on
  `argmax` of a learned objective has to answer this.
- **The action-ignoring twin picks better windows than the trained model.** Divergence peaks
  where a fault's *consequences* appear (~52 steps late), so the trained model is
  systematically wrong; the blind model is merely arbitrary, and arbitrary wins.
- Seed-ensembling helps a little (0.621 → 0.668, McNemar p=0.023). Mixing model *families*
  (residual on/off) hurts badly (0.551). `residual: false` scored 0/120 cells.

## Ideas considered and analysed — do not re-propose without reading these

**Q-learning / value-based edit selection.** Train `Q(s, a)`, flag timesteps with the lowest
Q as the mistake, replace with `argmax_a Q(s, a)`. Analysed 2026-08-06.

*Why it is attractive:* it is **predictive rather than retrospective**, which is exactly stage
1's disease — Q asks "was this action bad for the outcome", answerable at the moment of the
fault, whereas divergence only fires once consequences are visible. It also **eliminates the
rollout**, and compounding rollout error is what buries the edit signal (an edit moves the
prediction ~7% while 100 steps of error move it far more). And it optimises the target rather
than a proxy.

*Why it is not straightforward here:*
1. **Data.** ~30k transitions across ~154 episodes, with **sparse terminal reward** — one
   binary bit credit-assigned over ~200 steps, in a 7-DOF continuous action space. The ~1,400
   replayed edits cannot help: `run_replay.py` records actions and outcomes but **not frames**,
   so there are no states to learn from without re-replaying with capture.
2. **`argmax_a Q` is the exploitation problem renamed.** Offline RL's canonical failure is
   Q-overestimation on out-of-distribution actions, and we have *already measured* that
   dynamic in this codebase. Q would likely be worse than the dynamics model, which is at
   least anchored to predicting pixels.
3. **Scope.** `argmax_a Q(s,a)` is a greedy policy from a value function, and SPEC §9 excludes
   recovery policies.

*The split that matters:* **Q as a critic** (score the executed actions to locate the fault)
is in scope, cheap, and is where the value is concentrated. **Q as an actor** (generate the
replacement) is out of scope and hits the OOD wall.

*The cheap version, if pursued:* drop TD-learning entirely. Terminal-only reward means
Monte-Carlo estimation, which is just supervised learning — fit `f(z_t, a_t) -> P(success)`
with the episode outcome as the label. No bootstrapping, no Bellman instability, a small MLP
head on cached latents.

*The free experiment to run first — the ceiling test.* Every injected failure has its
uncorrupted source demo **at the identical initial state**, and the true fault window is known,
for 56 pairs. Train a classifier to separate clean from corrupted per timestep and measure
window IoU against those known windows. It is oracle-assisted and not deployable, but it
answers the question underneath all of this: **do these latents contain enough information to
localise a fault at all?** If no, no window-selection method can work and the negative result
is far stronger. If yes, divergence is simply the wrong readout. Pure GPU on cached latents,
no simulation.

## Scope discipline

Out of scope, per SPEC.md §9: recovery policies, VLM/language explanation, real hardware,
generative video, web UIs, multi-GPU, hyperparameter search infrastructure. If a task seems
to need one of these, that's a signal the scope has drifted — flag it instead of building it.
