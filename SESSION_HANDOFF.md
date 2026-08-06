# SESSION HANDOFF

Read this first when starting a session. Then `CLAUDE.md` and `SPEC.md` (SPEC.md is the contract
and wins any conflict), then `PROGRESS.md` for the full history.

Last updated: 2026-08-05, end of Phase 3.

## Where the project is

| | |
|---|---|
| Phase complete | **3** — failure set, two-stage search, replay, G3 |
| Gate state | **G0: PASS. G1: PASS. G2: PASS. G3: FAIL (4/6)** |
| Model of record | `p2-dyn-res` (`results/runs/p2-dyn-res/`), gate run `g2-20260804T185522Z` |
| G3 run | `g3-phase3`, reading `p3-replay-RD` (1060 replays) |
| Next phase | **4** — but read "Where G3 failed" below first; the next move is a decision, not a build |
| Last commit | `89744da` on `main` (Phase 3 work is uncommitted) |

## Read this before touching anything

**`cfe/eval/` is frozen and the freeze is now enforced by hooks, not by good intentions.**
`.claude/settings.json` installs two `PreToolUse` hooks:

- edits to `cfe/eval/**` are blocked, via Edit/Write *and* via the shell (`sed -i`, redirects,
  `rm`/`mv`/`cp`/`tee`, `git checkout -- cfe/eval/`);
- `git push` is blocked. Committing is fine; publishing is the user's call.

Reading frozen files is unaffected, and `tests/test_eval_metrics.py` is *not* frozen — only the
module is. If you hit the block and believe there is a genuine bug: **stop and tell the user.**
Do not route around it and do not reimplement the affected logic elsewhere. An approved change
is logged in `CHANGELOG.md` with its reason and rejected alternatives (CLAUDE.md rule 1).

## Where G3 failed, and what it points at

**The search does not beat `random_perturb`.** At K=8 on 10 injected failures it rescued one
episode of ten at one λ and zero elsewhere; `random_perturb` rescued none. The margin does not
survive the bootstrap at any λ. Full table in `PROGRESS.md` 2026-08-05.

**Both stages lose about half the signal.** Read `expert_oracle` minus `no_edit` at the same
window as the *headroom* an ideal edit could buy there. At K=8: **+0.60** at the injection
window, **+0.30** at the window stage 1 actually picked, and the search achieved **0.00-0.10**.
So stage 1 costs about half the headroom by placing the window a median of +52 steps past the
fault, and stage 2 then captures at most a third of what remains.

SPEC.md §8 warns that "search is broken" and "world model is inaccurate" are the two failure
modes you will spend the most time confusing. §8's analytic CEM test passes, so the machinery is
sound on a problem with a known optimum — but that does not show it extracts useful deltas from
*this* model at search scale. **The window-oracle ablation in PROGRESS.md's "Next" separates
them and is the first thing to run** (~30 min; strictly diagnostic, oracle-assisted, never
reportable as an RSR result).

**The action-ignoring twin selects better windows than the trained model.** It proposes the
identity edit in 240/240 cells (analytically forced — its `z_end` does not vary with the
candidate), so its RSR *is* "no_edit at its own window" — and it beats the search: 0.10 vs 0.03
at K=8 in experiment R, and 0.40 vs 0.00 in experiment D. A model that cannot see actions picks
earlier, more recoverable windows. This inverts what §6 expects of that baseline.

**`residual: false` is not the fix.** `p2-dyn` was the flagged fallback if the search turned out
starved of action signal. It scored 0.00 at every K and λ (0/120 cells) against R's 5/120. More
action-sensitivity at h96 did not produce better counterfactuals.

Three decisions are open and are the user's, not the assistant's:

1. **The `magnitude_match_tol` of 0.02 is too tight** for clipping this project had already
   documented — it failed on 1 of 120 cells (max diff 0.1176, mean 0.0065). The bias runs
   *toward* the search, so it cannot excuse criterion 1. Changing it is a gate change:
   `CHANGELOG.md` with reasoning and rejected alternatives.
2. **Whether stage 1's window criterion should change.** Divergence is retrospective — it ranks
   where the model was surprised, and a fault is only visible once its consequences are. A
   predictive criterion is the obvious alternative and is a change to SPEC.md §5.
3. **Whether to run the §4.4 expansion** (built, queued, unrun, ~4 h) or spend that compute on a
   stage-1 diagnostic instead.

The three obligations Phase 2 handed over were all discharged: `random_perturb` is matched on
realised magnitude per cell (verified as a gate criterion), `action_ignoring_dynamics` is
implemented in `cfe/search/arms.py` without mutating the frozen module, and `expert_oracle`
still raises `NotApplicable` on natural failures.

## Phase 2 results worth carrying forward

- **Action-conditioning is real but strongly horizon-dependent.** `rel_improvement` over the
  action-ignoring twin is 0.035 at h8 and 0.224 at h96. The rise is horizon-driven (4.5-6.1x),
  not episode-selection-driven (1.1-1.5x) — measured in `scripts/diagnose_window_selection.py`.
- **The G2 magnitude criterion was amended 2026-08-04** from h8 to h96. The threshold did not
  move; only the horizon it is measured at. Full reasoning and four rejected alternatives are in
  `CHANGELOG.md`. It was decided by the user on measurement, after the assistant argued against.
- **The model has a smooth action gradient** — quadratic edit response, detectable down to
  0.05 σ (`scripts/diagnose_edit_discrimination.py`). Search over small edits is viable in
  principle. But at search-scale edits the effect is small *relative to the model's own error*
  (a 1.0 σ edit moves h8 error by 6.7%), so the search likely needs long horizons to accumulate
  usable signal.
- **`residual: true` is on, and it is the weaker of the two on action-sensitivity.** It clears
  the 0.20 bar by 12% where `residual: false` clears it by 34%. If Phase 3's search turns out to
  be starved of action signal, this flag is the first thing to revisit — `p2-dyn` is already
  trained and on disk.
- **Rollouts, not encoding, are the bottleneck.** Rendering is CPU `llvmpipe` at ~9 steps/s, so
  a 300-step replay costs ~33 s. Budget accordingly; the search will replay a great many.

## What is not in the repo

`results/` is gitignored except `results/figures/` (SPEC.md §3). `setup_env.sh` recreates the
rest.

| What | Where | Size |
|---|---|---|
| `libero_object` demos | `~/data/libero/` | 7.0 GB |
| Encoding cache (154 episodes) | `~/data/cfe_cache/a005df8c057942c3/` | 9.7 GB |
| LIBERO simulation assets | `~/.cache/libero/assets` | 409 MB |
| DINOv2 weights | `~/.cache/huggingface` | 334 MB |
| Injected failure episodes (56) | same cache dir | 3.5 GB |
| Failure set records (70) | `results/failure_set/` | small, but gitignored |
| LIBERO global config | `~/.libero/config.yaml` | tiny, but see trap 3 |
| venv | `~/.venvs/failure-detection-WM` | — |

## Five traps that cost real time

1. **`mujoco` must stay `<= 3.3.7`.** robosuite declares it with *no upper bound*, so a fresh
   resolve grabs the latest. `MjData.qM` became `MjData.M` in 3.4.0 and the OSC controller dies
   on the first env construction. A plain `pip install -U` reintroduces this.
2. **`robosuite` must stay `1.4.x`.** 1.5 removed `SingleArmEnv`, which LIBERO imports, so
   `import libero` breaks outright.
3. **Importing `libero` runs `input()`** if `~/.libero/config.yaml` is absent, blocking
   interactively and dying with `EOFError` under any non-interactive runner.
4. **Demo image keys ≠ live env image keys.** Demos store `agentview_rgb`; the live env returns
   `agentview_image`. A silent-wrong-answer trap, not a crash.
5. **`configs/encoder.yaml` stores `cache_dir` with a literal `~`.** Python's `glob` does not
   expand it; use `os.path.expanduser`. This silently returns zero files.

## Unresolved: the machine hard power-offs

The box hard-rebooted six or more times during Phase 2 — Windows Kernel-Power Event 41, **no
bugcheck**, no WHEA, no TDR, on AC. **It is not thermal and not OOM**, both measured:

- 10 min of sustained GPU load: stable at 73-74°C, no crash. CPU saturation for 9 min: throughput
  flat (+2.3%), no crash. Chassis heat soaks +10°C and plateaus.
- Across the 97-minute training run, WSL `MemAvailable` never fell below 3671 MiB and process RSS
  was flat.

The signature (instant power loss with no dump, at idle *and* under load) points at power
delivery; the battery is 15.5% worn and the GPU transient-spikes to 110 W against a nominal
90 W cap. Unconfirmed. `~/run_with_telemetry.py` wraps any command with crash-survivable,
`fsync`'d logging plus GPU/memory telemetry — use it for long runs. Logs live in `~/runlogs/`
(**not** `/tmp`, which WSL clears on every boot).

Note also: `nvidia-smi`'s thermal-slowdown *counters* read 0 µs on this die and are unreliable.
Sample `clocks_throttle_reasons.active` instead.

## Rules that are easy to violate without noticing

- **Never weaken a metric, threshold, or gate to make a run pass.** A failing gate is
  information. Bring the measurement to the user and let them decide; log any change in
  `CHANGELOG.md`. This is the single most damaging thing that could happen to this project.
- **A number is only real if it came from simulator replay.** Latent goal distance is a search
  objective, never a result.
- Any experiment reporting a success rate also reports `random_perturb` and `no_edit`.
- Every result writes a config snapshot to `results/runs/<run_id>/` beside `metrics.jsonl`.
- **`cfe/eval/` is frozen** — see the top of this file. The hooks will stop you.
- One phase per session; append to `PROGRESS.md` at the end of each.
- **The shared encoding cache now holds three kinds of episode** — demos, BC rollouts, and
  injected failures. Select on `cfe/data/demos.py:episode_kind()`, never on a substring: an
  injected id is named after the demo it corrupts, and an `"_demo_" in cid` test classifies it
  as expert data. `train_policy.py` (demos, this task only), `train_dynamics.py` (demos +
  rollouts) and `probe_representation.py` all filter explicitly now; they did not before Phase 3
  and an unfiltered re-run would have trained on failures.
