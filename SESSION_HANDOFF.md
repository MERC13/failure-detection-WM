# SESSION HANDOFF

Read this first when starting a session. Then `CLAUDE.md` and `SPEC.md` (SPEC.md is the contract
and wins any conflict), then `PROGRESS.md` for the full history.

Last updated: 2026-08-04, end of Phase 2.

## Where the project is

| | |
|---|---|
| Phase complete | **2** — latent dynamics + G2, and the frozen `cfe/eval/` |
| Gate state | **G0: PASS. G1: PASS. G2: PASS (5/5)** |
| Model of record | `p2-dyn-res` (`results/runs/p2-dyn-res/`), gate run `g2-20260804T185522Z` |
| Next phase | **3** — failure set, two-stage CEM search, `run_replay.py`, the G3 gate |
| Last commit | `89744da` on `main` |

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

## What Phase 3 has to build

`build_failure_set.py` for §4.4's injected causes, the two-stage CEM of §5, `run_replay.py` on
the frozen eval code, and the **G3 gate**: RSR must exceed `random_perturb` by a margin that
survives a bootstrap CI, on injected failures.

Three obligations the frozen code cannot enforce for you:

1. **`random_perturb` must be matched on magnitude, not just on budget `K`.** Pass the search's
   *realised* `edit_magnitude(...)["mean_l2_ee"]` as `magnitude_scale`. This is the single
   easiest way to fake a positive result in this project — an under-scaled random baseline makes
   the search look good against nothing. §6 calls it "the baseline that decides whether the
   project is real". A pre-freeze bug that made this 2.45x too large was caught and fixed; the
   caller-side half is still on you.
2. **`action_ignoring_dynamics` is declared in `BASELINES` but raises `NotImplementedError`.**
   It *is* the search run against the trained twin (the `action_ignoring` key of a
   `train_dynamics.py` checkpoint). Phase 3 implements and registers it. §6: every table reports
   all four baselines or it does not ship.
3. **`expert_oracle` raises `NotApplicable` on natural failures** and that is correct — no demo
   shares a natural failure's initial state (verified in Phase 1), so there is no upper bound to
   compute. Do not substitute a nearest-neighbour demo. Report the gap.

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
| Encoding cache (98 episodes) | `~/data/cfe_cache/a005df8c057942c3/` | 7.0 GB |
| LIBERO simulation assets | `~/.cache/libero/assets` | 409 MB |
| DINOv2 weights | `~/.cache/huggingface` | 334 MB |
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
