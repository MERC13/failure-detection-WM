# CLAUDE.md

Research codebase. The contract is `@SPEC.md` — read it before writing code. Where this file
and SPEC.md conflict, SPEC.md wins and you should tell me about the conflict.

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
```

Training and experiment entrypoints are the scripts in `scripts/`, each taking
`--config configs/<name>.yaml --seed N --run-id <id>`.

## The five rules that matter

1. **Never edit `cfe/eval/` after Phase 2 is signed off.** Those files hold the metrics and
   the replay protocol. If you believe there is a bug in them, stop and ask — do not fix it
   inline. A change there rewrites every prior result. If I approve a change, log it in
   `CHANGELOG.md` with the reason.
2. **A number is only real if it came from simulator replay.** Latent-space goal distance is
   a search objective, never a result. Never report, log, or summarize latent distance as
   though it measured success.
3. **Never weaken a metric, threshold, or gate to make a run pass.** If a gate fails, say it
   failed and tell me why you think it failed. A failing gate is information. Silently
   relaxing it destroys the project. This is the single most damaging thing you could do here.
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
- `results/` is gitignored except `results/figures/`.
- Failure episodes carry a `cause` tag that is `null` for naturally occurring failures and a
  string for injected ones. Code that assumes non-null will crash on half the dataset.

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

## Scope discipline

Out of scope, per SPEC.md §9: recovery policies, VLM/language explanation, real hardware,
generative video, web UIs, multi-GPU, hyperparameter search infrastructure. If a task seems
to need one of these, that's a signal the scope has drifted — flag it instead of building it.
