# SESSION HANDOFF

Read this first when starting a session on a new machine. Then read `CLAUDE.md` and `SPEC.md`
(SPEC.md is the contract and wins any conflict), then `PROGRESS.md` for the full history.

Last updated: 2026-07-30, end of Phase 0.

## Where the project is

| | |
|---|---|
| Phase complete | **0** — pinned environment + G0 gate |
| Gate state | **G0: PASS (6/6)**. I1 verified bit-exact, I5 verified across processes |
| Next phase | **1** — frozen DINOv2 encoder + disk cache, demo loading, BC policy, G1 gate |
| Last commit | `8b8a80c` Phase 0: pinned environment and G0 gate |

**G0 passed on the original WSL2 CPU box only.** It has not been run anywhere else. See
"First thing to do" below — this is not a formality.

## First thing to do on a new machine

```bash
bash scripts/setup_env.sh     # idempotent; ends by running the G0 gate
make test-slow
make lint
```

`setup_env.sh` handles the toolchain check, venv, hardware-matched torch, pinned deps,
LIBERO's global config, and the demo download. It stops with the exact `apt` command if the
build toolchain is missing.

**If G0 fails on the new machine, stop and report it — do not start Phase 1.** In particular,
if `i1_exact_reset` or `i1_reset_under_stepping` fails, that is the project-design-invalidating
case SPEC.md §2 says to escalate immediately. Do not loosen the comparison to a tolerance; the
comparison being bit-exact is the whole point, and `tests/test_determinism.py` has a test
specifically guarding against that loosening.

Expect one legitimate difference: the original box rendered through **software rasterization**
(`kms_swrast`, because the user lacked `/dev/dri` access). A GPU box will use hardware EGL. That
is a different rendering path, which is exactly why G0 is re-run rather than assumed.

## What is not in the repo

`results/` is gitignored except `results/figures/` (SPEC.md §3), and the following live outside
the repo entirely. `setup_env.sh` recreates all of it:

| What | Where | Size |
|---|---|---|
| `libero_object` demos | `~/data/libero/` | 7.0 GB |
| LIBERO simulation assets | `~/.cache/libero/assets` | 409 MB |
| DINOv2 weights (Phase 1) | `~/.cache/huggingface` | 334 MB |
| LIBERO global config | `~/.libero/config.yaml` | tiny, but see trap 3 |
| venv | `~/.venvs/failure-detection-WM` | — |

## Five traps that cost real time on the first machine

1. **`mujoco` must stay `<= 3.3.7`.** robosuite declares `mujoco` with *no upper bound*, so a
   fresh resolve grabs the latest. `MjData.qM` became `MjData.M` in 3.4.0, and the OSC
   controller dies on the first env construction. A plain `pip install -U` reintroduces this.
2. **`robosuite` must stay `1.4.x`.** 1.5 removed `SingleArmEnv`, which LIBERO imports, so
   `import libero` breaks outright.
3. **Importing `libero` runs `input()`** if `~/.libero/config.yaml` is absent, blocking
   interactively and dying with `EOFError` under any non-interactive runner.
4. **`egl_probe` / `hf-egl-probe` publish no wheels** and must compile. There is no `--no-deps`
   shortcut that preserves EGL device probing — the exact path G0 verifies.
5. **Demo image keys ≠ live env image keys.** Demos store `agentview_rgb` / `eye_in_hand_rgb`;
   the live env returns `agentview_image` / `robot0_eye_in_hand_image`. Code reading both must
   translate. This is a silent-wrong-answer trap, not a crash.

## Two open questions Phase 1 must settle

1. **Demo replay fidelity — unresolved, and it gates Phase 3.** We replay under mujoco 3.3.7;
   the demos were recorded under an older MuJoCo. G0 proved only that *our* simulator is
   self-consistent — it did **not** test that replaying recorded `actions[t]` from `states[0]`
   reproduces the recorded `states[t]`. SPEC.md §8 requires that property. Check it before
   training anything on the demos. If it drifts, that is a finding to report, not to tune away.
2. **Encoding cache portability.** The cache key in SPEC.md §4.1 is
   `(episode_id, frame_idx, encoder_cfg_hash)` — it identifies neither machine nor device, so
   latents computed on different hardware collide under the same key. SPEC.md §4.1 has been
   amended (see the Phase 0 entry in PROGRESS.md); implement the fingerprint check in Phase 1
   before any cache is written.

## Rules that are easy to violate without noticing

From `CLAUDE.md`, repeated here because they matter most at phase boundaries:

- **Never weaken a metric, threshold, or gate to make a run pass.** A failing gate is
  information. This is the single most damaging thing that could happen to this project.
- **A number is only real if it came from simulator replay.** Latent goal distance is a search
  objective, never a result.
- Any experiment reporting a success rate also reports `random_perturb` and `no_edit`.
- Every result writes a config snapshot to `results/runs/<run_id>/` beside `metrics.jsonl`.
- Never edit `cfe/eval/` after Phase 2 sign-off. It does not exist yet.
- One phase per session; append to `PROGRESS.md` at the end of each.
