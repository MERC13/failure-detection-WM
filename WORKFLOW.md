# Claude Code setup and prompting playbook

## The one idea to internalize first

An agent asked to make a research result appear will make one appear. Not by lying — by
loosening a threshold, catching an exception, narrowing a test, substituting an approximate
reset, or reporting the search objective instead of the replayed outcome. Each step is locally
reasonable and the aggregate is a fabricated finding.

The defense is structural, not motivational: **write and freeze the measurement before the
method exists, and never let one session both implement a thing and define whether it
worked.** That is why `SPEC.md` fixes metrics in Phase 2 and the counterfactual search isn't
built until Phase 3. Keep that ordering even when it feels slow.

## Setup

Install per the current docs — https://code.claude.com/docs/en/overview — the npm package is
`@anthropic-ai/claude-code`. Then:

```bash
mkdir cfe && cd cfe && git init
# drop SPEC.md and CLAUDE.md in the root
claude
```

First session, in order:

1. `/init` — generates a starting CLAUDE.md from the codebase. Since you already have one, it
   proposes improvements rather than overwriting. Take what's useful, discard the rest.
2. `/context` — confirms which memory files actually loaded. If your CLAUDE.md isn't listed
   under **Memory files**, Claude cannot see it and nothing in it is in effect. Check this
   whenever behavior seems to ignore your rules.
3. `/memory` — browse and edit memory files, and see what Claude has saved for itself.

Worth knowing about how this actually works, because it changes how you write rules:

- CLAUDE.md is delivered as context, not as enforced configuration, and there's no guarantee
  of strict compliance — especially for vague or conflicting instructions. Specific and
  concrete beats emphatic.
- Target **under 200 lines**. Longer files consume more context and reduce adherence. The
  provided CLAUDE.md is deliberately short.
- Project-root CLAUDE.md **survives `/compact`** — it gets re-read from disk and re-injected.
  Instructions you only ever said in conversation do not survive. If you correct Claude on
  something twice, put it in the file.
- `/doctor` can propose trims: it cuts what Claude could derive from the codebase anyway
  (directory layouts, dependency lists) and keeps pitfalls and rationale. Useful once the repo
  is real.
- Alongside your CLAUDE.md, Claude keeps **auto memory** — its own notes on build commands and
  debugging insights, per repository, loaded each session. Audit it with `/memory`
  occasionally; a wrong entry there will quietly misdirect sessions.

### Hard enforcement for the things that matter

CLAUDE.md shapes behavior; it doesn't block anything. For the two rules whose violation would
be unrecoverable, use a **PreToolUse hook** instead, which runs as a shell command at a fixed
lifecycle event regardless of what Claude decides:

- block edits to `cfe/eval/**` after Phase 2 (protects I3)
- block `git push`

See https://code.claude.com/docs/en/hooks-guide. This is belt-and-braces, and worth the ten
minutes.

### Path-scoped rules

Put module-specific detail in `.claude/rules/*.md` with `paths` frontmatter so it loads only
when Claude touches matching files, keeping the always-loaded CLAUDE.md lean:

```markdown
---
paths:
  - "cfe/search/**/*.py"
---
# Search module rules
- CEM warm-starts at the executed actions. Never reinitialize from scratch.
- The edit window is enforced structurally, not by a soft penalty.
- z_goal takes the min distance over all matching successful demos — goal states are
  multimodal.
```

Good candidates: `cfe/search/`, `cfe/dynamics/`, `cfe/idm/`, `tests/`.

## Session hygiene

- **One phase per session.** Start fresh (`/clear`) between phases. Long sessions accumulate
  context that dilutes your rules.
- **Plan mode for anything multi-file.** It catches wrong-problem mistakes before code exists,
  which is most of what goes wrong. Skip it for one-line fixes.
- **Read every diff.** Not skim. The failure modes above are all visible in a diff and
  invisible in a summary.
- **Run it yourself at least once per phase.** Not "did the agent say the test passed" —
  actually run `make test` in your own terminal.
- **End each session by having Claude append to `PROGRESS.md`**: what landed, gate state, what
  surprised it, what's next. That file is what makes session N+1 cheap.

## Phase prompts

Paste these roughly as-is. They're written to be specific about the gate, since the gate is
what keeps the phase honest.

### P0 — Environment

```
Read SPEC.md. We're doing Phase 0 only: a pinned, reproducible environment and the G0 gate.

Install LIBERO with its robosuite/MuJoCo dependencies, pin every version in
requirements.txt with a comment on why each pin exists if you hit a conflict, and write
scripts/verify_env.py implementing the G0 gate from SPEC.md §7.

The critical check is invariant I1: reset the env to a demo's initial state twice and assert
the observations are identical. If exact-state reset doesn't work the way SPEC.md assumes,
stop and tell me — do not work around it.

Also correct SPEC.md's API guesses against the actually installed package and note the
corrections in PROGRESS.md. Don't write any modeling code this session.
```

### P1 — Encoder, data, failure-generating policy

```
Phase 1. Frozen DINOv2 encoder wrapper with the disk cache described in SPEC.md §4.1, demo
loading, and the BC policy from §4.3.

The BC policy is a means to an end — keep it simple and deliberately mediocre, targeting
40-70% success on one task from libero_object. Do not tune it upward.

G1 gate: encoder produces cached latents with a test asserting cache hits are bit-identical
to fresh encodes and that no encoder parameter has requires_grad=True; BC policy lands in
the success band. Report the actual success rate.
```

### P2 — Dynamics model, then freeze the metrics

```
Phase 2, in two parts, in this order.

Part A: the action-conditioned latent dynamics model from SPEC.md §4.2, plus the G2
diagnostics. I care most about one comparison: the model versus the same model with actions
zeroed out. If action-conditioning isn't clearly helping, the rest of the project is
meaningless, so report that number prominently and don't proceed if it's marginal.

Part B: write cfe/eval/ — metrics.py and replay_protocol.py per SPEC.md §6, with the tests
from §8 including the hand-constructed metric cases. These files get frozen after this
phase, so I want them reviewed carefully now. Implement all four mandatory baselines'
interfaces here even though the search doesn't exist yet.

Do not write any counterfactual search code this session.
```

Between P2 and P3: review `cfe/eval/` yourself, line by line. Then add the PreToolUse hook
blocking edits to it.

### P3 — Skeleton to the G3 gate

```
Phase 3. End-to-end skeleton on ONE task with 10 injected failures: build_failure_set.py for
the injected causes in SPEC.md §4.4, the two-stage CEM search in §5, and run_replay.py using
the frozen eval code.

Ugly and slow is fine. What I want is the G3 gate: replay success rate for the search versus
random_perturb at equal edit budget and versus no_edit, with a bootstrap CI.

If the search doesn't beat random_perturb, say so plainly and give me your best diagnosis of
why. Do not tune it until it wins. A failed G3 changes the project and I need to know on day
5, not day 20.
```

### P4 — Scale, C1

```
Phase 4. Scale to the full failure set in SPEC.md §4.4 — ~200 failures, 3+ tasks, natural
and injected. Run all four mandatory baselines.

Produce the C1 table with bootstrap CIs, and per invariant I4, report RSR as a curve over
edit budget K and over the λ grid rather than single numbers. Also report cause-identification
accuracy against the injected ground-truth windows.
```

### P5 — Latent actions, C2

```
Phase 5. The latent action arm from SPEC.md §4.5: IDM plus forward model, the three
regularization variants, and the action decoder.

Report decoder reconstruction error separately from the counterfactual results — I need to
distinguish "bad latent space" from "bad decoder." For the VQ variant, report codebook usage;
a collapsed codebook invalidates that row.

Then the C2 ablation table: raw actions vs each latent variant, on replay success rate, edit
sparsity, and edit magnitude, at matched edit budgets.
```

### P6 — Artifact

```
Phase 6. make_figures.py producing: the C1 baseline comparison, the RSR-vs-edit-budget
curves, the C2 ablation table, and side-by-side "what happened / what should have happened"
replay videos for 6-8 representative failures spanning the cause tags.

Then a README with the result up front, the method in a paragraph, and an honest limitations
section. Include Dream2Fix (arXiv 2603.13528) as the closest related work and state the
delta in two sentences: they synthesize failures from successes to train a corrector; we
search for minimal successful edits at inference as a diagnostic.
```

## Corrections worth keeping in your pocket

When a session goes sideways, these are the phrasings that reliably pull it back:

- "That's the search objective, not a result. Replay it."
- "You changed the threshold. Change it back and tell me why the run failed."
- "Don't add a fallback. Let it raise and show me the trace."
- "Is that number from the simulator or from the model?"
- "You're solving a different problem than SPEC.md describes. Re-read §5 and tell me where
  you diverged."

If you say any of these twice, it belongs in CLAUDE.md.

## When to distrust a green result

Be actively suspicious when:

- replay success is high on the *first* run of a phase
- `no_edit` isn't ~0% (replay is broken, or the success check is wrong)
- `random_perturb` is close to the method (the world model isn't contributing)
- `expert_oracle` isn't near-perfect (the replay protocol is broken)
- the numbers changed after a "small refactor" of anything under `cfe/eval/`

Every one of these has a boring mechanical cause, and finding it is the actual work.
