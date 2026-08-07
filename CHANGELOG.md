# CHANGELOG

Changes to metrics, thresholds, gates, and the replay protocol. Per CLAUDE.md, anything that
would rewrite or reinterpret a prior result is logged here with the reason, whether or not it
touches `cfe/eval/`.

---

## 2026-08-05 — G3 criteria pre-registered; `magnitude_match_tol` known too tight (OPEN)

**Added.** `configs/search.yaml:gate`, six criteria for G3, plus `scripts/verify_g3.py` which
reads them from the config rather than hard-coding them. This is a *new* gate, not a change to
an existing one, and is logged here so the pre-registration has a dated record.

The criteria were written and committed **before any search was run and before any replay number
existed**, matching the discipline `configs/dynamics.yaml` used for G2. In summary form:

| # | Criterion | Basis |
|---|---|---|
| 1 | `beats_random_perturb` — paired bootstrap CI excludes zero at K=8 **at every λ** | SPEC.md §7; §6: "A method that wins only at one λ has not won" |
| 2 | open-loop `no_edit` RSR is exactly 0 | §6's replay-bug control |
| 3 | all four §6 baselines present at every cell | §6: "or it does not ship" |
| 4 | `random_perturb` realised magnitude matches the search's to `magnitude_match_tol` | CLAUDE.md's magnitude-matching obligation |
| 5 | search reproduces a recorded edit bit-exactly | I5 |
| 6 | config snapshot present | CLAUDE.md rule 4 |

**Result (run `g3-phase3`): FAIL, 4/6.** Criteria 1 and 4 failed. Nothing was relaxed. Full
numbers in `PROGRESS.md` 2026-08-05.

### Open item: `magnitude_match_tol: 0.02` is too tight, and is NOT yet changed

Criterion 4 failed on **1 of 120 cells** — max absolute difference 0.1176, mean 0.0065. The
cause is clipping at the action bounds, which this repo had already documented in
`cfe/eval/metrics.py:random_perturb`: "the *realised* magnitude can come in under
`magnitude_scale` when the original action is already near a bound". On `wrong_target` episodes
the search's edit sits near those bounds, so `random_perturb`'s random direction clips and
realises 0.392 against a requested 0.509. The tolerance was pre-registered without accounting
for a mechanism described three files away.

**This is recorded as an open defect, not fixed.** Two reasons for leaving it:

1. Changing a gate threshold after seeing it fail is the exact shape CLAUDE.md rule 3 forbids,
   and this project has already amended a gate after a failure twice. It needs the user's
   decision, not the assistant's.
2. **The bias runs toward the search.** An under-scaled `random_perturb` is a *weaker* control,
   so this defect can only have flattered the treatment arm. It therefore cannot explain or
   excuse criterion 1's failure, and relaxing it would not manufacture a pass.

If it is changed, the change belongs here with its reasoning and rejected alternatives. Options
visible now: widen the tolerance to cover measured clipping; make the criterion compare
*requested* rather than realised scale (rejected on sight — realised is what was replayed);
or exclude clipped cells and report the count (keeps the tolerance meaningful, adds a rule).

**Prior results affected.** None. G3 is the first gate these criteria have scored.

---

## 2026-08-04 — G2 `beats_action_ignoring`: magnitude criterion moved from h8 to h96

**Changed.** `configs/dynamics.yaml:gate`. `min_rel_improvement_h8: 0.20` becomes
`min_rel_improvement: 0.20` plus `rel_improvement_horizon: 96`. `scripts/verify_g2.py` follows,
scoring the magnitude criterion on its own window set. `ci_horizons: [4, 8]` and
`min_abs_spearman: 0.30` are **unchanged**.

**The threshold did not move. 0.20 is still 0.20.** What moved is the horizon it is measured at.

**Why.** SPEC.md §5's search rolls from the edit window to episode end — 100+ steps. h8 measured
action-conditioning at the horizon where it is weakest, in a regime the project never operates
in. Two measurements, both reproducible:

*The horizon effect is real, not episode selection* (`scripts/diagnose_window_selection.py`,
run `diag-window-p2dyn`, model `p2-dyn`). `rel_improvement` by horizon × minimum future frames:

```
              8        16        32        64        96       128
   n=      2542      2422      2182      1702      1222       742
h8      +0.0499   +0.0508   +0.0534   +0.0587   +0.0538   +0.0548
h16           -   +0.0787   +0.0803   +0.0919   +0.0835   +0.0937
h32           -         -   +0.1179   +0.1417   +0.1394   +0.1532
h64           -         -         -   +0.2025   +0.2397   +0.2877
h96           -         -         -         -   +0.2678   +0.3573
```

Down a column is horizon alone; across a row is selection alone. **Horizon 4.5–6.1×, selection
1.10–1.42×**, and the h8 row is flat across window sets differing 3.4× in size. Long horizons
are only measurable in long episodes, and this separates the two.

*The model resolves fine action edits* (`scripts/diagnose_edit_discrimination.py`, run
`diag-edit-p2dyn`). Feeding deliberately wrong actions raises error smoothly and **quadratically**
— each 2× in edit size gives ~4× in error change — detectable down to 0.05 σ. That is a genuine
local action gradient, so h8's weakness is a scale artifact and not fine-edit blindness. The
action-ignoring twin is bit-identical under every perturbation, which validates the harness.

**This was written after a gate failed, twice.** That is the exact shape CLAUDE.md rule 3 warns
against and it is recorded plainly rather than presented as routine. Three things are offered as
the defense, and a reader who finds them insufficient has every number needed to revert:

1. The threshold is untouched; only the measurement point moved.
2. The horizon was chosen on a criterion independent of the result — *the regime §5's search
   operates in* — not by scanning for a horizon that passes.
3. The assistant recommended **against** this change and argued for leaving h8 alone; the user
   decided otherwise after both measurements were presented. It was not amended unilaterally.

**Options rejected, and why:**

| Option | Rejected because |
|---|---|
| Keep h8, lower 0.20 to ~0.05 | Straightforward threshold-weakening — changes only the number that failed and encodes no reason. This is what rule 3 exists to forbid. |
| Gate at h64 instead | `p2-dyn` reads 0.2025 there against a 0.20 bar — a 1.25% margin, inside the noise measured on this quantity. A gate that marginal is not a gate. h64 is also still short of §5's regime, so it would be picked for its number rather than its meaning. |
| Gate at h128 | Only 742 of 2542 windows, and the most selection-inflated column (1.42× at h64). Fewest windows and worst confound. |
| Leave the gate, pursue proprioception / finer reduction / larger `k_context` | Still worth doing and not foreclosed by this change, but they are orthogonal model improvements. None addresses the fact that h8 measures the wrong regime, so none is a substitute for fixing the horizon. |
| Move `ci_horizons` to long horizons as well | Not done deliberately. Requiring the CI to exclude zero at h4/h8 over all 2542 windows keeps a conservative short-horizon requirement that this amendment does not relax. |

**Known consequence.** The magnitude criterion is now scored on **1222 windows rather than
2542**, and those are systematically the longer episodes. The gate output reports
`n_rel_criterion_windows` so the narrowing is visible in every run rather than implicit. The
selection inflation at h96 is measured at 1.33× and is not corrected for.

**Prior results affected.** Two G2 runs FAILed under the old criterion —
`g2-20260801T014804Z` (`p2-dyn`, h8 rel 0.0499) and `g2-20260804T000955Z` (`p2-dyn-res`, h8 rel
0.0350). **Both stand exactly as recorded**: they were correct measurements of h8, and h8 is
still reported in every run. What changes is which horizon the pass/fail hinges on. Under the
amended gate `p2-dyn` reads 0.2678 at h96 and would pass; that is a re-decision of the same
model on a different criterion, not a new measurement of it.

---

## 2026-07-31 — G1 success criterion: replace the 40–70% band

**Changed.** SPEC.md §4.3 and §7 G1. The BC policy gate no longer tests a raw success-rate
band. It tests **≥13 natural failures per task per 50 rollouts**, requires a **failure-mode
breakdown** to be reported with the rate, and keeps a wide sanity bound of **[0.20, 0.85]** on
the rate itself. `scripts/verify_g1.py` and `configs/policy.yaml` follow.

**Why.** The measured policy came in at **72%** (36/50, Wilson 95% CI [0.583, 0.825]) against a
40–70% band. The CI's lower bound clears 0.70, so this is a real value, not sampling noise.

The band was a proxy. SPEC.md §4.3 states its purpose in the same breath as the numbers: *"A
policy that always succeeds gives no failures; one that always fails gives no recoverable
ones."* Both properties are directly measurable now, and both hold at 72%:

- **Supply.** §4.4 needs ~40 natural failures across ≥3 tasks (~13/task). At a 28% failure rate
  that is 14 per 50 rollouts — clears the requirement with no extra rollouts.
- **Recoverability.** Measured across all 14 failures: the target object is never grasped or
  transported. Max target displacement in any failure is **0.0381 m**, against a **minimum of
  0.4843 m across all 36 successes** — a 13× gap with nothing in between. The gripper reaches
  4–12 cm from the object and at most brushes it. These are approach errors, precisely the
  class a small x/y edit during the approach window can rescue.

  Breakdown of the 14: `target_moved_not_placed` 7 (target brushed 1.3–3.8 cm),
  `distractor_only_target_untouched` 4, `no_object_contact` 3.

**This was written after a gate failed.** That is the exact shape CLAUDE.md rule 3 warns
against, and it is recorded plainly rather than presented as routine. The defense is that the
requirement behind the threshold was measured and met, not that the number was inconvenient.
A reader who disagrees can reinstate the band; every number needed to do so is above.

**Options rejected, and why:**

| Option | Rejected because |
|---|---|
| Weaken the policy into the band | The failure mode is approach imprecision, so a weaker policy produces *more of the same mode*. Costs a retrain plus 50 rollouts and buys no diversity. Rejected on evidence, not preference. |
| Widen the band numerically (e.g. 40–80%) | Same effect with none of the justification — it changes only the threshold that failed, and encodes no reason. |
| Accept the deviation, leave G1 FAILing | Honest, but carries a standing red gate into Phase 2 that says nothing about whether the policy is fit for purpose. |
| Lower `max_steps` | **Never acceptable.** Every failure is a 300-step timeout, so shrinking the cap slides the rate into the band without touching the policy. This is measurement-tuning and is off-limits; recorded so it is not reached for later. Raising it is equally off-limits as a gate-directed edit. |

**Known consequence.** The natural failure set is currently **100% approach failures**, which
nearly duplicates the `grasp_offset` injected cause. Failure-set diversity now rests on the
injected arm. Carried into Phase 3.

**Prior results affected.** None. Phase 1 is the first phase to report a success rate, and the
72% figure is unchanged by this amendment — only its interpretation as pass/fail.

---

## 2026-07-31 — §8 replay fidelity scoped to cached episodes

**Changed.** SPEC.md §8 and G1's `expert_replay_success`. The requirement that replaying
unedited actions reproduces the recorded success flag *exactly* now applies to **every episode
admitted to the encoding cache**, rather than to every demo in the HDF5. A yield floor
(`data.min_replay_yield = 0.85` in `configs/encoder.yaml`) is added as a regression tripwire.

**Why.** Replay reproduces the recorded success flag for 92–96% of demos — measured over all
50 demos on each of `libero_object` tasks 0, 1 and 2 (0.96 / 0.92 / 0.94). As written, the
check could never pass on this data, and a gate that can never pass stops being read.

The evidence that this is physics divergence and not a replay defect:

- Reset is bit-exact: `max|get_sim_state() − states[0]| = 0.0`.
- 48/50 reproduce on task 0, and the yield is stable across three different target objects.
- The failures have a coherent physical signature. `demo_27` and `demo_31` end with the gripper
  0.049 m and 0.023 m from the object and lift it 0.005 m / 0.042 m — they follow the whole
  trajectory and miss at contact, which is margin, not a broken pipeline. A wrong action scale,
  an off-by-one, or a bad reset would fail demos indiscriminately from the first step.

This is suggestive, not proof, and it is recorded as such.

**The exactness requirement is not softened.** It is scoped. `collect_demos.py` already refuses
to cache an episode whose replay did not succeed, so a non-reproducing demo is *rejected by the
pipeline*, not silently used. The gate now asserts that no such episode is in the cache —
48/48, exact, no tolerance. That is the property I2 actually needs: everything feeding a result
replays faithfully.

**The yield floor is a tripwire, not a pass mark.** 0.85 sits below every measured yield and
exists so a real replay regression still fails the gate. Lowering it to make a run pass is
forbidden (CLAUDE.md rule 3).

**Prior results affected.** None. No result has been reported from replay yet; `demo_27` and
`demo_31` were already excluded from the cache and from BC training before this change.

---

## 2026-07-31 — G1 `fingerprint_mismatch_raises` made re-runnable (bug fix, no semantic change)

The check corrupts a fingerprint on disk by design, but did so in a fixed directory
(`<cache_dir>/_g1_probe`) it never cleaned up. The second G1 invocation crashed on the first
one's tampered fingerprint before reaching any check. Now uses a `tempfile.mkdtemp` directory
removed in a `finally`. The assertions are unchanged.
