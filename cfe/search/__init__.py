"""Two-stage counterfactual search (SPEC.md §5).

Stage 1 (`windows`) picks the edit window by per-step latent divergence; stage 2 (`cem`) runs
CEM over deltas inside it. `objective` holds `J(a')` and `arms` dispatches the search alongside
SPEC.md §6's four mandatory baselines.

**Everything in this package computes a search objective, not a result.** CLAUDE.md rule 2 and
SPEC.md I2: latent goal distance is what the search optimises against, and the only number that
may be reported is what the simulator says on replay. That is why this package sits outside the
frozen `cfe/eval/` and why nothing here imports the replay protocol except to construct an
`Edit` for it to execute.
"""
