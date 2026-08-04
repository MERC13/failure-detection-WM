"""Reported metrics and the replay protocol (SPEC.md §6). **FROZEN after Phase 2.**

Nothing in this package is edited after Phase 2 sign-off except to fix a demonstrated bug,
recorded in `CHANGELOG.md` with the reason (SPEC.md I3, CLAUDE.md rule 1). A change here
rewrites every result the project has ever reported, which is why it is written before any
search exists: an instrument built after the method it measures can be shaped by it.

Two things live here and nothing else:

- `replay_protocol` — how an edited episode is executed in the simulator. Per I2 this is the
  *only* source of ground truth in the project. Latent goal distance is a search objective and
  is never a result, so it does not appear here.
- `metrics` — the numbers §6 says to report, plus the four mandatory baselines' interfaces.

Deliberately **not** here: `cfe/dynamics/latents.py`'s latent distance (the search objective's
substrate, explicitly outside the freeze) and anything that proposes edits (that is
`cfe/search/`, Phase 3). Keeping the measuring instrument in a different directory from the
thing being measured is the point of the freeze.
"""
