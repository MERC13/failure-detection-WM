"""The failure set of SPEC.md §4.4, on disk as `results/failure_set/<id>.json`.

One record per failed episode, from either source. §4.4 lists what a record must carry: task
id, init state blob, executed actions, per-step observations *or encoding cache keys*, success
flag, cause tag (`null` for natural), and — for injected ones — the ground-truth edit window.
Observations are referenced by cache key rather than embedded: they are ~65 MB of latents per
episode and they already live in the encoding cache under a fingerprint that says which
machine produced them, which a copy inside a JSON file would not.

**`cause` is `None` for natural failures and a string for injected ones**, and CLAUDE.md is
explicit that code assuming non-null crashes on half the dataset. `FailureRecord.is_injected`
exists so that branch is spelled out at every site instead of being inferred from a truthiness
test on a field that is legitimately empty.

`expert_episode_id` is the uncorrupted source demo. It is what makes
`cfe.eval.metrics.expert_oracle` computable, and it is `None` for natural failures — Phase 1
verified no demo shares a benchmark init state, so there is no expert from the same state and
none may be substituted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NATURAL = "natural"
INJECTED = "injected"


@dataclass(frozen=True)
class FailureRecord:
    """One failed episode, with everything the search and the replay protocol need."""

    failure_id: str
    source: str  # NATURAL | INJECTED
    suite: str
    task_id: int
    task_name: str
    #: Cache key for this episode's latents, actions and proprio (`EncodingCache`).
    episode_id: str
    #: The exact simulator state to reset to (I1). 110 floats for LIBERO.
    init_state: np.ndarray
    #: What was actually executed, `(T, 7)`.
    actions: np.ndarray
    #: Always False — this is the failure set. Stored anyway so a record is self-describing
    #: and a bug that admits a success is visible in the file rather than only in a count.
    success: bool
    #: `None` for natural failures. See the module docstring.
    cause: str | None
    #: Ground-truth edit window `[start, end)`, injected failures only.
    edit_window: tuple[int, int] | None
    #: Uncorrupted source demo's cache key, injected failures only.
    expert_episode_id: str | None
    #: Injection parameters, or the natural failure's measured mode and displacements.
    params: dict

    def __post_init__(self) -> None:
        if self.source not in (NATURAL, INJECTED):
            raise ValueError(f"source must be {NATURAL!r} or {INJECTED!r}, got {self.source!r}")
        if self.success:
            raise ValueError(f"{self.failure_id}: the failure set holds failures only")
        if self.source == INJECTED and self.cause is None:
            raise ValueError(f"{self.failure_id}: an injected failure must carry a cause tag")
        if self.source == NATURAL and self.cause is not None:
            raise ValueError(
                f"{self.failure_id}: a natural failure has no ground-truth cause; got "
                f"{self.cause!r}. Guessing one would make cause accuracy score a guess."
            )
        if self.source == INJECTED and self.edit_window is None:
            raise ValueError(f"{self.failure_id}: an injected failure must carry its edit window")

    @property
    def is_injected(self) -> bool:
        return self.source == INJECTED

    @property
    def n_steps(self) -> int:
        return int(len(self.actions))

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "failure_id": self.failure_id,
            "source": self.source,
            "suite": self.suite,
            "task_id": int(self.task_id),
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "init_state": np.asarray(self.init_state, dtype=np.float64).tolist(),
            "actions": np.asarray(self.actions, dtype=np.float64).tolist(),
            "success": bool(self.success),
            "cause": self.cause,
            "edit_window": list(self.edit_window) if self.edit_window is not None else None,
            "expert_episode_id": self.expert_episode_id,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FailureRecord:
        w = d.get("edit_window")
        return cls(
            failure_id=d["failure_id"],
            source=d["source"],
            suite=d["suite"],
            task_id=int(d["task_id"]),
            task_name=d["task_name"],
            episode_id=d["episode_id"],
            init_state=np.asarray(d["init_state"], dtype=np.float64),
            actions=np.asarray(d["actions"], dtype=np.float64),
            success=bool(d["success"]),
            cause=d.get("cause"),
            edit_window=(int(w[0]), int(w[1])) if w is not None else None,
            expert_episode_id=d.get("expert_episode_id"),
            params=d.get("params", {}),
        )


def failure_set_dir(repo_root: str | Path) -> Path:
    return Path(repo_root) / "results" / "failure_set"


def write_record(directory: str | Path, record: FailureRecord) -> Path:
    """Write one record, atomically.

    Via a temp file and a rename because this runs inside a loop that has been interrupted by
    hard power-offs more than once: a half-written JSON that later parses as a valid record
    would put a truncated action sequence into a result.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{record.failure_id}.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record.to_dict()))
    tmp.replace(dest)
    return dest


def read_record(path: str | Path) -> FailureRecord:
    return FailureRecord.from_dict(json.loads(Path(path).read_text()))


def load_failure_set(
    directory: str | Path,
    *,
    source: str | None = None,
    task_id: int | None = None,
    cause: str | None = None,
) -> list[FailureRecord]:
    """Every record matching the filters, ordered by `failure_id` for reproducibility.

    Sorted rather than in directory order: G3 selects the first N failures, and a set that
    depends on filesystem enumeration order would not be the same set on two machines.
    """
    d = Path(directory)
    if not d.exists():
        return []
    out = [read_record(p) for p in sorted(d.glob("*.json"))]
    if source is not None:
        out = [r for r in out if r.source == source]
    if task_id is not None:
        out = [r for r in out if r.task_id == task_id]
    if cause is not None:
        out = [r for r in out if r.cause == cause]
    return sorted(out, key=lambda r: r.failure_id)


def summarise(records: list[FailureRecord]) -> dict:
    """Counts by source, cause and task — what §4.4's quotas are actually checked against."""
    by_cause: dict[str, int] = {}
    by_task: dict[int, int] = {}
    for r in records:
        key = r.cause if r.cause is not None else "natural (unlabelled)"
        by_cause[key] = by_cause.get(key, 0) + 1
        by_task[r.task_id] = by_task.get(r.task_id, 0) + 1
    return {
        "n_total": len(records),
        "n_injected": sum(1 for r in records if r.is_injected),
        "n_natural": sum(1 for r in records if not r.is_injected),
        "by_cause": dict(sorted(by_cause.items())),
        "by_task": dict(sorted(by_task.items())),
        "n_tasks": len(by_task),
    }
