"""Disk cache for encodings, keyed by `(episode_id, frame_idx, encoder_cfg_hash)`
(SPEC.md §4.1).

Layout, under `<cache_dir>/<encoder_cfg_hash>/`:

    fingerprint.json     the full payload the hash was computed from
    <episode_id>.npy     (T, N, D) tokens; frame_idx is the row index

One file per episode rather than one per frame: a task is ~7,800 frames, and 7,800 tiny
files is a bad shape for any filesystem. Reads are memory-mapped because a task's latents are
~3.1 GB and this machine has 7 GB of RAM.

**Why the fingerprint exists.** The key in SPEC.md §4.1 names neither machine nor device, so
latents computed on different hardware collide under the same key while differing in the low
bits. A mismatch is therefore a hard error, never a silent recompute — recomputing quietly
would hide that two runs' numbers came from different machines.

Beyond the §4.1 amendment (which lists torch version and a cpu/cuda tag) this also hashes the
**GL renderer string** and **obs_source**. Measured on 2026-07-30: two boxes both software-
rasterising with different mesa drivers render the same simulator state to different pixels,
so a cpu/cuda tag alone does not separate machines. See PROGRESS.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


class FingerprintMismatch(RuntimeError):
    """Raised when a cache's stored fingerprint disagrees with the current environment."""


@dataclass(frozen=True)
class Fingerprint:
    """Everything that can change what a frame encodes to."""

    encoder: dict  # EncoderConfig.hash_payload()
    torch_version: str
    device_kind: str  # "cpu" | "cuda"
    renderer: str  # GL_RENDERER of the context that produced the frames
    obs_source: str  # "replay" | "recorded"

    @classmethod
    def build(
        cls, encoder_payload: dict, device: torch.device | str, renderer: str, obs_source: str
    ) -> Fingerprint:
        kind = torch.device(device).type if not isinstance(device, torch.device) else device.type
        return cls(
            encoder=encoder_payload,
            torch_version=torch.__version__,
            device_kind=kind,
            renderer=renderer,
            obs_source=obs_source,
        )

    def payload(self) -> dict:
        """Canonical JSON form.

        Round-tripped through JSON on purpose: `asdict()` preserves tuples (norm_mean,
        norm_std) but reading fingerprint.json back yields lists, so comparing the raw
        dataclass form against a stored file would report a mismatch on every reopen and
        make the cache unusable. Canonicalising here keeps the hash stable too.
        """
        return json.loads(json.dumps(asdict(self), sort_keys=True))

    def digest(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class EncodingCache:
    """Encodings for one `encoder_cfg_hash`. Construction validates the fingerprint."""

    def __init__(self, cache_dir: str | Path, fingerprint: Fingerprint):
        self.fingerprint = fingerprint
        self.hash = fingerprint.digest()
        self.root = Path(os.path.expanduser(str(cache_dir))) / self.hash
        self.root.mkdir(parents=True, exist_ok=True)
        self._validate()

    @property
    def fingerprint_path(self) -> Path:
        return self.root / "fingerprint.json"

    def _validate(self) -> None:
        current = self.fingerprint.payload()
        if not self.fingerprint_path.exists():
            self.fingerprint_path.write_text(json.dumps(current, indent=2, sort_keys=True))
            return
        stored = json.loads(self.fingerprint_path.read_text())
        if stored != current:
            differing = sorted(
                k for k in set(stored) | set(current) if stored.get(k) != current.get(k)
            )
            raise FingerprintMismatch(
                f"cache at {self.root} was written under a different fingerprint; "
                f"differing fields: {differing}. This is a hard error, not a recompute: "
                f"reusing it would mix latents from different environments under one key. "
                f"stored={ {k: stored.get(k) for k in differing} } "
                f"current={ {k: current.get(k) for k in differing} }"
            )

    # -- episode-level access -------------------------------------------------

    def path_for(self, episode_id: str) -> Path:
        return self.root / f"{episode_id}.npy"

    def has(self, episode_id: str) -> bool:
        return self.path_for(episode_id).exists()

    def write(self, episode_id: str, tokens: np.ndarray) -> Path:
        """Store `(T, N, D)` tokens for one episode.

        Written to a temporary file and renamed, so an interrupted run cannot leave a
        truncated `.npy` that later reads as a valid cache hit.
        """
        if tokens.ndim != 3:
            raise ValueError(f"expected (T, N, D) tokens for {episode_id!r}, got {tokens.shape}")
        dest = self.path_for(episode_id)
        tmp = dest.with_suffix(".npy.tmp")
        # Write through a file handle: np.save(path, ...) appends ".npy" when the name does
        # not already end in it, which would silently write somewhere else.
        with open(tmp, "wb") as fh:
            np.save(fh, tokens)
        os.replace(tmp, dest)
        return dest

    def read(self, episode_id: str, frame_idx: int | slice | None = None) -> np.ndarray:
        """Memory-mapped read. `frame_idx=None` returns the whole `(T, N, D)` episode."""
        path = self.path_for(episode_id)
        if not path.exists():
            raise KeyError(f"no cache entry for episode {episode_id!r} at {path}")
        arr = np.load(path, mmap_mode="r")
        return arr if frame_idx is None else arr[frame_idx]

    def episode_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.npy"))

    # -- per-episode sidecar --------------------------------------------------
    # Actions, proprioception and the replayed success flag live beside the latents rather
    # than in their own store. They are small (tens of KB against 62 MB of tokens) and
    # keeping one episode's data in one place is easier to reason about than a second
    # directory keyed differently.

    def meta_path(self, episode_id: str) -> Path:
        return self.root / f"{episode_id}.meta.npz"

    def has_meta(self, episode_id: str) -> bool:
        return self.meta_path(episode_id).exists()

    def write_meta(self, episode_id: str, **arrays) -> Path:
        dest = self.meta_path(episode_id)
        tmp = dest.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, dest)
        return dest

    def read_meta(self, episode_id: str) -> dict:
        path = self.meta_path(episode_id)
        if not path.exists():
            raise KeyError(f"no metadata for episode {episode_id!r} at {path}")
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
