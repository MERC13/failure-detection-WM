"""Frozen DINOv2 encoder (SPEC.md §4.1).

Patch tokens `(N, D)` are the canonical representation. `pool()` is a module-level function
rather than a method or an implicit step inside `encode()`, so the pooling choice is always
visible at the point of use (CLAUDE.md convention).

The encoder *returns* the configured `output_dtype` rather than computing in fp32 and
narrowing when writing to disk. That keeps the G1 invariant "a cache hit is bit-identical to
a fresh encode" literally true instead of true-modulo-a-cast.

Weights never train: SPEC.md §4.1 requires `requires_grad == False` everywhere, and G1
asserts it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

# Fields of `encoder:` in configs/encoder.yaml, in the order they are hashed. Adding an
# encoder config field means adding it here too, or stale cache entries silently poison runs
# (CLAUDE.md conventions).
HASHED_FIELDS = (
    "model_id",
    "model_revision",
    "image_size",
    "flip_vertical",
    "resize_mode",
    "resize_antialias",
    "clamp_after_resize",
    "norm_mean",
    "norm_std",
    "output_dtype",
    "batch_size",
)

_DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str
    model_revision: str
    image_size: int
    flip_vertical: bool
    resize_mode: str
    resize_antialias: bool
    clamp_after_resize: bool
    norm_mean: tuple[float, ...]
    norm_std: tuple[float, ...]
    output_dtype: str
    batch_size: int = field(default=64)

    @classmethod
    def from_dict(cls, d: dict) -> EncoderConfig:
        missing = [f for f in HASHED_FIELDS if f not in d]
        if missing:
            raise KeyError(f"encoder config missing required fields: {missing}")
        unknown = set(d) - set(HASHED_FIELDS)
        if unknown:
            # A silently ignored field would not reach the hash, which is the exact failure
            # mode CLAUDE.md warns about. Refuse rather than drop it.
            raise KeyError(
                f"unknown encoder config fields (add to HASHED_FIELDS?): {sorted(unknown)}"
            )
        return cls(
            **{
                **d,
                "norm_mean": tuple(float(x) for x in d["norm_mean"]),
                "norm_std": tuple(float(x) for x in d["norm_std"]),
            }
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EncoderConfig:
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text())["encoder"])

    def hash_payload(self) -> dict:
        """The encoder half of the cache key. Runtime facts are added by cfe.encoders.cache."""
        return {k: v for k, v in asdict(self).items()}

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.output_dtype not in _DTYPES:
            raise ValueError(f"output_dtype {self.output_dtype!r} not in {sorted(_DTYPES)}")
        return _DTYPES[self.output_dtype]


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class Dinov2Encoder:
    """Frozen DINOv2. Call `encode()` with uint8 HWC frames; get `(B, N, D)` tokens back."""

    def __init__(self, cfg: EncoderConfig, device: str | torch.device = "auto"):
        from transformers import AutoModel

        self.cfg = cfg
        self.device = resolve_device(device) if isinstance(device, str) else device

        self.model = AutoModel.from_pretrained(cfg.model_id, revision=cfg.model_revision)
        self.model.eval().to(self.device)
        # SPEC.md §4.1: encoder weights never train.
        self.model.requires_grad_(False)

        self._mean = torch.tensor(cfg.norm_mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(cfg.norm_std, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """uint8 `(B, H, W, 3)` -> normalised float32 `(B, 3, S, S)` on the encoder's device."""
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected (B, H, W, 3) uint8 frames, got {frames.shape}")
        if frames.dtype != np.uint8:
            raise TypeError(f"expected uint8 frames, got {frames.dtype}")

        if self.cfg.flip_vertical:
            # MuJoCo's framebuffer is bottom-up; DINOv2 expects upright scenes.
            frames = frames[:, ::-1]

        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(
            x,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode=self.cfg.resize_mode,
            antialias=self.cfg.resize_antialias,
            align_corners=False,
        )
        if self.cfg.clamp_after_resize:
            x = x.clamp_(0.0, 1.0)
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """uint8 `(B, H, W, 3)` -> `(B, N, D)` patch tokens in `cfg.output_dtype`.

        Token 0 is DINOv2's CLS token; tokens 1: are the spatial patches. Batching uses
        `cfg.batch_size`, which is hashed into the cache key because GPU matmul kernels can
        vary with batch shape.
        """
        out = []
        for i in range(0, len(frames), self.cfg.batch_size):
            x = self.preprocess(frames[i : i + self.cfg.batch_size])
            tokens = self.model(pixel_values=x).last_hidden_state
            out.append(tokens.to(self.cfg.torch_dtype).cpu().numpy())
        if not out:
            return np.empty((0, 0, 0), dtype=self.cfg.output_dtype)
        return np.concatenate(out, axis=0)

    @property
    def n_params_requiring_grad(self) -> int:
        return sum(1 for p in self.model.parameters() if p.requires_grad)


def pool(tokens: np.ndarray | torch.Tensor, include_cls: bool = False):
    """Mean-pool patch tokens `(..., N, D)` -> `(..., D)`.

    Kept out of `encode()` on purpose: SPEC.md §4.1 makes tokens canonical and the pooled
    vector derived, and CLAUDE.md asks that the pooling choice be visible where it is used.

    `include_cls=False` averages only the spatial patches (tokens 1:), which is the standard
    DINOv2 mean-pooled representation. The CLS token is a different kind of summary and
    mixing it into the mean is a modelling choice, so it must be asked for explicitly.
    """
    sliced = tokens if include_cls else tokens[..., 1:, :]
    if isinstance(sliced, torch.Tensor):
        return sliced.mean(dim=-2)
    return sliced.mean(axis=-2)


#: Reductions the dynamics model may operate on, cheapest first. Which one to use is decided
#: by measurement (`scripts/probe_representation.py`), not by preference: full patch tokens are
#: 257x768 per frame, and a k=4 context with an H=8 horizon does not fit comfortably in 8.59 GB.
REDUCTIONS = ("cls", "mean", "grid2", "grid4", "full")


def grid_pool(tokens: np.ndarray, grid: int) -> np.ndarray:
    """Average the 16x16 DINOv2 patch grid down to `grid` x `grid`, keeping CLS as token 0.

    `(..., 257, D) -> (..., grid*grid + 1, D)`. Mean-pooling all 256 patches throws away
    *where* things are, which is most of what distinguishes "object on the table" from "object
    in the basket" — the thing this project is trying to measure. A coarse grid keeps location
    at a fraction of the cost.
    """
    *lead, n, d = tokens.shape
    side = int(round((n - 1) ** 0.5))
    if side * side != n - 1:
        raise ValueError(f"expected 1 CLS + a square patch grid, got {n} tokens")
    if side % grid:
        raise ValueError(f"{side}x{side} patch grid is not divisible into {grid}x{grid}")
    block = side // grid
    cls = tokens[..., :1, :]
    patches = tokens[..., 1:, :].reshape(*lead, grid, block, grid, block, d)
    pooled = patches.mean(axis=(-4, -2)).reshape(*lead, grid * grid, d)
    return np.concatenate([cls, pooled], axis=-2)


def reduce_latents(tokens: np.ndarray, mode: str) -> np.ndarray:
    """Apply one of `REDUCTIONS` to `(..., N, D)` tokens, returning `(..., R, D)`.

    Always returns a token axis, even when `R == 1`, so downstream code has one shape to
    reason about. Kept beside `pool()` and called at the point of use for the same reason:
    CLAUDE.md asks that the choice stay visible rather than buried in the encoder.
    """
    # Checked against the whitelist first: the mode comes from a config file, and a typo
    # should say so plainly rather than fail somewhere inside a parse.
    if mode not in REDUCTIONS:
        raise ValueError(f"unknown reduction {mode!r}; expected one of {REDUCTIONS}")
    if mode == "full":
        return tokens
    if mode == "cls":
        return tokens[..., :1, :]
    if mode == "mean":
        return pool(tokens, include_cls=False)[..., None, :]
    return grid_pool(tokens, int(mode.removeprefix("grid")))
