"""Action-conditioned latent dynamics (SPEC.md §4.2).

A small transformer that takes the last `k` frames of latent context plus an action chunk of
length `H`, and predicts the next `H` frames of latent.

Two design points that are easy to get subtly wrong, so they are structural here:

- **The model works in LayerNormed latent space.** §4.2 puts the loss on LayerNormed latents;
  targets are normalised once when the windows are built (`cfe/dynamics/data.py`) and the model
  predicts directly in that space. That keeps the autoregressive feedback consistent — the
  model consumes exactly the kind of vector it produces — and it means Phase 3's
  `D(z_end, z_goal)` must be measured in the same space to be optimising what was trained.
- **The action-ignoring control zeroes actions inside `forward`.** §4.2 calls the comparison
  "the most important diagnostic in the repo", and a control that accidentally sees actions at
  evaluation time would quietly invalidate it. Making it an attribute of the module rather than
  something the training loop remembers to do means it cannot be forgotten in one of the two
  places.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from cfe.encoders.dinov2 import REDUCTIONS

ACTION_DIM = 7


def tokens_per_frame(reduction: str, n_tokens: int = 257) -> int:
    """How many tokens one frame becomes under `reduction`."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"unknown reduction {reduction!r}; expected one of {REDUCTIONS}")
    if reduction == "full":
        return n_tokens
    if reduction in ("cls", "mean"):
        return 1
    grid = int(reduction.removeprefix("grid"))
    return grid * grid + 1


@dataclass(frozen=True)
class DynamicsConfig:
    k_context: int
    horizon: int
    reduction: str
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    ffn_mult: int
    latent_dim: int
    residual: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> DynamicsConfig:
        return cls(
            k_context=int(d["k_context"]),
            horizon=int(d["horizon"]),
            reduction=str(d["reduction"]),
            d_model=int(d["d_model"]),
            n_layers=int(d["n_layers"]),
            n_heads=int(d["n_heads"]),
            dropout=float(d["dropout"]),
            ffn_mult=int(d["ffn_mult"]),
            latent_dim=int(d["latent_dim"]),
            # Defaults False so run dirs written before this field existed (`p2-dyn`) still load
            # and still describe the architecture that actually produced their numbers.
            residual=bool(d.get("residual", False)),
        )

    @property
    def r(self) -> int:
        return tokens_per_frame(self.reduction)


class LatentDynamics(nn.Module):
    """`(context latents, action chunk) -> next H frames of latent`.

    Encoder-only: the `H*R` output positions are learned query tokens appended to the sequence,
    so one forward pass produces the whole chunk. No causal mask — the chunk is predicted
    jointly, not token by token, which is what §4.2's "output: predicted latents for the next H
    steps" describes and is far cheaper than autoregressive decoding inside a chunk.
    """

    def __init__(self, cfg: DynamicsConfig, ignore_actions: bool = False):
        super().__init__()
        self.cfg = cfg
        self.ignore_actions = ignore_actions
        r, d = cfg.r, cfg.d_model

        self.latent_in = nn.Linear(cfg.latent_dim, d)
        self.action_in = nn.Linear(ACTION_DIM, d)
        self.pos_context = nn.Parameter(torch.zeros(cfg.k_context * r, d))
        self.pos_action = nn.Parameter(torch.zeros(cfg.horizon, d))
        self.queries = nn.Parameter(torch.zeros(cfg.horizon * r, d))

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=d * cfg.ffn_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and torch warns about it on
        # every construction; there is no padding here for it to help with anyway.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d, cfg.latent_dim)

        for p in (self.pos_context, self.pos_action, self.queries):
            nn.init.normal_(p, std=0.02)

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """`context (B, k, R, D)`, `actions (B, H, 7)` -> `(B, H, R, D)`."""
        cfg = self.cfg
        b = context.shape[0]
        if context.shape[1:] != (cfg.k_context, cfg.r, cfg.latent_dim):
            raise ValueError(
                f"expected context (B, {cfg.k_context}, {cfg.r}, {cfg.latent_dim}), "
                f"got {tuple(context.shape)}"
            )
        if actions.shape[1:] != (cfg.horizon, ACTION_DIM):
            raise ValueError(
                f"expected actions (B, {cfg.horizon}, {ACTION_DIM}), got {tuple(actions.shape)}"
            )

        if self.ignore_actions:
            # The control. Zeroed here so it holds at eval as well as during training.
            actions = torch.zeros_like(actions)

        ctx = self.latent_in(context).reshape(b, cfg.k_context * cfg.r, cfg.d_model)
        ctx = ctx + self.pos_context
        act = self.action_in(actions) + self.pos_action
        qry = self.queries.expand(b, -1, -1)

        out = self.encoder(torch.cat([ctx, act, qry], dim=1))
        pred = self.head(out[:, -cfg.horizon * cfg.r :])
        pred = pred.reshape(b, cfg.horizon, cfg.r, cfg.latent_dim)

        if cfg.residual:
            # Predict a delta off the last context frame rather than the whole latent. Measured,
            # not assumed: at matched scale and matched step budget this is worth +22.4% at 600
            # steps and +14.5% at 2400, so it is not merely faster convergence (PROGRESS.md
            # 2026-08-01). It also starts the model at copy-last-latent, which is the baseline
            # `beats_copy_last_latent` compares against and which the non-residual model lost to
            # at short horizons. Broadcasts (B, 1, R, D) across the H predicted frames.
            pred = pred + context[:, -1:]
        return pred

    def rollout(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Autoregressive open-loop rollout.

        `actions (B, n_chunks*H, 7)` -> `(B, n_chunks*H, R, D)`. Each chunk after the first is
        conditioned on the model's own predictions, which is the regime the counterfactual
        search will actually run in.
        """
        cfg = self.cfg
        if actions.shape[1] % cfg.horizon:
            raise ValueError(f"{actions.shape[1]} actions is not a multiple of H={cfg.horizon}")
        if cfg.horizon < cfg.k_context:
            raise ValueError(
                f"rollout needs H >= k to refill the context from predictions alone; "
                f"got H={cfg.horizon}, k={cfg.k_context}"
            )
        chunks = []
        for i in range(actions.shape[1] // cfg.horizon):
            pred = self(context, actions[:, i * cfg.horizon : (i + 1) * cfg.horizon])
            chunks.append(pred)
            context = pred[:, -cfg.k_context :]
        return torch.cat(chunks, dim=1)


def chunk_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "smooth_l1") -> torch.Tensor:
    """SPEC.md §4.2: L1 or smooth-L1 on LayerNormed latents.

    Both arguments are already in LayerNormed space — targets by construction, predictions
    because that is what the model is trained to emit. Re-normalising the prediction here would
    make the loss blind to its scale, which is supervision we want to keep.
    """
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if kind == "l1":
        return F.l1_loss(pred, target)
    raise ValueError(f"unknown loss {kind!r}; expected 'smooth_l1' or 'l1'")


def rollout_loss(
    model: LatentDynamics,
    context: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    ar_chunks: int,
    kind: str = "smooth_l1",
) -> tuple[torch.Tensor, list[float]]:
    """Teacher-forced first chunk plus `ar_chunks - 1` autoregressive chunks.

    §4.2 asks for "teacher-forced one-step loss plus a short autoregressive rollout loss (2
    steps)". Read here as *chunk*-level: chunk 1 is conditioned on real latents, and each later
    chunk is conditioned on the model's own output. The alternative reading (two single-frame
    autoregressive steps) trains nothing about compounding error across a whole chunk, which is
    the regime §5's search runs in. The interpretation is recorded in PROGRESS.md.

    Chunks are weighted equally: the later ones are harder, and down-weighting them would train
    away from exactly the compounding error this term exists to blunt.
    """
    h = model.cfg.horizon
    per_chunk, total = [], 0.0
    cur = context
    for i in range(ar_chunks):
        sl = slice(i * h, (i + 1) * h)
        pred = model(cur, actions[:, sl])
        loss = chunk_loss(pred, targets[:, sl], kind)
        per_chunk.append(loss)
        total = total + loss
        cur = pred[:, -model.cfg.k_context :]
    return total / ar_chunks, [float(x.detach()) for x in per_chunk]
