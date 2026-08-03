"""Train the action-conditioned latent dynamics model and its action-ignoring twin.

Two models, identical config and identical seed, differing only in whether actions reach the
network. SPEC.md §4.2 describes the control as "same model with actions zeroed"; training a
twin *blind* is the fair reading of that question — "could a model that never sees actions do
as well?" — whereas zeroing actions at evaluation time on an action-trained model only measures
how far out of distribution that puts it. Both numbers are reported by `verify_g2.py`; the twin
is the one that gates. SPEC.md §6's `action_ignoring_dynamics` baseline needs this trained twin
regardless, so it is not extra work.

Data is demos *plus* rollouts, successes and failures, per §4.2.

Usage:
    python scripts/train_dynamics.py --config configs/dynamics.yaml --seed 0 --run-id p2-dyn
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train latent dynamics + action-ignoring twin")
    p.add_argument("--config", required=True, type=Path, help="configs/dynamics.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--epochs", type=int, default=None, help="override config, for smoke tests")
    p.add_argument("--limit-episodes", type=int, default=None)
    return p.parse_args()


def open_cache(cfg: dict, seed: int):
    """Open the encoding cache, taking the renderer string from a live GL context."""
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    enc_yaml = yaml.safe_load(enc_path.read_text())
    env_cfg = yaml.safe_load(env_path.read_text())
    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    import torch

    from cfe.data.demos import gl_renderer, make_replay_env
    from cfe.encoders.cache import EncodingCache, Fingerprint
    from cfe.encoders.dinov2 import EncoderConfig
    from scripts.verify_env import load_suite

    data_cfg = enc_yaml["data"]
    enc_cfg = EncoderConfig.from_dict(enc_yaml["encoder"])
    suite = load_suite({"suite": data_cfg["suite"]})
    task = suite.get_task(int(data_cfg["task_id"]))
    env = make_replay_env(env_cfg, task, seed)
    try:
        renderer = gl_renderer()
    finally:
        env.close()

    device_kind = "cuda" if torch.cuda.is_available() else "cpu"
    fp = Fingerprint.build(enc_cfg.hash_payload(), device_kind, renderer, obs_source="replay")
    cache = EncodingCache(enc_yaml["runtime"]["cache_dir"], fp)
    return cache, renderer, enc_path, env_path


def evaluate(model, index, batch_size: int, ar_chunks: int, loss_kind: str, device) -> float:
    """Mean validation loss over every window, in eval mode."""
    import torch

    from cfe.dynamics.model import rollout_loss

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = np.arange(start, min(start + batch_size, len(index)))
            b = index.gather(rows)
            loss, _ = rollout_loss(
                model,
                torch.from_numpy(b["context"]).to(device),
                torch.from_numpy(b["actions"]).to(device),
                torch.from_numpy(b["targets"]).to(device),
                ar_chunks,
                loss_kind,
            )
            total += float(loss) * len(rows)
            n += len(rows)
    return total / max(n, 1)


def train_one(model, train_idx, val_idx, tr: dict, ar_chunks: int, device, seed: int, tag: str):
    """Train one model. Returns (history, seconds)."""
    import torch

    from cfe.dynamics.model import rollout_loss

    # Reseeded per model so the twin sees the same init draw and the same batch order: the
    # comparison between them must not be confounded by a different shuffle.
    torch.manual_seed(seed)
    np.random.seed(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()
    for epoch in range(int(tr["epochs"])):
        model.train()
        order = rng.permutation(len(train_idx))
        total, n = 0.0, 0
        for start in range(0, len(order), int(tr["batch_size"])):
            rows = order[start : start + int(tr["batch_size"])]
            b = train_idx.gather(rows)
            loss, _ = rollout_loss(
                model,
                torch.from_numpy(b["context"]).to(device),
                torch.from_numpy(b["actions"]).to(device),
                torch.from_numpy(b["targets"]).to(device),
                ar_chunks,
                tr["loss"],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
            opt.step()
            total += float(loss.detach()) * len(rows)
            n += len(rows)
        val = evaluate(model, val_idx, int(tr["batch_size"]), ar_chunks, tr["loss"], device)
        rec = {
            "model": tag,
            "epoch": epoch,
            "train_loss": round(total / max(n, 1), 6),
            "val_loss": round(val, 6),
        }
        history.append(rec)
        print(f"  [{tag:16s}] epoch {epoch:3d}  train {rec['train_loss']:.6f}  val {val:.6f}")
    return history, time.time() - t0


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run_id = args.run_id or f"p2-dyn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    cache, renderer, enc_path, env_path = open_cache(cfg, args.seed)

    import torch

    from cfe.dynamics.data import WindowIndex, load_episode, split_by_episode
    from cfe.dynamics.model import DynamicsConfig, LatentDynamics

    dyn_cfg = DynamicsConfig.from_dict(cfg["dynamics"])
    tr = dict(cfg["train"])
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    ar_chunks = int(tr["ar_chunks"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ids = cache.episode_ids()
    if args.limit_episodes:
        ids = ids[: args.limit_episodes]
    n_demo = sum(1 for e in ids if "_demo_" in e)
    n_roll = len(ids) - n_demo

    print(f"train_dynamics  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"cache={cache.root}")
    print(f"episodes: {len(ids)} ({n_demo} demo, {n_roll} rollout)")
    print(f"reduction={dyn_cfg.reduction} -> R={dyn_cfg.r} tokens/frame")
    print("-" * 78)

    t0 = time.time()
    train_ids, val_ids = split_by_episode(ids, float(tr["val_fraction"]), args.seed)
    train_eps = [load_episode(cache, e, dyn_cfg.reduction) for e in train_ids]
    val_eps = [load_episode(cache, e, dyn_cfg.reduction) for e in val_ids]
    mb = sum(e.z.nbytes for e in train_eps + val_eps) / 1e6
    print(f"loaded {len(train_eps)} train / {len(val_eps)} val episodes in {time.time() - t0:.1f}s")
    print(f"reduced latents in RAM: {mb:.0f} MB")

    train_idx = WindowIndex(train_eps, dyn_cfg.k_context, dyn_cfg.horizon, ar_chunks)
    val_idx = WindowIndex(val_eps, dyn_cfg.k_context, dyn_cfg.horizon, ar_chunks)
    n_frames = int(sum(len(e.z) for e in train_eps + val_eps))
    print(f"windows: {len(train_idx)} train / {len(val_idx)} val   frames: {n_frames}")
    if not len(train_idx) or not len(val_idx):
        raise RuntimeError("no usable windows; episodes are shorter than k + ar_chunks*H")
    print("-" * 78)

    models, histories, seconds = {}, {}, {}
    for tag, ignore in (("dynamics", False), ("action_ignoring", True)):
        model = LatentDynamics(dyn_cfg, ignore_actions=ignore).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{tag}: {n_params:,} params  ignore_actions={ignore}")
        hist, secs = train_one(model, train_idx, val_idx, tr, ar_chunks, device, args.seed, tag)
        models[tag], histories[tag], seconds[tag] = model, hist, secs
        print(f"{tag} done in {secs:.1f}s")
        print("-" * 78)

    summary = {
        "run_id": run_id,
        "seed": args.seed,
        "device": str(device),
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "reduction": dyn_cfg.reduction,
        "tokens_per_frame": dyn_cfg.r,
        "k_context": dyn_cfg.k_context,
        "horizon": dyn_cfg.horizon,
        "ar_chunks": ar_chunks,
        "n_episodes": len(ids),
        "n_demo_episodes": n_demo,
        "n_rollout_episodes": n_roll,
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_train_windows": len(train_idx),
        "n_val_windows": len(val_idx),
        "n_params": int(sum(p.numel() for p in models["dynamics"].parameters())),
        "latents_ram_mb": round(mb, 1),
        "final_val_loss": {k: histories[k][-1]["val_loss"] for k in histories},
        "train_seconds": {k: round(v, 1) for k, v in seconds.items()},
    }

    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "dynamics.yaml")
    shutil.copyfile(enc_path, run_dir / "encoder.yaml")
    shutil.copyfile(env_path, run_dir / "env.yaml")
    torch.save(
        {
            "dynamics": models["dynamics"].state_dict(),
            "action_ignoring": models["action_ignoring"].state_dict(),
            "dynamics_config": cfg["dynamics"],
            "train_config": tr,
            "val_episode_ids": val_ids,
            "train_episode_ids": train_ids,
        },
        run_dir / "dynamics.pt",
    )
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for tag in histories:
            for rec in histories[tag]:
                fh.write(json.dumps({"run_id": run_id, "kind": "epoch", **rec}) + "\n")
        fh.write(json.dumps({"run_id": run_id, "kind": "summary", **summary}) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"artifacts: {run_dir}")
    print("\nOpen-loop error and the G2 verdict come from: make g2-check DYN_RUN=" + run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
