"""Encoder and cache invariants (SPEC.md §4.1, §8).

The two that matter and would silently poison results if they broke:
  - no encoder parameter trains
  - a cache hit is bit-identical to a fresh encode

Everything that does not need the 334 MB model on disk is a fast test.
"""

import json

import numpy as np
import pytest
import torch

from cfe.encoders.cache import EncodingCache, Fingerprint, FingerprintMismatch
from cfe.encoders.dinov2 import HASHED_FIELDS, EncoderConfig, pool

ENCODER_YAML = "configs/encoder.yaml"


@pytest.fixture
def enc_cfg(repo_root):
    return EncoderConfig.from_yaml(repo_root / ENCODER_YAML)


@pytest.fixture
def fingerprint(enc_cfg):
    return Fingerprint.build(enc_cfg.hash_payload(), "cpu", renderer="test", obs_source="replay")


# --- config -----------------------------------------------------------------


def test_unknown_encoder_field_is_rejected(enc_cfg):
    """A field that never reaches the hash is exactly how stale cache entries poison runs."""
    d = enc_cfg.hash_payload() | {"brightness": 1.2}
    with pytest.raises(KeyError, match="unknown encoder config fields"):
        EncoderConfig.from_dict(d)


def test_missing_encoder_field_is_rejected(enc_cfg):
    d = enc_cfg.hash_payload()
    del d["image_size"]
    with pytest.raises(KeyError, match="missing required fields"):
        EncoderConfig.from_dict(d)


def test_every_hashed_field_changes_the_digest(enc_cfg, fingerprint):
    """If a field can change what a frame encodes to, it must change the cache key."""
    base = fingerprint.digest()
    bumps = {
        "model_id": "other/model",
        "model_revision": "deadbeef",
        "image_size": 196,
        "flip_vertical": not enc_cfg.flip_vertical,
        "resize_mode": "bilinear",
        "resize_antialias": not enc_cfg.resize_antialias,
        "clamp_after_resize": not enc_cfg.clamp_after_resize,
        "norm_mean": (0.1, 0.2, 0.3),
        "norm_std": (0.4, 0.5, 0.6),
        "output_dtype": "float32",
        "batch_size": enc_cfg.batch_size + 1,
    }
    assert set(bumps) == set(HASHED_FIELDS), "a hashed field is untested here"
    for field, value in bumps.items():
        altered = EncoderConfig.from_dict(enc_cfg.hash_payload() | {field: value})
        fp = Fingerprint.build(altered.hash_payload(), "cpu", renderer="test", obs_source="replay")
        assert fp.digest() != base, f"changing {field} did not change the cache key"


@pytest.mark.parametrize(
    "override",
    [
        {"device": "cuda"},
        {"renderer": "egl:some other driver"},
        {"obs_source": "recorded"},
    ],
    ids=["device_kind", "renderer", "obs_source"],
)
def test_runtime_facts_change_the_digest(enc_cfg, fingerprint, override):
    """Renderer and obs_source go beyond SPEC.md §4.1's cpu/cuda tag — see PROGRESS.md.

    Two boxes that both software-rasterise with different mesa drivers render the same
    simulator state to different pixels, so the device tag alone does not separate machines.
    """
    kwargs = {"device": "cpu", "renderer": "test", "obs_source": "replay"} | override
    other = Fingerprint.build(enc_cfg.hash_payload(), **kwargs)
    assert other.digest() != fingerprint.digest()


# --- cache ------------------------------------------------------------------


def test_cache_round_trip_is_bit_identical(tmp_path, fingerprint):
    cache = EncodingCache(tmp_path, fingerprint)
    tokens = np.random.default_rng(0).standard_normal((7, 257, 768)).astype(np.float16)
    cache.write("demo_0", tokens)
    assert np.array_equal(np.asarray(cache.read("demo_0")), tokens)
    assert np.array_equal(np.asarray(cache.read("demo_0", 3)), tokens[3])
    assert cache.episode_ids() == ["demo_0"]


def test_reopening_with_the_same_fingerprint_does_not_raise(tmp_path, fingerprint):
    """Guards the tuple-vs-list JSON round-trip that would make every reopen a false alarm."""
    EncodingCache(tmp_path, fingerprint).write("demo_0", np.zeros((2, 4, 8), np.float16))
    reopened = EncodingCache(tmp_path, fingerprint)
    assert reopened.read("demo_0").shape == (2, 4, 8)


def test_fingerprint_mismatch_is_a_hard_error(tmp_path, fingerprint):
    """SPEC.md §4.1: a mismatch must raise, never silently recompute."""
    cache = EncodingCache(tmp_path, fingerprint)
    stored = json.loads(cache.fingerprint_path.read_text())
    stored["renderer"] = "a different renderer"
    cache.fingerprint_path.write_text(json.dumps(stored))
    with pytest.raises(FingerprintMismatch, match="renderer"):
        EncodingCache(tmp_path, fingerprint)


def test_missing_episode_raises(tmp_path, fingerprint):
    with pytest.raises(KeyError):
        EncodingCache(tmp_path, fingerprint).read("nope")


def test_write_rejects_wrong_rank(tmp_path, fingerprint):
    with pytest.raises(ValueError, match=r"\(T, N, D\)"):
        EncodingCache(tmp_path, fingerprint).write("demo_0", np.zeros((4, 8), np.float16))


# --- pooling ----------------------------------------------------------------


def test_pool_excludes_cls_by_default():
    tokens = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    assert np.allclose(pool(tokens), tokens[:, 1:, :].mean(axis=1))
    assert np.allclose(pool(tokens, include_cls=True), tokens.mean(axis=1))
    assert not np.allclose(pool(tokens), pool(tokens, include_cls=True))


def test_pool_matches_between_numpy_and_torch():
    tokens = np.random.default_rng(0).standard_normal((3, 6, 4)).astype(np.float32)
    assert np.allclose(pool(tokens), pool(torch.from_numpy(tokens)).numpy(), atol=1e-6)


# --- the model itself (needs the weights on disk) ---------------------------


@pytest.mark.slow
def test_encoder_has_no_trainable_parameters(enc_cfg):
    """SPEC.md §4.1: encoder weights never train."""
    from cfe.encoders import Dinov2Encoder

    encoder = Dinov2Encoder(enc_cfg, device="cpu")
    offenders = [n for n, p in encoder.model.named_parameters() if p.requires_grad]
    assert offenders == [], f"{len(offenders)} encoder parameters require grad: {offenders[:5]}"
    assert encoder.n_params_requiring_grad == 0
    assert not encoder.model.training


@pytest.mark.slow
def test_cache_hit_is_bit_identical_to_a_fresh_encode(tmp_path, enc_cfg):
    """SPEC.md §8. Bit-exact, no tolerance — the encoder returns output_dtype directly."""
    from cfe.encoders import Dinov2Encoder

    encoder = Dinov2Encoder(enc_cfg, device="cpu")
    frames = np.random.default_rng(0).integers(0, 256, (3, 128, 128, 3), dtype=np.uint8)
    fresh = encoder.encode(frames)

    fp = Fingerprint.build(enc_cfg.hash_payload(), "cpu", renderer="test", obs_source="replay")
    cache = EncodingCache(tmp_path, fp)
    cache.write("demo_0", fresh)
    assert np.array_equal(np.asarray(cache.read("demo_0")), fresh)
    assert cache.read("demo_0").dtype == np.dtype(enc_cfg.output_dtype)


@pytest.mark.slow
def test_encode_is_deterministic(enc_cfg):
    from cfe.encoders import Dinov2Encoder

    encoder = Dinov2Encoder(enc_cfg, device="cpu")
    frames = np.random.default_rng(1).integers(0, 256, (2, 128, 128, 3), dtype=np.uint8)
    assert np.array_equal(encoder.encode(frames), encoder.encode(frames))


@pytest.mark.slow
def test_flip_vertical_actually_changes_the_encoding(enc_cfg):
    """MuJoCo renders bottom-up; the flip must be real, not a no-op config field."""
    from cfe.encoders import Dinov2Encoder

    frames = np.random.default_rng(2).integers(0, 256, (1, 128, 128, 3), dtype=np.uint8)
    flipped = Dinov2Encoder(enc_cfg, device="cpu").preprocess(frames)
    unflipped_cfg = EncoderConfig.from_dict(enc_cfg.hash_payload() | {"flip_vertical": False})
    unflipped = Dinov2Encoder(unflipped_cfg, device="cpu").preprocess(frames)
    # The flip must have a large effect...
    assert not torch.equal(flipped, unflipped)
    assert (flipped - unflipped).abs().max() > 1.0

    # ...and must be exactly a vertical flip, up to interpolation numerics. float32 bicubic
    # resize is not bit-symmetric under flipping (summation order differs): measured max
    # deviation 5.9e-5 against a tensor std of 1.05, while the flip itself moves values by
    # 4.44. 1e-4 sits ~44,000x below the effect being tested, so this cannot pass vacuously.
    assert torch.allclose(flipped, unflipped.flip(-2), atol=1e-4)
