"""Frozen visual encoder wrappers (SPEC.md §4.1)."""

from cfe.encoders.dinov2 import Dinov2Encoder, EncoderConfig, pool

__all__ = ["Dinov2Encoder", "EncoderConfig", "pool"]
