"""TensorFlow interop helpers."""

from __future__ import annotations

from ._optional import require_optional

__all__ = ["to_tf", "from_tf"]


def to_tf(ts, dtype=None):
    """Convert a time series to a TensorFlow tensor."""
    tf = require_optional("tensorflow")
    from .base import to_plain_array

    return tf.convert_to_tensor(to_plain_array(ts), dtype=dtype)


def from_tf(cls, tensor, t0, dt, unit=None):
    """Create a time series from a TensorFlow tensor."""
    # Eager execution assumed
    data = tensor.numpy()
    return cls(data, t0=t0, dt=dt, unit=unit)
