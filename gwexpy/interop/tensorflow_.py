from __future__ import annotations

from ._optional import require_optional


def to_tf(ts, dtype=None):
    """
    Convert TimeSeries to tensorflow.Tensor.
    """
    tf = require_optional("tensorflow")
    from .base import to_plain_array

    return tf.convert_to_tensor(to_plain_array(ts), dtype=dtype)


def from_tf(cls, tensor, t0, dt, unit=None):
    """
    Create TimeSeries from tensorflow.Tensor.
    """
    # Eager execution assumed
    data = tensor.numpy()
    return cls(data, t0=t0, dt=dt, unit=unit)
