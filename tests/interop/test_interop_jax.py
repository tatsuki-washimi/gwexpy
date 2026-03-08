"""Tests for JAX interop adapter."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from gwexpy.interop.jax_ import from_jax, to_jax
from gwexpy.timeseries import TimeSeries


def _make_ts(n=100):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=1000000000.0, dt=0.01, unit="m", name="test",
    )


class TestToJax:
    def test_returns_jax_array(self):
        ts = _make_ts()
        arr = to_jax(ts)
        assert hasattr(arr, "shape")
        assert arr.shape == (100,)

    def test_values_preserved(self):
        ts = _make_ts()
        arr = to_jax(ts)
        np.testing.assert_allclose(np.array(arr), ts.value)

    def test_dtype_override(self):
        ts = _make_ts()
        arr = to_jax(ts, dtype=jax.numpy.float32)
        assert arr.dtype == jax.numpy.float32


class TestFromJax:
    def test_roundtrip(self):
        ts = _make_ts()
        arr = to_jax(ts)
        ts2 = from_jax(TimeSeries, arr, t0=ts.t0, dt=ts.dt, unit="m")
        np.testing.assert_allclose(ts2.value, ts.value)

    def test_metadata(self):
        arr = jax.numpy.ones(50)
        ts = from_jax(TimeSeries, arr, t0=99.0, dt=0.5, unit="V")
        assert np.isclose(ts.t0.value, 99.0)
        assert np.isclose(ts.dt.value, 0.5)
        assert str(ts.unit) == "V"
