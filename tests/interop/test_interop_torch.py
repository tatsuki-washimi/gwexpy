"""Tests for PyTorch interop adapter."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gwexpy.interop.torch_ import from_torch, to_torch
from gwexpy.timeseries import TimeSeries


def _make_ts(n=100):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=1000000000.0, dt=0.01, unit="m", name="test",
    )


class TestToTorch:
    def test_returns_tensor(self):
        ts = _make_ts()
        t = to_torch(ts)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (100,)

    def test_values_preserved(self):
        ts = _make_ts()
        t = to_torch(ts)
        np.testing.assert_allclose(t.numpy(), ts.value)

    def test_dtype_override(self):
        ts = _make_ts()
        t = to_torch(ts, dtype=torch.float32)
        assert t.dtype == torch.float32

    def test_requires_grad(self):
        ts = _make_ts()
        t = to_torch(ts, requires_grad=True, dtype=torch.float64)
        assert t.requires_grad

    def test_copy_flag(self):
        ts = _make_ts()
        t = to_torch(ts, copy=True)
        assert isinstance(t, torch.Tensor)


class TestFromTorch:
    def test_roundtrip(self):
        ts = _make_ts()
        t = to_torch(ts, dtype=torch.float64)
        ts2 = from_torch(TimeSeries, t, t0=ts.t0, dt=ts.dt, unit="m")
        np.testing.assert_allclose(ts2.value, ts.value)

    def test_metadata(self):
        t = torch.randn(50, dtype=torch.float64)
        ts = from_torch(TimeSeries, t, t0=12345.0, dt=0.002, unit="V")
        assert np.isclose(ts.t0.value, 12345.0)
        assert np.isclose(ts.dt.value, 0.002)
        assert str(ts.unit) == "V"
