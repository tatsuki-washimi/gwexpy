"""Tests for gwexpy/interop/torch_.py and gwexpy/interop/tensorflow_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


def _make_ts(n=8, t0=0.0, dt=1.0, unit="m"):
    return TimeSeries(np.arange(float(n)), t0=t0, dt=dt, unit=unit)


# ============================================================
# torch_
# ============================================================

class TestToTorchRequiresPackage:
    def test_raises_import_error_without_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            from gwexpy.interop.torch_ import to_torch
            with pytest.raises(ImportError):
                to_torch(_make_ts())


class TestToTorchWithMock:
    def _fake_torch(self):
        class FakeTensor:
            def __init__(self, data, **kw):
                self._data = np.asarray(data)
                self.requires_grad = False

            def requires_grad_(self, val):
                self.requires_grad = val
                return self

            def numpy(self):
                return self._data

        return SimpleNamespace(
            tensor=lambda data, device=None, dtype=None, requires_grad=False: FakeTensor(data),
            as_tensor=lambda data, device=None, dtype=None: FakeTensor(data),
        )

    def test_copy_false_uses_as_tensor(self):
        torch_mod = self._fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop import torch_
            import importlib
            importlib.reload(torch_)
            ts = _make_ts()
            result = torch_.to_torch(ts, copy=False)
        assert result is not None

    def test_copy_true_uses_tensor(self):
        torch_mod = self._fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop import torch_
            import importlib
            importlib.reload(torch_)
            ts = _make_ts()
            result = torch_.to_torch(ts, copy=True)
        assert result is not None

    def test_requires_grad_sets_flag(self):
        torch_mod = self._fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop import torch_
            import importlib
            importlib.reload(torch_)
            ts = _make_ts()
            result = torch_.to_torch(ts, requires_grad=True, copy=False)
        assert result.requires_grad is True


class TestFromTorch:
    def _make_tensor(self, data):
        class FakeTensor:
            def __init__(self, d):
                self._data = np.asarray(d)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._data

        return FakeTensor(data)

    def test_basic_conversion(self):
        from gwexpy.interop.torch_ import from_torch
        tensor = self._make_tensor(np.arange(5.0))
        ts = from_torch(TimeSeries, tensor, t0=0.0, dt=1.0, unit="m")
        assert isinstance(ts, TimeSeries)
        assert len(ts) == 5

    def test_t0_dt_unit_preserved(self):
        from gwexpy.interop.torch_ import from_torch
        tensor = self._make_tensor(np.ones(4))
        ts = from_torch(TimeSeries, tensor, t0=10.0, dt=0.5, unit="s")
        assert ts.t0.value == pytest.approx(10.0)
        assert ts.dt.value == pytest.approx(0.5)
        assert str(ts.unit) == "s"

    def test_resolve_conj_called_if_exists(self):
        from gwexpy.interop.torch_ import from_torch
        resolved = {"conj": False, "neg": False}

        class FakeTensor:
            def detach(self): return self
            def cpu(self): return self
            def resolve_conj(self):
                resolved["conj"] = True
                return self
            def resolve_neg(self):
                resolved["neg"] = True
                return self
            def numpy(self): return np.ones(3)

        ts = from_torch(TimeSeries, FakeTensor(), t0=0.0, dt=1.0)
        assert resolved["conj"] is True
        assert resolved["neg"] is True


# ============================================================
# tensorflow_
# ============================================================

class TestToTfRequiresPackage:
    def test_raises_import_error_without_tensorflow(self):
        with patch.dict(sys.modules, {"tensorflow": None}):
            from gwexpy.interop.tensorflow_ import to_tf
            with pytest.raises(ImportError):
                to_tf(_make_ts())


class TestToTfWithMock:
    def test_basic_conversion(self):
        class FakeTensor:
            def __init__(self, data):
                self._data = np.asarray(data)
            def numpy(self):
                return self._data

        fake_tf = SimpleNamespace(
            convert_to_tensor=lambda data, dtype=None: FakeTensor(data)
        )
        with patch.dict(sys.modules, {"tensorflow": fake_tf}):
            from gwexpy.interop import tensorflow_
            import importlib
            importlib.reload(tensorflow_)
            ts = _make_ts(n=5)
            tensor = tensorflow_.to_tf(ts)
        np.testing.assert_array_equal(tensor.numpy(), ts.value)

    def test_raises_import_error_without_tensorflow(self):
        with patch.dict(sys.modules, {"tensorflow": None}):
            from gwexpy.interop.tensorflow_ import to_tf
            with pytest.raises(ImportError):
                to_tf(_make_ts())


class TestFromTf:
    def test_basic_conversion(self):
        from gwexpy.interop.tensorflow_ import from_tf
        fake_tensor = SimpleNamespace(numpy=lambda: np.arange(5.0))
        ts = from_tf(TimeSeries, fake_tensor, t0=0.0, dt=1.0)
        assert isinstance(ts, TimeSeries)
        assert len(ts) == 5

    def test_t0_dt_unit(self):
        from gwexpy.interop.tensorflow_ import from_tf
        fake_tensor = SimpleNamespace(numpy=lambda: np.ones(3))
        ts = from_tf(TimeSeries, fake_tensor, t0=5.0, dt=0.25, unit="V")
        assert ts.t0.value == pytest.approx(5.0)
        assert ts.dt.value == pytest.approx(0.25)
        assert str(ts.unit) == "V"
