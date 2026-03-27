"""Tests for gwexpy/interop/cupy_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gwexpy.interop.cupy_ import is_cupy_available, to_cupy, from_cupy
from gwexpy.timeseries import TimeSeries


class TestIsCupyAvailable:
    def test_returns_false_when_cupy_not_installed(self):
        with patch.dict(sys.modules, {"cupy": None}):
            result = is_cupy_available()
        assert result is False

    def test_returns_false_when_runtime_error(self):
        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(
                    getDeviceCount=lambda: (_ for _ in ()).throw(RuntimeError("no CUDA"))
                )
            )
        )
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            result = is_cupy_available()
        assert result is False

    def test_returns_false_when_attribute_error(self):
        fake_cupy = SimpleNamespace()  # no .cuda attribute
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            result = is_cupy_available()
        assert result is False

    def test_returns_true_when_device_available(self):
        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=lambda: 1)
            )
        )
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            result = is_cupy_available()
        assert result is True

    def test_returns_false_when_no_devices(self):
        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=lambda: 0)
            )
        )
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            result = is_cupy_available()
        assert result is False


class TestToCupyRequiresPackage:
    def test_raises_import_error_when_cupy_not_installed(self):
        with patch.dict(sys.modules, {"cupy": None}):
            with pytest.raises(ImportError):
                to_cupy([1.0, 2.0, 3.0])

    def test_cuda_driver_error_reraised_as_runtime(self):
        import numpy as np
        fake_cupy = SimpleNamespace(
            asarray=lambda x, dtype=None: (_ for _ in ()).throw(
                RuntimeError("cudaErrorInsufficientDriver")
            )
        )
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            with pytest.raises(RuntimeError, match="CuPy is installed"):
                to_cupy(np.ones(3))

    def test_other_runtime_error_reraised(self):
        import numpy as np
        fake_cupy = SimpleNamespace(
            asarray=lambda x, dtype=None: (_ for _ in ()).throw(
                RuntimeError("some other error")
            )
        )
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            with pytest.raises(RuntimeError, match="some other error"):
                to_cupy(np.ones(3))

    def test_converts_successfully_with_mock(self):
        import numpy as np
        arr = np.ones(4)
        fake_cupy = SimpleNamespace(asarray=lambda x, dtype=None: x)
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            result = to_cupy(arr)
        assert result is arr


class TestFromCupyRequiresPackage:
    def test_raises_import_error_when_cupy_not_installed(self):
        with patch.dict(sys.modules, {"cupy": None}):
            with pytest.raises(ImportError):
                from_cupy(TimeSeries, [1.0, 2.0], t0=0, dt=1)

    def test_converts_successfully_with_mock(self):
        import numpy as np
        arr = np.ones(5)
        fake_cupy = SimpleNamespace(asnumpy=lambda x: x)
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            ts = from_cupy(TimeSeries, arr, t0=0.0, dt=1.0, unit="m")
        assert len(ts) == 5
        assert str(ts.unit) == "m"
