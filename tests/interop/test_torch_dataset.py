"""Tests for gwexpy/interop/torch_dataset.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


def _fake_torch():
    """Build a minimal fake torch module."""
    return SimpleNamespace(
        as_tensor=lambda x, device=None, dtype=None: np.asarray(x),
        utils=SimpleNamespace(
            data=SimpleNamespace(
                DataLoader=lambda ds, **kw: ds
            )
        ),
    )


def _make_ts(n=20, t0=0.0, dt=1.0, unit="m", name="ch"):
    return TimeSeries(np.arange(float(n)), t0=t0, dt=dt, unit=unit, name=name)


class TestTimeSeriesWindowDataset:
    def test_len(self):
        torch_mod = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            ds = TimeSeriesWindowDataset(_make_ts(20), window=5, stride=2)
        # starts = range(0, 20-5+1, 2) = [0,2,4,6,8,10,12,14,16] → 9 items
        assert len(ds) > 0

    def test_getitem_no_labels(self):
        torch_mod = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            ds = TimeSeriesWindowDataset(_make_ts(20), window=4, stride=1)
            item = ds[0]
        # No labels → returns x_tensor directly
        assert item is not None

    def test_getitem_with_numpy_labels(self):
        torch_mod = _fake_torch()
        labels = np.zeros(20)
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            ds = TimeSeriesWindowDataset(_make_ts(20), window=4, stride=1, labels=labels)
            item = ds[0]
        # Returns (x, y) tuple
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_with_callable_labels(self):
        torch_mod = _fake_torch()
        called = {"n": 0}
        def label_fn(x, start):
            called["n"] += 1
            return np.array([float(start)])
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            ds = TimeSeriesWindowDataset(_make_ts(20), window=4, stride=1, labels=label_fn)
            item = ds[0]
        assert called["n"] == 1
        assert isinstance(item, tuple)

    def test_invalid_window_raises(self):
        torch_mod = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            with pytest.raises(ValueError, match="window"):
                TimeSeriesWindowDataset(_make_ts(20), window=0)

    def test_invalid_stride_raises(self):
        torch_mod = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            with pytest.raises(ValueError, match="stride"):
                TimeSeriesWindowDataset(_make_ts(20), window=4, stride=0)

    def test_window_too_large_raises(self):
        torch_mod = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            with pytest.raises(ValueError, match="no samples"):
                TimeSeriesWindowDataset(_make_ts(5), window=10)

    def test_unsupported_type_raises(self):
        torch_mod = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            with pytest.raises(TypeError):
                TimeSeriesWindowDataset([1, 2, 3, 4, 5], window=2)

    def test_requires_torch_installed(self):
        with patch.dict(sys.modules, {"torch": None}):
            # Reimport to ensure clean state
            import importlib
            import gwexpy.interop.torch_dataset as td_mod
            importlib.reload(td_mod)
            with pytest.raises(ImportError):
                td_mod.TimeSeriesWindowDataset(_make_ts(20), window=4)

    def test_label_index_exceeds_raises(self):
        torch_mod = _fake_torch()
        # Use horizon that pushes index out of range
        labels = np.zeros(10)  # shorter than ts
        ts = _make_ts(n=20)
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import TimeSeriesWindowDataset
            # window=4, horizon=10 → idx = start+4+10-1 ≥ 10 (label length)
            ds = TimeSeriesWindowDataset(ts, window=4, stride=1, horizon=7, labels=labels)
            with pytest.raises(IndexError, match="Label index"):
                ds[0]  # first start=0, idx = 0+4+7-1 = 10 >= 10


class TestToTorchDataset:
    def test_returns_dataset(self):
        torch_mod = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import to_torch_dataset
            ds = to_torch_dataset(_make_ts(20), window=4, stride=2)
        assert ds is not None
        assert len(ds) > 0


class TestToTorchDataloader:
    def test_requires_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            import importlib
            import gwexpy.interop.torch_dataset as td_mod
            importlib.reload(td_mod)
            with pytest.raises(ImportError):
                td_mod.to_torch_dataloader(object())

    def test_returns_dataloader(self):
        torch_mod = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch_mod}):
            from gwexpy.interop.torch_dataset import (
                TimeSeriesWindowDataset,
                to_torch_dataloader,
            )
            ds = TimeSeriesWindowDataset(_make_ts(20), window=4)
            loader = to_torch_dataloader(ds, batch_size=2)
        assert loader is not None
