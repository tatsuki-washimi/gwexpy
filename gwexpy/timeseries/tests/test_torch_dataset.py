import importlib

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries
from gwexpy.interop import torch_dataset


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_torch_window_dataset_shapes():
    torch = importlib.import_module("torch")
    ts = TimeSeries(np.arange(10, dtype=float), dt=0.1)
    labels = TimeSeries(np.arange(10, dtype=float) * 2, dt=0.1)

    ds = torch_dataset.TimeSeriesWindowDataset(ts, window=4, stride=2, horizon=1, labels=labels)
    assert len(ds) == 3

    x, y = ds[1]
    assert x.shape == (1, 4)
    assert y.shape == (1,)
    expected_label = labels.value[2 + 4 + 1 - 1]
    assert pytest.approx(y.item()) == expected_label

    loader = torch_dataset.to_torch_dataloader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch[0].shape[0] == 2


def test_torch_dataset_importerror(monkeypatch):
    ts = TimeSeries(np.arange(5, dtype=float), dt=1.0)

    def fake_require(name):
        raise ImportError("The 'torch' package is required for this feature but is not installed.")

    monkeypatch.setattr(torch_dataset, "require_optional", fake_require)
    with pytest.raises(ImportError):
        _ = torch_dataset.TimeSeriesWindowDataset(ts, window=2)
