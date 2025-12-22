from typing import Callable, Optional, Union, TYPE_CHECKING

import numpy as np

from ._optional import require_optional

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix, TimeSeriesDict, TimeSeriesList


class TimeSeriesWindowDataset:
    """
    Simple windowed Dataset wrapper for torch training loops.
    """

    def __init__(
        self,
        series,
        *,
        window: int,
        stride: int = 1,
        horizon: int = 0,
        labels: Optional[Union["TimeSeries", "TimeSeriesMatrix", np.ndarray, Callable]] = None,
        multivariate: bool = False,
        align: str = "intersection",
        device=None,
        dtype=None,
    ):
        torch = require_optional("torch")
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.window = int(window)
        self.stride = int(stride)
        self.horizon = int(horizon)
        if self.window <= 0 or self.stride <= 0:
            raise ValueError("window and stride must be positive integers.")

        from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix, TimeSeriesDict, TimeSeriesList

        from .base import to_plain_array

        data_obj = series
        if multivariate and isinstance(series, (TimeSeriesDict, TimeSeriesList)):
            data_obj = series.to_matrix(align=align)

        if isinstance(data_obj, TimeSeriesMatrix):
            self.t0 = data_obj.t0
            self.dt = data_obj.dt
            vals = to_plain_array(data_obj)
            self._feature_names = getattr(data_obj, "channel_names", None)
            self.data = vals.reshape(-1, vals.shape[-1])
            self.unit = None
        elif isinstance(data_obj, TimeSeries):
            self.t0 = data_obj.t0
            self.dt = data_obj.dt
            self.data = to_plain_array(data_obj)[None, :]
            self.unit = getattr(data_obj, "unit", None)
            self._feature_names = [data_obj.name] if getattr(data_obj, "name", None) else None
        else:
            raise TypeError(f"Unsupported type for TimeSeriesWindowDataset: {type(data_obj)}")

        self.labels = labels
        if isinstance(labels, (TimeSeries, TimeSeriesMatrix)):
            label_vals = to_plain_array(labels)
            self.label_array = label_vals.reshape(-1, label_vals.shape[-1])
        elif isinstance(labels, np.ndarray):
            arr = labels
            self.label_array = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr[None, :]
        else:
            self.label_array = None

        max_start = self.data.shape[-1] - self.window - self.horizon + 1
        if max_start <= 0:
            raise ValueError("window/horizon configuration yields no samples.")
        self.starts = list(range(0, max_start, self.stride))

    def __len__(self):
        return len(self.starts)

    def _slice_x(self, start: int):
        end = start + self.window
        x_np = self.data[:, start:end]
        return self.torch.as_tensor(x_np, device=self.device, dtype=self.dtype)

    def _slice_label(self, start: int, x_tensor):
        if self.labels is None:
            return None
        if callable(self.labels):
            return self.labels(x_tensor, start)
        if self.label_array is None:
            return None
        idx = start + self.window + self.horizon - 1
        if idx >= self.label_array.shape[-1]:
            raise IndexError("Label index exceeds label array length.")
        y_np = self.label_array[:, idx]
        return self.torch.as_tensor(y_np, device=self.device, dtype=self.dtype)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        x_tensor = self._slice_x(start)
        y = self._slice_label(start, x_tensor)
        return (x_tensor, y) if self.labels is not None else x_tensor


def to_torch_dataset(
    obj,
    *,
    window: int,
    stride: int = 1,
    horizon: int = 0,
    labels: Optional[Union["TimeSeries", "TimeSeriesMatrix", np.ndarray, Callable]] = None,
    multivariate: bool = False,
    align: str = "intersection",
    device=None,
    dtype=None,
):
    """
    Convenience wrapper to build a TimeSeriesWindowDataset.
    """
    return TimeSeriesWindowDataset(
        obj,
        window=window,
        stride=stride,
        horizon=horizon,
        labels=labels,
        multivariate=multivariate,
        align=align,
        device=device,
        dtype=dtype,
    )


def to_torch_dataloader(dataset, *, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, **kwargs):
    """
    Create a torch DataLoader from the provided dataset.
    """
    torch = require_optional("torch")
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
