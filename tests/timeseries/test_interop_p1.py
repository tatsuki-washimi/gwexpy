
import numpy as np
import pytest
from gwexpy.timeseries import TimeSeries

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import zarr
except ImportError:
    zarr = None

# Import order matters: dask -> torch/tf avoids segfaults in some envs.
try:
    import torch  # noqa: F401 - availability check
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_interop():
    data = np.arange(10, dtype=np.float32)
    ts = TimeSeries(data, dt=0.01)

    # 1. to_torch
    tensor = ts.to_torch()
    assert torch.is_tensor(tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.numpy(), data)

    # Check copy=False (sharing memory if possible)
    # Numpy->Torch sharing works if array is writable and aligned.
    # We test functionality.

    # 2. from_torch
    ts2 = TimeSeries.from_torch(tensor, t0=0, dt=0.01)
    assert np.allclose(ts2.value, data)

@pytest.mark.skipif(tf is None, reason="tensorflow not installed")
def test_tf_interop():
    data = np.arange(10, dtype=np.float32)
    ts = TimeSeries(data, dt=0.01)

    # 1. to_tensorflow
    tensor = ts.to_tensorflow()
    assert tf.is_tensor(tensor)
    assert np.allclose(tensor.numpy(), data)

    # 2. from_tensorflow
    ts2 = TimeSeries.from_tensorflow(tensor, t0=0, dt=0.01)
    assert np.allclose(ts2.value, data)

@pytest.mark.skipif(da is None, reason="dask not installed")
def test_dask_interop():
    data = np.arange(100, dtype=float)
    ts = TimeSeries(data, dt=0.1)

    # 1. to_dask
    darr = ts.to_dask(chunks=10)
    assert isinstance(darr, da.Array)
    assert darr.chunks == ((10,)*10,)

    # 2. from_dask (compute=True)
    ts2 = TimeSeries.from_dask(darr, t0=0, dt=0.1, compute=True)
    assert isinstance(ts2.value, np.ndarray)
    assert np.allclose(ts2.value, data)

@pytest.mark.skipif(zarr is None, reason="zarr not installed")
def test_zarr_interop(tmp_path):
    data = np.arange(100, dtype=float)
    ts = TimeSeries(data, dt=0.1, name="zts")

    store_path = str(tmp_path / "test.zarr")

    # 1. to_zarr
    ts.to_zarr(store_path, "my_array", overwrite=True)

    # 2. from_zarr
    ts2 = TimeSeries.from_zarr(store_path, "my_array")
    assert np.allclose(ts2.value, data)
    assert ts2.dt.value == 0.1
    # Check if name persisted (if zarr attrs supported)
    assert ts2.name == "zts"
