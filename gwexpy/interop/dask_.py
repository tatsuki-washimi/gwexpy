from ._optional import require_optional


def to_dask(ts, chunks="auto"):
    """
    Convert TimeSeries to dask.array.Array.
    """
    da = require_optional("dask.array")
    return da.from_array(ts.value, chunks=chunks)


def from_dask(cls, array, t0, dt, unit=None, compute=True):
    """
    Create TimeSeries from dask array.

    Parameters
    ----------
    compute : bool
        If True, compute the array to numpy immediately.
        If False, TimeSeries will hold the dask array (if underlying class supports it,
        gwpy TimeSeries usually expects numpy, so caution).
        Default True for safety.
    """
    if compute:
        data = array.compute()
    else:
        data = array

    return cls(data, t0=t0, dt=dt, unit=unit)
