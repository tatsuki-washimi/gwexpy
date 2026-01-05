import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries


def test_timeseries_crop_numeric_bounds():
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, name="demo")
    cropped = ts.crop(2, 5)

    assert isinstance(cropped, TimeSeries)
    assert len(cropped) < len(ts)
    t0 = cropped.t0
    if hasattr(t0, "to_value"):
        unit = getattr(t0, "unit", None)
        t0_val = t0.to_value(u.s) if unit is not None else t0.to_value()
    else:
        t0_val = float(t0)
    assert t0_val == pytest.approx(2.0)


def test_timeseries_to_from_pandas_roundtrip():
    pytest.importorskip("pandas")

    ts = TimeSeries(np.arange(3.0), t0=0, dt=1, name="demo")
    series = ts.to_pandas(index="seconds")

    restored = TimeSeries.from_pandas(series, t0=ts.t0, dt=ts.dt, unit=ts.unit)
    assert isinstance(restored, TimeSeries)
    assert len(restored) == len(ts)
