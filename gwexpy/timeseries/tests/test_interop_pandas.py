
import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

# Optional dependencies check
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_pandas_interop():
    t0 = LIGOTimeGPS(1234567890, 123000000)
    dt = 0.01 * u.s
    data = np.arange(100)
    ts = TimeSeries(data, t0=t0, dt=dt, name="test_ts", unit="V")

    # 1. to_pandas (datetime)
    s_dt = ts.to_pandas(index="datetime")
    assert isinstance(s_dt, pd.Series)
    assert s_dt.index.tz is not None # aware
    assert len(s_dt) == 100
    assert s_dt.name == "test_ts"

    # 2. from_pandas
    ts_restored = TimeSeries.from_pandas(s_dt, unit="V")
    # allow small precision difference due to float64/datetime conversions
    assert np.isclose(ts_restored.t0.value, float(t0), atol=1e-6)
    np.testing.assert_allclose(ts_restored.dt.value, dt.value, atol=1e-6)
    np.testing.assert_array_equal(ts_restored.value, data)

    # 3. TimeSeriesDict -> DataFrame
    tsd = TimeSeriesDict()
    tsd["ch1"] = ts
    tsd["ch2"] = TimeSeries(data * 2, t0=t0, dt=dt, name="ch2")

    df = tsd.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 2)
    assert list(df.columns) == ["ch1", "ch2"]

    # 4. DataFrame -> TimeSeriesDict
    tsd_restored = TimeSeriesDict.from_pandas(df)
    assert "ch1" in tsd_restored
    assert "ch2" in tsd_restored

