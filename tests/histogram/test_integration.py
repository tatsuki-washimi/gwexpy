import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.histogram import Histogram, HistogramDict
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def test_timeseries_histogram():
    data = np.random.normal(0, 1, 1000)
    ts = TimeSeries(data, sample_rate=100, unit="m", name="test")

    h = ts.histogram(bins=10, range=(-5, 5))

    assert isinstance(h, Histogram)
    assert h.unit == u.dimensionless_unscaled
    assert h.xunit == u.m
    assert h.name == "test"
    assert len(h) == 10
    assert h.edges[0].value == -5
    assert h.edges[-1].value == 5


def test_timeseries_histogram_density():
    ts = TimeSeries([1, 2, 3], sample_rate=1, unit="m")
    h = ts.histogram(bins=2, range=(1, 3), density=True)

    assert h.unit == u.dimensionless_unscaled
    assert h.to_density().unit == u.m**-1
    assert pytest.approx(h.integral().value) == 1.0


def test_frequencyseries_histogram():
    fs = FrequencySeries([1, 10, 100], frequencies=[10, 20, 30], unit="m/Hz")
    h = fs.histogram(bins=3, range=(0, 150))

    assert isinstance(h, Histogram)
    assert h.xunit == u.m / u.Hz
    assert h.values[0] == 2  # value 1 and 10
    assert h.values[1] == 0
    assert h.values[2] == 1  # value 100


def test_timeseriesdict_histogram():
    tsd = TimeSeriesDict()
    tsd["A"] = TimeSeries(np.random.normal(0, 1, 100), sample_rate=10)
    tsd["B"] = TimeSeries(np.random.normal(5, 1, 100), sample_rate=10)

    h_dict = tsd.histogram(bins=10, range=(-5, 10))

    assert isinstance(h_dict, HistogramDict)
    assert "A" in h_dict
    assert "B" in h_dict
    assert h_dict["A"].mean().value < 1.0
    assert h_dict["B"].mean().value > 4.0


def test_frequencyseriesdict_histogram():
    fsd = FrequencySeriesDict()
    fsd["A"] = FrequencySeries([1, 2, 3], frequencies=[10, 20, 30])

    h_dict = fsd.histogram(bins=2)
    assert isinstance(h_dict, HistogramDict)
    assert "A" in h_dict
