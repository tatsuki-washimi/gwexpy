import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.noise.wave import from_asd
from gwexpy.timeseries import TimeSeries


def test_from_asd_returns_timeseries_with_metadata():
    freqs = np.linspace(1.0, 10.0, 10)
    asd_values = np.ones_like(freqs) * 2.0
    unit = u.m / (u.Hz ** 0.5)
    asd = FrequencySeries(
        asd_values,
        frequencies=freqs,
        unit=unit,
        name="TEST_ASD",
        channel="X1:TEST",
    )

    duration = 4.0
    sample_rate = 16.0
    t0 = 123.0
    ts = from_asd(
        asd,
        duration=duration,
        sample_rate=sample_rate,
        t0=t0,
        rng=np.random.default_rng(0),
    )

    assert isinstance(ts, TimeSeries)
    assert len(ts) == int(duration * sample_rate)
    assert ts.sample_rate.value == pytest.approx(sample_rate)
    assert ts.t0.value == pytest.approx(t0)
    assert ts.unit == unit * (u.Hz ** 0.5)
    assert ts.name == "TEST_ASD"
    assert getattr(ts.channel, "name", None) == "X1:TEST"
