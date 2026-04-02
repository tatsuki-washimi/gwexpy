"""Tests for GBD (GRAPHTEC) reader."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io.gbd import read_timeseriesdict_gbd

FIXTURE_DATA = Path(__file__).parent.parent / "fixtures" / "data" / "test.gbd"

@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gbd fixture not found")
def test_read_gbd_dict():
    # GBD requires a timezone
    with pytest.raises(ValueError, match="timezone is required"):
        read_timeseriesdict_gbd(FIXTURE_DATA)
    
    # Read with JST timezone
    tsd = TimeSeriesDict.read(FIXTURE_DATA, format="gbd", timezone="Asia/Tokyo")
    
    assert isinstance(tsd, TimeSeriesDict)
    # Expected channels from generate_fixtures.py: ["CH1", "Alarm", "AlarmOut"]
    assert "CH1" in tsd
    assert "Alarm" in tsd
    assert "AlarmOut" in tsd
    
    # Check metadata
    ts = tsd["CH1"]
    assert ts.sample_rate == 1.0 * u.Hz
    assert ts.unit == u.V  # Default for non-digital
    
    # Check data content (sin wave was generated for CH1)
    # Just check it's not all zeros or NaNs
    assert not np.all(ts.value == 0)
    assert not np.any(np.isnan(ts.value))

def test_read_gbd_single():
    # Read single channel
    ts = TimeSeries.read(FIXTURE_DATA, format="gbd", timezone="UTC", channels=["CH1"])
    assert ts.name == "CH1"
    assert len(ts) == 100

def test_gbd_digital_channels():
    # Alarm/AlarmOut should be binarized (0 or 1)
    tsd = TimeSeriesDict.read(FIXTURE_DATA, format="gbd", timezone="UTC")
    ts_alarm = tsd["Alarm"]
    assert ts_alarm.unit == u.dimensionless_unscaled
    assert set(np.unique(ts_alarm.value)).issubset({0.0, 1.0})
