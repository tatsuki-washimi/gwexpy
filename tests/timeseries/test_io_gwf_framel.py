"""Tests for Frame (GWF) reading using the framel backend."""

from pathlib import Path

import pytest

from gwexpy.timeseries import TimeSeries

FIXTURE_DATA = Path(__file__).parent.parent.parent / "fixtures" / "data" / "test.gwf"
CHANNEL = "K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"

def has_framel():
    try:
        import framel
        return True
    except (ImportError, RuntimeError):
        return False

@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_framel(), reason="framel backend not available")
def test_read_gwf_framel():
    # Force the use of framel backend
    ts = TimeSeries.read(FIXTURE_DATA, CHANNEL, format="gwf.framel")

    assert isinstance(ts, TimeSeries)
    assert ts.name == CHANNEL
    assert len(ts) > 0
    assert ts.sample_rate.value == 16384.0
