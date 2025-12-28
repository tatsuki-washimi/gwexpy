
import pytest
import numpy as np
from gwexpy.frequencyseries import FrequencySeries

try:
    import obspy
except ImportError:
    obspy = None

@pytest.mark.skipif(obspy is None, reason="obspy not installed")
def test_to_obspy_trace():
    """Test conversion from FrequencySeries to Obspy Trace."""
    data = np.arange(10).astype(float)
    freqs = np.linspace(0, 9, 10)
    fs = FrequencySeries(data, frequencies=freqs, unit='m', name='TEST_FS')
    
    tr = fs.to_obspy()
    
    assert isinstance(tr, obspy.Trace)
    assert tr.stats.npts == 10
    assert tr.stats.delta == 1.0 # f1-f0
    assert tr.stats.station == 'TEST_FS'
    np.testing.assert_array_equal(tr.data, data)
    
    # Check if we can reconstruct
    fs_rec = FrequencySeries.from_obspy(tr)
    np.testing.assert_array_equal(fs_rec.value, fs.value)
    assert fs_rec.df.value == fs.df.value
    # f0 might be starttime timestamp, which is 0.0 + starttime
    # Our impl: starttime = f0
    # Obspy default starttime is 1970-01-01T00:00:00
    # If f0=0, starttime is 1970...
    assert fs_rec.f0.value == fs.f0.value

@pytest.mark.skipif(obspy is None, reason="obspy not installed")
def test_from_obspy_trace():
    """Test conversion from Obspy Trace to FrequencySeries."""
    # Create a trace that mimics spectral data
    header = {'delta': 0.5, 'starttime': 100.0, 'station': 'OBSPY_TR', 'channel': 'CH1'}
    data = np.random.rand(20)
    tr = obspy.Trace(data=data, header=header)
    
    fs = FrequencySeries.from_obspy(tr)
    
    assert isinstance(fs, FrequencySeries)
    assert fs.df.value == 0.5
    assert fs.f0.value == 100.0 # starttime timestamp
    # tr.id is typically NET.STA.LOC.CHAN, here '.
    expected_id = '.OBSPY_TR..CH1'
    assert fs.name == expected_id
    # Our impl defaults to id if name_policy='id', which includes network.station...
    # If 'id', it's usually network.station.location.channel
    # obspy default network is empty, so it might be '.OBSPY_TR..CH1'
    # Checking default behavior of from_obspy (legacy uses name_policy='id')
    # Let's check what ID looks like
    print(fs.name)
    
    np.testing.assert_array_equal(fs.value, data)
