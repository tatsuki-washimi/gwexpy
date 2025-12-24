
import numpy as np
import pytest
from gwexpy.timeseries import TimeSeries

try:
    import astropy
except ImportError:
    astropy = None

try:
    import mne
except ImportError:
    mne = None

try:
    import neo
except ImportError:
    neo = None

@pytest.mark.skipif(astropy is None, reason="astropy not installed")
def test_astropy_interop():
    ts = TimeSeries([1, 2, 3], dt=1.0, t0=0, unit="m")

    # 1. to_astropy
    ap_ts = ts.to_astropy_timeseries()
    assert len(ap_ts) == 3
    # Astropy TimeSeries access
    assert np.all(ap_ts['value'].value == [1, 2, 3])

    # 2. from_astropy
    ts2 = TimeSeries.from_astropy_timeseries(ap_ts)
    assert np.allclose(ts2.value, [1, 2, 3])
    assert np.isclose(ts2.dt.value, 1.0)

@pytest.mark.skipif(mne is None, reason="mne not installed")
def test_mne_interop():
    # Construct a dict for testing
    from gwexpy.timeseries import TimeSeriesDict
    tsd = TimeSeriesDict()
    tsd['ch1'] = TimeSeries(np.arange(100), dt=0.01)
    tsd['ch2'] = TimeSeries(np.arange(100)*2, dt=0.01)

    # 1. to_mne (multi-channel)
    raw = tsd.to_mne_raw()
    assert isinstance(raw, mne.io.RawArray)
    assert raw.info['nchan'] == 2
    assert raw.info['sfreq'] == 100.0

    raw_picked = tsd.to_mne_raw(picks=["ch2"])
    assert isinstance(raw_picked, mne.io.RawArray)
    assert raw_picked.info["nchan"] == 1
    assert raw_picked.ch_names == ["ch2"]

    # 2. from_mne
    tsd2 = TimeSeriesDict.from_mne_raw(raw)
    assert 'ch1' in tsd2
    assert 'ch2' in tsd2
    assert np.allclose(tsd2['ch1'].value, tsd['ch1'].value)

    # 3. single-channel convenience
    ts = TimeSeries(np.arange(100), dt=0.01, name="ch1")
    raw_single = ts.to_mne_raw()
    assert isinstance(raw_single, mne.io.RawArray)
    assert raw_single.info["nchan"] == 1
    assert raw_single.info["sfreq"] == 100.0

@pytest.mark.skipif(neo is None, reason="neo not installed")
def test_neo_interop():
    # Neo usually deals with multiple channels in 2D
    # Using TimeSeriesMatrix path if available, or just testing low-level?
    # TimeSeries has no to_neo? Only Matrix/Dict?
    # Spec implied TimeSeriesMatrix. But we put to_neo_analogsignal on SeriesMatrix.

    from gwexpy.timeseries import TimeSeriesMatrix

    data = np.random.randn(2, 1, 100) # 2 channels, 100 samples
    tsm = TimeSeriesMatrix(data, dt=0.01, t0=0)
    tsm.channel_names = ['C1', 'C2']

    # 1. to_neo
    sig = tsm.to_neo_analogsignal()
    # Check shape: neo is (time, channels)
    assert sig.shape == (100, 2)
    assert sig.sampling_rate.magnitude == 100.0

    # 2. from_neo
    tsm2 = TimeSeriesMatrix.from_neo_analogsignal(sig)
    # Check shape restoration (channels, 1, time)
    assert tsm2.shape == (2, 1, 100)
    assert np.allclose(tsm2.value, data)
    assert tsm2.channel_names == ['C1', 'C2']
