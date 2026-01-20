
import numpy as np
import pytest

from gwexpy.spectrogram import Spectrogram

try:
    import obspy
except ImportError:
    obspy = None

@pytest.mark.skipif(obspy is None, reason="obspy not installed")
def test_to_obspy_stream():
    """Test conversion from Spectrogram to Obspy Stream."""
    nf = 5
    nt = 10
    data = np.random.rand(nt, nf)
    freqs = np.linspace(0, 4, nf)
    times = np.linspace(0, 9, nt)

    spec = Spectrogram(data, frequencies=freqs, times=times, unit='m', name='TEST_SPEC')

    st = spec.to_obspy()

    assert isinstance(st, obspy.Stream)
    assert len(st) == nf

    # Check first trace (first frequency bin)
    tr0 = st[0]
    assert tr0.stats.npts == nt
    assert tr0.stats.delta == 1.0 # t1-t0
    assert tr0.stats.station == 'TEST_SPEC'
    # Check if frequency info is preserved in our custom header
    assert hasattr(tr0.stats, 'frequency')
    assert tr0.stats.frequency == freqs[0]

    np.testing.assert_array_equal(tr0.data, data[:, 0])

    # Check reconstruction
    spec_rec = Spectrogram.from_obspy(st)

    np.testing.assert_array_equal(spec_rec.value, spec.value)
    np.testing.assert_array_equal(spec_rec.frequencies.value, spec.frequencies.value)
    # Check time axis
    # t0 should be starttime
    assert spec_rec.t0.value == spec.t0.value
    assert spec_rec.dt.value == spec.dt.value

@pytest.mark.skipif(obspy is None, reason="obspy not installed")
def test_from_obspy_stream_mixed():
    """Test conversion from mixed Stream (should fail or filter?)."""
    # Should from_obspy only handle Streams that are "Spectrogram-like"?
    # Yes, our implementation expects they can form a matrix.
    pass
