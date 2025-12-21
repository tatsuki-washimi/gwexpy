
import numpy as np
import pytest
from astropy import units as u
from gwpy.spectrogram import Spectrogram
from gwexpy.spectrogram import SpectrogramList, SpectrogramDict, SpectrogramMatrix

def create_mock_spectrogram(name="spec", shape=(10, 10)):
    # Create a 10x10 spectrogram
    # Time: 0 to 10s (dt=1), Freq: 0 to 100Hz (df=10)
    data = np.random.random(shape)
    spec = Spectrogram(data, t0=0, dt=1, f0=0, df=10, unit='strain', name=name)
    return spec

def test_spectrogram_list():
    s1 = create_mock_spectrogram("s1")
    s2 = create_mock_spectrogram("s2")
    
    sl = SpectrogramList([s1])
    sl.append(s2)
    assert len(sl) == 2
    assert sl[0].name == "s1"

    # Type check
    try:
        sl.append("invalid")
        raise AssertionError("Type check failed - allowed string append")
    except TypeError:
        pass

    # Crop
    sl_cropped = sl.crop(t0=2, t1=8)
    # t0=0, dt=1. Indices 2 to 8.
    # Result time axis should start >= 2
    assert sl_cropped[0].times[0].value >= 2
    assert len(sl_cropped) == 2
    
    # Crop frequencies
    # f0=0, df=10. 20Hz is index 2. 80Hz is index 8.
    sl_freq = sl.crop_frequencies(f0=20, f1=80)
    # Check freq axis
    assert sl_freq[0].frequencies[0].value >= 20
    assert sl_freq[0].frequencies[-1].value <= 80

    # to_matrix
    mat = sl.to_matrix()
    # Expect (2, 10, 10)
    assert isinstance(mat, SpectrogramMatrix)
    assert mat.shape == (2, 10, 10)
    assert mat.times is not None
    assert len(mat.times) == 10
    assert mat.frequencies is not None
    assert len(mat.frequencies) == 10
    assert mat.unit == u.Unit('strain')

    # Plot check (dry run) - skip if matplotlib not available
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use('Agg')
    p = sl.plot()
    p.close()
    
    # Shape mismatch for matrix
    s3 = create_mock_spectrogram("s3", shape=(5, 5))
    sl.append(s3)
    try:
        sl.to_matrix()
        raise AssertionError("Should fail with shape mismatch")
    except ValueError:
        pass

def test_spectrogram_dict():
    s1 = create_mock_spectrogram("s1")
    s2 = create_mock_spectrogram("s2")
    
    sd = SpectrogramDict({'a': s1})
    sd['b'] = s2
    
    assert len(sd) == 2
    
    # Type check
    try:
        sd['c'] = "invalid"
        raise AssertionError("Type check failed")
    except TypeError:
         pass
         
    # Crop
    sd_cropped = sd.crop(2, 8)
    assert sd_cropped['a'].times[0].value >= 2
    
    # Matrix
    mat = sd.to_matrix()
    assert mat.shape == (2, 10, 10)
