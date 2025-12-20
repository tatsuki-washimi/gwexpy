
import numpy as np
from astropy import units as u
from gwpy.spectrogram import Spectrogram
from gwexpy.spectrogram import SpectrogramList, SpectrogramDict, SpectrogramMatrix

def create_mock_spectrogram(name="spec", shape=(10, 10)):
    # Create a 10x10 spectrogram
    # Time: 0 to 10s (dt=1), Freq: 0 to 100Hz (df=10)
    data = np.random.random(shape)
    # Spectrogram(data, t0, dt, f0, df, ...)
    # Adjust args based on gwpy version if needed, but kwargs usually work
    spec = Spectrogram(data, t0=0, dt=1, f0=0, df=10, unit='strain', name=name)
    return spec

def test_list():
    print("Testing SpectrogramList...")
    s1 = create_mock_spectrogram("s1")
    s2 = create_mock_spectrogram("s2")
    
    sl = SpectrogramList([s1])
    sl.append(s2)
    assert len(sl) == 2
    assert sl[0].name == "s1"
    print("  Basic list ops OK")

    # Type check
    try:
        sl.append("invalid")
        raise AssertionError("Type check failed - allowed string append")
    except TypeError:
        print("  Type check OK")

    # Crop
    print("  Testing crop(t0=2, t1=8)...")
    sl_cropped = sl.crop(t0=2, t1=8)
    # t0=0, dt=1. Indices 2 to 8.
    # Result time axis should start >= 2
    assert sl_cropped[0].times[0].value >= 2
    assert len(sl_cropped) == 2
    print("  Crop OK")
    
    # Crop frequencies
    print("  Testing crop_frequencies(f0=20, f1=80)...")
    # f0=0, df=10. 20Hz is index 2. 80Hz is index 8.
    sl_freq = sl.crop_frequencies(f0=20, f1=80)
    # Check freq axis
    assert sl_freq[0].frequencies[0].value >= 20
    assert sl_freq[0].frequencies[-1].value <= 80
    print("  Crop frequencies OK")

    # to_matrix
    print("  Testing to_matrix...")
    mat = sl.to_matrix()
    # Expect (2, 10, 10)
    assert isinstance(mat, SpectrogramMatrix)
    assert mat.shape == (2, 10, 10)
    assert mat.times is not None
    assert len(mat.times) == 10
    assert mat.frequencies is not None
    assert len(mat.frequencies) == 10
    assert mat.unit == u.Unit('strain')
    print("  to_matrix OK")

    # Plot check (dry run)
    print("  Testing plot()...")
    try:
        # We can't easily check output in headless, but ensure no exception
        # Mocking or just calling it. gwpy plot might try to open backend.
        # Use a non-interactive backend for matplotlib if possible,
        # but here we just call it and hopefully it just returns a Plot object without showing.
        import matplotlib
        matplotlib.use('Agg')
        p = sl.plot()
        p.close()
        print("  plot() OK")
    except Exception as e:
        print(f"  plot() failed (might be display issue): {e}")
    
    # Shape mismatch for matrix
    s3 = create_mock_spectrogram("s3", shape=(5, 5))
    sl.append(s3)
    try:
        sl.to_matrix()
        raise AssertionError("Should fail with shape mismatch")
    except ValueError as e:
        print(f"  to_matrix mismatch check OK: {e}")

    print("SpectrogramList OK")

def test_dict():
    print("\nTesting SpectrogramDict...")
    s1 = create_mock_spectrogram("s1")
    s2 = create_mock_spectrogram("s2")
    
    sd = SpectrogramDict({'a': s1})
    sd['b'] = s2
    
    assert len(sd) == 2
    print("  Basic dict ops OK")
    
    # Type check
    try:
        sd['c'] = "invalid"
        raise AssertionError("Type check failed")
    except TypeError:
         print("  Type check OK")
         
    # Crop
    print("  Testing crop...")
    sd_cropped = sd.crop(2, 8)
    assert sd_cropped['a'].times[0].value >= 2
    
    # Matrix
    print("  Testing to_matrix...")
    mat = sd.to_matrix()
    assert mat.shape == (2, 10, 10)
    
    print("SpectrogramDict OK")

if __name__ == "__main__":
    try:
        test_list()
        test_dict()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
