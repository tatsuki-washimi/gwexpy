import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from gwpy.spectrogram import Spectrogram
from gwexpy.spectrogram import SpectrogramList, SpectrogramDict

def test_features():
    print("Testing Extra Features...")
    
    # Dummy Data (Time=10, Freq=5)
    data1 = np.ones((10, 5))
    data2 = np.ones((10, 5)) * 2
    # dt=1, df=1. t0=0, f0=0
    s1 = Spectrogram(data1, dt=1, df=1, name='s1', t0=0, f0=0)
    s2 = Spectrogram(data2, dt=1, df=1, name='s2', t0=0, f0=0)
    
    # 1. Matrix Extra
    sl = SpectrogramList([s1, s2])
    mat = sl.to_matrix() # (2, 10, 5)
    print(f"Matrix shape: {mat.shape}")
    
    # Mean
    mean_spec_arr = mat.mean(axis=0) # (10, 5)
    assert mean_spec_arr.shape == (10, 5)
    print("Matrix.mean shape OK")
    
    # Plot (Dry run)
    # This calls mat.plot -> mean -> Spectrogram.plot
    p = mat.plot()
    p.close()
    p2 = mat.plot(monitor=0)
    p2.close()
    print("Matrix.plot OK")
    
    # 2. IO
    filename = 'test_spec_io.h5'
    if os.path.exists(filename): os.remove(filename)
    
    try:
        import h5py
        sl.write(filename)
        sl_read = SpectrogramList().read(filename)
        assert len(sl_read) == 2
        # Check values
        assert np.allclose(sl_read[0].value, data1)
        print("List IO OK")
        
        sd = SpectrogramDict({'a': s1, 'b': s2})
        sd.write(filename, mode='w') # overwrite
        sd_read = SpectrogramDict().read(filename)
        assert 'a' in sd_read
        assert np.allclose(sd_read['a'].value, data1)
        print("Dict IO OK")
    except ImportError:
        print("h5py not installed, skipping IO tests")
    except Exception as e:
        print(f"IO Test Failed: {e}")
        # Clean up
        if os.path.exists(filename): os.remove(filename)
        raise e
    
    if os.path.exists(filename): os.remove(filename)

    # 3. Inplace
    # crop returns new spectrogram but inplace should update reference in list
    old_id = id(sl)
    # Crop time: 2 to 8 -> 6 samples? (includes end? gwpy behavior varies, usually [start, end))
    sl_new = sl.crop(2, 8, inplace=True)
    assert id(sl_new) == old_id
    # Check updated content
    # original length 10. crop(2,8) should be smaller.
    print(f"Cropped shape: {sl[0].shape}")
    assert sl[0].shape[0] < 10
    print("Inplace Crop OK")
    
    # Inplace Dict
    sd = SpectrogramDict({'a': s1, 'b': s2})
    old_id_d = id(sd)
    sd_new = sd.crop(2, 8, inplace=True)
    assert id(sd_new) == old_id_d
    assert sd['a'].shape[0] < 10
    print("Inplace Dict Crop OK")
    
    print("ALL EXTRA TESTS PASSED")

if __name__ == "__main__":
    test_features()
