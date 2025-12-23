
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from gwpy.spectrogram import Spectrogram
from gwexpy.spectrogram import SpectrogramList, SpectrogramDict

def test_spectrogram_extra_features():
    # Dummy Data (Time=10, Freq=5)
    data1 = np.ones((10, 5))
    data2 = np.ones((10, 5)) * 2
    # dt=1, df=1. t0=0, f0=0
    s1 = Spectrogram(data1, dt=1, df=1, name='s1', t0=0, f0=0)
    s2 = Spectrogram(data2, dt=1, df=1, name='s2', t0=0, f0=0)
    
    # 1. Matrix Extra
    sl = SpectrogramList([s1, s2])
    mat = sl.to_matrix() # (2, 10, 5)
    
    # Mean
    mean_spec_arr = mat.mean(axis=0) # (10, 5)
    assert mean_spec_arr.shape == (10, 5)
    
    # Plot (Dry run)
    p = mat.plot()
    p.close()
    p2 = mat.plot(monitor=0)
    p2.close()
    
    # 2. IO
    filename = 'test_spec_io.h5'
    if os.path.exists(filename):
        os.remove(filename)
    
    try:
        import h5py
        sl.write(filename)
        sl_read = SpectrogramList().read(filename)
        assert len(sl_read) == 2
        # Check values
        assert np.allclose(sl_read[0].value, data1)
        
        sd = SpectrogramDict({'a': s1, 'b': s2})
        sd.write(filename, mode='w') # overwrite
        sd_read = SpectrogramDict().read(filename)
        assert 'a' in sd_read
        assert np.allclose(sd_read['a'].value, data1)
    except ImportError:
        pass
    finally:
        if os.path.exists(filename):
            os.remove(filename)
    
    # 3. Inplace
    sl = SpectrogramList([s1, s2])
    id(sl[0])
    sl.crop(2, 8, inplace=True)
    # Check updated content
    assert sl[0].shape[0] < 10
    
    # Inplace Dict
    sd = SpectrogramDict({'a': s1, 'b': s2})
    sd.crop(2, 8, inplace=True)
    assert sd['a'].shape[0] < 10
