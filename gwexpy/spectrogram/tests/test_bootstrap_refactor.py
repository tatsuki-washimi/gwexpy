import pytest
import numpy as np
from astropy import units as u
from gwexpy.spectrogram import Spectrogram, SpectrogramList, SpectrogramDict

def test_spectrogram_bootstrap_method():
    # Create valid dummy spectrogram
    # Time x Frequency
    times = np.arange(10) # 10s
    frequencies = np.array([10, 20, 30]) # 3 bins
    
    data = np.random.rand(10, 3).astype(float)
    unit = 'V'
    
    spec = Spectrogram(data, times=times, frequencies=frequencies, unit=unit, name='TestSpec')
    
    # Test new method name "bootstrap" and parameter "method='median'" (default)
    bs_median = spec.bootstrap(method='median', n_boot=10)
    assert bs_median.name == "TestSpec (Bootstrap median)"
    assert bs_median.shape == (3,)
    assert hasattr(bs_median, 'error_low')
    assert hasattr(bs_median, 'error_high')
    
    # Test "method='mean'"
    bs_mean = spec.bootstrap(method='mean', n_boot=10)
    assert bs_mean.name == "TestSpec (Bootstrap mean)"
    assert hasattr(bs_mean, 'error_low')
    
    # Check that 'average' kwarg raises TypeError (implicit due to not being in signature and not using **kwargs)
    # OR if it's caught as unknown argument.
    # The signature is def bootstrap(..., method='median', ...)
    # So passing average='...' effectively passes it as keyword argument? No, it's not accepted.
    with pytest.raises(TypeError):
        spec.bootstrap(average='mean', n_boot=10)

def test_collections_bootstrap():
    times = np.arange(10)
    frequencies = np.array([10, 20, 30])
    data1 = np.random.rand(10, 3)
    data2 = np.random.rand(10, 3)
    
    s1 = Spectrogram(data1, times=times, frequencies=frequencies, name='S1')
    s2 = Spectrogram(data2, times=times, frequencies=frequencies, name='S2')
    
    # List
    sl = SpectrogramList([s1, s2])
    res_list = sl.bootstrap(method='mean', n_boot=5)
    from gwexpy.frequencyseries import FrequencySeriesList
    assert isinstance(res_list, FrequencySeriesList)
    assert len(res_list) == 2
    assert res_list[0].name == 'S1 (Bootstrap mean)'
    
    # Dict
    sd = SpectrogramDict({'a': s1, 'b': s2})
    res_dict = sd.bootstrap(method='median', n_boot=5)
    from gwexpy.frequencyseries import FrequencySeriesDict
    assert isinstance(res_dict, FrequencySeriesDict)
    assert len(res_dict) == 2
    assert res_dict['a'].name == 'S1 (Bootstrap median)'

if __name__ == "__main__":
    test_spectrogram_bootstrap_method()
    test_collections_bootstrap()
    print("All tests passed!")
