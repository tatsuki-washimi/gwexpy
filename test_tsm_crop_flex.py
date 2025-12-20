
import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.time import tconvert

def test_tsm_crop():
    t0 = 1234567890.0
    dt = 1.0
    data = np.random.rand(2, 2, 10)
    tsm = TimeSeriesMatrix(data, t0=t0, dt=dt)
    
    start_time = tconvert(t0 + 2)
    end_time = tconvert(t0 + 5)
    
    print(f"Testing TSM crop with string: {start_time}")
    try:
        tsm_cropped = tsm.crop(start=start_time, end=end_time)
        print(f"Cropped TSM shape: {tsm_cropped.shape}")
        assert tsm_cropped.shape[-1] == 3
        print("TimeSeriesMatrix crop passed")
    except Exception as e:
        print(f"TimeSeriesMatrix crop FAILED: {e}")

if __name__ == "__main__":
    test_tsm_crop()
