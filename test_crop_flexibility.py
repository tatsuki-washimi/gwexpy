
import numpy as np
import pandas as pd
from datetime import datetime
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList

def test_crop_flexible():
    # 1. Setup TimeSeries
    t0 = 1234567890.0
    dt = 1.0
    data = np.arange(10)
    ts = TimeSeries(data, t0=t0, dt=dt)
    
    print(f"Original TimeSeries: t0={ts.t0}, span={ts.span}")
    
    # Crop with string
    # t0 + 2s = 1234567892.0
    start_str = "2019-02-24 03:31:32" # This is exactly t0+2 in GPS? 
    # Wait, let's use tconvert to be sure
    from gwexpy.time import tconvert
    start_time = tconvert(t0 + 2)
    end_time = tconvert(t0 + 5)
    
    print(f"Testing crop with string: {start_time} to {end_time}")
    ts_cropped = ts.crop(start=start_time, end=end_time)
    
    print(f"Cropped TimeSeries: t0={ts_cropped.t0}, len={len(ts_cropped)}")
    assert len(ts_cropped) == 3
    assert ts_cropped.t0.value == t0 + 2
    
    # 2. Setup TimeSeriesDict
    tsd = TimeSeriesDict()
    tsd['A'] = ts
    tsd['B'] = ts * 2
    
    tsd_cropped = tsd.crop(start=start_time, end=end_time)
    assert len(tsd_cropped['A']) == 3
    assert len(tsd_cropped['B']) == 3
    print("TimeSeriesDict crop passed")
    
    # 3. Setup TimeSeriesList
    tsl = TimeSeriesList()
    tsl.append(ts)
    tsl.append(ts * 2)
    tsl_cropped = tsl.crop(start=start_time, end=end_time)
    assert len(tsl_cropped[0]) == 3
    assert len(tsl_cropped[1]) == 3
    print("TimeSeriesList crop passed")

if __name__ == "__main__":
    try:
        test_crop_flexible()
        print("\nAll Crop Flexibility Tests Passed!")
    except Exception as e:
        print(f"\nTests FAILED: {e}")
        import traceback
        traceback.print_exc()
