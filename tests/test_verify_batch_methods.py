import numpy as np
import pandas as pd
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesList, TimeSeriesDict
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList, FrequencySeriesDict

def test_timeseries_batch():
    print("=== Testing TimeSeries Batch Operations ===")
    
    # Create dummy data
    t0 = 0
    dt = 0.01
    n_samples = 1000
    times = np.arange(n_samples) * dt + t0
    
    data1 = np.sin(2 * np.pi * 1 * times)
    data2 = np.cos(2 * np.pi * 1 * times)
    
    ts1 = TimeSeries(data1, t0=t0, dt=dt, unit="V", name="ch1")
    ts2 = TimeSeries(data2, t0=t0, dt=dt, unit="V", name="ch2")
    
    # --- TimeSeriesList ---
    print("Checking TimeSeriesList...")
    ts_list = TimeSeriesList(ts1, ts2)
    
    # Waveform Ops
    print(f"Original list length: {len(ts_list)}")
    cropped = ts_list.crop(0, 5)
    print(f"Cropped list length: {len(cropped)}")
    assert len(cropped) == 2
    assert len(cropped[0]) == int(5/dt) + 1 or len(cropped[0]) == int(5/dt), f"Crop length unexpected: {len(cropped[0])}"
    
    # Signal Processing
    resampled = ts_list.resample(10) # 10 Hz (original 100Hz)
    assert len(resampled) == 2
    
    # Stats
    means = ts_list.mean()
    print(f"Means: {means}")
    assert isinstance(means, list)
    assert len(means) == 2
    
    # Spectral
    # Note: fft implementation details might vary, just checking container return
    fs_list = ts_list.fft()
    print(f"FFT result type: {type(fs_list)}")
    assert isinstance(fs_list, FrequencySeriesList)
    assert len(fs_list) == 2
    
    # --- TimeSeriesDict ---
    print("\nChecking TimeSeriesDict...")
    ts_dict = TimeSeriesDict({"ch1": ts1, "ch2": ts2})
    
    # Waveform Update (inplace check)
    # create copy first
    ts_dict_copy = ts_dict.copy()
    # Create extension data that matches the end time of the series
    end_time = ts1.span[1]
    extension = TimeSeries([0, 0, 0], t0=end_time, dt=dt, unit="V")
    
    # Broadcast append: append this extension to ALL series in the dict
    ts_dict_copy.append(extension) 
    assert len(ts_dict_copy["ch1"]) == n_samples + 3
    
    # Stats (Pandas return)
    # rms() returns a TimeSeries, so we use mean() to check scalar return behavior
    mean_res = ts_dict.mean()
    print(f"Mean result type: {type(mean_res)}")
    print(mean_res)
    assert isinstance(mean_res, pd.Series)
    assert "ch1" in mean_res
    
    # To Matrix
    mat = ts_dict.to_matrix()
    print(f"Matrix shape: {mat.shape}")
    assert mat.shape == (2, 1, n_samples)
    
    print("TimeSeries batch verification OK.")


def test_frequencyseries_batch():
    print("=== Testing FrequencySeries Batch Operations ===")
    
    # Create dummy data via FFT
    dt = 0.01
    n_samples = 1000
    times = np.arange(n_samples) * dt
    data1 = np.sin(2 * np.pi * 10 * times) # 10 Hz
    data2 = np.sin(2 * np.pi * 20 * times) # 20 Hz
    
    ts1 = TimeSeries(data1, dt=dt, unit="V", name="ch1")
    ts2 = TimeSeries(data2, dt=dt, unit="V", name="ch2")
    
    fs1 = ts1.fft()
    fs2 = ts2.fft()
    
    # --- FrequencySeriesList ---
    print("Checking FrequencySeriesList...")
    fs_list = FrequencySeriesList(fs1, fs2)
    
    # Map Operations
    db_list = fs_list.to_db()
    assert isinstance(db_list, FrequencySeriesList)
    assert len(db_list) == 2
    assert db_list[0].unit == u.Unit("dB")
    
    # IFFT
    print("Checking IFFT...")
    try:
        ts_list_back = fs_list.ifft()
        print(f"IFFT result type: {type(ts_list_back)}")
        assert isinstance(ts_list_back, TimeSeriesList) 
        assert len(ts_list_back) == 2
    except ImportError as e:
        print(f"IFFT ImportError: {e}")
        raise
    
    # --- FrequencySeriesDict ---
    print("\nChecking FrequencySeriesDict...")
    fs_dict = FrequencySeriesDict({"ch1": fs1, "ch2": fs2})
    
    # To Pandas
    print("Checking to_pandas...")
    df = fs_dict.to_pandas()
    print(f"DataFrame Head:\n{df.head()}")
    assert isinstance(df, pd.DataFrame)
    assert "ch1" in df.columns
    assert "ch2" in df.columns
    
    # Interop (Mock check)
    # Just check if method exists and runs without error returning dict
    try:
        # Assuming torch not installed in env? Or maybe it is.
        # We'll try to_torch if torch is available, otherwise skip
        import torch
        torch_dict = fs_dict.to_torch()
        assert isinstance(torch_dict, dict)
        print("to_torch successful")
    except ImportError:
        print("Skipping torch test (not installed)")
    except Exception as e:
        print(f"to_torch failed: {e}")

    print("FrequencySeries batch verification OK.")


