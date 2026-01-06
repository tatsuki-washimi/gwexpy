import numpy as np
from gwexpy.analysis.bruco import Bruco
from gwpy.timeseries import TimeSeries, TimeSeriesDict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Mock Data
start = 1000000000
duration = 10
end = start + duration
fs = 256
t = np.linspace(0, duration, duration * fs)
target_data = TimeSeries(np.random.randn(len(t)), t0=start, sample_rate=fs, name="Target")

# Bruco setup
# Case 4: Init with some channels (Auto fetch) + Pass manual dict
bruco = Bruco("Target", ["NDS:AutoFetch1"])
# Mock fetching for target and NDS channel (monkeypatch)
def mock_get(channel, start, end, **kwargs):
    print(f"DEBUG: Fetching {channel} from {start} to {end}")
    return TimeSeries(np.random.randn(int((end-start)*fs)), t0=start, sample_rate=fs, name=channel)
TimeSeries.get = mock_get
TimeSeriesDict.get = lambda channels, s, e, **k: TimeSeriesDict({c: mock_get(c, s, e) for c in channels})

print("--- Test 1: Valid Mixed Mode ---")
# Manual Data (Valid Span)
manual_data = TimeSeriesDict()
manual_data["Manual:Ch1"] = TimeSeries(np.random.randn(len(t)), t0=start, sample_rate=fs, name="Manual:Ch1")

try:
    result = bruco.compute(start, duration, target_data=target_data, aux_data=manual_data)
    print("SUCCESS: Mixed mode computed without error.")
    print("Results:", result.channel_names)
except Exception as e:
    print(f"FAILURE: {e}")

print("\n--- Test 2: Invalid Span (ValueError Expected) ---")
# Manual Data (Invalid Span)
offset_start = start + 5 # overlap but incomplete
manual_data_bad = TimeSeriesDict()
manual_data_bad["Manual:Bad"] = TimeSeries(np.random.randn(len(t)), t0=offset_start, sample_rate=fs, name="Manual:Bad")

try:
    result = bruco.compute(start, duration, target_data=target_data, aux_data=manual_data_bad)
    print("FAILURE: Should have raised ValueError but didn't.")
except ValueError as e:
    print(f"SUCCESS: Caught expected ValueError: {e}")
except Exception as e:
    print(f"FAILURE: Caught unexpected exception: {e}")
