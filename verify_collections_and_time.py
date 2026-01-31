
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict
from gwexpy.timeseries import TimeSeries as GwexTimeSeries
from gwexpy.timeseries import TimeSeriesDict as GwexTimeSeriesDict
from gwexpy.timeseries import TimeSeriesList as GwexTimeSeriesList
from gwexpy.time import to_gps as gwex_to_gps

def log_result(test_name, success, message=""):
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {test_name}")
    if message:
        print(f"       Details: {message}")
    return success

def test_collection_type_preservation():
    print("\n--- Test: Collection Type Preservation ---")
    
    # Setup data
    ts1 = GwexTimeSeries(np.random.randn(1000), t0=0, sample_rate=1, name="Ch1")
    ts2 = GwexTimeSeries(np.random.randn(1000), t0=0, sample_rate=1, name="Ch2")
    
    # 1. Dict
    tsd = GwexTimeSeriesDict()
    tsd["Ch1"] = ts1
    tsd["Ch2"] = ts2
    
    # Operation: crop
    tsd_crop = tsd.crop(0, 50)
    is_gwex_dict = isinstance(tsd_crop, GwexTimeSeriesDict)
    
    # Operation: resample
    # Note: gwexpy resample is in-place for Dict? Static analysis said:
    # "Time-bin logic: replace items in-place... Native gwpy: super().resample (in-place)"
    # Let's check return type just in case logic returns self
    tsd_res = tsd.resample(0.5) 
    is_gwex_dict_res = isinstance(tsd_res, GwexTimeSeriesDict)

    # 2. List
    tsl = GwexTimeSeriesList(*[ts1, ts2])
    tsl_crop = tsl.crop(0, 50)
    is_gwex_list = isinstance(tsl_crop, GwexTimeSeriesList)
    
    success = is_gwex_dict and is_gwex_list and is_gwex_dict_res
    msg = []
    if not is_gwex_dict: msg.append(f"Dict.crop returned {type(tsd_crop)}")
    if not is_gwex_dict_res: msg.append(f"Dict.resample returned {type(tsd_res)}")
    if not is_gwex_list: msg.append(f"List.crop returned {type(tsl_crop)}")
    
    return log_result("Collection Type Preservation", success, ", ".join(msg))

def test_time_array_handling():
    print("\n--- Test: Time Array Handling (to_gps) ---")
    
    # Input list of mixed types (str, float)
    inputs = ["2023-01-01 12:00:00", 1234567890.0]
    
    try:
        gps_arr = gwex_to_gps(inputs)
        is_array = isinstance(gps_arr, (np.ndarray, list))
        # Check values roughly
        success = is_array and len(gps_arr) == 2
        msg = f"Output type: {type(gps_arr)}"
    except Exception as e:
        success = False
        msg = str(e)
        
    return log_result("gwexpy.time.to_gps([list])", success, msg)

def test_monkey_patching():
    print("\n--- Test: Monkey Patching of gwpy classes ---")
    # gwexpy is already imported, so patches should be applied
    
    base_dict = GwpyTimeSeriesDict()
    has_csd = hasattr(base_dict, "csd_matrix")
    
    return log_result("Monkey Patch (csd_matrix on gwpy.TimeSeriesDict)", has_csd, "Patch applied" if has_csd else "Patch missing")

if __name__ == "__main__":
    failures = []
    
    if not test_collection_type_preservation(): failures.append("Collection Type Preservation")
    if not test_time_array_handling(): failures.append("Time Array Handling")
    if not test_monkey_patching(): failures.append("Monkey Patching")
    
    print("\n========================================")
    if failures:
        print(f"SUMMARY: Found {len(failures)} issues.")
        print("Please record these in FIX_REQUIRED_LIST.md")
    else:
        print("SUMMARY: All collection/time checks passed.")
