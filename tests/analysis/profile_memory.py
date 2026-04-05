import gc
import os

import numpy as np
import psutil

from gwexpy.analysis.coupling import CouplingFunctionAnalysis, PercentileThreshold
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def get_rss():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MB

def profile_bkg_memory():
    fs = 16384.0
    duration = 100.0  # 100s of high-rate data
    t = np.arange(0, duration, 1/fs)

    print(f"Initial RSS: {get_rss():.2f} MB")

    # Create large-ish data
    data = np.random.normal(0, 1, len(t))
    ts = TimeSeries(data, sample_rate=fs, name="Large")
    print(f"Data created. RSS: {get_rss():.2f} MB")

    analysis = CouplingFunctionAnalysis()

    # 1. Without limit (should take significant memory)
    # 100s, fft=1.0, stride=1.0 -> 100 rows.
    # fs=16384 -> 8193 bins.
    # 100 * 8193 * 8 bytes approx 6.5 MB pure data.
    # Plus overhead, easily 10-20 MB.

    # 2. With tight limit (force stride increase)
    print("\n--- Running with 1MB limit ---")
    # This should force stride to increase to inter-leave results
    analysis.compute(
        TimeSeriesDict({"W": ts, "T": ts}),
        TimeSeriesDict({"W": ts, "T": ts}),
        fftlength=1.0,
        threshold_target=PercentileThreshold(),
        memory_limit=1 * 1024**3, # 1MB limit (Wait, 1024**3 is 1GB, 1024**2 is 1MB. Input is in bytes)
        bkg_stride=1.0
    )
    print(f"RSS after 1MB limit run: {get_rss():.2f} MB")

    # 3. With extremely tight limit (force even more stride)
    print("\n--- Running with 1KB limit (Extreme) ---")
    analysis.compute(
        TimeSeriesDict({"W": ts, "T": ts}),
        TimeSeriesDict({"W": ts, "T": ts}),
        fftlength=1.0,
        threshold_target=PercentileThreshold(),
        memory_limit=1024, # 1KB
        bkg_stride=1.0
    )
    print(f"RSS after 1KB limit run: {get_rss():.2f} MB")

if __name__ == "__main__":
    profile_bkg_memory()
