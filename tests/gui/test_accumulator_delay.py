import time

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwexpy.gui.streaming import SpectralAccumulator


def test_spectral_accumulator_channel_delay():
    """
    Verify behavior when one channel lags behind another.
    Current expectation: System waits for the slowest channel (blocking behavior).
    """
    acc = SpectralAccumulator()

    # Configure
    params = {
        "bw": 1.0, # 1s FFT
        "overlap": 0.0,
        "window": "boxcar",
        "avg_type": "Infinite"
    }

    # Trace 1: CH_A (Fast)
    # Trace 2: CH_B (Slow/Missing)
    active_traces = [
        {"active": True, "ch_a": "CH_A", "ch_b": None, "graph_type": "Time Series"},
        {"active": True, "ch_a": "CH_B", "ch_b": None, "graph_type": "Time Series"}
    ]

    # Available channels: Both A and B are expected
    available = ["CH_A", "CH_B"]

    acc.configure(params, active_traces, available_channels=available)

    # Create dummy data packet
    # 1 second of data
    fs = 16
    n_samples = 16
    t0 = 1000.0
    dt = 1.0/fs

    data_a = np.random.normal(0, 1, n_samples)

    packet_a = {
        "CH_A": {
            "data": data_a,
            "step": dt,
            "gps_start": t0
        }
    }

    # 1. Feed CH_A ONLY
    acc.add_chunk(packet_a)

    # Check results
    results = acc.get_results()

    # Expectation: CH_A should NOT produce results yet because CH_B is blocking the buffer processing
    # logical step: _process_buffers checks ALL active traces for readiness.

    res_a = results[0]
    _ = results[1]

    print(f"DEBUG: Res A is {res_a}")

    # Assertion to confirm BLOCKING behavior
    if res_a is None:
        print("Confirmed: CH_A result is blocked by missing CH_B data.")
    else:
        pytest.fail("Unexpected: CH_A result updated despite missing CH_B data.")

    # 2. Feed CH_B
    data_b = np.random.normal(0, 1, n_samples)
    packet_b = {
        "CH_B": {
            "data": data_b,
            "step": dt,
            "gps_start": t0
        }
    }
    acc.add_chunk(packet_b)

    results_after = acc.get_results()
    if results_after[0] is not None:
         print("Confirmed: Results updated after CH_B arrived.")
    else:
         pytest.fail("Failed: Results still not updated even after CH_B arrived.")

if __name__ == "__main__":
    test_spectral_accumulator_channel_delay()
