"""Tests for SDyPy/pyuff interoperability.

Uses mock UFF dataset dicts. Does NOT require pyuff to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries
from gwexpy.interop.sdypy_ import from_uff_dataset55, from_uff_dataset58

N_SAMPLES = 256


# ---------------------------------------------------------------------------
# Mock UFF data
# ---------------------------------------------------------------------------


def _make_dataset58_time():
    """UFF dataset 58 in time domain (func_type=1)."""
    dt = 1e-3
    x = np.arange(N_SAMPLES) * dt
    data = np.random.default_rng(0).random(N_SAMPLES)
    return {
        "type": 58,
        "func_type": 1,
        "x": x,
        "data": data,
        "id1": "Time response test",
        "rsp_dir": 1,
        "ref_dir": 3,
    }


def _make_dataset58_frf():
    """UFF dataset 58 in frequency domain (func_type=4 = FRF)."""
    df = 1.0
    x = np.arange(N_SAMPLES) * df
    rng = np.random.default_rng(1)
    data = rng.random(N_SAMPLES) + 1j * rng.random(N_SAMPLES)
    return {
        "type": 58,
        "func_type": 4,
        "x": x,
        "data": data,
        "id1": "FRF test",
    }


def _make_dataset55():
    """UFF dataset 55 — modal parameters for 2 modes, 4 nodes × 3 DOFs."""
    return {
        "type": 55,
        "modal_freq": np.array([12.5, 37.8]),
        "modal_damp": np.array([0.015, 0.028]),
        "node_nums": np.array([1, 2, 3, 4]),
        "r1": np.random.default_rng(2).random((2, 4)),  # X per mode
        "r2": np.random.default_rng(3).random((2, 4)),  # Y per mode
        "r3": np.random.default_rng(4).random((2, 4)),  # Z per mode
    }


# ---------------------------------------------------------------------------
# from_uff_dataset58
# ---------------------------------------------------------------------------


class TestFromUffDataset58:
    def test_time_domain_returns_ts(self):
        uff = _make_dataset58_time()
        result = from_uff_dataset58(TimeSeries, uff)
        assert isinstance(result, TimeSeries)

    def test_time_domain_length(self):
        uff = _make_dataset58_time()
        result = from_uff_dataset58(TimeSeries, uff)
        assert len(result) == N_SAMPLES

    def test_frf_returns_fs(self):
        uff = _make_dataset58_frf()
        result = from_uff_dataset58(FrequencySeries, uff)
        assert isinstance(result, FrequencySeries)

    def test_frf_complex(self):
        uff = _make_dataset58_frf()
        result = from_uff_dataset58(FrequencySeries, uff)
        assert np.iscomplexobj(result.value)

    def test_name_preserved(self):
        uff = _make_dataset58_time()
        result = from_uff_dataset58(TimeSeries, uff)
        assert result.name == "Time response test"

    def test_unit_from_response_type(self):
        uff = _make_dataset58_time()
        result = from_uff_dataset58(TimeSeries, uff, response_type="accel")
        assert result.unit is not None

    def test_auto_detect_time(self):
        uff = _make_dataset58_time()
        # func_type=1 → time domain, even if cls=FrequencySeries passed
        # (cls overrides auto-detect)
        result = from_uff_dataset58(TimeSeries, uff)
        assert isinstance(result, TimeSeries)


# ---------------------------------------------------------------------------
# from_uff_dataset55
# ---------------------------------------------------------------------------


class TestFromUffDataset55:
    def test_returns_dataframe(self):
        uff = _make_dataset55()
        df = from_uff_dataset55(uff)
        assert hasattr(df, "columns")

    def test_frequency_in_attrs(self):
        uff = _make_dataset55()
        df = from_uff_dataset55(uff)
        assert "frequency_Hz" in df.attrs
        np.testing.assert_allclose(df.attrs["frequency_Hz"], [12.5, 37.8])

    def test_mode_columns(self):
        uff = _make_dataset55()
        df = from_uff_dataset55(uff)
        assert "mode_1" in df.columns
        assert "mode_2" in df.columns

    def test_row_count(self):
        uff = _make_dataset55()
        df = from_uff_dataset55(uff)
        # 4 nodes × 3 directions = 12 DOFs
        assert len(df) == 12
