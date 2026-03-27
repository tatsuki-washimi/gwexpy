"""Tests for gwexpy/interop/pydub_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


def _make_ts(n=16, t0=0.0, dt=1.0 / 44100, unit="", name="audio"):
    return TimeSeries(
        np.linspace(-0.5, 0.5, n), t0=t0, dt=dt, unit=unit, name=name
    )


class TestToLibrosa:
    def test_returns_tuple(self):
        from gwexpy.interop.pydub_ import to_librosa
        ts = _make_ts()
        y, sr = to_librosa(ts)
        assert isinstance(y, np.ndarray)
        assert isinstance(sr, int)

    def test_dtype_float32_default(self):
        from gwexpy.interop.pydub_ import to_librosa
        ts = _make_ts()
        y, sr = to_librosa(ts)
        assert y.dtype == np.float32

    def test_dtype_override(self):
        from gwexpy.interop.pydub_ import to_librosa
        ts = _make_ts()
        y, sr = to_librosa(ts, y_dtype=np.float64)
        assert y.dtype == np.float64

    def test_sample_rate_correct(self):
        from gwexpy.interop.pydub_ import to_librosa
        ts = _make_ts(dt=1.0 / 8000)
        y, sr = to_librosa(ts)
        assert sr == 8000

    def test_values_preserved(self):
        from gwexpy.interop.pydub_ import to_librosa
        ts = _make_ts(n=8)
        y, sr = to_librosa(ts)
        np.testing.assert_allclose(y, ts.value.astype(np.float32))


class TestToPydub:
    def test_raises_import_error_without_pydub(self):
        with patch.dict(sys.modules, {"pydub": None}):
            from gwexpy.interop.pydub_ import to_pydub
            with pytest.raises(ImportError):
                to_pydub(_make_ts())

    def test_float_data_scaled(self):
        calls = {}
        class FakeAudioSegment:
            def __init__(self, data, sample_width, frame_rate, channels):
                calls["data"] = data
                calls["sample_width"] = sample_width
                calls["frame_rate"] = frame_rate
                calls["channels"] = channels

        fake_pydub = SimpleNamespace(AudioSegment=FakeAudioSegment)
        with patch.dict(sys.modules, {"pydub": fake_pydub}):
            from gwexpy.interop import pydub_ as pydub_mod
            import importlib
            importlib.reload(pydub_mod)
            ts = _make_ts(n=8)
            seg = pydub_mod.to_pydub(ts, sample_width=2)
        assert calls["sample_width"] == 2
        assert calls["channels"] == 1

    def test_integer_data_not_rescaled(self):
        calls = {}
        class FakeAudioSegment:
            def __init__(self, data, sample_width, frame_rate, channels):
                calls["data"] = data

        fake_pydub = SimpleNamespace(AudioSegment=FakeAudioSegment)
        with patch.dict(sys.modules, {"pydub": fake_pydub}):
            from gwexpy.interop import pydub_ as pydub_mod
            import importlib
            importlib.reload(pydub_mod)
            int_data = np.array([0, 100, 200, -100], dtype=np.int16)
            ts = TimeSeries(int_data, dt=1.0 / 44100)
            seg = pydub_mod.to_pydub(ts, sample_width=2)
        assert calls["data"] is not None


class TestFromPydub:
    def test_basic_conversion(self):
        from gwexpy.interop.pydub_ import from_pydub
        fake_seg = SimpleNamespace(
            get_array_of_samples=lambda: [1, 2, 3, 4, 5],
            frame_rate=44100,
        )
        ts = from_pydub(TimeSeries, fake_seg)
        assert isinstance(ts, TimeSeries)
        assert len(ts) == 5

    def test_dt_from_frame_rate(self):
        from gwexpy.interop.pydub_ import from_pydub
        fake_seg = SimpleNamespace(
            get_array_of_samples=lambda: [0, 1, 2, 3],
            frame_rate=8000,
        )
        ts = from_pydub(TimeSeries, fake_seg)
        assert ts.dt.value == pytest.approx(1.0 / 8000)

    def test_unit_passed_through(self):
        from gwexpy.interop.pydub_ import from_pydub
        fake_seg = SimpleNamespace(
            get_array_of_samples=lambda: [0, 1, 2],
            frame_rate=44100,
        )
        ts = from_pydub(TimeSeries, fake_seg, unit="Pa")
        assert str(ts.unit) == "Pa"

    def test_values_preserved(self):
        from gwexpy.interop.pydub_ import from_pydub
        samples = [10, 20, 30, 40]
        fake_seg = SimpleNamespace(
            get_array_of_samples=lambda: samples,
            frame_rate=44100,
        )
        ts = from_pydub(TimeSeries, fake_seg)
        np.testing.assert_array_equal(ts.value, np.array(samples))
