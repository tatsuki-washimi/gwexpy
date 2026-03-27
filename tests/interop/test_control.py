"""Tests for gwexpy/interop/control_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gwexpy.interop.control_ import from_control_frd, from_control_response, to_control_frd
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries


def _make_fs(n=8, f0=1.0, df=1.0):
    return FrequencySeries(np.ones(n, dtype=complex), f0=f0, df=df)


def _make_fake_ctl():
    """Return a fake `control` module with frd() support."""
    class FakeFRD:
        def __init__(self, data, omega):
            self.frdata = np.array(data)[np.newaxis, np.newaxis, :]
            self.omega = np.asarray(omega)
            self.noutputs = 1
            self.ninputs = 1

    fake = SimpleNamespace(frd=lambda data, omega: FakeFRD(data, omega))
    return fake


class TestToControlFrd:
    def test_raises_import_error_without_control(self):
        with patch.dict(sys.modules, {"control": None}):
            with pytest.raises(ImportError):
                to_control_frd(_make_fs())

    def test_rad_per_s_conversion(self):
        fake_ctl = _make_fake_ctl()
        fs = _make_fs(n=4, f0=1.0, df=1.0)
        with patch.dict(sys.modules, {"control": fake_ctl}):
            frd = to_control_frd(fs, frequency_unit="rad/s")
        # Frequencies in rad/s should be 2*pi times Hz values
        expected_omega = fs.frequencies.value * 2 * np.pi
        np.testing.assert_allclose(frd.omega, expected_omega)

    def test_hz_unit_no_conversion(self):
        fake_ctl = _make_fake_ctl()
        fs = _make_fs(n=4, f0=1.0, df=1.0)
        with patch.dict(sys.modules, {"control": fake_ctl}):
            frd = to_control_frd(fs, frequency_unit="Hz")
        np.testing.assert_allclose(frd.omega, fs.frequencies.value)

    def test_invalid_frequency_unit_raises(self):
        fake_ctl = _make_fake_ctl()
        fs = _make_fs()
        with patch.dict(sys.modules, {"control": fake_ctl}):
            with pytest.raises(ValueError, match="frequency_unit"):
                to_control_frd(fs, frequency_unit="kHz")

    def test_sysname_set_from_name(self):
        class FakeFRDWithSysname:
            def __init__(self, data, omega):
                self.frdata = np.array(data)[np.newaxis, np.newaxis, :]
                self.omega = np.asarray(omega)
                self.sysname = None

        fake_ctl = SimpleNamespace(frd=lambda d, o: FakeFRDWithSysname(d, o))
        from gwexpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(4, dtype=complex), f0=1.0, df=1.0, name="myfs")
        with patch.dict(sys.modules, {"control": fake_ctl}):
            frd = to_control_frd(fs)
        assert frd.sysname == "myfs"


class TestFromControlFrd:
    def _make_frd(self, n=8, f0=1.0, df=1.0):
        omega = (np.arange(n) * df + f0) * 2 * np.pi
        data = np.ones((1, 1, n), dtype=complex)
        frd = SimpleNamespace(
            omega=omega,
            frdata=data,
            noutputs=1,
            ninputs=1,
        )
        return frd

    def test_basic_roundtrip(self):
        frd = self._make_frd(n=8)
        fs = from_control_frd(FrequencySeries, frd, frequency_unit="rad/s")
        assert isinstance(fs, FrequencySeries)
        assert len(fs) == 8

    def test_hz_unit_passthrough(self):
        # frequency_unit="Hz" still divides by 2*pi (FRD.omega always rad/s internally)
        frd = self._make_frd(n=4)
        fs = from_control_frd(FrequencySeries, frd, frequency_unit="Hz")
        assert len(fs) == 4

    def test_invalid_frequency_unit_raises(self):
        frd = self._make_frd()
        with pytest.raises(ValueError, match="frequency_unit"):
            from_control_frd(FrequencySeries, frd, frequency_unit="kHz")

    def test_uses_fresp_fallback(self):
        frd = self._make_frd(n=4)
        del frd.frdata
        frd.fresp = np.ones((1, 1, 4), dtype=complex)
        fs = from_control_frd(FrequencySeries, frd, frequency_unit="rad/s")
        assert len(fs) == 4

    def test_uses_underscore_fresp_fallback(self):
        frd = self._make_frd(n=4)
        del frd.frdata
        frd._fresp = np.ones((1, 1, 4), dtype=complex)
        fs = from_control_frd(FrequencySeries, frd, frequency_unit="rad/s")
        assert len(fs) == 4

    def test_irregular_frequencies(self):
        # Irregular spacing (non-uniform)
        omega = np.array([1.0, 3.0, 8.0, 20.0]) * 2 * np.pi
        frd = SimpleNamespace(
            omega=omega,
            frdata=np.ones((1, 1, 4), dtype=complex),
        )
        fs = from_control_frd(FrequencySeries, frd, frequency_unit="rad/s")
        assert len(fs) == 4

    def test_single_point(self):
        frd = SimpleNamespace(
            omega=np.array([2 * np.pi]),
            frdata=np.ones((1, 1, 1), dtype=complex),
        )
        fs = from_control_frd(FrequencySeries, frd, frequency_unit="rad/s")
        assert len(fs) == 1


class TestFromControlResponse:
    def _make_siso_response(self, n=10, dt=0.1, t0=0.0, label=None):
        time = np.arange(n) * dt + t0
        outputs = np.ones((1, n))
        resp = SimpleNamespace(
            time=time,
            outputs=outputs,
            noutputs=1,
            output_labels=[label] if label else None,
        )
        return resp

    def _make_mimo_response(self, nout=2, n=10, dt=0.1):
        time = np.arange(n) * dt
        outputs = np.ones((nout, n))
        resp = SimpleNamespace(
            time=time,
            outputs=outputs,
            noutputs=nout,
            output_labels=[f"ch{i}" for i in range(nout)],
        )
        return resp

    def test_siso_returns_timeseries(self):
        resp = self._make_siso_response(n=8)
        ts = from_control_response(TimeSeries, resp)
        assert isinstance(ts, TimeSeries)
        assert len(ts) == 8

    def test_siso_dt_and_t0(self):
        resp = self._make_siso_response(n=5, dt=0.5, t0=10.0)
        ts = from_control_response(TimeSeries, resp)
        assert ts.dt.value == pytest.approx(0.5)
        assert ts.t0.value == pytest.approx(10.0)

    def test_siso_with_label(self):
        resp = self._make_siso_response(n=4, label="myout")
        ts = from_control_response(TimeSeries, resp)
        assert ts.name == "myout"

    def test_siso_no_label_defaults_to_output(self):
        resp = self._make_siso_response(n=4)
        ts = from_control_response(TimeSeries, resp)
        assert ts.name == "output"

    def test_siso_single_point_dt_defaults(self):
        resp = SimpleNamespace(
            time=np.array([0.0]),
            outputs=np.array([[1.0]]),
            noutputs=1,
            output_labels=None,
        )
        ts = from_control_response(TimeSeries, resp)
        assert len(ts) == 1
        assert ts.dt.value == pytest.approx(1.0)

    def test_mimo_returns_dict(self):
        from gwexpy.timeseries import TimeSeriesDict
        resp = self._make_mimo_response(nout=3, n=6)
        tsd = from_control_response(TimeSeries, resp)
        assert isinstance(tsd, TimeSeriesDict)
        assert "ch0" in tsd
        assert "ch2" in tsd
        assert len(tsd["ch0"]) == 6

    def test_mimo_no_labels(self):
        from gwexpy.timeseries import TimeSeriesDict
        resp = SimpleNamespace(
            time=np.arange(5) * 0.1,
            outputs=np.ones((2, 5)),
            noutputs=2,
            output_labels=None,
        )
        tsd = from_control_response(TimeSeries, resp)
        assert "output_0" in tsd
        assert "output_1" in tsd
