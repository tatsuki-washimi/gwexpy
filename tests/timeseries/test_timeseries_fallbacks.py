"""
Tests for gwexpy.timeseries core fallbacks and API shortcuts.

Covers:
- TimeSeries.__new__ coercion fallbacks (invalid xunit, non-convertible units)
- TimeSeriesCore.append fallback for raw GWpy objects
- TimeSeries finalize(None) and _get_meta_for_constructor
- Pickle support (__reduce_ex__)
- API Shortcuts (SimPEG, Control, ARIMA) delegation and kwargs transparency
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries._core import TimeSeriesCore


# ---------------------------------------------------------------------------
# TimeSeries.__new__ Coercion Fallbacks
# ---------------------------------------------------------------------------

class TestTimeSeriesNewFallbacks:
    def test_invalid_xunit_disables_coercion(self):
        # We want to check that should_coerce becomes False and skip normalization.
        with patch("gwexpy.timeseries.utils._coerce_t0_gps") as mock_coerce:
            ts = TimeSeries([1, 2, 3], t0="1234567890.0", xunit="m")
            # should_coerce is False because "m" isn't "s".
            mock_coerce.assert_not_called()
            assert ts.t0.value == 1234567890.0

    def test_xunit_value_error_disables_coercion(self):
        # Trigger ValueError in u.Unit(xunit) at line 104 in timeseries.py
        with patch("gwpy.timeseries.TimeSeries.__new__") as mock_super_new:
            mock_super_new.return_value = np.array([1, 2, 3]).view(TimeSeries)
            with patch("gwexpy.timeseries.utils._coerce_t0_gps") as mock_coerce:
                ts = TimeSeries([1, 2, 3], t0=100.0, xunit="invalid_unit_string")
                # coercion should be skipped
                mock_coerce.assert_not_called()
                # super().__new__ should still be called with the invalid unit
                args, kwargs = mock_super_new.call_args
                assert kwargs["xunit"] == "invalid_unit_string"

    def test_target_unit_resolution_value_error(self):
        # Trigger ValueError in u.Unit(xunit) at line 124 in timeseries.py
        # should_coerce=True (default), but target_unit resolution fails.
        with patch("gwpy.timeseries.TimeSeries.__new__") as mock_super_new:
            mock_super_new.return_value = np.array([1, 2, 3]).view(TimeSeries)
            # Use data that doesn't trigger should_coerce=False early (no dt)
            ts = TimeSeries([1, 2, 3], t0=1234.5, xunit="invalid_again")
            # It should fallback to target_unit = u.s
            args, kwargs = mock_super_new.call_args
            # t0 coerced to float via float(t0_q.to(u.s).value) if coerced
            assert isinstance(kwargs["t0"], float)

    def test_non_convertible_target_unit_keeps_quantity(self):
        # Case where should_coerce is True, but normalization to target_unit fails.
        # This triggers UnitConversionError at line 135/143.
        with patch("gwpy.timeseries.TimeSeries.__new__") as mock_super_new:
            mock_super_new.return_value = np.array([1, 2, 3]).view(TimeSeries)
            
            with patch("gwexpy.timeseries.utils._coerce_t0_gps") as mock_coerce:
                mock_coerce.return_value = 10.0 * u.m  # non-convertible to 's'
                # Pass dt='s' to set target_unit='s'
                ts = TimeSeries([1, 2, 3], t0=1234.5, dt=1.0*u.s)
                
                # Check what was passed to super().__new__
                args, kwargs = mock_super_new.call_args
                assert isinstance(kwargs["t0"], u.Quantity)
                assert kwargs["t0"].unit == u.m

    def test_epoch_coercion_fallback(self):
        # Same for epoch
        with patch("gwpy.timeseries.TimeSeries.__new__") as mock_super_new:
            mock_super_new.return_value = np.array([1, 2, 3]).view(TimeSeries)
            with patch("gwexpy.timeseries.utils._coerce_t0_gps") as mock_coerce:
                mock_coerce.return_value = 20.0 * u.kg
                ts = TimeSeries([1, 2, 3], epoch=5678.9)
                args, kwargs = mock_super_new.call_args
                assert kwargs["epoch"].unit == u.kg

    def test_finalize_with_none_obj(self):
        # Triggers line 156 in timeseries.py
        ts = TimeSeries([1, 2, 3])
        ts.__array_finalize__(None)
        # Should return safely

    def test_get_meta_for_constructor(self):
        # Triggers line 76 in timeseries.py
        ts = TimeSeries([1, 2, 3], t0=100.0, dt=0.5)
        meta = ts._get_meta_for_constructor()
        assert meta["t0"].value == 100.0
        assert meta["dt"].value == 0.5

    def test_pickle_reduce(self):
        # Triggers line 164-166 in timeseries.py
        import pickle
        ts = TimeSeries([1, 2, 3], name="pickle_test")
        pickled = pickle.dumps(ts)
        unpickled = pickle.loads(pickled)
        assert unpickled.name == "pickle_test"
        np.testing.assert_array_equal(unpickled.value, ts.value)


# ---------------------------------------------------------------------------
# TimeSeriesCore.append Fallback
# ---------------------------------------------------------------------------

class TestTimeSeriesCoreAppendFallback:
    def test_append_gwpy_object_fallback(self):
        # Create a mock object representing a raw GWpy TimeSeries
        mock_res = MagicMock()
        mock_res.value = np.array([1, 2, 3, 4, 5, 6])
        mock_res.times = np.arange(6) * 0.1
        mock_res.unit = u.m
        mock_res.name = "combined"
        mock_res.channel = "L1:TEST"
        
        ts1 = TimeSeries([1, 2, 3], dt=0.1, unit=u.m, name="ch1")
        ts2 = TimeSeries([4, 5, 6], dt=0.1, unit=u.m, name="ch2")
        
        with patch("gwpy.timeseries.TimeSeries.append") as mock_append:
            mock_append.return_value = mock_res
            result = ts1.append(ts2, inplace=False)
            
            assert isinstance(result, TimeSeries)
            assert result.name == "combined"
            assert result.unit == u.m
            np.testing.assert_array_equal(result.value, [1, 2, 3, 4, 5, 6])


# ---------------------------------------------------------------------------
# API Shortcuts (Delegation & Spies)
# ---------------------------------------------------------------------------

class TestTimeSeriesAPIShortcuts:
    def test_to_simpeg_delegation(self):
        with patch("gwexpy.interop.to_simpeg") as mock_to:
            ts = TimeSeries([1, 2, 3])
            ts.to_simpeg(location=[1, 2, 3], rx_type="TestRx", extra_param="val")
            mock_to.assert_called_once_with(
                ts, location=[1, 2, 3], rx_type="TestRx", orientation="x", extra_param="val"
            )

    def test_from_simpeg_delegation(self):
        with patch("gwexpy.interop.from_simpeg") as mock_from:
            TimeSeries.from_simpeg("mock_obj", param=1)
            mock_from.assert_called_once_with(TimeSeries, "mock_obj", param=1)

    def test_from_control_delegation(self):
        with patch("gwexpy.interop.from_control_response") as mock_from:
            TimeSeries.from_control("mock_resp", opt="A")
            mock_from.assert_called_once_with(TimeSeries, "mock_resp", opt="A")

    def test_arima_delegation(self):
        with patch("gwexpy.timeseries.arima.fit_arima") as mock_fit:
            ts = TimeSeries([1, 2, 3])
            ts.arima(order=(2, 1, 0), auto=True, my_kw="test")
            mock_fit.assert_called_once_with(
                ts, order=(2, 1, 0), seasonal_order=None, auto=True, my_kw="test"
            )

    def test_ar_ma_arma_spies_and_kwargs(self):
        ts = TimeSeries([1, 2, 3])
        with patch("gwexpy.timeseries.arima.fit_arima") as mock_fit:
            # AR(2)
            ts.ar(p=2, trend="c")
            mock_fit.assert_called_with(ts, order=(2, 0, 0), seasonal_order=None, auto=False, trend="c")
            
            # MA(3)
            ts.ma(q=3, method="mle")
            mock_fit.assert_called_with(ts, order=(0, 0, 3), seasonal_order=None, auto=False, method="mle")
            
            # ARMA(1, 2)
            ts.arma(p=1, q=2, solver="newton")
            mock_fit.assert_called_with(ts, order=(1, 0, 2), seasonal_order=None, auto=False, solver="newton")
