"""
Direct unit tests for gwexpy/interop/netcdf4_.py using MagicMock.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries


@pytest.fixture
def mock_netcdf4():
    """Mock the netCDF4 module."""
    mock_nc = MagicMock()
    with patch.dict(sys.modules, {"netCDF4": mock_nc}):
        # Mock require_optional to always pass
        with patch("gwexpy.interop.netcdf4_.require_optional", return_value=mock_nc):
            yield mock_nc


class TestToNetCDF4Direct:
    def test_normal_write(self, mock_netcdf4):
        from gwexpy.interop.netcdf4_ import to_netcdf4
        
        ds = MagicMock()
        ds.dimensions = {}
        ds.variables = {}
        
        ts = TimeSeries([1, 2, 3], t0=100 * u.s, dt=0.5 * u.s, unit="m", name="test_ch")
        
        to_netcdf4(ts, ds, "ch_data")
        
        # Verify dimension creation
        ds.createDimension.assert_called_with("time", 3)
        
        # Verify variable creation
        ds.createVariable.assert_called_with("ch_data", ts.dtype, ("time",))
        
        # Verify attribute assignment
        var = ds.createVariable.return_value
        assert var.t0 == 100.0
        assert var.dt == 0.5
        assert var.units == "m"
        assert var.long_name == "test_ch"

    def test_overwrite_false_raises(self, mock_netcdf4):
        from gwexpy.interop.netcdf4_ import to_netcdf4
        
        ds = MagicMock()
        ds.variables = {"exists": MagicMock()}
        
        ts = TimeSeries([1], t0=0, dt=1)
        with pytest.raises(ValueError, match="Variable exists exists"):
            to_netcdf4(ts, ds, "exists", overwrite=False)

    def test_overwrite_true_reuses(self, mock_netcdf4):
        from gwexpy.interop.netcdf4_ import to_netcdf4
        
        ds = MagicMock()
        existing_var = MagicMock()
        ds.variables = {"exists": existing_var}
        ds.dimensions = {"time": MagicMock()}
        
        ts = TimeSeries([1, 2], t0=0, dt=1)
        # Should not raise
        to_netcdf4(ts, ds, "exists", overwrite=True)
        # Should reuse existing_var
        existing_var.__setitem__.assert_called()


class TestFromNetCDF4Direct:
    def test_normal_read(self, mock_netcdf4):
        from gwexpy.interop.netcdf4_ import from_netcdf4
        
        var = MagicMock()
        var.t0 = 50.0
        var.dt = 0.1
        var.units = "V"
        var.long_name = "voltage"
        var.__getitem__.return_value = np.array([1.0, 2.0, 3.0])
        
        ds = MagicMock()
        ds.variables = {"ch1": var}
        
        ts = from_netcdf4(TimeSeries, ds, "ch1")
        assert isinstance(ts, TimeSeries)
        assert ts.t0.value == 50.0
        assert ts.dt.value == 0.1
        assert ts.unit == u.V
        assert ts.name == "voltage"
        np.testing.assert_array_equal(ts.value, [1.0, 2.0, 3.0])

    def test_masked_array_filling(self, mock_netcdf4):
        from gwexpy.interop.netcdf4_ import from_netcdf4
        
        # Create a masked array
        data = np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        
        var = MagicMock()
        var.__getitem__.return_value = data
        # No attributes -> defaults
        # We must delete them to avoid MagicMock returning sub-mocks
        del var.t0
        del var.dt
        del var.units
        del var.long_name
        
        ds = MagicMock()
        ds.variables = {"ch1": var}
        
        ts = from_netcdf4(TimeSeries, ds, "ch1")
        # Masked value (index 1) should be filled with NaN
        assert np.isnan(ts.value[1])
        assert ts.value[0] == 1.0
        assert ts.value[2] == 3.0

    def test_attribute_defaults(self, mock_netcdf4):
        from gwexpy.interop.netcdf4_ import from_netcdf4
        
        var = MagicMock()
        var.__getitem__.return_value = np.array([1.0])
        # Clear all attributes
        del var.t0
        del var.dt
        del var.units
        del var.long_name
        
        ds = MagicMock()
        ds.variables = {"ch": var}
        
        ts = from_netcdf4(TimeSeries, ds, "ch")
        assert ts.t0.value == 0.0
        assert ts.dt.value == 1.0
        assert ts.unit == u.dimensionless_unscaled
        assert ts.name == "ch"  # var_name as fallback
