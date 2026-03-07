"""Tests that optional-dep readers raise clear ImportError when deps are missing."""

import sys
from unittest import mock

import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


class TestZarrImportGuard:
    def test_zarr_import_error(self):
        with mock.patch.dict(sys.modules, {"zarr": None}):
            from gwexpy.timeseries.io import zarr_ as zarr_mod
            # Force reimport of the lazy import
            with pytest.raises(ImportError, match="zarr"):
                zarr_mod._import_zarr()


class TestNetcdf4ImportGuard:
    def test_xarray_import_error(self):
        with mock.patch.dict(sys.modules, {"xarray": None}):
            from gwexpy.timeseries.io import netcdf4_ as nc_mod
            with pytest.raises(ImportError, match="xarray"):
                nc_mod._import_xarray()


class TestTdmsImportGuard:
    def test_nptdms_import_error(self):
        with mock.patch.dict(sys.modules, {"nptdms": None}):
            from gwexpy.timeseries.io import tdms as tdms_mod
            with pytest.raises(ImportError, match="npTDMS"):
                tdms_mod._import_nptdms()


class TestAudioImportGuard:
    def test_pydub_import_error(self):
        with mock.patch.dict(sys.modules, {"pydub": None}):
            from gwexpy.timeseries.io import audio as audio_mod
            with pytest.raises(ImportError, match="pydub"):
                audio_mod._import_pydub()
