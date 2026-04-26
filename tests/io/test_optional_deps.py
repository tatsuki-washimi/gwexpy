"""Tests that optional-dep readers raise clear ImportError when deps are missing."""

import sys
from unittest import mock

import pytest

import gwexpy

gwexpy.register_all()


class TestZarrImportGuard:
    def test_zarr_import_error(self):
        with mock.patch.dict(sys.modules, {"zarr": None}):
            from gwexpy.timeseries.io import zarr_ as zarr_mod
            with pytest.raises(ImportError, match="zarr"):
                zarr_mod._import_zarr()

    def test_zarr_error_mentions_extra(self):
        with mock.patch.dict(sys.modules, {"zarr": None}):
            from gwexpy.timeseries.io import zarr_ as zarr_mod
            with pytest.raises(ImportError, match=r"gwexpy\[zarr\]"):
                zarr_mod._import_zarr()


class TestNetcdf4ImportGuard:
    def test_xarray_import_error(self):
        with mock.patch.dict(sys.modules, {"xarray": None}):
            from gwexpy.timeseries.io import netcdf4_ as nc_mod
            with pytest.raises(ImportError, match="xarray"):
                nc_mod._import_xarray()

    def test_xarray_error_mentions_extra(self):
        with mock.patch.dict(sys.modules, {"xarray": None}):
            from gwexpy.timeseries.io import netcdf4_ as nc_mod
            with pytest.raises(ImportError, match=r"gwexpy\[netcdf4\]"):
                nc_mod._import_xarray()


class TestTdmsImportGuard:
    def test_nptdms_import_error(self):
        with mock.patch.dict(sys.modules, {"nptdms": None}):
            from gwexpy.timeseries.io import tdms as tdms_mod
            with pytest.raises(ImportError, match="npTDMS"):
                tdms_mod._import_nptdms()

    def test_nptdms_error_mentions_extra(self):
        with mock.patch.dict(sys.modules, {"nptdms": None}):
            from gwexpy.timeseries.io import tdms as tdms_mod
            with pytest.raises(ImportError, match=r"gwexpy\[io\]"):
                tdms_mod._import_nptdms()


class TestAudioImportGuard:
    def test_pydub_import_error(self):
        with mock.patch.dict(sys.modules, {"pydub": None}):
            from gwexpy.timeseries.io import audio as audio_mod
            with pytest.raises(ImportError, match="pydub"):
                audio_mod._import_pydub()

    def test_pydub_error_mentions_extra(self):
        with mock.patch.dict(sys.modules, {"pydub": None}):
            from gwexpy.timeseries.io import audio as audio_mod
            with pytest.raises(ImportError, match=r"gwexpy\[audio\]"):
                audio_mod._import_pydub()


class TestSeismicImportGuard:
    def test_obspy_import_error(self):
        with mock.patch.dict(sys.modules, {"obspy": None}):
            from gwexpy.timeseries.io import seismic as seismic_mod
            with pytest.raises(ImportError, match="(?i)obspy"):
                seismic_mod._import_obspy()

    def test_obspy_error_mentions_extra(self):
        with mock.patch.dict(sys.modules, {"obspy": None}):
            from gwexpy.timeseries.io import seismic as seismic_mod
            with pytest.raises(ImportError, match=r"gwexpy\[seismic\]"):
                seismic_mod._import_obspy()
