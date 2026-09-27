"""Tests that optional-dep readers raise clear ImportError when deps are missing."""

import sys
from unittest import mock

import numpy as np
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

    def test_public_auto_zarr_reads_preserve_missing_backend_importerror(
        self, tmp_path
    ):
        from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

        path = tmp_path / "missing-backend.zarr"
        path.mkdir()
        readers = (
            lambda: TimeSeries.read(path),
            lambda: TimeSeriesDict.read(path),
            lambda: TimeSeriesMatrix.read(path),
        )

        with mock.patch.dict(sys.modules, {"zarr": None}):
            for reader in readers:
                with pytest.raises(ImportError, match="zarr is required"):
                    reader()

    def test_public_matrix_auto_zarr_write_preserves_missing_backend_importerror(
        self, tmp_path
    ):
        from gwexpy.timeseries import TimeSeriesMatrix

        path = tmp_path / "missing-backend-write.zarr"
        matrix = TimeSeriesMatrix(
            np.ones((1, 1, 3)), t0=1_234_567_890.25, sample_rate=16.0
        )

        with mock.patch.dict(sys.modules, {"zarr": None}):
            with pytest.raises(ImportError, match="zarr is required"):
                matrix.write(path)


@pytest.mark.parametrize(
    "format_name,suffix,missing_module,required_modules",
    [
        pytest.param("nc", ".nc", "xarray", ("xarray", "netCDF4"), id="nc"),
        pytest.param("zarr", ".zarr", "zarr", ("zarr",), id="zarr"),
    ],
)
def test_public_matrix_auto_read_preserves_optional_backend_importerror(
    tmp_path, format_name, suffix, missing_module, required_modules
):
    from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

    for required_module in required_modules:
        pytest.importorskip(required_module)
    path = tmp_path / f"valid-store{suffix}"
    source = TimeSeries(
        np.array([1.0, 2.0, 3.0]),
        t0=1_234_567_890.25,
        sample_rate=16.0,
        name="signal",
        unit="V",
    )
    TimeSeriesDict({"signal": source}).write(path, format=format_name)

    with mock.patch.dict(sys.modules, {missing_module: None}):
        with pytest.raises(ImportError):
            TimeSeriesMatrix.read(path)


@pytest.mark.parametrize(
    "suffix,missing_module,required_modules",
    [
        pytest.param(".nc", "xarray", ("xarray", "netCDF4"), id="nc"),
        pytest.param(".zarr", "zarr", ("zarr",), id="zarr"),
    ],
)
def test_public_matrix_auto_write_preserves_optional_backend_importerror(
    tmp_path, suffix, missing_module, required_modules
):
    from gwexpy.timeseries import TimeSeriesMatrix

    for required_module in required_modules:
        pytest.importorskip(required_module)
    path = tmp_path / f"output{suffix}"
    matrix = TimeSeriesMatrix(
        np.ones((1, 1, 3)), t0=1_234_567_890.25, sample_rate=16.0, unit="V"
    )

    with mock.patch.dict(sys.modules, {missing_module: None}):
        with pytest.raises(ImportError):
            matrix.write(path)


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
