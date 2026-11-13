"""SeriesMatrix read/write must not report I/O errors as format-inference failures (#775)."""

from __future__ import annotations

import io

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix

IDENTIFY_MSG = "Could not identify format"


def _make_tsm():
    data = np.arange(2 * 2 * 8, dtype=float).reshape(2, 2, 8)
    return TimeSeriesMatrix(data, t0=0.0, dt=1.0)


# -- read: failures after a reader was identified -------------------------


@pytest.mark.parametrize("suffix", [".csv", ".nc", ".h5"])
def test_read_missing_file_autodetect_reports_missing_file(tmp_path, suffix):
    pytest.importorskip("pandas")
    if suffix == ".nc":
        pytest.importorskip("netCDF4")
    missing = tmp_path / f"missing{suffix}"

    with pytest.warns() if suffix == ".csv" else _no_warning_filter():
        with pytest.raises(OSError) as excinfo:
            TimeSeriesMatrix.read(missing)

    assert IDENTIFY_MSG not in str(excinfo.value)
    assert "missing" in str(excinfo.value)


@pytest.mark.parametrize("fmt", ["csv", "hdf5"])
def test_read_missing_file_explicit_format_unchanged(tmp_path, fmt):
    pytest.importorskip("pandas")
    missing = tmp_path / "missing.dat"

    with pytest.raises(OSError) as excinfo:
        _read_quiet(TimeSeriesMatrix, missing, format=fmt)

    assert IDENTIFY_MSG not in str(excinfo.value)


def test_read_corrupt_identified_file_keeps_backend_error(tmp_path):
    pytest.importorskip("pandas")
    bad = tmp_path / "empty.csv"
    bad.write_text("")

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - backend-defined type
        _read_quiet(TimeSeriesMatrix, bad)

    assert IDENTIFY_MSG not in str(excinfo.value)


# -- read: genuine identification failure --------------------------------


def test_read_unknown_extension_reports_identification_failure(tmp_path):
    unknown = tmp_path / "data.xyz_unknown"
    unknown.write_bytes(b"\x00\x01not a known format")

    with pytest.raises(ValueError, match=IDENTIFY_MSG) as excinfo:
        TimeSeriesMatrix.read(unknown)

    assert excinfo.value.__cause__ is not None


def test_read_unknown_explicit_format_is_not_rewritten(tmp_path):
    target = tmp_path / "data.bin"
    target.write_bytes(b"x")

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - registry-defined type
        TimeSeriesMatrix.read(target, format="xyz_unknown")

    assert IDENTIFY_MSG not in str(excinfo.value)
    assert "xyz_unknown" in str(excinfo.value)


# -- read: non-path sources ------------------------------------------------


def test_read_list_source_autodetect_does_not_fail_on_path_conversion(tmp_path):
    first = tmp_path / "a.h5"
    _make_tsm().write(first, format="hdf5")

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - contract under test
        TimeSeriesMatrix.read([first, first])

    # A list is not a single path: it must reach format identification rather
    # than fail inside ``Path(list)``.
    assert "expected str, bytes or os.PathLike" not in str(excinfo.value)


def test_read_filelike_source_autodetect_does_not_fail_on_path_conversion():
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - contract under test
        TimeSeriesMatrix.read(io.BytesIO(b"not a known format"))

    assert "expected str, bytes or os.PathLike" not in str(excinfo.value)


def test_read_hdf5_filelike_explicit_format_roundtrip(tmp_path):
    target = tmp_path / "a.h5"
    tsm = _make_tsm()
    tsm.write(target, format="hdf5")

    with open(target, "rb") as fobj:
        back = TimeSeriesMatrix.read(fobj, format="hdf5")

    np.testing.assert_array_equal(back.value, tsm.value)


# -- write -----------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".csv", ".h5", ".nc"])
def test_write_into_missing_directory_reports_os_error(tmp_path, suffix):
    pytest.importorskip("pandas")
    if suffix == ".nc":
        pytest.importorskip("netCDF4")
    target = tmp_path / "no_such_dir" / f"out{suffix}"

    with pytest.raises(OSError) as excinfo:
        _make_tsm().write(target)

    assert IDENTIFY_MSG not in str(excinfo.value)


def test_write_unknown_extension_reports_identification_failure(tmp_path):
    target = tmp_path / "out.xyz_unknown"

    with pytest.raises(ValueError, match=IDENTIFY_MSG) as excinfo:
        _make_tsm().write(target)

    assert excinfo.value.__cause__ is not None
    assert not target.exists()


def test_write_unknown_explicit_format_is_not_rewritten(tmp_path):
    target = tmp_path / "out.bin"

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - registry-defined type
        _make_tsm().write(target, format="xyz_unknown")

    assert IDENTIFY_MSG not in str(excinfo.value)
    assert "xyz_unknown" in str(excinfo.value)


def test_write_filelike_target_autodetect_does_not_fail_on_path_conversion():
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - contract under test
        _make_tsm().write(io.BytesIO())

    assert "expected str, bytes or os.PathLike" not in str(excinfo.value)


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_missing_optional_backend_import_error_is_preserved(
    tmp_path, monkeypatch, suffix
):
    import gwexpy.interop._optional as optional

    def _missing(name, *args, **kwargs):
        raise ImportError(f"{name} is required; pip install gwexpy[io]")

    monkeypatch.setattr(optional, "require_optional", _missing)
    target = tmp_path / f"out{suffix}"

    try:
        _make_tsm().write(target)
    except ImportError as exc:
        assert IDENTIFY_MSG not in str(exc)
    except ValueError as exc:  # pragma: no cover - the defect under test
        pytest.fail(f"backend ImportError was rewritten: {exc}")


def test_frequencyseriesmatrix_shares_the_contract(tmp_path):
    pytest.importorskip("pandas")
    missing = tmp_path / "missing.csv"

    with pytest.raises(OSError) as excinfo:
        _read_quiet(FrequencySeriesMatrix, missing)

    assert IDENTIFY_MSG not in str(excinfo.value)


# -- helpers ---------------------------------------------------------------


def _read_quiet(cls, source, **kwargs):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cls.read(source, **kwargs)


class _no_warning_filter:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
