from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gwexpy
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList
from tests.io_conformance.contract import load_public_io_contract

gwexpy.register_all()


_CONTRACT_FORMATS = {
    entry["canonical"]: entry for entry in load_public_io_contract()["formats"]
}


def _contract_aliases(canonical: str) -> tuple[str, ...]:
    return tuple(_CONTRACT_FORMATS[canonical]["aliases"])


def _skip_missing_gwf_backend(exc: Exception, *, alias: str | None = None) -> None:
    message = str(exc)
    target = f"alias {alias!r}" if alias is not None else "format 'gwf'"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        pytest.skip(f"GWF backend unavailable for {target}: {exc}")
    if "Missing optional dependency" in message:
        pytest.skip(f"GWF backend unavailable for {target}: {exc}")
    if "No writer defined for format 'gwf'" in message:
        pytest.skip(f"GWF backend unavailable for {target}: {exc}")
    if "No reader defined for format 'gwf'" in message:
        pytest.skip(f"GWF backend unavailable for {target}: {exc}")
    raise exc


def _make_series(
    *,
    name: str = "H1:ROUNDTRIP",
    sample_rate: float = 4.0,
    t0: float = 1_000_000_000.0,
    values: np.ndarray | None = None,
) -> TimeSeries:
    data = np.arange(8.0) if values is None else values
    return TimeSeries(data, sample_rate=sample_rate, t0=t0, unit="m", name=name)


def _assert_series_close(actual: TimeSeries, expected: TimeSeries) -> None:
    np.testing.assert_allclose(actual.value, expected.value)
    np.testing.assert_allclose(actual.times.value, expected.times.value)
    assert str(actual.unit) == str(expected.unit)
    assert float(actual.sample_rate.value) == pytest.approx(
        float(expected.sample_rate.value)
    )
    assert float(actual.t0.value) == pytest.approx(float(expected.t0.value))


def test_csv_timeseriesdict_roundtrip(tmp_path: Path) -> None:
    expected = TimeSeriesDict(
        {
            "H1:CSV": _make_series(name="H1:CSV"),
            "L1:CSV": _make_series(name="L1:CSV", values=np.arange(8.0) * 2.0),
        }
    )
    outdir = tmp_path / "csv"

    expected.write(outdir, format="csv")
    actual = TimeSeriesDict.read(outdir, format="csv")

    assert list(actual.keys()) == list(expected.keys())
    for key in expected:
        _assert_series_close(actual[key], expected[key])


def test_txt_timeserieslist_roundtrip(tmp_path: Path) -> None:
    expected = TimeSeriesList(
        _make_series(name="H1:TXT"),
        _make_series(name="L1:TXT", values=np.arange(8.0) * 3.0),
    )
    outdir = tmp_path / "txt"

    expected.write(outdir, format="txt")
    actual = TimeSeriesList.read(outdir, format="txt")

    assert len(actual) == len(expected)
    for index, series in enumerate(expected):
        _assert_series_close(actual[index], series)


def test_hdf5_timeseries_roundtrip(tmp_path: Path) -> None:
    expected = _make_series()
    path = tmp_path / "series.h5"

    expected.write(path, format="hdf5")
    actual = TimeSeries.read(path, format="hdf5")

    _assert_series_close(actual, expected)


def test_hdf_ndscope_timeseriesdict_roundtrip(tmp_path: Path) -> None:
    expected = TimeSeriesDict({"H1:NDSCOPE": _make_series(name="H1:NDSCOPE")})
    path = tmp_path / "ndscope.hdf5"

    expected.write(path, format="hdf.ndscope")
    actual = TimeSeriesDict.read(path, format="hdf.ndscope")

    assert list(actual.keys()) == list(expected.keys())
    _assert_series_close(actual["H1:NDSCOPE"], expected["H1:NDSCOPE"])


@pytest.mark.parametrize("alias", _contract_aliases("gwf"), ids=str)
def test_gwf_alias_write_roundtrip(tmp_path: Path, alias: str) -> None:
    expected = TimeSeriesDict({"K1:GWF": _make_series(name="K1:GWF")})
    path = tmp_path / f"{alias.replace('.', '_')}.gwf"

    try:
        expected.write(path, format=alias)
        actual = TimeSeriesDict.read(path, format="gwf")
    except Exception as exc:  # pragma: no cover - depends on optional backend
        _skip_missing_gwf_backend(exc, alias=alias)

    assert list(actual.keys()) == list(expected.keys())
    _assert_series_close(actual["K1:GWF"], expected["K1:GWF"])


@pytest.mark.parametrize("alias", _contract_aliases("hdf.ndscope"), ids=str)
def test_hdf_ndscope_alias_write_roundtrip(tmp_path: Path, alias: str) -> None:
    expected = TimeSeriesDict(
        {
            "H1:NDSCOPE": _make_series(name="H1:NDSCOPE"),
            "L1:NDSCOPE": _make_series(
                name="L1:NDSCOPE",
                values=np.arange(8.0) * 2.0,
            ),
        }
    )
    path = tmp_path / f"{alias.replace('.', '_')}.hdf5"

    expected.write(path, format=alias)
    actual = TimeSeriesDict.read(path, format="hdf.ndscope")

    assert list(actual.keys()) == list(expected.keys())
    for key in expected:
        _assert_series_close(actual[key], expected[key])


def test_wav_timeseries_roundtrip(tmp_path: Path) -> None:
    expected = _make_series(
        name="H1:WAV",
        sample_rate=8_000.0,
        t0=0.0,
        values=np.sin(np.linspace(0.0, 4.0 * np.pi, 512)),
    )
    path = tmp_path / "signal.wav"

    expected.write(path, format="wav")
    actual = TimeSeries.read(path, format="wav")

    assert len(actual) == len(expected)
    assert float(actual.sample_rate.value) == pytest.approx(
        float(expected.sample_rate.value)
    )
    assert float(actual.t0.value) == pytest.approx(float(expected.t0.value))
    assert np.corrcoef(actual.value, expected.value)[0, 1] > 0.99


def test_gwf_timeseriesdict_roundtrip(tmp_path: Path) -> None:
    expected = TimeSeriesDict({"K1:GWF": _make_series(name="K1:GWF")})
    path = tmp_path / "frame.gwf"

    try:
        expected.write(path, format="gwf")
        actual = TimeSeriesDict.read(path, format="gwf")
    except Exception as exc:  # pragma: no cover - depends on optional backend
        _skip_missing_gwf_backend(exc)

    assert list(actual.keys()) == list(expected.keys())
    _assert_series_close(actual["K1:GWF"], expected["K1:GWF"])
