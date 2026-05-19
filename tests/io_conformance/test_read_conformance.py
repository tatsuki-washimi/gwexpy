from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

import gwexpy

gwexpy.register_all()

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from tests.io_conformance.contract import load_public_io_contract
from tests.io_conformance.generators import audio, gwf, hdf5, hdf_ndscope
from tests.io_conformance.validators import (
    assert_timeseries_close,
    assert_timeseriesdict_close,
)

_CONTRACT_FORMATS = {
    entry["canonical"]: entry for entry in load_public_io_contract()["formats"]
}

_TEXT_VALUES = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)


@dataclass(frozen=True)
class SeriesCase:
    path: Path
    values: np.ndarray | None
    sample_rate: float
    t0: float
    name: str | None = None
    channel: str | None = None
    unit: str | None = None


@dataclass(frozen=True)
class DictCase:
    path: Path
    auto_path: Path | None
    values_by_key: dict[str, np.ndarray]
    sample_rate: float
    t0: float
    unit_by_key: dict[str, str] | None = None


def _contract_aliases(canonical: str) -> tuple[str, ...]:
    return tuple(_CONTRACT_FORMATS[canonical]["aliases"])


def _skip_missing_gwf_backend(exc: Exception, *, alias: str | None = None) -> None:
    message = str(exc)
    if isinstance(exc, ImportError) or "Missing optional dependency" in message:
        target = f"alias {alias!r}" if alias is not None else "format 'gwf'"
        pytest.skip(f"GWF backend unavailable for {target}: {exc}")
    if "No reader defined for format 'gwf'" in message:
        target = f"alias {alias!r}" if alias is not None else "format 'gwf'"
        pytest.skip(f"GWF backend unavailable for {target}: {exc}")
    raise exc


def _make_text_fixture(tmp_path: Path, suffix: str, fmt: str) -> SeriesCase:
    series = GWpyTimeSeries(
        _TEXT_VALUES,
        sample_rate=1.0,
        t0=123.0,
        name="X1:TEST",
        channel="X1:TEST",
        unit="m",
    )
    path = tmp_path / f"sample.{suffix}"
    series.write(path, format=fmt)
    return SeriesCase(
        path=path,
        values=_TEXT_VALUES,
        sample_rate=1.0,
        t0=123.0,
        name="ch1" if fmt == "csv" else None,
        channel=None,
        unit=None,
    )


def _gwf_expected_values() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(8_183)
    return {"K1:CONFORMANCE-GWF": rng.normal(loc=0.0, scale=1.0, size=32)}


def _ndscope_expected_values() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20_241)
    return {
        "H1:CONFORMANCE-NDSCOPE": rng.normal(loc=0.0, scale=0.1, size=32),
        "L1:CONFORMANCE-NDSCOPE": rng.normal(loc=1.0, scale=0.1, size=32),
    }


def _hdf5_expected_values() -> np.ndarray:
    rng = np.random.default_rng(15_934)
    return rng.normal(loc=0.0, scale=1.0, size=32)


@pytest.fixture(scope="module")
def gwf_case(tmp_path_factory: pytest.TempPathFactory) -> DictCase:
    path = tmp_path_factory.mktemp("gwf-read") / "frame.gwf"
    generated = gwf.generate(path.parent)
    gwf_path = generated["gwf"]
    try:
        TimeSeriesDict.read(gwf_path, format="gwf")
    except Exception as exc:  # pragma: no cover - depends on optional backend
        _skip_missing_gwf_backend(exc)

    return DictCase(
        path=gwf_path,
        auto_path=None,
        values_by_key=_gwf_expected_values(),
        sample_rate=16.0,
        t0=1_000_000_000.0,
        unit_by_key={"K1:CONFORMANCE-GWF": "m"},
    )


@pytest.fixture(scope="module")
def ndscope_case(tmp_path_factory: pytest.TempPathFactory) -> DictCase:
    path = tmp_path_factory.mktemp("ndscope-read") / "ndscope.hdf"
    generated = hdf_ndscope.generate(path.parent)
    auto_path = generated["hdf"].with_suffix(".hdf5")
    shutil.copyfile(generated["hdf"], auto_path)
    return DictCase(
        path=generated["hdf"],
        auto_path=auto_path,
        values_by_key=_ndscope_expected_values(),
        sample_rate=16.0,
        t0=1_000_000_000.0,
    )


@pytest.fixture(scope="module")
def hdf5_case(tmp_path_factory: pytest.TempPathFactory) -> SeriesCase:
    path = tmp_path_factory.mktemp("hdf5-read") / "sample.h5"
    generated = hdf5.generate(path.parent)
    return SeriesCase(
        path=generated["hdf5"],
        values=_hdf5_expected_values(),
        sample_rate=8.0,
        t0=1_000_000_000.0,
        name="H1:CONFORMANCE-HDF5",
        channel=None,
        unit="m",
    )


@pytest.fixture(scope="module")
def csv_case(tmp_path_factory: pytest.TempPathFactory) -> SeriesCase:
    return _make_text_fixture(tmp_path_factory.mktemp("csv-read"), "csv", "csv")


@pytest.fixture(scope="module")
def txt_case(tmp_path_factory: pytest.TempPathFactory) -> SeriesCase:
    return _make_text_fixture(tmp_path_factory.mktemp("txt-read"), "txt", "txt")


@pytest.fixture(scope="module")
def wav_case(tmp_path_factory: pytest.TempPathFactory) -> SeriesCase:
    path = tmp_path_factory.mktemp("wav-read") / "tone.wav"
    generated = audio.generate(path.parent)
    return SeriesCase(
        path=generated["wav"],
        values=None,
        sample_rate=8_000.0,
        t0=0.0,
        name="channel_0",
        channel="channel_0",
        unit="",
    )


def test_gwf_explicit_and_auto_identify(gwf_case: DictCase) -> None:
    explicit = TimeSeriesDict.read(gwf_case.path, format="gwf")
    auto = TimeSeriesDict.read(gwf_case.path)

    assert_timeseriesdict_close(
        explicit,
        gwf_case.values_by_key,
        sample_rate=gwf_case.sample_rate,
        t0=gwf_case.t0,
        unit_by_key=gwf_case.unit_by_key,
    )
    assert_timeseriesdict_close(
        auto,
        gwf_case.values_by_key,
        sample_rate=gwf_case.sample_rate,
        t0=gwf_case.t0,
        unit_by_key=gwf_case.unit_by_key,
    )


@pytest.mark.parametrize("alias", _contract_aliases("gwf"), ids=str)
def test_gwf_alias_reads(gwf_case: DictCase, alias: str) -> None:
    try:
        actual = TimeSeriesDict.read(gwf_case.path, format=alias)
    except Exception as exc:  # pragma: no cover - depends on optional backend
        _skip_missing_gwf_backend(exc, alias=alias)

    assert_timeseriesdict_close(
        actual,
        gwf_case.values_by_key,
        sample_rate=gwf_case.sample_rate,
        t0=gwf_case.t0,
        unit_by_key=gwf_case.unit_by_key,
        check_channel=False,
    )


def test_hdf_ndscope_explicit_and_auto_identify(ndscope_case: DictCase) -> None:
    explicit = TimeSeriesDict.read(ndscope_case.path, format="hdf.ndscope")
    assert ndscope_case.auto_path is not None
    auto = TimeSeriesDict.read(ndscope_case.auto_path)

    assert_timeseriesdict_close(
        explicit,
        ndscope_case.values_by_key,
        sample_rate=ndscope_case.sample_rate,
        t0=ndscope_case.t0,
    )
    assert_timeseriesdict_close(
        auto,
        ndscope_case.values_by_key,
        sample_rate=ndscope_case.sample_rate,
        t0=ndscope_case.t0,
    )


@pytest.mark.parametrize("alias", _contract_aliases("hdf.ndscope"), ids=str)
def test_hdf_ndscope_alias_reads(ndscope_case: DictCase, alias: str) -> None:
    actual = TimeSeriesDict.read(ndscope_case.path, format=alias)

    assert_timeseriesdict_close(
        actual,
        ndscope_case.values_by_key,
        sample_rate=ndscope_case.sample_rate,
        t0=ndscope_case.t0,
    )


def test_hdf5_explicit_read(hdf5_case: SeriesCase) -> None:
    actual = TimeSeries.read(hdf5_case.path, format="hdf5")

    assert_timeseries_close(
        actual,
        hdf5_case.values,
        sample_rate=hdf5_case.sample_rate,
        t0=hdf5_case.t0,
        name=hdf5_case.name,
        channel=hdf5_case.channel,
        unit=hdf5_case.unit,
    )


def test_hdf5_requires_explicit_format(hdf5_case: SeriesCase) -> None:
    with pytest.raises(IORegistryError, match="Format could not be identified"):
        TimeSeries.read(hdf5_case.path)


def test_csv_explicit_and_auto_identify(csv_case: SeriesCase) -> None:
    explicit = TimeSeries.read(csv_case.path, format="csv")
    auto = TimeSeries.read(csv_case.path)

    assert_timeseries_close(
        explicit,
        csv_case.values,
        sample_rate=csv_case.sample_rate,
        t0=csv_case.t0,
        name=csv_case.name,
        channel=csv_case.channel,
        unit=csv_case.unit,
    )
    assert_timeseries_close(
        auto,
        csv_case.values,
        sample_rate=csv_case.sample_rate,
        t0=csv_case.t0,
        name=csv_case.name,
        channel=csv_case.channel,
        unit=csv_case.unit,
    )


def test_txt_explicit_read_and_explicit_requirement(txt_case: SeriesCase) -> None:
    actual = TimeSeries.read(txt_case.path, format="txt")

    assert_timeseries_close(
        actual,
        txt_case.values,
        sample_rate=txt_case.sample_rate,
        t0=txt_case.t0,
        name=txt_case.name,
        channel=txt_case.channel,
        unit=txt_case.unit,
    )

    with pytest.raises(IORegistryError, match="Format could not be identified"):
        TimeSeries.read(txt_case.path)


def test_wav_explicit_and_auto_identify(wav_case: SeriesCase) -> None:
    explicit = TimeSeries.read(wav_case.path, format="wav")
    auto = TimeSeries.read(wav_case.path)

    assert_timeseries_close(
        explicit,
        wav_case.values,
        sample_rate=wav_case.sample_rate,
        t0=wav_case.t0,
        name=wav_case.name,
        channel=wav_case.channel,
        unit=wav_case.unit,
    )
    assert_timeseries_close(
        auto,
        wav_case.values,
        sample_rate=wav_case.sample_rate,
        t0=wav_case.t0,
        name=wav_case.name,
        channel=wav_case.channel,
        unit=wav_case.unit,
    )
