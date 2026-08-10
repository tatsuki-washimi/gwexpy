"""Contracts for the public single-series ``ats.mth5`` reader (#619)."""

from __future__ import annotations

import datetime
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from astropy import units as u

from gwexpy.io.utils import datetime_to_gps
from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io import ats as ats_io

_START = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.UTC)
_VALUES = np.array([1.25, -2.5, 7.75], dtype=np.float64)
_ROOT = Path(__file__).resolve().parents[2]


def _mth5_channel(
    *,
    data: object = _VALUES,
    sample_rate: object = 8.0,
    start: object = _START,
    component: object = "ex",
    unit: object = "milliVolt per kilometer",
) -> SimpleNamespace:
    return SimpleNamespace(
        ts=data,
        sample_rate=sample_rate,
        start=start,
        component=component,
        channel_metadata=SimpleNamespace(units=unit),
    )


def _install_fake_mth5(
    monkeypatch: pytest.MonkeyPatch,
    channel: SimpleNamespace,
) -> Mock:
    read_file = Mock(return_value=channel)
    module = SimpleNamespace(__version__="0.6.8", read_file=read_file)
    monkeypatch.setattr(ats_io, "ensure_dependency", lambda name: module)
    return read_file


def _write_independent_atss_fixture(
    tmp_path,
    component: str,
    unit: str,
):
    """Write raw float64 ATSS plus independently specified JSON metadata."""
    channel_number = {"ex": 0, "ey": 1, "hx": 2, "hy": 3, "hz": 4}[component]
    run_dir = tmp_path / "survey01" / "stations" / "station01" / "run01"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / (f"084_ADU08e_C{channel_number:03d}_T{component}_8Hz.atss")
    path.write_bytes(_VALUES.tobytes())
    metadata = {
        "angle": 0.0,
        "datetime": _START.isoformat(),
        "elevation": 12.0,
        "filter": "",
        "latitude": 35.0,
        "longitude": 135.0,
        "resistance": 1000.0,
        "sensor_calibration": {
            "a": [],
            "chopper": 0,
            "datetime": "2024-01-01T00:00:00+00:00",
            "f": [],
            "p": [],
            "sensor": "fixture-sensor",
            "serial": f"fixture-{channel_number}",
            "units_amplitude": unit,
            "units_phase": "degrees",
        },
        "tilt": 0.0,
        "units": unit,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("component", "source_unit", "expected_unit"),
    [
        ("ex", "mV/km", u.mV / u.km),
        ("ey", "mV/km", u.mV / u.km),
        ("hx", "nT", u.nT),
        ("hy", "nT", u.nT),
        ("hz", "nT", u.nT),
    ],
)
def test_ats_mth5_reads_independent_float64_json_fixture_without_scaling(
    tmp_path,
    component,
    source_unit,
    expected_unit,
):
    pytest.importorskip("mth5")
    source = _write_independent_atss_fixture(tmp_path, component, source_unit)

    result = TimeSeries.read(source, format="ats.mth5")

    np.testing.assert_array_equal(result.value, _VALUES)
    assert result.unit == expected_unit
    assert result.sample_rate == 8 * u.Hz
    assert float(result.t0.value) == pytest.approx(
        datetime_to_gps(_START), rel=0, abs=1e-6
    )
    assert result.name == component
    assert str(result.channel) == component


def test_ats_mth5_calls_top_level_read_file_and_treats_naive_start_as_utc(
    monkeypatch,
    tmp_path,
):
    naive_start = _START.replace(tzinfo=None)
    read_file = _install_fake_mth5(
        monkeypatch,
        _mth5_channel(start=naive_start),
    )
    source = tmp_path / "input.atss"

    result = ats_io.read_timeseries_ats_mth5(source)

    read_file.assert_called_once_with(str(source), file_type="metronix")
    np.testing.assert_array_equal(result.value, _VALUES)
    assert float(result.t0.value) == pytest.approx(
        datetime_to_gps(_START), rel=0, abs=1e-6
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"data": np.array([], dtype=float)}, "non-empty"),
        ({"data": np.ones((2, 2))}, "one-dimensional"),
        ({"sample_rate": 0.0}, "sample_rate"),
        ({"sample_rate": np.inf}, "sample_rate"),
        ({"sample_rate": None}, "sample_rate"),
        ({"start": None}, "start"),
        ({"component": None}, "component"),
        ({"unit": None}, "unit"),
    ],
)
def test_ats_mth5_rejects_missing_or_invalid_channel_data(
    monkeypatch,
    changes,
    message,
):
    channel = _mth5_channel(**changes)
    _install_fake_mth5(monkeypatch, channel)

    with pytest.raises(ValueError, match=message):
        ats_io.read_timeseries_ats_mth5("input.atss")


@pytest.mark.parametrize(
    ("component", "unit"),
    [
        ("ez", "milliVolt per kilometer"),
        ("ex", "nanoTesla"),
        ("hx", "milliVolt per kilometer"),
    ],
)
def test_ats_mth5_rejects_unsupported_component_or_unit(
    monkeypatch,
    component,
    unit,
):
    _install_fake_mth5(
        monkeypatch,
        _mth5_channel(component=component, unit=unit),
    )

    with pytest.raises(ValueError, match="component|unit"):
        ats_io.read_timeseries_ats_mth5("input.atss")


def test_timeseriesdict_rejects_ats_mth5_before_dependency_lookup(
    monkeypatch,
    tmp_path,
):
    def forbidden_lookup(name):
        raise AssertionError(f"dependency lookup must not run: {name}")

    monkeypatch.setattr(ats_io, "ensure_dependency", forbidden_lookup)

    with pytest.raises(TypeError, match="TimeSeries.*only"):
        TimeSeriesDict.read(tmp_path / "input.atss", format="ats.mth5")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"epoch": 0}, ValueError, "epoch override"),
        ({"unit": "nT"}, TypeError, "unit"),
    ],
)
def test_ats_mth5_rejects_unsupported_overrides_before_dependency_lookup(
    monkeypatch,
    kwargs,
    error,
    message,
):
    def forbidden_lookup(name):
        raise AssertionError(f"dependency lookup must not run: {name}")

    monkeypatch.setattr(ats_io, "ensure_dependency", forbidden_lookup)

    with pytest.raises(error, match=message):
        ats_io.read_timeseries_ats_mth5("input.atss", **kwargs)


def test_seismic_and_all_extras_require_mth5_0_6_8_or_newer():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    for extra in ("seismic", "all"):
        assert "mth5>=0.6.8" in extras[extra]
