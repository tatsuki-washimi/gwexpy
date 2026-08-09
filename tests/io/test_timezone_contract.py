"""Machine-readable timezone contracts and reader routing regressions."""

from __future__ import annotations

import datetime as dt
import inspect
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gwpy.time import to_gps

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_io_contract.json"

EXPECTED_TIME_CONTRACT = {
    "gwf": ("absolute", "none", "not_accepted"),
    "hdf.ndscope": ("absolute", "none", "not_accepted"),
    "hdf5": ("absolute", "none", "not_accepted"),
    "xml.diaggui": ("absolute", "override", "epoch_localize_only"),
    "csv": ("absolute", "none", "component_localize"),
    "txt": ("absolute", "none", "not_accepted"),
    "sdb": ("absolute", "none", "rejected"),
    "wav": ("relative", "override", "not_accepted"),
    "flac": ("relative", "override", "not_accepted"),
    "ogg": ("relative", "override", "not_accepted"),
    "mp3": ("relative", "override", "not_accepted"),
    "m4a": ("relative", "override", "not_accepted"),
    "gbd": ("naive_civil", "override", "required"),
    "tdms": ("absolute", "override", "epoch_localize_only"),
    "mseed": ("absolute", "override", "epoch_localize_only"),
    "sac": ("absolute", "override", "epoch_localize_only"),
    "gse2": ("absolute", "override", "epoch_localize_only"),
    "knet": ("absolute", "override", "epoch_localize_only"),
    "win": ("fixed_zone", "none", "rejected"),
    "ats": ("absolute", "override", "epoch_localize_only"),
    "ats.mth5": ("absolute", "none", "rejected"),
    "nc": ("absolute", "none", "not_accepted"),
    "zarr": ("absolute", "t0_override", "not_accepted"),
}

EXPECTED_TIME_ALIASES = {
    "frame": "gwf",
    "framecpp": "gwf",
    "framel": "gwf",
    "lalframe": "gwf",
    "gwf.framecpp": "gwf",
    "gwf.framel": "gwf",
    "gwf.lalframe": "gwf",
    "ndscope-hdf5": "hdf.ndscope",
    "ndscope_hdf5": "hdf.ndscope",
    "ndscopehdf5": "hdf.ndscope",
    "dttxml": "xml.diaggui",
    "miniseed": "mseed",
    "win32": "win",
    "netcdf4": "nc",
}


def _time_contract_entries() -> dict[str, dict[str, object]]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return {
        entry["canonical"]: entry
        for entry in contract["formats"]
        if "time_semantics" in entry
    }


def test_all_time_bearing_formats_have_machine_readable_contracts() -> None:
    entries = _time_contract_entries()

    assert set(entries) == set(EXPECTED_TIME_CONTRACT)
    for canonical, expected in EXPECTED_TIME_CONTRACT.items():
        entry = entries[canonical]
        assert (
            entry["time_semantics"],
            entry["epoch_arg"],
            entry["timezone_arg"],
        ) == expected


def test_time_contract_enums_and_aliases_are_closed() -> None:
    entries = _time_contract_entries()
    alias_contracts = {}

    for canonical, entry in entries.items():
        assert entry["time_semantics"] in {
            "absolute",
            "fixed_zone",
            "naive_civil",
            "relative",
        }
        assert entry["epoch_arg"] in {"none", "override", "t0_override"}
        assert entry["timezone_arg"] in {
            "rejected",
            "required",
            "epoch_localize_only",
            "component_localize",
            "not_accepted",
        }
        for alias in entry["aliases"]:
            assert alias not in alias_contracts
            alias_contracts[alias] = (
                canonical,
                entry["time_semantics"],
                entry["epoch_arg"],
                entry["timezone_arg"],
            )

    assert {alias: value[0] for alias, value in alias_contracts.items()} == (
        EXPECTED_TIME_ALIASES
    )
    for alias, canonical in EXPECTED_TIME_ALIASES.items():
        assert alias_contracts[alias][1:] == EXPECTED_TIME_CONTRACT[canonical]


def test_csv_contract_enumerates_component_numeric_and_index_routes() -> None:
    csv_contract = _time_contract_entries()["csv"]

    assert csv_contract["time_routes"] == [
        {
            "route": "component",
            "time_semantics": "naive_civil",
            "timezone_behavior": "localize",
        },
        {
            "route": "numeric",
            "time_semantics": "absolute",
            "timezone_behavior": "ignored_with_warning",
        },
        {
            "route": "index",
            "time_semantics": "relative",
            "timezone_behavior": "ignored_with_warning",
        },
    ]


def test_timezone_helper_validates_before_epoch_branch() -> None:
    from gwexpy.io.utils import _reject_timezone_reinterpretation

    with pytest.raises(ValueError, match="Could not parse timezone"):
        _reject_timezone_reinterpretation("ats", "Not/AZone", 123.0)


def test_timezone_helper_rejects_source_reinterpretation() -> None:
    from gwexpy.io.utils import _reject_timezone_reinterpretation

    with pytest.raises(ValueError, match="format 'mseed'"):
        _reject_timezone_reinterpretation("mseed", "Asia/Tokyo", None)


@pytest.mark.parametrize(
    "epoch",
    [
        123.0,
        dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
    ],
)
def test_timezone_helper_warns_for_already_absolute_epoch(epoch) -> None:
    from gwexpy.io.utils import _reject_timezone_reinterpretation

    with pytest.warns(UserWarning, match="timezone.*ignored.*format 'ats'"):
        tzinfo = _reject_timezone_reinterpretation("ats", "+09:00", epoch)

    assert tzinfo is None


@pytest.mark.parametrize("epoch_type", [np.int64, np.float64])
def test_timezone_helper_classifies_numpy_numeric_epochs(epoch_type) -> None:
    from gwexpy.io.utils import _coerce_numeric_epoch, _is_numeric_epoch

    epoch = epoch_type(123)
    assert _is_numeric_epoch(epoch)
    assert _coerce_numeric_epoch(epoch) == 123.0


@pytest.mark.parametrize("epoch", [True, False, np.bool_(True), np.bool_(False)])
def test_timezone_helper_rejects_boolean_epochs(epoch) -> None:
    from gwexpy.io.utils import _coerce_numeric_epoch, _is_numeric_epoch

    assert not _is_numeric_epoch(epoch)
    with pytest.raises(TypeError, match="epoch must be numeric"):
        _coerce_numeric_epoch(epoch)


def test_timezone_helper_returns_zone_for_naive_epoch() -> None:
    from gwexpy.io.utils import _reject_timezone_reinterpretation

    timezone = _reject_timezone_reinterpretation(
        "ats", "+09:00", dt.datetime(2024, 1, 1, 9)
    )

    assert timezone is not None
    assert timezone.utcoffset(None) == dt.timedelta(hours=9)


def test_timezone_helper_treats_aware_iso_epoch_as_absolute() -> None:
    from gwexpy.io.utils import _reject_timezone_reinterpretation, ensure_datetime

    epoch = "2024-01-01T00:00:00+00:00"
    with pytest.warns(UserWarning, match="timezone.*ignored.*format 'ats'"):
        timezone = _reject_timezone_reinterpretation("ats", "Asia/Tokyo", epoch)

    assert timezone is None
    assert ensure_datetime(epoch) == dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def _fake_obspy_stream(seismic_io):
    trace = SimpleNamespace(
        data=np.arange(4, dtype=np.int32),
        id="XX.STAT..BHZ",
        stats=SimpleNamespace(
            starttime=SimpleNamespace(
                datetime=dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC)
            ),
            delta=0.25,
            channel="BHZ",
        ),
    )
    return [trace]


def test_mseed_source_timezone_is_rejected_before_read(monkeypatch) -> None:
    from gwexpy.timeseries.io import seismic as seismic_io

    called = False

    def _reader(*args, **kwargs):
        nonlocal called
        called = True
        return _fake_obspy_stream(seismic_io)

    monkeypatch.setattr(seismic_io, "_read_obspy_stream", _reader)

    with pytest.raises(ValueError, match="format 'mseed'"):
        seismic_io.read_miniseed_timeseriesdict("unused.mseed", timezone="Asia/Tokyo")

    assert called is False


def test_mseed_source_epoch_remains_utc(monkeypatch) -> None:
    from gwexpy.timeseries.io import seismic as seismic_io

    monkeypatch.setattr(
        seismic_io,
        "_read_obspy_stream",
        lambda *args, **kwargs: _fake_obspy_stream(seismic_io),
    )

    result = seismic_io.read_miniseed_timeseriesdict("unused.mseed")
    series = next(iter(result.values()))

    assert float(series.t0.value) == float(
        to_gps(dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))
    )
    assert result._gwexpy_io["timezone"] is None


def test_mseed_naive_epoch_localizes_and_records_timezone(monkeypatch) -> None:
    from gwexpy.timeseries.io import seismic as seismic_io

    monkeypatch.setattr(
        seismic_io,
        "_read_obspy_stream",
        lambda *args, **kwargs: _fake_obspy_stream(seismic_io),
    )
    naive_epoch = dt.datetime(2024, 1, 1, 12)

    result = seismic_io.read_miniseed_timeseriesdict(
        "unused.mseed", epoch=naive_epoch, timezone="Asia/Tokyo"
    )
    series = next(iter(result.values()))

    assert float(series.t0.value) == float(
        to_gps(naive_epoch.replace(tzinfo=dt.timezone(dt.timedelta(hours=9))))
    )
    assert result._gwexpy_io["timezone"] == "Asia/Tokyo"


def test_mseed_numeric_epoch_warns_once_and_drops_timezone_provenance(
    monkeypatch,
) -> None:
    from gwexpy.timeseries.io import seismic as seismic_io

    stream = _fake_obspy_stream(seismic_io) * 2
    monkeypatch.setattr(
        seismic_io, "_read_obspy_stream", lambda *args, **kwargs: stream
    )

    with pytest.warns(UserWarning, match="timezone.*ignored.*format 'mseed'") as caught:
        result = seismic_io.read_miniseed_timeseriesdict(
            "unused.mseed", epoch=123.0, timezone="UTC"
        )

    assert len(caught) == 1
    assert all(float(series.t0.value) == 123.0 for series in result.values())
    assert result._gwexpy_io["timezone"] is None


def test_knet_preserves_obspy_utc_start_without_extra_shift(monkeypatch) -> None:
    from gwexpy.timeseries.io import seismic as seismic_io

    monkeypatch.setattr(
        seismic_io,
        "_read_obspy_stream",
        lambda *args, **kwargs: _fake_obspy_stream(seismic_io),
    )

    result = seismic_io.read_knet_timeseriesdict("unused.knet")
    series = next(iter(result.values()))

    assert float(series.t0.value) == float(
        to_gps(dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))
    )


def test_sdb_rejects_timezone_even_with_dummy_epoch(tmp_path) -> None:
    import sqlite3

    from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

    path = tmp_path / "sample.sdb"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp REAL)")
    connection.execute("INSERT INTO archive VALUES (1704067200, 32.0)")
    connection.commit()
    connection.close()

    for kwargs in (
        {"timezone": "Asia/Tokyo"},
        {"timezone": "Not/AZone", "epoch": 0.0},
    ):
        with pytest.raises(ValueError, match="format 'sdb'"):
            read_timeseriesdict_sdb(path, **kwargs)


@pytest.mark.parametrize(
    "module_name",
    [
        "gwexpy.timeseries.io.dttxml",
        "gwexpy.frequencyseries.io.dttxml",
    ],
)
def test_dttxml_epoch_builder_validates_and_routes_timezone(module_name) -> None:
    import importlib

    module = importlib.import_module(module_name)

    with pytest.raises(ValueError, match="Could not parse timezone"):
        module._build_epoch(123.0, "Not/AZone")
    with pytest.warns(UserWarning, match="timezone.*ignored.*xml.diaggui"):
        assert module._build_epoch(123.0, "UTC") == 123.0
    with pytest.warns(UserWarning, match="timezone.*ignored.*xml.diaggui"):
        aware = module._build_epoch("2024-01-01T00:00:00+00:00", "Asia/Tokyo")
    assert aware == float(to_gps(dt.datetime(2024, 1, 1, tzinfo=dt.UTC)))

    expected = float(
        to_gps(dt.datetime(2024, 1, 1, 9, tzinfo=dt.timezone(dt.timedelta(hours=9))))
    )
    assert module._build_epoch("2024-01-01T09:00:00", "+09:00") == expected


@pytest.mark.parametrize(
    ("module_name", "reader_name", "products"),
    [
        (
            "gwexpy.timeseries.io.dttxml",
            "read_timeseriesdict_dttxml",
            "TS",
        ),
        (
            "gwexpy.frequencyseries.io.dttxml",
            "read_frequencyseriesdict_dttxml",
            "PSD",
        ),
    ],
)
def test_dttxml_file_epoch_rejects_timezone_before_loading(
    monkeypatch, module_name, reader_name, products
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    called = False

    def _loader(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(module, "load_dttxml_products", _loader)
    reader = getattr(module, reader_name)

    with pytest.raises(ValueError, match="format 'xml.diaggui'"):
        reader("unused.xml", products=products, timezone="Asia/Tokyo")

    assert called is False


def test_forged_timezone_routing_state_cannot_bypass_validation(
    monkeypatch,
) -> None:
    from gwexpy.timeseries.io import dttxml as dttxml_io

    monkeypatch.setattr(dttxml_io, "load_dttxml_products", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="Could not parse timezone"):
        dttxml_io.read_timeseriesdict_dttxml(
            "unused.xml",
            products="TS",
            epoch=np.int64(123),
            timezone="+09:60",
            _timezone_checked=True,
            _epoch_timezone=dt.UTC,
            _timezone_routing_state=(object(), dt.UTC),
        )


def _ats_fixture() -> Path:
    return ROOT / "tests/fixtures/data/test.ats"


@pytest.mark.skipif(not _ats_fixture().exists(), reason="ATS fixture not found")
def test_ats_native_epoch_timezone_policy() -> None:
    from gwexpy.timeseries.io.ats import read_timeseries_ats

    with pytest.raises(ValueError, match="format 'ats'"):
        read_timeseries_ats(_ats_fixture(), timezone="Asia/Tokyo")
    with pytest.raises(ValueError, match="Could not parse timezone"):
        read_timeseries_ats(_ats_fixture(), epoch=123.0, timezone="Not/AZone")
    with pytest.warns(UserWarning, match="timezone.*ignored.*format 'ats'"):
        absolute = read_timeseries_ats(
            _ats_fixture(), epoch=123.0, timezone="Asia/Tokyo"
        )
    assert float(absolute.t0.value) == 123.0

    naive = dt.datetime(2024, 1, 1, 9)
    localized = read_timeseries_ats(_ats_fixture(), epoch=naive, timezone="+09:00")
    assert float(localized.t0.value) == float(
        to_gps(naive.replace(tzinfo=dt.timezone(dt.timedelta(hours=9))))
    )


def test_ats_mth5_rejects_timezone_before_dependency_lookup(monkeypatch) -> None:
    from gwexpy.timeseries.io import ats as ats_io

    called = False

    def _dependency(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dependency lookup must not run")

    monkeypatch.setattr(ats_io, "ensure_dependency", _dependency)

    with pytest.raises(ValueError, match="format 'ats.mth5'"):
        ats_io.read_timeseries_ats_mth5("unused.atss", timezone="Not/AZone", epoch=0.0)

    assert called is False


class _FakeTdmsChannel:
    name = "Signal"
    properties = {"wf_increment": 0.25, "wf_start_time": 0.0}

    @staticmethod
    def read_data():
        return np.arange(4, dtype=float)


class _FakeTdmsGroup:
    name = "Group"

    @staticmethod
    def channels():
        return [_FakeTdmsChannel()]


class _FakeTdmsFile:
    properties = {}

    @staticmethod
    def groups():
        return [_FakeTdmsGroup()]


class _FakeTdmsReader:
    @staticmethod
    def read(source):
        return _FakeTdmsFile()


def test_tdms_epoch_timezone_policy(monkeypatch) -> None:
    from gwexpy.timeseries.io import tdms as tdms_io

    monkeypatch.setattr(tdms_io, "_import_nptdms", lambda: _FakeTdmsReader)

    with pytest.raises(ValueError, match="format 'tdms'"):
        tdms_io.read_timeseriesdict_tdms("unused.tdms", timezone="UTC")
    with pytest.raises(ValueError, match="Could not parse timezone"):
        tdms_io.read_timeseriesdict_tdms(
            "unused.tdms", epoch=123.0, timezone="Not/AZone"
        )
    with pytest.warns(UserWarning, match="timezone.*ignored.*format 'tdms'"):
        absolute = tdms_io.read_timeseriesdict_tdms(
            "unused.tdms", epoch=123.0, timezone="UTC"
        )
    assert float(absolute["Group/Signal"].t0.value) == 123.0

    naive = dt.datetime(2024, 1, 1, 9)
    localized = tdms_io.read_timeseriesdict_tdms(
        "unused.tdms", epoch=naive, timezone="+09:00"
    )
    assert float(localized["Group/Signal"].t0.value) == float(
        to_gps(naive.replace(tzinfo=dt.timezone(dt.timedelta(hours=9))))
    )


@pytest.mark.skipif(
    not (ROOT / "tests/fixtures/data/test.gbd").exists(),
    reason="GBD fixture not found",
)
def test_gbd_absolute_epoch_warns_and_provenance_matches_use() -> None:
    from gwexpy.timeseries.io.gbd import read_timeseriesdict_gbd

    fixture = ROOT / "tests/fixtures/data/test.gbd"
    with pytest.warns(UserWarning, match="timezone.*ignored.*format 'gbd'") as caught:
        result = read_timeseriesdict_gbd(
            fixture,
            epoch=123.0,
            timezone="Asia/Tokyo",
        )

    assert len(caught) == 1
    assert all(float(series.t0.value) == 123.0 for series in result.values())
    assert result._gwexpy_io["timezone"] is None


def _win_stream(win_io, channel: str):
    trace = win_io.Trace(data=np.array([100, 101], dtype=np.int32))
    trace.stats.channel = channel
    trace.stats.sampling_rate = 1.0
    trace.stats.starttime = win_io.UTCDateTime(2024, 1, 1)
    return win_io.Stream(traces=[trace])


def test_win_warns_once_for_single_and_multi_source(monkeypatch) -> None:
    pytest.importorskip("obspy")
    from gwexpy.timeseries.io import win as win_io

    monkeypatch.setattr(
        win_io,
        "_read_win_fixed",
        lambda source, **kwargs: _win_stream(win_io, Path(source).stem),
    )
    expected = "WIN header time is timezone-naive; interpreting as UTC (#632)"

    with pytest.warns(UserWarning, match=r"WIN header time.*#632") as single:
        win_io.read_win_file("one.win")
    assert len(single) == 1
    assert str(single[0].message) == expected

    with pytest.warns(UserWarning, match=r"WIN header time.*#632") as multi:
        result = win_io.read_win_file(["one.win", "two.win"])
    assert len(multi) == 1
    assert len(result) == 2


def test_forged_win_warning_marker_cannot_suppress_warning(monkeypatch) -> None:
    pytest.importorskip("obspy")
    from gwexpy.timeseries.io import win as win_io

    monkeypatch.setattr(
        win_io,
        "_read_win_fixed",
        lambda source, **kwargs: _win_stream(win_io, "forged"),
    )

    with pytest.warns(UserWarning, match=r"WIN header time.*#632") as caught:
        win_io.read_win_file("unused.win", _utc_warning_marker=[True])

    assert len(caught) == 1


def test_win_invalid_timezone_precedes_reader_even_with_dummy_epoch(
    monkeypatch,
) -> None:
    pytest.importorskip("obspy")
    from gwexpy.timeseries.io import win as win_io

    called = False

    def _reader(*args, **kwargs):
        nonlocal called
        called = True
        return _win_stream(win_io, "unused")

    monkeypatch.setattr(win_io, "_read_win_fixed", _reader)
    with pytest.raises(ValueError, match="Could not parse timezone"):
        win_io.read_win_file("unused.win", timezone="Not/AZone", epoch=0.0)
    assert called is False


def _write_numeric_csv(path: Path, start: float) -> None:
    path.write_text(
        "\n".join(f"{start + offset},{offset}" for offset in (0.0, 1.0)) + "\n",
        encoding="utf-8",
    )


def test_csv_numeric_timezone_warning_once_across_multi_source(tmp_path) -> None:
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_numeric_csv(first, 0.0)
    _write_numeric_csv(second, 2.0)

    with pytest.warns(UserWarning, match="timezone.*ignored.*CSV") as caught:
        result = read_timeseriesdict_csv([first, second], timezone="Asia/Tokyo")

    assert len(caught) == 1
    assert len(result["ch1"]) == 4
    assert float(result["ch1"].t0.value) == 0.0


def test_csv_index_timezone_warning_once_across_multi_source(
    monkeypatch, tmp_path
) -> None:
    from gwexpy.timeseries.io import _multi
    from gwexpy.timeseries.io.csv_config import ColumnSpec, CSVFormatConfig
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("1\n2\n", encoding="utf-8")
    second.write_text("3\n4\n", encoding="utf-8")
    config = CSVFormatConfig(
        columns=[ColumnSpec(name="value", column_index=0, role="data")],
        sample_rate=1.0,
    )

    def _read_without_merging(reader, sources, format_name, **kwargs):
        results = [reader(source, **kwargs) for source in sources]
        return results[0]

    monkeypatch.setattr(_multi, "read_multi_dict", _read_without_merging)

    with pytest.warns(UserWarning, match="timezone.*ignored.*CSV") as caught:
        result = read_timeseriesdict_csv(
            [first, second],
            config=config,
            timezone="UTC",
        )

    assert len(caught) == 1
    assert float(result["value"].t0.value) == 0.0


@pytest.mark.parametrize("route", ["numeric", "index"])
@pytest.mark.parametrize("multi", [False, True])
def test_csv_invalid_timezone_is_rejected_before_ignored_warning(
    tmp_path, route, multi
) -> None:
    from gwexpy.timeseries.io.csv_config import ColumnSpec, CSVFormatConfig
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    if route == "numeric":
        _write_numeric_csv(first, 0.0)
        _write_numeric_csv(second, 2.0)
        config = None
    else:
        first.write_text("1\n2\n", encoding="utf-8")
        second.write_text("3\n4\n", encoding="utf-8")
        config = CSVFormatConfig(
            columns=[ColumnSpec(name="value", column_index=0, role="data")],
            sample_rate=1.0,
        )
    source = [first, second] if multi else first

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Could not parse timezone"):
            read_timeseriesdict_csv(
                source,
                config=config,
                timezone="Not/AZone",
                epoch=0.0,
            )

    assert caught == []


@pytest.mark.parametrize("contents", ["", "header,only\n"])
def test_csv_empty_routes_still_validate_timezone_without_warning(
    tmp_path, contents
) -> None:
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    path = tmp_path / "empty.csv"
    path.write_text(contents, encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Could not parse timezone.*format 'csv'"):
            read_timeseriesdict_csv(path, timezone="+09:60")

    assert caught == []


@pytest.mark.parametrize("contents", ["", "header,only\n"])
def test_csv_empty_routes_do_not_warn_for_valid_timezone(tmp_path, contents) -> None:
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    path = tmp_path / "empty.csv"
    path.write_text(contents, encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = read_timeseriesdict_csv(path, timezone="UTC")

    assert not result
    assert caught == []


def test_csv_empty_route_preserves_explicit_zero_timezone(
    monkeypatch, tmp_path
) -> None:
    from gwexpy.timeseries.io import csv_enhanced as module
    from gwexpy.timeseries.io.csv_config import CSVFormatConfig

    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    observed = []

    def record_timezone(format_name, timezone):
        observed.append((format_name, timezone))
        return dt.UTC

    monkeypatch.setattr(module, "_parse_timezone_for_format", record_timezone)

    result = module.read_timeseriesdict_csv(
        path,
        config=CSVFormatConfig(timezone="Asia/Tokyo"),
        timezone=0,
    )

    assert not result
    assert observed == [("csv", 0)]


def test_csv_empty_route_does_not_replace_explicit_empty_timezone(
    tmp_path,
) -> None:
    from gwexpy.timeseries.io.csv_config import CSVFormatConfig
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse timezone ''.*format 'csv'"):
        read_timeseriesdict_csv(
            path,
            config=CSVFormatConfig(timezone="UTC"),
            timezone="",
        )


def test_csv_empty_component_route_requires_timezone_before_return(tmp_path) -> None:
    from gwexpy.timeseries.io.csv_config import ColumnSpec, CSVFormatConfig
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    config = CSVFormatConfig(
        columns=[
            ColumnSpec(
                name="year",
                column_index=0,
                role="time_component",
                time_component="year",
            )
        ]
    )

    with pytest.raises(
        ValueError, match="timezone is required when using time_component columns"
    ):
        read_timeseriesdict_csv(path, config=config)


def test_forged_csv_warning_marker_cannot_suppress_warning(tmp_path) -> None:
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    path = tmp_path / "numeric.csv"
    _write_numeric_csv(path, 0.0)

    with pytest.warns(UserWarning, match="timezone.*ignored.*CSV") as caught:
        read_timeseriesdict_csv(
            path,
            timezone="UTC",
            _timezone_warning_marker=[True],
        )

    assert len(caught) == 1


def test_win_validates_timezone_before_optional_backend(monkeypatch) -> None:
    from gwexpy.timeseries.io import win as module

    monkeypatch.setattr(module, "HAS_OBSPY", False)

    with pytest.raises(ValueError, match="Could not parse timezone.*format 'win'"):
        module.read_win_file("unused.win", timezone="+09:60")


@pytest.mark.parametrize(
    ("reader_kind", "timezone", "message"),
    [
        ("wav", "UTC", "format 'wav'"),
        ("wav", "+09:60", "Could not parse timezone.*format 'wav'"),
        ("audio", "UTC", "format 'mp3'"),
        ("audio", "+09:60", "Could not parse timezone.*format 'mp3'"),
        ("ndscope", "UTC", "format 'hdf.ndscope'"),
        (
            "ndscope",
            "+09:60",
            "Could not parse timezone.*format 'hdf.ndscope'",
        ),
    ],
)
def test_not_accepted_timezone_fails_before_optional_backend(
    monkeypatch, reader_kind, timezone, message
) -> None:
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("backend must not be called")

    if reader_kind == "wav":
        from gwexpy.timeseries.io import wav as module

        monkeypatch.setattr(module.wavfile, "read", fail_if_called)
        reader = module.read_timeseriesdict_wav
        kwargs = {}
    elif reader_kind == "audio":
        from gwexpy.timeseries.io import audio as module

        monkeypatch.setattr(module, "_import_pydub", fail_if_called)
        reader = module.read_timeseriesdict_audio
        kwargs = {"format_hint": "mp3"}
    else:
        from gwexpy.timeseries.io import ndscope_hdf5 as module

        monkeypatch.setattr(module.h5py, "File", fail_if_called)
        reader = module.read_timeseriesdict_ndscope_hdf5
        kwargs = {}

    with pytest.raises(ValueError, match=message):
        reader("unused", timezone=timezone, **kwargs)

    assert called is False


@pytest.mark.parametrize(
    ("format_name", "timezone", "message"),
    [
        ("hdf5", "UTC", "format 'hdf5'"),
        ("hdf5", "+09:60", "Could not parse timezone.*format 'hdf5'"),
        ("txt", "UTC", "format 'txt'"),
        ("txt", "+09:60", "Could not parse timezone.*format 'txt'"),
    ],
)
def test_collection_direct_routes_reject_timezone_before_traversal(
    tmp_path, format_name, timezone, message
) -> None:
    from gwexpy.timeseries import TimeSeriesDict

    source = tmp_path / ("collection" if format_name == "txt" else "unused.h5")
    if format_name == "txt":
        source.mkdir()

    with pytest.raises(ValueError, match=message):
        TimeSeriesDict.read(source, format=format_name, timezone=timezone)


@pytest.mark.parametrize("epoch_type", [np.int64, np.float64])
@pytest.mark.parametrize("reader_kind", ["wav", "audio"])
def test_audio_readers_accept_numpy_numeric_epochs(
    monkeypatch, epoch_type, reader_kind
) -> None:
    epoch = epoch_type(123)
    if reader_kind == "wav":
        from gwexpy.timeseries.io import wav as module

        monkeypatch.setattr(
            module.wavfile,
            "read",
            lambda *args, **kwargs: (4, np.arange(4, dtype=np.int16)),
        )
        result = module.read_timeseriesdict_wav("unused.wav", epoch=epoch)
    else:
        from gwexpy.timeseries.io import audio as module

        segment = SimpleNamespace(
            channels=1,
            frame_rate=4,
            sample_width=2,
            get_array_of_samples=lambda: [0, 1, 2, 3],
        )
        audio_segment = SimpleNamespace(
            from_file=lambda *args, **kwargs: segment,
        )
        monkeypatch.setattr(module, "_import_pydub", lambda: audio_segment)
        result = module.read_timeseriesdict_audio(
            "unused.mp3", format_hint="mp3", epoch=epoch
        )

    assert float(result["channel_0"].t0.value) == 123.0


@pytest.mark.parametrize("epoch", [0, np.int64(0), np.float64(0)])
@pytest.mark.parametrize("reader_kind", ["wav", "audio"])
def test_audio_zero_epoch_is_recorded_as_user_provenance(
    monkeypatch, epoch, reader_kind
) -> None:
    if reader_kind == "wav":
        from gwexpy.timeseries.io import wav as module

        monkeypatch.setattr(
            module.wavfile,
            "read",
            lambda *args, **kwargs: (4, np.arange(4, dtype=np.int16)),
        )
        result = module.read_timeseriesdict_wav("unused.wav", epoch=epoch)
    else:
        from gwexpy.timeseries.io import audio as module

        segment = SimpleNamespace(
            channels=1,
            frame_rate=4,
            sample_width=2,
            get_array_of_samples=lambda: [0, 1, 2, 3],
        )
        audio_segment = SimpleNamespace(
            from_file=lambda *args, **kwargs: segment,
        )
        monkeypatch.setattr(module, "_import_pydub", lambda: audio_segment)
        result = module.read_timeseriesdict_audio(
            "unused.mp3", format_hint="mp3", epoch=epoch
        )

    assert result._gwexpy_io["epoch_source"] == "user"


def test_public_reader_signatures_hide_internal_timezone_state() -> None:
    from gwpy.io.registry import default_registry as io_registry

    from gwexpy.frequencyseries import FrequencySeriesDict, FrequencySeriesMatrix
    from gwexpy.frequencyseries.io import dttxml as frequency_dttxml
    from gwexpy.timeseries import TimeSeries, TimeSeriesDict
    from gwexpy.timeseries.io import ats, csv_enhanced, dttxml, gbd, tdms, win

    readers = [
        ats.read_timeseriesdict_ats,
        ats.read_timeseries_ats,
        ats.read_timeseries_ats_mth5,
        csv_enhanced.read_timeseriesdict_csv,
        dttxml.read_timeseriesdict_dttxml,
        frequency_dttxml.read_frequencyseriesdict_dttxml,
        frequency_dttxml.read_frequencyseriesmatrix_dttxml,
        gbd.read_timeseriesdict_gbd,
        tdms.read_timeseriesdict_tdms,
        win.read_win_file,
    ]
    readers.extend(
        [
            io_registry.get_reader("ats", TimeSeries),
            io_registry.get_reader("ats", TimeSeriesDict),
            io_registry.get_reader("csv", TimeSeriesDict),
            io_registry.get_reader("xml.diaggui", TimeSeriesDict),
            io_registry.get_reader("gbd", TimeSeriesDict),
            io_registry.get_reader("tdms", TimeSeriesDict),
            io_registry.get_reader("xml.diaggui", FrequencySeriesDict),
            io_registry.get_reader("xml.diaggui", FrequencySeriesMatrix),
        ]
    )
    if win.HAS_OBSPY:
        readers.append(io_registry.get_reader("win", TimeSeriesDict))

    for reader in readers:
        internal = {
            name
            for name in inspect.signature(reader).parameters
            if name.startswith("_")
        }
        assert internal == set(), f"{reader.__qualname__}: {sorted(internal)}"


@pytest.mark.parametrize(
    ("timezone", "offset"),
    [("UTC", 0), ("Asia/Tokyo", 9), ("+09:00", 9)],
)
def test_csv_component_route_localizes_timezone(tmp_path, timezone, offset) -> None:
    from gwexpy.timeseries.io.csv_config import ColumnSpec, CSVFormatConfig
    from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

    path = tmp_path / "component.csv"
    path.write_text("2024,1,1,9,0,0,1\n", encoding="utf-8")
    roles = ("year", "month", "day", "hour", "minute", "second")
    config = CSVFormatConfig(
        columns=[
            *[
                ColumnSpec(
                    name=role,
                    column_index=index,
                    role="time_component",
                    time_component=role,
                )
                for index, role in enumerate(roles)
            ],
            ColumnSpec(name="value", column_index=6, role="data"),
        ]
    )

    result = read_timeseriesdict_csv(path, config=config, timezone=timezone)
    expected = dt.datetime(
        2024,
        1,
        1,
        9,
        tzinfo=dt.timezone(dt.timedelta(hours=offset)),
    )
    assert float(result["value"].t0.value) == float(to_gps(expected))
