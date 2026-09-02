"""GWpy-native CSV routing contracts for the v0.2.3 Phase 4 fixes."""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.io.registry import default_registry as io_registry
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

import gwexpy
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries

ENHANCED_KEYS = (
    "config",
    "channels",
    "timezone",
    "resample",
    "resample_method",
)

SERIES_TYPES = (
    pytest.param(TimeSeries, GwpyTimeSeries, "timeseries", id="timeseries"),
    pytest.param(
        FrequencySeries,
        GwpyFrequencySeries,
        "frequencyseries",
        id="frequencyseries",
    ),
)


def _series(cls, kind: str, values=None):
    values = np.asarray([10.0, 20.0] if values is None else values)
    if kind == "timeseries":
        return cls(values, t0=0, dt=1)
    return cls(values, f0=0, df=1)


def _assert_native_series_equal(actual, expected) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    assert actual.xindex.unit == expected.xindex.unit
    assert actual.name == expected.name
    assert actual.channel == expected.channel
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.xindex.value, expected.xindex.value)
    if isinstance(expected, GwpyFrequencySeries):
        assert actual.epoch == expected.epoch


def _read_csv(cls, path, *, explicit: bool, **kwargs):
    if explicit:
        return cls.read(path, format="csv", **kwargs)
    return cls.read(path, **kwargs)


@pytest.mark.parametrize("explicit", [False, True], ids=["suffix", "explicit"])
def test_plain_timeseries_csv_uses_native_reader(tmp_path, explicit):
    path = tmp_path / ("data.csv" if not explicit else "data.dat")
    path.write_text("0,10\n1,20\n2,30\n", encoding="utf-8")

    expected = _read_csv(GwpyTimeSeries, path, explicit=explicit)
    actual = _read_csv(TimeSeries, path, explicit=explicit)

    _assert_native_series_equal(actual, expected)


@pytest.mark.parametrize(
    ("content", "kwargs"),
    [
        pytest.param("0;10\n1;20\n", {"delimiter": ";"}, id="delimiter"),
        pytest.param("# x,y\n0,10\n1,20\n", {}, id="comment-header"),
        pytest.param("x,y\n0,10\n1,20\n", {"skiprows": 1}, id="skiprows"),
        pytest.param("0,10\n1,20\n", {"dtype": np.float32}, id="dtype-float32"),
        pytest.param("0,99,10\n1,88,20\n", {"usecols": (0, 2)}, id="usecols"),
        pytest.param("0,10\n1,20\n2,30\n", {"start": 1, "end": 2}, id="crop"),
    ],
)
def test_timeseries_reader_kwargs_match_native_gwpy(tmp_path, content, kwargs):
    path = tmp_path / "kwargs.csv"
    path.write_text(content, encoding="utf-8")

    expected = GwpyTimeSeries.read(path, format="csv", **kwargs)
    actual = TimeSeries.read(path, format="csv", **kwargs)

    _assert_native_series_equal(actual, expected)


def test_plain_text_header_preserves_native_failure(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("x,y\n0,10\n1,20\n", encoding="utf-8")

    with pytest.raises(ValueError):
        GwpyTimeSeries.read(path, format="csv")
    with pytest.raises(ValueError):
        TimeSeries.read(path, format="csv")


@pytest.mark.parametrize("key", ENHANCED_KEYS)
@pytest.mark.parametrize(
    "format_mode",
    ["absent", "none", "csv"],
    ids=["format-absent", "format-none", "format-csv"],
)
def test_explicit_none_selects_enhanced_reader(tmp_path, key, format_mode):
    path = tmp_path / ("enhanced.csv" if format_mode == "csv" else "enhanced.dat")
    path.write_text("time,value\n0,10\n1,20\n", encoding="utf-8")
    kwargs = {key: None}
    if format_mode == "none":
        kwargs["format"] = None
    elif format_mode == "csv":
        kwargs["format"] = "csv"

    result = TimeSeries.read(path, **kwargs)

    np.testing.assert_array_equal(result.value, [10.0, 20.0])
    np.testing.assert_array_equal(result.xindex.value, [0.0, 1.0])


@pytest.mark.parametrize("key", ENHANCED_KEYS)
def test_explicit_non_csv_format_is_authoritative(tmp_path, key):
    path = tmp_path / "data.txt"
    path.write_text("0 10\n1 20\n", encoding="utf-8")

    with pytest.raises(TypeError):
        TimeSeries.read(path, format="txt", **{key: None})


def test_direct_helpers_remain_enhanced(tmp_path):
    from gwexpy.timeseries.io.csv_enhanced import (
        read_timeseries_csv,
        write_timeseries_csv,
    )

    source = tmp_path / "enhanced-input.dat"
    source.write_text("time,value\n0,10\n1,20\n", encoding="utf-8")
    result = read_timeseries_csv(source)
    np.testing.assert_array_equal(result.value, [10.0, 20.0])

    target = tmp_path / "enhanced-output.dat"
    returned = write_timeseries_csv(_series(TimeSeries, "timeseries"), target)
    assert returned == target
    assert target.read_text(encoding="utf-8").startswith("# gwexpy.timeseries.csv v1\n")


@pytest.mark.parametrize("actual_cls,parent_cls,kind", SERIES_TYPES)
@pytest.mark.parametrize("explicit", [False, True], ids=["suffix", "explicit"])
def test_native_writer_route_and_output(
    tmp_path, actual_cls, parent_cls, kind, explicit
):
    actual_path = tmp_path / f"actual-{kind}.csv"
    parent_path = tmp_path / f"parent-{kind}.csv"
    actual = _series(actual_cls, kind)
    expected = _series(parent_cls, kind)

    if explicit:
        actual_return = actual.write(actual_path, format="csv")
        parent_return = expected.write(parent_path, format="csv")
    else:
        actual_return = actual.write(actual_path)
        parent_return = expected.write(parent_path)

    assert actual_return is parent_return is None
    assert actual_path.read_text(encoding="utf-8") == parent_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("actual_cls,parent_cls,kind", SERIES_TYPES)
def test_native_writer_kwargs_are_forwarded(actual_cls, parent_cls, kind):
    kwargs = {
        "delimiter": ";",
        "header": "x;y",
        "comments": "# ",
        "fmt": ["%.1f", "%.2f"],
    }
    actual_buffer = io.StringIO()
    parent_buffer = io.StringIO()

    actual_return = _series(actual_cls, kind).write(
        actual_buffer, format="csv", **kwargs
    )
    parent_return = _series(parent_cls, kind).write(
        parent_buffer, format="csv", **kwargs
    )

    assert actual_return is parent_return is None
    assert actual_buffer.getvalue() == parent_buffer.getvalue()


@pytest.mark.parametrize("actual_cls,parent_cls,kind", SERIES_TYPES)
def test_invalid_writer_kwarg_preserves_native_typeerror(actual_cls, parent_cls, kind):
    with pytest.raises(TypeError):
        _series(parent_cls, kind).write(io.StringIO(), format="csv", not_a_savetxt_kw=1)
    with pytest.raises(TypeError):
        _series(actual_cls, kind).write(io.StringIO(), format="csv", not_a_savetxt_kw=1)


@pytest.mark.parametrize("actual_cls,parent_cls,kind", SERIES_TYPES)
@pytest.mark.parametrize("explicit", [False, True], ids=["suffix", "explicit"])
def test_extra_writer_positional_preserves_native_typeerror(
    tmp_path, actual_cls, parent_cls, kind, explicit
):
    actual_path = tmp_path / f"actual-extra-{kind}.csv"
    parent_path = tmp_path / f"parent-extra-{kind}.csv"

    parent_kwargs = {"format": "csv"} if explicit else {}
    actual_kwargs = {"format": "csv"} if explicit else {}
    with pytest.raises(TypeError):
        _series(parent_cls, kind).write(parent_path, "extra", **parent_kwargs)
    with pytest.raises(TypeError):
        _series(actual_cls, kind).write(actual_path, "extra", **actual_kwargs)


@pytest.mark.parametrize("actual_cls,parent_cls,kind", SERIES_TYPES)
def test_uppercase_csv_suffix_preserves_native_identification_failure(
    tmp_path, actual_cls, parent_cls, kind
):
    parent_read_path = tmp_path / f"parent-read-{kind}.CSV"
    actual_read_path = tmp_path / f"actual-read-{kind}.CSV"
    parent_read_path.write_text("0,10\n1,20\n", encoding="utf-8")
    actual_read_path.write_text("0,10\n1,20\n", encoding="utf-8")

    with pytest.raises(IORegistryError):
        parent_cls.read(parent_read_path)
    with pytest.raises(IORegistryError):
        actual_cls.read(actual_read_path)

    with pytest.raises(IORegistryError):
        _series(parent_cls, kind).write(tmp_path / f"parent-{kind}.CSV")
    with pytest.raises(IORegistryError):
        _series(actual_cls, kind).write(tmp_path / f"actual-{kind}.CSV")


@pytest.mark.parametrize("actual_cls,parent_cls,kind", SERIES_TYPES)
def test_complex_csv_matches_native_write_and_supported_roundtrip(
    tmp_path, actual_cls, parent_cls, kind
):
    values = np.array([1 + 2j, 3 + 4j])
    actual_path = tmp_path / f"actual-complex-{kind}.csv"
    parent_path = tmp_path / f"parent-complex-{kind}.csv"

    actual_return = _series(actual_cls, kind, values).write(actual_path, format="csv")
    parent_return = _series(parent_cls, kind, values).write(parent_path, format="csv")

    assert actual_return is parent_return is None
    assert actual_path.read_text(encoding="utf-8") == parent_path.read_text(
        encoding="utf-8"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            parent_cls.read(parent_path, format="csv")
        with pytest.raises(ValueError):
            actual_cls.read(parent_path, format="csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = parent_cls.read(parent_path, format="csv", dtype=complex)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = actual_cls.read(parent_path, format="csv", dtype=complex)
    _assert_native_series_equal(actual, expected)


@pytest.mark.parametrize(
    ("content", "kwargs"),
    [
        pytest.param(
            "0,99,10\n1,88,20\n",
            {"usecols": (0, 2), "dtype": np.float32},
            id="dtype-usecols",
        ),
        pytest.param(
            "0,10\n1,20\n2,30\n",
            {"start": 1, "end": 2},
            id="crop",
        ),
    ],
)
@pytest.mark.parametrize(
    ("actual_cls", "parent_cls"),
    [
        pytest.param(TimeSeries, GwpyTimeSeries, id="timeseries"),
        pytest.param(FrequencySeries, GwpyFrequencySeries, id="frequencyseries"),
    ],
)
def test_frequency_and_time_reader_kwargs_follow_native_gwpy(
    tmp_path, actual_cls, parent_cls, content, kwargs
):
    path = tmp_path / "reader-kwargs.csv"
    path.write_text(content, encoding="utf-8")

    if parent_cls is GwpyFrequencySeries and "start" in kwargs:
        with pytest.raises(TypeError):
            parent_cls.read(path, format="csv", **kwargs)
        with pytest.raises(TypeError):
            actual_cls.read(path, format="csv", **kwargs)
        return

    expected = parent_cls.read(path, format="csv", **kwargs)
    actual = actual_cls.read(path, format="csv", **kwargs)

    _assert_native_series_equal(actual, expected)


def test_frequencyseries_text_header_preserves_native_failure(tmp_path):
    path = tmp_path / "frequency-header.csv"
    path.write_text("frequency,value\n0,10\n1,20\n", encoding="utf-8")

    with pytest.raises(ValueError):
        GwpyFrequencySeries.read(path, format="csv")
    with pytest.raises(ValueError):
        FrequencySeries.read(path, format="csv")


@pytest.mark.parametrize("actual_cls,parent_cls,kind", SERIES_TYPES)
def test_explicit_csv_file_like_behavior_matches_native(actual_cls, parent_cls, kind):
    content = "0,10\n1,20\n2,30\n"

    expected = parent_cls.read(io.StringIO(content), format="csv")
    actual = actual_cls.read(io.StringIO(content), format="csv")

    _assert_native_series_equal(actual, expected)


def test_repeated_bootstrap_keeps_native_single_series_handlers():
    gwexpy.register_all()
    first = {
        cls: (
            io_registry.get_reader("csv", cls),
            io_registry.get_writer("csv", cls),
        )
        for cls in (TimeSeries, FrequencySeries)
    }

    gwexpy.register_all()

    for cls, handlers in first.items():
        assert io_registry.get_reader("csv", cls) is handlers[0]
        assert io_registry.get_writer("csv", cls) is handlers[1]
        assert handlers[0].__module__.startswith("gwpy.")
        assert handlers[1].__module__.startswith("gwpy.")


def test_collection_and_matrix_csv_readers_remain_enhanced():
    from gwexpy.timeseries import TimeSeriesDict, TimeSeriesMatrix

    for cls in (TimeSeriesDict, TimeSeriesMatrix):
        assert io_registry.get_reader("csv", cls).__module__.startswith(
            "gwexpy.timeseries.io.csv_enhanced"
        )

    assert (
        io_registry.identify_format(
            "read", TimeSeries, "data.CSV", None, ("data.CSV",), {}
        )
        == []
    )
    assert io_registry.identify_format(
        "read", TimeSeriesMatrix, "data.CSV", None, ("data.CSV",), {}
    ) == ["csv"]
