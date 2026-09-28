"""B1-compatible multi-source merge and bounded-copy regression tests."""

from __future__ import annotations

import gc
import warnings
import weakref
from collections import Counter

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io import _multi


def _read(files):
    return lambda source, **kwargs: files[source]


def _series(values, *, t0=0, dt=1, unit=None, dtype=None, name=None):
    return TimeSeries(
        np.asarray(values, dtype=dtype), t0=t0, dt=dt, unit=unit, name=name
    )


def test_regular_64_segment_merge_places_each_input_once(monkeypatch):
    files = {
        index: TimeSeriesDict(
            {
                "merge": TimeSeries(
                    np.full(4096, index, dtype=np.float64),
                    t0=index * 1024,
                    dt=0.25,
                    name="merge",
                    unit="m",
                )
            }
        )
        for index in range(64)
    }
    append_calls = []
    placements = []
    original_append = TimeSeries.append
    original_place = _multi._place_segment

    def count_append(self, other, **kwargs):
        if not kwargs.get("inplace", True):
            append_calls.append((self, other))
        return original_append(self, other, **kwargs)

    def count_place(destination, start, source):
        placements.append((id(source), source.nbytes))
        return original_place(destination, start, source)

    monkeypatch.setattr(TimeSeries, "append", count_append)
    monkeypatch.setattr(_multi, "_place_segment", count_place)

    result = _multi.read_multi_dict(_read(files), list(reversed(files)), "test")
    expected_ids = {id(files[index]["merge"]) for index in files}

    assert append_calls == []
    assert Counter(source_id for source_id, _ in placements) == Counter(expected_ids)
    assert sum(size for _, size in placements) <= 2_097_152
    np.testing.assert_array_equal(
        result["merge"].value,
        np.repeat(np.arange(64, dtype=np.float64), 4096),
    )
    assert all(
        not np.shares_memory(result["merge"].value, files[index]["merge"].value)
        for index in files
    )


def test_stable_channel_and_equal_epoch_source_order():
    files = {
        "later": TimeSeriesDict(
            {
                "b": _series([3, 4], t0=2, name="first-b"),
                "a": _series([10, 11], t0=0),
            }
        ),
        "earlier": TimeSeriesDict(
            {
                "a": _series([12, 13], t0=0),
                "b": _series([1, 2], t0=0),
                "unique": _series([9]),
            }
        ),
    }

    result = _multi.read_multi_dict(
        _read(files), ["later", "earlier"], "test", gap="ignore"
    )

    assert list(result) == ["b", "a", "unique"]
    np.testing.assert_array_equal(result["b"].value, [1, 2, 3, 4])
    np.testing.assert_array_equal(result["a"].value, [10, 11, 12, 13])
    assert result["unique"] is files["earlier"]["unique"]
    assert result["b"].name is None  # earliest segment supplies metadata


@pytest.mark.parametrize(
    ("gap", "pad", "expected"),
    [
        ("pad", -7.0, [1.0, 2.0, -7.0, -7.0, 3.0, 4.0]),
        ("ignore", -7.0, [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_gap_modes_keep_values_and_regular_axis(gap, pad, expected):
    files = {
        0: TimeSeriesDict({"ch": _series([1.0, 2.0], t0=0, dt=1)}),
        1: TimeSeriesDict({"ch": _series([3.0, 4.0], t0=4, dt=1)}),
    }

    merged = _multi.read_multi_dict(_read(files), [0, 1], "test", gap=gap, pad=pad)[
        "ch"
    ]

    np.testing.assert_array_equal(merged.value, expected)
    np.testing.assert_array_equal(merged.times.value, np.arange(len(expected)))
    assert merged.dtype == np.dtype("float64")


def test_integer_nan_padding_preserves_b1_cast_warning_and_dtype():
    files = {
        0: TimeSeriesDict({"ch": _series([1, 2], t0=0, dtype=np.int16)}),
        1: TimeSeriesDict({"ch": _series([3, 4], t0=4, dtype=np.int16)}),
    }

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        merged = _multi.read_multi_dict(_read(files), [0, 1], "test")["ch"]

    np.testing.assert_array_equal(merged.value, [1, 2, 0, 0, 3, 4])
    assert merged.dtype == np.dtype("int16")
    assert [(w.category, str(w.message)) for w in seen] == [
        (RuntimeWarning, "invalid value encountered in cast")
    ]


def test_equivalent_units_convert_to_first_segment_unit():
    files = {
        0: TimeSeriesDict({"ch": _series([1.0, 2.0], unit="m")}),
        1: TimeSeriesDict({"ch": _series([300.0, 400.0], t0=2, unit="cm")}),
    }

    merged = _multi.read_multi_dict(_read(files), [0, 1], "test")["ch"]

    np.testing.assert_array_equal(merged.value, [1.0, 2.0, 3.0, 4.0])
    assert str(merged.unit) == "m"


def test_gap_ignore_concatenates_overlapping_samples_without_deduplication():
    files = {
        0: TimeSeriesDict({"ch": _series([1, 2, 3], t0=0)}),
        1: TimeSeriesDict({"ch": _series([4, 5], t0=2)}),
    }

    merged = _multi.read_multi_dict(_read(files), [0, 1], "test", gap="ignore")["ch"]

    np.testing.assert_array_equal(merged.value, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(merged.times.value, [0, 1, 2, 3, 4])


def test_safe_dtype_cast_keeps_first_segment_dtype():
    files = {
        0: TimeSeriesDict({"ch": _series([1, 2], dtype=np.int64)}),
        1: TimeSeriesDict({"ch": _series([3, 4], t0=2, dtype=np.int16)}),
    }

    merged = _multi.read_multi_dict(_read(files), [0, 1], "test")["ch"]

    assert merged.dtype == np.dtype("int64")
    np.testing.assert_array_equal(merged.value, [1, 2, 3, 4])


def test_incompatible_dtype_keeps_unwrapped_b1_type_error():
    files = {
        0: TimeSeriesDict({"ch": _series([1, 2], dtype=np.int16)}),
        1: TimeSeriesDict({"ch": _series([3.5, 4.5], t0=2)}),
    }

    with pytest.raises(TypeError, match="array data types do not match"):
        _multi.read_multi_dict(_read(files), [0, 1], "test")


def test_incompatible_cadence_replays_b1_value_error_context():
    files = {
        0: TimeSeriesDict({"ch": _series([1.0, 2.0], dt=1)}),
        1: TimeSeriesDict({"ch": _series([3.0, 4.0], t0=2, dt=0.5)}),
    }

    with pytest.raises(
        ValueError,
        match="failed to merge channel 'ch' across test files: "
        "TimeSeries x-axis sample sizes do not match",
    ):
        _multi.read_multi_dict(_read(files), [0, 1], "test")


def test_quantity_padding_uses_append_error_behavior():
    files = {
        0: TimeSeriesDict({"ch": _series([1.0, 2.0], unit="m")}),
        1: TimeSeriesDict({"ch": _series([3.0, 4.0], t0=4, unit="m")}),
    }

    with pytest.raises(AttributeError, match="no 'xspan' member"):
        _multi.read_multi_dict(_read(files), [0, 1], "test", pad=1 * u.m)


def test_negative_cadence_uses_gwpy_gap_geometry():
    files = {
        0: TimeSeriesDict({"ch": _series([1.0], t0=0, dt=-1)}),
        1: TimeSeriesDict({"ch": _series([3.0, 4.0], t0=0, dt=-1)}),
    }

    merged = _multi.read_multi_dict(_read(files), [0, 1], "test", pad=0.0)["ch"]

    np.testing.assert_array_equal(merged.value, [1, 0, 0, 3, 4])


@pytest.mark.parametrize(
    ("second_offset", "gap", "pad"),
    [
        (0.3, "pad", -3.0),
        (0.3 + 2**-19, "raise", -3.0),  # within GWpy's 2**-18 tolerance
        (0.3 + 2**-17, "pad", -3.0),  # outside tolerance, rounds to zero
        (0.4, "pad", np.float32(-4.5)),
        (0.4, "pad", np.array([5.0])),
        (0.35, "pad", 7.0),  # half-sample gap rounding
    ],
)
def test_regular_metadata_and_gap_rounding_match_b1_append(second_offset, gap, pad):
    epoch_ns = 1_000_000_000_001
    first = TimeSeries(
        [1.0, 2.0, 3.0],
        t0_ns=epoch_ns,
        dt=0.1,
        unit="m",
        name="first name",
        channel="H1:FIRST",
    )
    first._gwex_note = "keep"
    second = TimeSeries(
        [400.0, 500.0],
        t0=first.t0.value + second_offset,
        dt=0.1,
        unit="cm",
        name="second name",
        channel="H1:SECOND",
    )
    files = {0: TimeSeriesDict({"ch": first}), 1: TimeSeriesDict({"ch": second})}

    with warnings.catch_warnings(record=True) as before_warnings:
        warnings.simplefilter("always")
        try:
            expected = first.append(second, inplace=False, gap=gap, pad=pad)
            expected_error = None
        except Exception as exc:
            expected = None
            expected_error = exc

    with warnings.catch_warnings(record=True) as after_warnings:
        warnings.simplefilter("always")
        try:
            actual = _multi.read_multi_dict(
                _read(files), [0, 1], "test", gap=gap, pad=pad
            )["ch"]
            actual_error = None
        except Exception as exc:
            actual = None
            actual_error = exc

    assert [(w.category, str(w.message)) for w in after_warnings] == [
        (w.category, str(w.message)) for w in before_warnings
    ]
    if expected_error is not None:
        assert type(actual_error) is type(expected_error)
        assert str(actual_error).endswith(str(expected_error))
        return

    assert actual_error is None
    assert expected is not None
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.times.value, expected.times.value)
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    assert actual.t0 == expected.t0
    assert actual.dt == expected.dt
    assert actual.t0_gps_ns == expected.t0_gps_ns
    assert actual._gwex_dt_gps_ns == expected._gwex_dt_gps_ns
    assert actual.name == expected.name
    assert str(actual.channel) == str(expected.channel)
    assert actual._gwex_note == expected._gwex_note


def test_exact_gps_nanoseconds_survive_regular_merge():
    epoch_ns = 1_000_000_000_001
    first = TimeSeries([1.0, 2.0], t0_ns=epoch_ns, dt=1e-9)
    second = TimeSeries([3.0, 4.0], t0_ns=epoch_ns + 2, dt=1e-9)
    files = {0: TimeSeriesDict({"ch": first}), 1: TimeSeriesDict({"ch": second})}

    merged = _multi.read_multi_dict(_read(files), [0, 1], "test")["ch"]

    np.testing.assert_array_equal(merged.value, [1.0, 2.0, 3.0, 4.0])
    assert merged.t0_gps_ns == epoch_ns
    assert merged._gwex_dt_gps_ns == 1


@pytest.mark.parametrize(
    ("gap", "second_t0", "message"),
    [
        ("pad", 0, "Cannot append TimeSeries that starts before this one"),
        ("raise", 4, "Cannot append discontiguous TimeSeries"),
        ("raise", 1, "Cannot append overlapping TimeSeriess"),
    ],
)
def test_merge_errors_keep_b1_type_and_context(gap, second_t0, message):
    files = {
        0: TimeSeriesDict({"ch": _series([1.0, 2.0])}),
        1: TimeSeriesDict({"ch": _series([3.0, 4.0], t0=second_t0)}),
    }

    with pytest.raises(ValueError) as caught:
        _multi.read_multi_dict(_read(files), [0, 1], "test", gap=gap)

    assert str(caught.value).startswith(
        f"failed to merge channel 'ch' across test files: {message}"
    )


def test_first_file_provenance_and_single_segment_identity():
    first = TimeSeriesDict({"only": _series([1, 2])})
    first._gwexpy_io = {"source": "first"}
    second = TimeSeriesDict({"other": _series([3, 4])})
    second._gwexpy_io = {"source": "second"}

    result = _multi.read_multi_dict(_read({0: first, 1: second}), [0, 1], "test")

    assert result["only"] is first["only"]
    assert result["other"] is second["other"]
    assert result._gwexpy_io == {"source": "first", "n_sources": 2}


def test_materialized_index_uses_original_append_path(monkeypatch):
    first = _series([1.0, 2.0])
    second = _series([3.0, 4.0], t0=2)
    _ = first.xindex
    _ = second.xindex
    files = {0: TimeSeriesDict({"ch": first}), 1: TimeSeriesDict({"ch": second})}
    calls = []
    original = TimeSeries.append

    def count_append(self, other, **kwargs):
        calls.append(kwargs)
        return original(self, other, **kwargs)

    monkeypatch.setattr(TimeSeries, "append", count_append)
    merged = _multi.read_multi_dict(_read(files), [0, 1], "test")["ch"]

    assert calls and calls[0]["inplace"] is False
    np.testing.assert_array_equal(merged.value, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(merged.times.value, [0.0, 1.0, 2.0, 3.0])


def test_finished_channel_releases_intermediate_input_parts(monkeypatch):
    seen = []

    def reader(index):
        first = _series([index], t0=index, name="first")
        if index == 1:
            seen.append(weakref.ref(first))
        return TimeSeriesDict(
            {"first": first, "second": _series([index], t0=index, name="second")}
        )

    original = _multi._place_segment
    observed = []

    def inspect_lifetime(destination, start, source):
        if source.name == "second":
            gc.collect()
            observed.append(seen[0]() is None)
        return original(destination, start, source)

    monkeypatch.setattr(_multi, "_place_segment", inspect_lifetime)
    result = _multi.read_multi_dict(reader, [0, 1, 2], "test")

    assert observed and all(observed)
    np.testing.assert_array_equal(result["first"].value, [0, 1, 2])
