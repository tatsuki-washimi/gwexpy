"""Contracts for parallel GWF filesystem reads (#588)."""

from __future__ import annotations

import copy
import importlib.util
import os
import pickle
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

import gwexpy.timeseries._gwf_io as gwf_io
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

CHANNEL = "K1:TEST-CHANNEL"


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"parallel": None}, 1),
        ({"parallel": False}, 1),
        ({"parallel": np.bool_(False)}, 1),
        ({"parallel": 1}, 1),
        ({"parallel": np.int64(1)}, 1),
        ({"nproc": None}, 1),
        ({"nproc": 1}, 1),
        ({"nproc": np.int32(1)}, 1),
    ],
)
def test_gwf_serial_parallel_options_are_normalized(options, expected) -> None:
    kwargs = dict(options)

    assert gwf_io._consume_gwf_parallel_kwargs(kwargs, number_of_spans=4) == expected
    assert kwargs == {}


@pytest.mark.parametrize("option", ["parallel", "nproc"])
@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_gwf_true_requests_capped_automatic_workers(monkeypatch, option, value) -> None:
    monkeypatch.setattr(gwf_io.os, "cpu_count", lambda: 32)

    assert gwf_io._consume_gwf_parallel_kwargs({option: value}, number_of_spans=20) == 8


def test_gwf_true_uses_one_when_cpu_count_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr(gwf_io.os, "cpu_count", lambda: None)

    assert (
        gwf_io._consume_gwf_parallel_kwargs({"parallel": True}, number_of_spans=20) == 1
    )


@pytest.mark.parametrize("option", ["parallel", "nproc"])
def test_gwf_explicit_integer_worker_request_accepts_two_through_eight(
    option,
) -> None:
    assert (
        gwf_io._consume_gwf_parallel_kwargs({option: np.int64(2)}, number_of_spans=3)
        == 2
    )
    assert (
        gwf_io._consume_gwf_parallel_kwargs({option: np.int64(8)}, number_of_spans=3)
        == 8
    )


@pytest.mark.parametrize("option", ["parallel", "nproc"])
@pytest.mark.parametrize("value", [9, 20, np.int64(9)])
def test_gwf_explicit_integer_worker_request_above_eight_is_rejected(
    option, value
) -> None:
    with pytest.raises(ValueError, match="at most 8"):
        gwf_io._consume_gwf_parallel_kwargs({option: value}, number_of_spans=20)


@pytest.mark.parametrize(
    "options",
    [
        {"parallel": None, "nproc": None},
        {"parallel": False, "nproc": 1},
        {"parallel": True, "nproc": 2},
    ],
)
def test_gwf_parallel_aliases_always_conflict(options) -> None:
    with pytest.raises(TypeError, match="both"):
        gwf_io._consume_gwf_parallel_kwargs(dict(options), number_of_spans=2)


@pytest.mark.parametrize(
    "value",
    [0, -1, np.int64(0), np.int64(-1)],
)
def test_gwf_nonpositive_worker_count_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        gwf_io._consume_gwf_parallel_kwargs({"parallel": value}, number_of_spans=2)


@pytest.mark.parametrize("value", [1.0, 2.0, "2", object()])
def test_gwf_non_integral_worker_count_is_rejected_with_type_error(value) -> None:
    with pytest.raises(TypeError):
        gwf_io._consume_gwf_parallel_kwargs({"nproc": value}, number_of_spans=2)


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
@pytest.mark.parametrize(
    ("option", "value"),
    [
        (option, value)
        for option in ("parallel", "nproc")
        for value in (1.0, "2", object())
    ],
)
def test_public_gwf_read_rejects_invalid_parallel_values_with_type_error(
    reader, option, value
) -> None:
    with pytest.raises(TypeError):
        reader("invalid.gwf", CHANNEL, format="gwf", **{option: value})


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
@pytest.mark.parametrize("option", ["parallel", "nproc"])
@pytest.mark.parametrize("value", [9, 20])
def test_public_gwf_read_rejects_explicit_integer_workers_above_eight(
    reader, option, value
) -> None:
    with pytest.raises(ValueError, match="at most 8"):
        reader("invalid.gwf", CHANNEL, format="gwf", **{option: value})


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
def test_public_gwf_read_rejects_parallel_alias_conflict_with_type_error(
    reader,
) -> None:
    with pytest.raises(TypeError, match="both"):
        reader(
            "invalid.gwf",
            CHANNEL,
            format="gwf",
            parallel=None,
            nproc=None,
        )


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
@pytest.mark.parametrize("source_factory", [list, tuple])
@pytest.mark.parametrize(
    "options",
    [
        {"parallel": None, "nproc": None},
        {"parallel": False, "nproc": False},
        {"parallel": None, "nproc": False},
        {"parallel": False, "nproc": None},
    ],
)
def test_public_gwf_read_checks_alias_conflict_before_empty_source_guard(
    reader, source_factory, options
) -> None:
    with pytest.raises(TypeError, match="both"):
        reader(source_factory(), CHANNEL, format="gwf", **options)


@pytest.mark.parametrize(
    "options",
    [{"parallel": True}, {"parallel": 2}, {"nproc": 2}],
)
def test_gwf_scalar_parallel_request_is_rejected_before_read(
    monkeypatch, options
) -> None:
    def unexpected_read(*args, **kwargs):
        raise AssertionError("scalar parallel input must be rejected before reading")

    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", unexpected_read
    )

    with pytest.raises(TypeError, match="list or tuple"):
        gwf_io.read_gwf_timeseriesdict(
            Path("one.gwf"),
            [CHANNEL],
            **options,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


@pytest.mark.parametrize("source_factory", [list, tuple])
def test_gwf_one_element_parallel_list_validates_span_without_executor(
    monkeypatch, source_factory
) -> None:
    events = []

    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("one-element parallel input may bypass executor")

    def resolve_span(source, channels, backend):
        events.append("resolve")
        return (0.0, 1.0)

    def read_one(source, channels, **kwargs):
        events.append("read")
        return {
            CHANNEL: TimeSeries(
                [42.0],
                sample_rate=1,
                t0=0.0,
                channel=CHANNEL,
                name=CHANNEL,
            )
        }

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", resolve_span)
    monkeypatch.setattr("gwpy.timeseries.io.gwf.core.read_timeseriesdict", read_one)

    result = gwf_io.read_gwf_timeseriesdict(
        source_factory([Path("one.gwf")]),
        [CHANNEL],
        parallel=True,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )

    assert events == ["resolve", "read"]
    assert result[CHANNEL].value.tolist() == [42.0]


class _ImmediateExecutor:
    instances = []

    def __init__(self, *, max_workers, mp_context):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.futures = []
        self.submit_calls = []
        self.returned_futures = []
        self.shutdown_calls = []
        self.__class__.instances.append(self)

    def submit(self, function, *args):
        self.submit_calls.append((function, args))
        future = Future()
        future.set_result(function(*args))
        self.futures.append(future)
        self.returned_futures.append(future)
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


def _fake_span(source, *args, **kwargs):
    starts = {"bad.gwf": 0.0, "late.gwf": 3.0, "early.gwf": 1.0, "middle.gwf": 2.0}
    start = starts[Path(source).name]
    return (start, start + 1.0)


def _fake_gwf_read(source, channels, **kwargs):
    start = {"late.gwf": 3.0, "early.gwf": 1.0, "middle.gwf": 2.0}[Path(source).name]
    return {
        CHANNEL: TimeSeries(
            [start],
            sample_rate=1,
            t0=start,
            channel=CHANNEL,
            name=CHANNEL,
        )
    }


def _spawn_pid_probe_worker(
    source,
    channels,
    start,
    end,
    backend,
    dict_class,
    series_class,
    read_kwargs,
):
    """Return one source-tagged result carrying the real spawn child PID."""
    del channels, start, end, backend, dict_class, series_class, read_kwargs
    source_path = Path(source)
    ordinal = int(source_path.stem.removeprefix("pid"))
    channel = f"{CHANNEL}:{source_path.stem}"
    series = GwpyTimeSeries(
        [float(ordinal), float(ordinal) + 0.5],
        sample_rate=1,
        t0=float(ordinal),
        channel=channel,
        name=channel,
    )
    series._gwexpy_io = {
        "metadata": {
            "spawn_probe": {
                "source": source_path.name,
                "pid": os.getpid(),
            }
        }
    }
    result = GwpyTimeSeriesDict({channel: series})
    result._gwexpy_io = {
        "metadata": {
            "spawn_probe": {
                "source": source_path.name,
                "pid": os.getpid(),
            }
        }
    }
    return result


def test_gwf_parallel_uses_spawn_and_merges_by_resolved_span(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    calls = []

    def recording_read(source, channels, **kwargs):
        calls.append((source, channels, kwargs))
        return _fake_gwf_read(source, channels, **kwargs)

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "as_completed", lambda futures: reversed(futures))
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", recording_read
    )

    sources = [Path("late.gwf"), Path("early.gwf"), Path("middle.gwf")]
    result = gwf_io.read_gwf_timeseriesdict(
        sources,
        [CHANNEL],
        backend="frameCPP",
        scaled=True,
        parallel=8,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )

    executor = _ImmediateExecutor.instances[0]
    assert executor.max_workers == 8
    assert executor.mp_context.get_start_method() == "spawn"
    assert result[CHANNEL].value.tolist() == [1.0, 2.0, 3.0]
    assert len(executor.submit_calls) == len(sources)
    assert [call[1][0] for call in executor.submit_calls] == sources
    assert all(
        call[0] is gwf_io._read_gwf_timeseriesdict_worker
        for call in executor.submit_calls
    )
    assert len(executor.returned_futures) == len(sources)
    assert executor.returned_futures == executor.futures
    assert executor.shutdown_calls == [{"wait": True}]
    assert [call[1] for call in calls] == [[CHANNEL]] * 3
    assert all(call[2]["backend"] == "frameCPP" for call in calls)
    assert all(call[2]["scaled"] is True for call in calls)


def test_public_gwf_parallel_spawn_probe_reports_child_pids(
    monkeypatch, tmp_path
) -> None:
    """Public reading must execute every source in real spawn child processes."""
    sources = [tmp_path / "pid0.gwf", tmp_path / "pid1.gwf", tmp_path / "pid2.gwf"]

    def probe_span(source, *args, **kwargs):
        ordinal = int(Path(source).stem.removeprefix("pid"))
        return float(ordinal), float(ordinal) + 2.0

    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", probe_span)
    monkeypatch.setattr(
        gwf_io, "_read_gwf_timeseriesdict_worker", _spawn_pid_probe_worker
    )

    result = TimeSeriesDict.read(
        sources,
        CHANNEL,
        format="gwf",
        backend="frameCPP",
        gap="ignore",
        parallel=2,
    )

    probes = [
        series._gwexpy_io["metadata"]["spawn_probe"] for series in result.values()
    ]
    assert len(probes) == len(sources)
    assert {probe["source"] for probe in probes} == {source.name for source in sources}
    assert all(probe["pid"] != os.getpid() for probe in probes)


def test_gwf_parallel_identical_spans_keep_input_order_after_reversed_completion(
    monkeypatch,
) -> None:
    _ImmediateExecutor.instances.clear()

    def identical_span(source, *args, **kwargs):
        return (10.0, 11.0)

    def distinguishable_read(source, channels, **kwargs):
        value = {"first.gwf": 10.0, "second.gwf": 20.0}[Path(source).name]
        return {
            CHANNEL: TimeSeries(
                [value],
                sample_rate=1,
                t0=10.0,
                channel=CHANNEL,
                name=CHANNEL,
            )
        }

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "as_completed", lambda futures: reversed(futures))
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", identical_span)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", distinguishable_read
    )

    result = gwf_io.read_gwf_timeseriesdict(
        [Path("first.gwf"), Path("second.gwf")],
        [CHANNEL],
        gap="ignore",
        parallel=2,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )

    assert result[CHANNEL].value.tolist() == [10.0, 20.0]


def test_gwf_one_worker_keeps_serial_path_without_executor(monkeypatch) -> None:
    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("executor must not be constructed")

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", _fake_gwf_read
    )

    result = gwf_io.read_gwf_timeseriesdict(
        [Path("late.gwf"), Path("middle.gwf")],
        [CHANNEL],
        parallel=1,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )

    assert result[CHANNEL].value.tolist() == [2.0, 3.0]


def test_gwf_parallel_rejects_non_path_source_before_executor(monkeypatch) -> None:
    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("executor must not be constructed")

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)

    with pytest.raises(TypeError, match="filesystem paths"):
        gwf_io.read_gwf_timeseriesdict(
            ["single.gwf", object()],
            [CHANNEL],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


def test_gwf_parallel_rejects_unknown_span_before_executor(monkeypatch) -> None:
    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("executor must not be constructed")

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(
        gwf_io,
        "_resolve_gwf_path_span",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unknown span")),
    )

    with pytest.raises(ValueError, match="unknown span"):
        gwf_io.read_gwf_timeseriesdict(
            [Path("one.gwf"), Path("two.gwf")],
            [CHANNEL],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
@pytest.mark.parametrize(
    "source",
    [Path("one.gwf"), range(2), [Path("one.gwf"), object()]],
)
def test_public_gwf_parallel_source_preflight_precedes_channel_discovery(
    monkeypatch, reader, source
) -> None:
    calls = []

    def unexpected_discovery(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("channel discovery must not run during parallel preflight")

    monkeypatch.setattr("gwpy.io.gwf.core.get_channel_names", unexpected_discovery)

    with pytest.raises(TypeError, match="list or tuple|filesystem paths"):
        reader(source, format="gwf", parallel=2)

    assert calls == []


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
def test_public_gwf_empty_parallel_source_preflight_precedes_channel_discovery(
    monkeypatch, reader
) -> None:
    calls = []
    monkeypatch.setattr(
        "gwpy.io.gwf.core.get_channel_names",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="non-empty"):
        reader([], format="gwf", parallel=True)

    assert calls == []


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
def test_public_gwf_parallel_option_type_preflight_precedes_channel_discovery(
    monkeypatch, reader
) -> None:
    calls = []
    monkeypatch.setattr(
        "gwpy.io.gwf.core.get_channel_names",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(TypeError):
        reader(Path("one.gwf"), format="gwf", parallel=2.0)

    assert calls == []


def test_gwf_invalid_resolved_span_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr("gwpy.io.cache.file_segment", lambda source: (2.0, 1.0))

    with pytest.raises(ValueError, match="Invalid GWF frame span"):
        gwf_io._resolve_gwf_path_span(Path("one.gwf"), [CHANNEL], None)


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
def test_public_gwf_read_translates_span_backend_import_error(
    monkeypatch, reader
) -> None:
    monkeypatch.setattr("gwpy.io.cache.file_segment", lambda source: None)

    def missing_backend(*args, **kwargs):
        raise ImportError("frameCPP is unavailable")

    monkeypatch.setattr("gwpy.io.gwf.core.data_segments", missing_backend)

    with pytest.raises(ImportError, match="backend hint") as exc_info:
        reader(
            [Path("one.gwf")],
            CHANNEL,
            format="gwf.framecpp",
            parallel=2,
        )

    assert "package='frameCPP'" in str(exc_info.value)


class _PicklingExecutor:
    def __init__(self, *, max_workers, mp_context):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.shutdown_calls = []

    def submit(self, function, *args):
        future = Future()
        worker_result = function(*args)
        future.set_result(pickle.loads(pickle.dumps(worker_result)))
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


def _snapshot_gwf_result(result):
    entry = (
        result if isinstance(result, (TimeSeries, GwpyTimeSeries)) else result[CHANNEL]
    )
    return {
        "collection": (
            None
            if isinstance(result, (TimeSeries, GwpyTimeSeries))
            else copy.deepcopy(result._gwexpy_io)
        ),
        "entry": copy.deepcopy(entry._gwexpy_io),
        "values": entry.value.copy(),
        "xindex": entry.xindex.value.copy(),
    }


def _iter_mutable_metadata_nodes(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _iter_mutable_metadata_nodes(child, (*path, key))
    elif isinstance(value, list):
        yield path, value
        for index, child in enumerate(value):
            yield from _iter_mutable_metadata_nodes(child, (*path, index))
    elif isinstance(value, np.ndarray):
        yield path, value


def _assert_pairwise_deep_metadata_isolation(values):
    node_maps = [dict(_iter_mutable_metadata_nodes(value)) for value in values]
    for left_index, left_nodes in enumerate(node_maps):
        for right_nodes in node_maps[left_index + 1 :]:
            assert left_nodes.keys() == right_nodes.keys()
            for path in left_nodes:
                left_node = left_nodes[path]
                right_node = right_nodes[path]
                assert left_node is not right_node, path
                if isinstance(left_node, np.ndarray):
                    assert not np.shares_memory(left_node, right_node), path


def _assert_pairwise_entry_buffer_isolation(entries):
    for left_index, left in enumerate(entries):
        for right in entries[left_index + 1 :]:
            assert not np.shares_memory(left.value, right.value)
            assert not np.shares_memory(left.xindex.value, right.xindex.value)


def _assert_nested_value_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_value_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_value_equal(actual_item, expected_item)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def test_gwf_parallel_reconstructs_gwexpy_types_after_pickle_boundary(
    monkeypatch,
) -> None:
    """Serialization-aware worker results retain deeply isolated metadata."""
    source_results = []

    def fake_span(source, *args, **kwargs):
        start = 1.0 if Path(source).name == "early.gwf" else 3.0
        return start, start + 2.0

    def fake_read(source, channels, **kwargs):
        start = 1.0 if Path(source).name == "early.gwf" else 3.0
        series = GwpyTimeSeries(
            [start, start + 0.5],
            sample_rate=1,
            t0=start,
            unit="V",
            channel=CHANNEL,
            name="requested-name",
        )
        series._gwex_t0_gps_ns = int(start * 1_000_000_000)
        series._gwex_t0_gps_precision = "exact"
        series._gwexpy_io = {
            "provenance": {
                "source": Path(source).name,
                "details": {
                    "span_start": start,
                    "nested": {
                        "kind": "frame",
                        "items": [{"buffer": np.array([start, start + 0.25])}],
                    },
                },
            },
            "metadata": {
                "labels": {
                    "channel": CHANNEL,
                    "nested": {
                        "role": "entry",
                        "tags": ["entry", Path(source).name],
                    },
                }
            },
        }
        result = GwpyTimeSeriesDict({CHANNEL: series})
        result._gwexpy_io = {
            "provenance": {
                "source": Path(source).name,
                "details": {
                    "span_start": start,
                    "nested": {
                        "kind": "collection",
                        "items": [{"buffer": np.array([start, start + 0.5])}],
                    },
                },
            },
            "metadata": {
                "labels": {
                    "channel": CHANNEL,
                    "nested": {
                        "role": "collection",
                        "tags": ["collection", Path(source).name],
                    },
                }
            },
        }
        source_results.append(result)
        return result

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _PicklingExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", fake_span)
    monkeypatch.setattr("gwpy.timeseries.io.gwf.core.read_timeseriesdict", fake_read)

    sources = [Path("late.gwf"), Path("early.gwf")]
    serial = gwf_io.read_gwf_timeseriesdict(
        sources,
        [CHANNEL],
        gap="ignore",
        parallel=1,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )
    parallel = gwf_io.read_gwf_timeseriesdict(
        sources,
        [CHANNEL],
        gap="ignore",
        parallel=2,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )
    delegated = TimeSeries.read(
        sources,
        CHANNEL,
        format="gwf",
        gap="ignore",
        parallel=2,
    )

    for result in (serial, parallel):
        series = result[CHANNEL]
        assert type(result) is TimeSeriesDict
        assert type(series) is TimeSeries
        assert series.value.tolist() == [1.0, 1.5, 3.0, 3.5]
        assert series.unit == GwpyTimeSeries([1], unit="V").unit
        assert series.channel.name == CHANNEL
        assert series.name == "requested-name"
        assert float(series.t0.value) == pytest.approx(1.0)
        assert float(series.dt.value) == pytest.approx(1.0)
        assert series._gwex_t0_gps_ns == 1_000_000_000
        assert series._gwex_t0_gps_precision == "exact"
        assert series._gwexpy_io["metadata"]["labels"]["channel"] == CHANNEL
        assert result._gwexpy_io["provenance"]["source"] in {
            "early.gwf",
            "late.gwf",
        }
    assert type(delegated) is TimeSeries
    assert delegated.value.tolist() == [1.0, 1.5, 3.0, 3.5]

    results = [serial, parallel, *source_results]
    entries = [result[CHANNEL] for result in results] + [delegated]
    _assert_pairwise_deep_metadata_isolation([result._gwexpy_io for result in results])
    _assert_pairwise_deep_metadata_isolation([entry._gwexpy_io for entry in entries])
    _assert_pairwise_entry_buffer_isolation(entries)
    for result in results:
        assert result._gwexpy_io is not result[CHANNEL]._gwexpy_io

    snapshots = [_snapshot_gwf_result(result) for result in results]
    entry_snapshots = [_snapshot_gwf_result(entry) for entry in entries]
    serial_snapshot = snapshots[0]
    parallel_snapshot = snapshots[1]
    source_snapshots = snapshots[2:]

    parallel_entry = parallel[CHANNEL]
    parallel._gwexpy_io["provenance"]["details"]["nested"]["mutated"] = True
    parallel._gwexpy_io["provenance"]["details"]["nested"]["items"][0]["buffer"][0] += (
        100.0
    )
    parallel._gwexpy_io["metadata"]["labels"]["nested"]["mutated"] = True
    parallel._gwexpy_io["metadata"]["labels"]["nested"]["tags"].append("mutated")
    parallel_entry._gwexpy_io["provenance"]["details"]["nested"]["mutated"] = True
    parallel_entry._gwexpy_io["provenance"]["details"]["nested"]["items"][0]["buffer"][
        0
    ] += 100.0
    parallel_entry._gwexpy_io["metadata"]["labels"]["nested"]["mutated"] = True
    parallel_entry._gwexpy_io["metadata"]["labels"]["nested"]["tags"].append("mutated")
    parallel_entry.value[0] += 100.0
    # `.times` is a public view; mutate the authoritative writable cached xindex.
    parallel_entry.xindex.value[0] += 100.0

    assert parallel._gwexpy_io["provenance"]["details"]["nested"]["mutated"]
    assert parallel._gwexpy_io["metadata"]["labels"]["nested"]["mutated"]
    assert parallel_entry._gwexpy_io["provenance"]["details"]["nested"]["mutated"]
    assert parallel_entry._gwexpy_io["metadata"]["labels"]["nested"]["mutated"]
    assert parallel_entry.value[0] == pytest.approx(
        parallel_snapshot["values"][0] + 100.0
    )
    assert parallel_entry.xindex.value[0] == pytest.approx(
        parallel_snapshot["xindex"][0] + 100.0
    )

    serial_entry = serial[CHANNEL]
    _assert_nested_value_equal(serial._gwexpy_io, serial_snapshot["collection"])
    _assert_nested_value_equal(serial_entry._gwexpy_io, serial_snapshot["entry"])
    np.testing.assert_array_equal(serial_entry.value, serial_snapshot["values"])
    np.testing.assert_array_equal(serial_entry.xindex.value, serial_snapshot["xindex"])
    for source, snapshot in zip(source_results, source_snapshots, strict=True):
        _assert_nested_value_equal(source._gwexpy_io, snapshot["collection"])
        _assert_nested_value_equal(source[CHANNEL]._gwexpy_io, snapshot["entry"])
        np.testing.assert_array_equal(source[CHANNEL].value, snapshot["values"])
        np.testing.assert_array_equal(source[CHANNEL].xindex.value, snapshot["xindex"])

    delegated_snapshot = entry_snapshots[-1]
    _assert_nested_value_equal(delegated._gwexpy_io, delegated_snapshot["entry"])
    np.testing.assert_array_equal(delegated.value, delegated_snapshot["values"])
    np.testing.assert_array_equal(delegated.xindex.value, delegated_snapshot["xindex"])

    assert type(serial[CHANNEL]) is type(parallel[CHANNEL])


def _write_lalframe_gwf_parts(tmp_path: Path) -> tuple[list[Path], np.ndarray]:
    """Write two contiguous GWF files through the lalframe writer."""
    sample_rate = 4.0
    samples_per_file = 4
    expected_parts = []
    sources = []
    for index, start in enumerate((1000.0, 1001.0)):
        values = np.arange(
            index * samples_per_file,
            (index + 1) * samples_per_file,
            dtype=float,
        )
        TimeSeriesDict(
            {
                CHANNEL: TimeSeries(
                    values,
                    sample_rate=sample_rate,
                    t0=start,
                    channel=CHANNEL,
                    name=CHANNEL,
                )
            }
        ).write(
            tmp_path / f"part{index}.gwf",
            format="gwf",
            backend="lalframe",
        )
        sources.append(tmp_path / f"part{index}.gwf")
        expected_parts.append(values)
    return sources, np.concatenate(expected_parts)


@pytest.mark.skipif(
    importlib.util.find_spec("lalframe") is None,
    reason="lalframe module is not installed",
)
def test_public_lalframe_gwf_parallel_read_uses_real_spawn_and_reconstructs_state(
    tmp_path: Path,
) -> None:
    """Exercise real lalframe payloads, not arbitrary provenance persistence.

    Nested metadata/provenance isolation belongs to the serialization-aware test
    above because real lalframe GWF carries public payload state, not arbitrary
    Python provenance mappings.
    """
    # The structural spawn assertion in
    # test_gwf_parallel_uses_spawn_and_merges_by_resolved_span complements this
    # real child-process check and guards against a synchronous regression.
    sources, expected = _write_lalframe_gwf_parts(tmp_path)

    result = TimeSeriesDict.read(
        sources[::-1],
        CHANNEL,
        format="gwf",
        backend="lalframe",
        parallel=2,
    )
    delegated = TimeSeries.read(
        sources[::-1],
        CHANNEL,
        format="gwf",
        backend="lalframe",
        parallel=2,
    )

    assert type(result) is TimeSeriesDict
    assert type(result[CHANNEL]) is TimeSeries
    assert type(delegated) is TimeSeries

    expected_times = np.array(
        [
            1000.0,
            1000.25,
            1000.5,
            1000.75,
            1001.0,
            1001.25,
            1001.5,
            1001.75,
        ]
    )
    for series in (result[CHANNEL], delegated):
        np.testing.assert_allclose(series.value, expected)
        np.testing.assert_allclose(series.times.to_value(u.s), expected_times)
        assert series.times.unit == u.s
        assert series.dt == 0.25 * u.s
        assert series.t0 == 1000.0 * u.s
        assert series._gwex_t0_gps_precision == "exact"
        assert series.t0_gps_ns == 1_000_000_000_000
    assert delegated.name == CHANNEL

    serial = TimeSeriesDict.read(
        sources,
        CHANNEL,
        format="gwf",
        backend="lalframe",
        parallel=1,
    )
    for series in (serial[CHANNEL],):
        np.testing.assert_allclose(series.value, expected)
        np.testing.assert_allclose(series.times.to_value(u.s), expected_times)
        assert series.times.unit == u.s
        assert series.dt == 0.25 * u.s
        assert series.t0 == 1000.0 * u.s
        assert series._gwex_t0_gps_precision == "exact"
        assert series.t0_gps_ns == 1_000_000_000_000

    # `.times` is the public view of the cached `xindex`; use that authoritative
    # writable index buffer for the no-aliasing and mutation contract.
    dict_axis = result[CHANNEL].xindex.value
    delegated_axis = delegated.xindex.value
    serial_axis = serial[CHANNEL].xindex.value
    dict_values = result[CHANNEL].value
    delegated_values = delegated.value
    serial_values = serial[CHANNEL].value
    assert not np.shares_memory(dict_values, delegated_values)
    assert not np.shares_memory(dict_values, serial_values)
    assert not np.shares_memory(delegated_values, serial_values)
    assert not np.shares_memory(dict_axis, delegated_axis)
    assert not np.shares_memory(dict_axis, serial_axis)
    assert not np.shares_memory(delegated_axis, serial_axis)
    delegated_value_snapshot = delegated_values.copy()
    serial_value_snapshot = serial_values.copy()
    delegated_snapshot = delegated_axis.copy()
    serial_snapshot = serial_axis.copy()
    dict_values[0] += 10.0
    dict_axis[0] += 10.0
    assert dict_values[0] == pytest.approx(10.0)
    np.testing.assert_array_equal(delegated_values, delegated_value_snapshot)
    np.testing.assert_array_equal(serial_values, serial_value_snapshot)
    assert dict_axis[0] == pytest.approx(1010.0)
    np.testing.assert_array_equal(delegated_axis, delegated_snapshot)
    np.testing.assert_array_equal(serial_axis, serial_snapshot)


class _FailingFuture:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.cancelled = False

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        self.cancelled = True
        return True


def test_gwf_worker_failure_cancels_pending_and_returns_no_partial_result(monkeypatch):
    error = RuntimeError("worker failed")
    futures = []

    class FailingExecutor:
        instance = None

        def __init__(self, **kwargs):
            self.shutdown_calls = []
            self.__class__.instance = self

        def submit(self, function, source, *args):
            if Path(source).name == "bad.gwf":
                future = _FailingFuture(error=error)
            else:
                future = _FailingFuture(result={})
            futures.append(future)
            return future

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(gwf_io, "as_completed", lambda items: iter(items[:1]))

    with pytest.raises(RuntimeError, match="worker failed"):
        gwf_io.read_gwf_timeseriesdict(
            [Path("bad.gwf"), Path("early.gwf")],
            [CHANNEL],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )

    assert futures[1].cancelled
    assert FailingExecutor.instance.shutdown_calls == [
        {"wait": True, "cancel_futures": True}
    ]
