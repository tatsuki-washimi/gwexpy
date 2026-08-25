"""Contracts for parallel GWF filesystem reads (#588)."""

from __future__ import annotations

import os
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

import gwexpy.timeseries._gwf_io as gwf_io
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

CHANNEL_A = "K1:TEST-A"
CHANNEL_B = "K1:TEST-B"


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"parallel": None}, 1),
        ({"parallel": False}, 1),
        ({"parallel": np.bool_(False)}, 1),
        ({"parallel": 1}, 1),
        ({"nproc": None}, 1),
        ({"nproc": 1}, 1),
    ],
)
def test_gwf_serial_parallel_options_are_normalized(options, expected) -> None:
    kwargs = dict(options)
    assert gwf_io._consume_gwf_parallel_kwargs(kwargs, number_of_spans=4) == expected
    assert kwargs == {}


@pytest.mark.parametrize("option", ["parallel", "nproc"])
def test_gwf_true_uses_capped_automatic_workers(monkeypatch, option) -> None:
    monkeypatch.setattr(gwf_io.os, "cpu_count", lambda: 32)
    assert gwf_io._consume_gwf_parallel_kwargs({option: True}, number_of_spans=20) == 8


def test_gwf_true_uses_one_when_cpu_count_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr(gwf_io.os, "cpu_count", lambda: None)
    assert (
        gwf_io._consume_gwf_parallel_kwargs({"parallel": True}, number_of_spans=20) == 1
    )


@pytest.mark.parametrize("option", ["parallel", "nproc"])
@pytest.mark.parametrize("value", [0, -1, 9, np.int64(20)])
def test_gwf_invalid_integral_worker_count_is_rejected(option, value) -> None:
    with pytest.raises(ValueError):
        gwf_io._consume_gwf_parallel_kwargs({option: value}, number_of_spans=20)


@pytest.mark.parametrize("value", [1.0, "2", object()])
def test_gwf_non_integral_worker_count_is_rejected_with_type_error(value) -> None:
    with pytest.raises(TypeError):
        gwf_io._consume_gwf_parallel_kwargs({"nproc": value}, number_of_spans=2)


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
def test_public_alias_conflict_raises_type_error_before_io_or_channel_discovery(
    monkeypatch, reader
) -> None:
    calls = []
    monkeypatch.setattr(
        "gwpy.io.gwf.core.get_channel_names",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("read called")),
    )
    with pytest.raises(TypeError, match="either 'parallel' or 'nproc'"):
        reader("invalid.gwf", format="gwf", parallel=None, nproc=None)
    assert calls == []


@pytest.mark.parametrize("reader", [TimeSeriesDict.read, TimeSeries.read])
def test_public_empty_input_is_rejected_after_parallel_argument_validation(
    reader,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        reader([], CHANNEL_A, format="gwf", parallel=True)


class _ImmediateExecutor:
    instances = []

    def __init__(self, *, max_workers, mp_context):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.submit_calls = []
        self.shutdown_calls = []
        self.__class__.instances.append(self)

    def submit(self, function, *args):
        self.submit_calls.append((function, args))
        future = Future()
        future.set_result(function(*args))
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


def _source_start(source: str | Path) -> float:
    return {"early.gwf": 1.0, "middle.gwf": 2.0, "late.gwf": 3.0}[Path(source).name]


def _fake_span(source, *args, **kwargs):
    start = _source_start(source)
    return start, start + 1.0


def _fake_gwf_read(source, channels, **kwargs):
    start = _source_start(source)
    entries = {}
    for channel in (CHANNEL_A, CHANNEL_B):
        series = GwpyTimeSeries(
            [start if channel == CHANNEL_A else start + 100.0],
            sample_rate=1,
            t0=start,
            unit="V",
            channel=channel,
            name=f"{channel}-name",
        )
        series._gwexpy_io = {"source": Path(source).name, "channel": channel}
        entries[channel] = series
    result = GwpyTimeSeriesDict(entries)
    result._gwexpy_io = {"source": Path(source).name, "kind": "collection"}
    return result


def test_gwf_parallel_uses_spawn_and_deterministic_source_channel_time_merge(
    monkeypatch,
) -> None:
    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "as_completed", lambda futures: reversed(futures))
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", _fake_gwf_read
    )
    sources = [Path("late.gwf"), Path("early.gwf"), Path("middle.gwf")]
    result = gwf_io.read_gwf_timeseriesdict(
        sources,
        [CHANNEL_B, CHANNEL_A],
        backend="frameCPP",
        parallel=8,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )
    serial_result = gwf_io.read_gwf_timeseriesdict(
        sources,
        [CHANNEL_B, CHANNEL_A],
        backend="frameCPP",
        parallel=1,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )
    nproc_result = gwf_io.read_gwf_timeseriesdict(
        sources,
        [CHANNEL_B, CHANNEL_A],
        backend="frameCPP",
        nproc=2,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )
    executor = _ImmediateExecutor.instances[0]
    assert executor.max_workers == 8
    assert executor.mp_context.get_start_method() == "spawn"
    assert all(
        call[0] is gwf_io._read_gwf_timeseriesdict_worker
        for call in executor.submit_calls
    )
    assert executor.shutdown_calls == [{"wait": True}]
    for merged in (result, serial_result, nproc_result):
        assert type(merged) is TimeSeriesDict
        assert list(merged) == [CHANNEL_B, CHANNEL_A]
        assert merged[CHANNEL_A].value.tolist() == [1.0, 2.0, 3.0]
        assert merged[CHANNEL_B].value.tolist() == [101.0, 102.0, 103.0]
        for channel in (CHANNEL_A, CHANNEL_B):
            series = merged[channel]
            assert type(series) is TimeSeries
            assert str(series.unit) == "V"
            assert series.name == f"{channel}-name"
            assert series.channel.name == channel
            assert float(series.t0.value) == pytest.approx(1.0)
            assert float(series.dt.value) == pytest.approx(1.0)
            assert series._gwexpy_io == {"source": "early.gwf", "channel": channel}
        assert merged._gwexpy_io == {"source": "early.gwf", "kind": "collection"}
    result[CHANNEL_A]._gwexpy_io["mutated"] = True
    assert result[CHANNEL_A]._gwexpy_io is not nproc_result[CHANNEL_A]._gwexpy_io
    assert "mutated" not in nproc_result[CHANNEL_A]._gwexpy_io


def test_gwf_parallel_uses_input_order_for_identical_spans(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "as_completed", lambda futures: reversed(futures))
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", lambda *args: (10.0, 11.0))

    def read_identical(source, channels, **kwargs):
        value = {"early.gwf": 10.0, "late.gwf": 20.0}[Path(source).name]
        return {
            CHANNEL_A: GwpyTimeSeries([value], sample_rate=1, t0=10, channel=CHANNEL_A)
        }

    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", read_identical
    )
    result = gwf_io.read_gwf_timeseriesdict(
        [Path("early.gwf"), Path("late.gwf")],
        [CHANNEL_A],
        gap="ignore",
        parallel=2,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )
    assert result[CHANNEL_A].value.tolist() == [10.0, 20.0]


def test_gwf_parallel_overlap_conflict_returns_no_result(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", lambda *args: (10.0, 11.0))

    def overlapping_read(source, channels, **kwargs):
        return {
            CHANNEL_A: GwpyTimeSeries(
                [float(Path(source).name == "late.gwf")],
                sample_rate=1,
                t0=10,
                channel=CHANNEL_A,
            )
        }

    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", overlapping_read
    )

    with pytest.raises(ValueError):
        gwf_io.read_gwf_timeseriesdict(
            [Path("early.gwf"), Path("late.gwf")],
            [CHANNEL_A],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


def test_gwf_one_worker_keeps_existing_serial_path(monkeypatch) -> None:
    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("serial path created an executor")

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", _fake_gwf_read
    )
    result = gwf_io.read_gwf_timeseriesdict(
        [Path("late.gwf"), Path("middle.gwf")],
        [CHANNEL_A],
        parallel=1,
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )
    assert result[CHANNEL_A].value.tolist() == [2.0, 3.0]


def test_gwf_parallel_rejects_scalar_source_before_backend_read(monkeypatch) -> None:
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backend read")),
    )

    with pytest.raises(TypeError, match="list or tuple"):
        gwf_io.read_gwf_timeseriesdict(
            Path("early.gwf"),
            [CHANNEL_A],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


def test_gwf_parallel_preflight_rejects_unknown_span_before_executor(
    monkeypatch,
) -> None:
    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("executor was created before preflight")

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(
        gwf_io,
        "_resolve_gwf_path_span",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unknown span")),
    )
    with pytest.raises(ValueError, match="unknown span"):
        gwf_io.read_gwf_timeseriesdict(
            [Path("early.gwf"), Path("late.gwf")],
            [CHANNEL_A],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


def test_gwf_parallel_rejects_partial_empty_worker_results(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)

    def partial_read(source, channels, **kwargs):
        if Path(source).name == "late.gwf":
            return {}
        return _fake_gwf_read(source, channels, **kwargs)

    monkeypatch.setattr("gwpy.timeseries.io.gwf.core.read_timeseriesdict", partial_read)
    with pytest.raises(ValueError, match="partial|empty"):
        gwf_io.read_gwf_timeseriesdict(
            [Path("early.gwf"), Path("late.gwf")],
            [CHANNEL_A],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


def test_gwf_parallel_rejects_empty_requested_series(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)

    def empty_series_read(source, channels, **kwargs):
        if Path(source).name == "late.gwf":
            return {
                CHANNEL_A: GwpyTimeSeries([], sample_rate=1, t0=3, channel=CHANNEL_A)
            }
        return _fake_gwf_read(source, channels, **kwargs)

    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", empty_series_read
    )

    with pytest.raises(ValueError, match="partial|empty"):
        gwf_io.read_gwf_timeseriesdict(
            [Path("early.gwf"), Path("late.gwf")],
            [CHANNEL_A],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


def test_public_parallel_worker_type_error_is_not_translated(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)

    def worker_type_error(*args):
        raise TypeError("backend worker type error")

    monkeypatch.setattr(gwf_io, "_read_gwf_timeseriesdict_worker", worker_type_error)

    with pytest.raises(TypeError, match="backend worker type error"):
        TimeSeriesDict.read(
            [Path("early.gwf"), Path("late.gwf")],
            CHANNEL_A,
            format="gwf",
            parallel=2,
        )


def test_gwf_serial_read_preserves_custom_metadata_without_aliasing(
    monkeypatch,
) -> None:
    source_results = []

    def metadata_read(source, channels, **kwargs):
        result = _fake_gwf_read(source, channels, **kwargs)
        result[CHANNEL_A].custom_metadata = {"source": Path(source).name, "tags": []}
        source_results.append(result)
        return result

    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", metadata_read
    )
    result = gwf_io.read_gwf_timeseriesdict(
        [Path("early.gwf"), Path("late.gwf")],
        [CHANNEL_A],
        parallel=1,
        gap="ignore",
        dict_class=TimeSeriesDict,
        series_class=TimeSeries,
    )

    assert result[CHANNEL_A].custom_metadata == {"source": "early.gwf", "tags": []}
    result[CHANNEL_A].custom_metadata["tags"].append("mutated")
    assert source_results[0][CHANNEL_A].custom_metadata == {
        "source": "early.gwf",
        "tags": [],
    }


def test_gwf_parallel_normalizes_unusual_pickle_failures(monkeypatch) -> None:
    class BrokenPickle:
        def __reduce__(self):
            raise AttributeError("unpickleable state")

    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("executor was created before pickle validation")

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)

    with pytest.raises(TypeError, match="picklable"):
        gwf_io.read_gwf_timeseriesdict(
            [Path("early.gwf"), Path("late.gwf")],
            [CHANNEL_A],
            parallel=2,
            broken=BrokenPickle(),
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )


class _FailingFuture:
    def __init__(self, *, error=None):
        self.error = error
        self.cancelled = False

    def result(self):
        if self.error is not None:
            raise self.error
        return {}

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
            future = _FailingFuture(
                error=error if Path(source).name == "late.gwf" else None
            )
            futures.append(future)
            return future

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setattr(gwf_io, "as_completed", lambda items: iter(items[:1]))
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    with pytest.raises(RuntimeError, match="worker failed"):
        gwf_io.read_gwf_timeseriesdict(
            [Path("late.gwf"), Path("early.gwf")],
            [CHANNEL_A],
            parallel=2,
            dict_class=TimeSeriesDict,
            series_class=TimeSeries,
        )
    assert futures[1].cancelled
    assert FailingExecutor.instance.shutdown_calls == [
        {"wait": True, "cancel_futures": True}
    ]


def _spawn_probe_worker(source, channels, start, end, backend, read_kwargs):
    """Importable top-level worker used to demonstrate real spawn execution."""
    del channels, start, end, backend, read_kwargs
    source_path = Path(source)
    ordinal = int(source_path.stem.removeprefix("pid"))
    channel = CHANNEL_A
    series = GwpyTimeSeries(
        [float(ordinal)], sample_rate=1, t0=float(ordinal), unit="V", channel=channel
    )
    series._gwexpy_io = {"pid": os.getpid(), "source": source_path.name}
    return GwpyTimeSeriesDict({channel: series})


def test_public_gwf_parallel_runs_module_level_worker_in_spawn_child(
    monkeypatch, tmp_path
):
    sources = [tmp_path / "pid0.gwf", tmp_path / "pid1.gwf"]
    monkeypatch.setattr(
        gwf_io,
        "_resolve_gwf_path_span",
        lambda source, *args: (
            float(int(Path(source).stem.removeprefix("pid"))),
            float(int(Path(source).stem.removeprefix("pid"))) + 1.0,
        ),
    )
    monkeypatch.setattr(gwf_io, "_read_gwf_timeseriesdict_worker", _spawn_probe_worker)
    result = TimeSeriesDict.read(
        sources, CHANNEL_A, format="gwf", gap="ignore", parallel=2
    )
    assert list(result) == [CHANNEL_A]
    assert result[CHANNEL_A].value.tolist() == [0.0, 1.0]
    assert result[CHANNEL_A]._gwexpy_io["pid"] != os.getpid()
