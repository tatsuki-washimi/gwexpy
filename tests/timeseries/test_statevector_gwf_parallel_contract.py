"""Public StateVector GWF parallel-read contract regressions (#588)."""

from __future__ import annotations

import os
import warnings
from concurrent.futures import Future
from contextlib import nullcontext
from inspect import signature
from io import BytesIO
from pathlib import Path, PureWindowsPath

import pytest
from gwpy.timeseries import StateVector, StateVectorDict
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

import gwexpy.timeseries._gwf_io as gwf_io
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

CHANNEL_A = "K1:STATE-A"
CHANNEL_B = "K1:STATE-B"
FRAMEL_FIXTURE = Path(__file__).parent.parent / "fixtures" / "data" / "test.gwf"
FRAMEL_CHANNEL = "K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"


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


class _StringPath(os.PathLike[str]):
    """PathLike test double for structural parallel-source validation."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __fspath__(self) -> str:
        return self.value


def _source_start(source: str | Path) -> float:
    return {"early.gwf": 1.0, "late.gwf": 2.0}[Path(source).name]


def _fake_span(source, *args, **kwargs):
    start = _source_start(source)
    return start, start + 1.0


def _fake_statevector_read(source, channels, **kwargs):
    start = _source_start(source)
    bits = kwargs.get("bits", ["ready", "active"])
    result = StateVectorDict()
    for channel in channels:
        channel_bits = bits.get(channel) if isinstance(bits, dict) else bits
        series = StateVector(
            [int(start), int(start) + 1],
            bits=channel_bits,
            sample_rate=1,
            t0=start,
            unit="count",
            channel=channel,
            name=f"{channel}-name",
        )
        series.custom_metadata = {"source": Path(source).name}
        result[channel] = series
    result.custom_metadata = {"source": Path(source).name, "kind": "state"}
    return result


def _patch_connector_open(monkeypatch) -> None:
    """Keep legacy connector reads mocked without opening frame files."""
    monkeypatch.setattr(
        "gwpy.io.registry.open_remote_file",
        lambda source, **kwargs: nullcontext(source),
    )


@pytest.mark.parametrize(
    ("reader", "selector"),
    [
        (StateVector.read, CHANNEL_A),
        (StateVectorDict.read, [CHANNEL_A]),
    ],
)
def test_statevector_alias_conflict_preflights_before_connector_or_backend(
    monkeypatch, reader, selector
) -> None:
    calls = []

    def unexpected_registry_read(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("connector/backend I/O ran")

    monkeypatch.setattr(StateVector.read.registry, "read", unexpected_registry_read)

    with pytest.raises(TypeError, match="either 'parallel' or 'nproc'"):
        reader("invalid.gwf", selector, format="gwf", parallel=None, nproc=None)
    assert calls == []


def test_statevectordict_parallel_nproc_and_serial_reads_preserve_order_and_state(
    monkeypatch,
) -> None:
    _ImmediateExecutor.instances.clear()
    connector_calls = []
    _patch_connector_open(monkeypatch)

    def connector_read(cls, source, *args, **kwargs):
        connector_calls.append((cls, source, args, kwargs))
        result = _fake_statevector_read(source, list(args[0]), **kwargs)
        return result if cls is StateVectorDict else result[args[0]]

    monkeypatch.setattr(StateVector.read.registry, "read", connector_read)
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "as_completed", lambda futures: reversed(futures))
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_statevectordict", _fake_statevector_read
    )
    sources = [Path("late.gwf"), Path("early.gwf")]

    parallel = StateVectorDict.read(
        sources,
        [CHANNEL_B, CHANNEL_A],
        format="gwf",
        gap="ignore",
        parallel=2,
        bits=["ready", "active"],
    )
    nproc = StateVectorDict.read(
        sources,
        [CHANNEL_B, CHANNEL_A],
        format="gwf",
        gap="ignore",
        nproc=2,
        bits=["ready", "active"],
    )
    serial = StateVectorDict.read(
        sources,
        [CHANNEL_B, CHANNEL_A],
        format="gwf",
        gap="ignore",
        parallel=1,
        bits=["ready", "active"],
    )

    assert _ImmediateExecutor.instances[0].mp_context.get_start_method() == "spawn"
    assert all(
        call[0] is gwf_io._read_gwf_statevectordict_worker
        for call in _ImmediateExecutor.instances[0].submit_calls
    )
    assert _ImmediateExecutor.instances[0].shutdown_calls == [{"wait": True}]
    assert connector_calls  # effective one worker remains GWpy's serial path
    for result in (parallel, nproc, serial):
        assert type(result) is StateVectorDict
        assert list(result) == [CHANNEL_B, CHANNEL_A]
        for channel in (CHANNEL_A, CHANNEL_B):
            series = result[channel]
            assert type(series) is StateVector
            assert series.value.tolist() == [1, 2, 2, 3]
            assert str(series.unit) == "ct"
            assert series.name == f"{channel}-name"
            assert series.channel.name == channel
            assert float(series.t0.value) == pytest.approx(1.0)
            assert list(series.bits) == ["ready", "active"]
    for result in (parallel, nproc):
        for channel in (CHANNEL_A, CHANNEL_B):
            assert result[channel].custom_metadata == {"source": "early.gwf"}
        assert result.custom_metadata == {"source": "early.gwf", "kind": "state"}


def test_statevector_parallel_uses_same_statevector_dict_worker(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    _patch_connector_open(monkeypatch)

    def connector_read(cls, source, *args, **kwargs):
        result = _fake_statevector_read(source, [args[0]], **kwargs)
        return result[args[0]]

    monkeypatch.setattr(StateVector.read.registry, "read", connector_read)
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_statevectordict", _fake_statevector_read
    )

    result = StateVector.read(
        [Path("late.gwf"), Path("early.gwf")],
        CHANNEL_A,
        format="gwf",
        gap="ignore",
        parallel=2,
        bits=["ready", "active"],
    )

    assert type(result) is StateVector
    assert result.value.tolist() == [1, 2, 2, 3]
    assert list(result.bits) == ["ready", "active"]
    assert _ImmediateExecutor.instances[0].submit_calls


@pytest.mark.parametrize(
    ("reader", "selector"),
    [
        (StateVector.read, CHANNEL_A),
        (StateVectorDict.read, [CHANNEL_A]),
    ],
)
def test_statevector_empty_input_and_invalid_worker_count_fail_before_io(
    monkeypatch, reader, selector
) -> None:
    _patch_connector_open(monkeypatch)

    def unexpected_registry_read(*args, **kwargs):
        raise AssertionError("connector/backend I/O ran")

    monkeypatch.setattr(StateVector.read.registry, "read", unexpected_registry_read)
    with pytest.raises(ValueError, match="non-empty"):
        reader([], selector, format="gwf", parallel=True)
    with pytest.raises(ValueError, match="positive"):
        reader("invalid.gwf", selector, format="gwf", parallel=0)


def test_statevector_single_input_parallel_true_keeps_serial_connector_path(
    monkeypatch,
) -> None:
    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("one source created a process executor")

    calls = []
    _patch_connector_open(monkeypatch)

    def connector_read(cls, source, *args, **kwargs):
        calls.append((cls, source, args, kwargs))
        result = _fake_statevector_read(source, [args[0]], **kwargs)
        return result[args[0]]

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(StateVector.read.registry, "read", connector_read)
    result = StateVector.read(
        Path("early.gwf"),
        CHANNEL_A,
        format="gwf",
        parallel=True,
        bits=["ready", "active"],
    )
    assert result.value.tolist() == [1, 2]
    assert calls


def test_statevectordict_worker_failure_cancels_without_partial_result(
    monkeypatch,
) -> None:
    futures = []

    class FailingExecutor:
        instance = None

        def __init__(self, **kwargs):
            self.shutdown_calls = []
            self.__class__.instance = self

        def submit(self, function, source, *args):
            future = Future()
            if Path(source).name == "late.gwf":
                future.set_exception(RuntimeError("state worker failed"))
            else:
                future.set_result(_fake_statevector_read(source, list(args[0])))
            futures.append(future)
            return future

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(gwf_io, "as_completed", lambda values: iter(values))
    _patch_connector_open(monkeypatch)
    monkeypatch.setattr(
        StateVector.read.registry,
        "read",
        lambda cls, source, *args, **kwargs: _fake_statevector_read(
            source, list(args[0]), **kwargs
        ),
    )

    with pytest.raises(RuntimeError, match="state worker failed"):
        StateVectorDict.read(
            [Path("late.gwf"), Path("early.gwf")],
            [CHANNEL_A],
            format="gwf",
            parallel=2,
        )
    assert FailingExecutor.instance.shutdown_calls == [
        {"wait": True, "cancel_futures": True}
    ]
    assert all(future.cancelled() or future.done() for future in futures)


def _statevector_spawn_probe_worker(source, channels, start, end, backend, read_kwargs):
    """Importable worker proving StateVector dispatch uses a spawn child."""
    del start, end, backend, read_kwargs
    ordinal = int(Path(source).stem.removeprefix("pid"))
    result = StateVectorDict()
    for channel in channels:
        series = StateVector(
            [ordinal],
            bits=["ready"],
            sample_rate=1,
            t0=ordinal,
            channel=channel,
        )
        series.worker_pid = os.getpid()
        result[channel] = series
    return result


def test_statevectordict_parallel_runs_state_worker_in_spawn_child(
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
    monkeypatch.setattr(
        gwf_io, "_read_gwf_statevectordict_worker", _statevector_spawn_probe_worker
    )
    monkeypatch.setattr(
        StateVector.read.registry,
        "read",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("connector ran")),
    )
    _patch_connector_open(monkeypatch)

    result = StateVectorDict.read(
        sources, [CHANNEL_A], format="gwf", gap="ignore", parallel=2
    )
    assert result[CHANNEL_A].value.tolist() == [0, 1]
    assert list(result[CHANNEL_A].bits) == ["ready"]
    assert result[CHANNEL_A].worker_pid != os.getpid()


def test_statevector_hook_preserves_descriptor_and_delegates_non_gwf(monkeypatch):
    _patch_connector_open(monkeypatch)
    calls = []

    def connector_read(cls, source, *args, **kwargs):
        calls.append((cls, source, args, kwargs))
        return StateVector([1], bits=["ready"], sample_rate=1, channel=args[0])

    monkeypatch.setattr(StateVector.read.registry, "read", connector_read)
    result = StateVector.read(
        Path("state.custom"), CHANNEL_A, format="custom", parallel=1
    )
    assert type(StateVector.read).__name__ == "StateVectorRead"
    assert type(StateVectorDict.read).__name__ == "StateVectorDictRead"
    assert StateVector.read.registry is StateVectorDict.read.registry
    assert result.value.tolist() == [1]
    assert calls


def test_all_public_gwf_readers_advertise_parallel_and_nproc() -> None:
    for reader in (
        TimeSeries.read,
        TimeSeriesDict.read,
        StateVector.read,
        StateVectorDict.read,
    ):
        parameters = signature(reader).parameters
        assert {"parallel", "nproc"} <= set(parameters)
        assert "nproc" in (reader.__doc__ or "")


@pytest.mark.parametrize("format_name", ["frame", None])
def test_statevectordict_parallel_recognizes_gwf_aliases_and_extension(
    monkeypatch, format_name
) -> None:
    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_statevectordict", _fake_statevector_read
    )
    kwargs = {"parallel": 2, "gap": "ignore"}
    if format_name is not None:
        kwargs["format"] = format_name
    result = StateVectorDict.read(
        [Path("late.gwf"), Path("early.gwf")], [CHANNEL_A], **kwargs
    )
    assert result[CHANNEL_A].value.tolist() == [1, 2, 2, 3]
    assert _ImmediateExecutor.instances


def test_statevector_nproc_is_warning_free_and_pickle_preflight_is_atomic(monkeypatch):
    class BrokenPickle:
        def __reduce__(self):
            raise AttributeError("broken state")

    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("executor created before pickle preflight")

    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_statevectordict", _fake_statevector_read
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = StateVector.read(
            [Path("late.gwf"), Path("early.gwf")],
            CHANNEL_A,
            format="gwf",
            gap="ignore",
            nproc=2,
        )
    assert result.value.tolist() == [1, 2, 2, 3]
    assert not [item for item in caught if item.category is DeprecationWarning]

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    with pytest.raises(TypeError, match="picklable"):
        StateVectorDict.read(
            [Path("early.gwf"), Path("late.gwf")],
            [CHANNEL_A],
            format="gwf",
            parallel=2,
            bits=BrokenPickle(),
        )


def _decoded_start(source: str | Path) -> float:
    """Return a payload span deliberately different from the filename span."""
    return {100: 11.0, 200: 10.0}[int(Path(source).stem.rsplit("-", 2)[-2])]


def _decoded_timeseries_worker(source, channels, start, end, backend, read_kwargs):
    """Spawn-safe TimeSeries worker with decoded spans opposite file spans."""
    del start, end, backend, read_kwargs
    decoded = _decoded_start(source)
    result = GwpyTimeSeriesDict()
    for channel in channels:
        series = GwpyTimeSeries(
            [decoded],
            sample_rate=1,
            t0=decoded,
            unit="V",
            channel=channel,
            name=f"{channel}-decoded",
        )
        series.source_metadata = {"decoded": decoded}
        result[channel] = series
    result.source_metadata = {"decoded": decoded, "kind": "timeseries"}
    return result


def _decoded_statevector_worker(source, channels, start, end, backend, read_kwargs):
    """Spawn-safe StateVector worker with decoded spans opposite file spans."""
    del start, end, backend, read_kwargs
    decoded = _decoded_start(source)
    result = StateVectorDict()
    for channel in channels:
        series = StateVector(
            [int(decoded)],
            bits=["decoded"],
            sample_rate=1,
            t0=decoded,
            unit="count",
            channel=channel,
            name=f"{channel}-decoded",
        )
        series.source_metadata = {"decoded": decoded}
        result[channel] = series
    result.source_metadata = {"decoded": decoded, "kind": "statevector"}
    return result


def test_real_spawn_uses_decoded_payload_spans_for_all_public_gwf_readers(
    monkeypatch, tmp_path
) -> None:
    sources = [tmp_path / "K1-span-100-1.gwf", tmp_path / "K1-span-200-1.gwf"]
    monkeypatch.setattr(
        gwf_io, "_read_gwf_timeseriesdict_worker", _decoded_timeseries_worker
    )
    monkeypatch.setattr(
        gwf_io, "_read_gwf_statevectordict_worker", _decoded_statevector_worker
    )
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict",
        lambda source, channels, **kwargs: _decoded_timeseries_worker(
            source, channels, None, None, None, {}
        ),
    )
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_statevectordict",
        lambda source, channels, **kwargs: _decoded_statevector_worker(
            source, channels, None, None, None, {}
        ),
    )
    _patch_connector_open(monkeypatch)

    def statevector_connector_read(cls, source, *args, **kwargs):
        channels = args[0] if isinstance(args[0], list) else [args[0]]
        result = _decoded_statevector_worker(source, channels, None, None, None, {})
        return result if cls is StateVectorDict else result[channels[0]]

    monkeypatch.setattr(StateVector.read.registry, "read", statevector_connector_read)
    families = (
        (TimeSeriesDict.read, [CHANNEL_A], False),
        (TimeSeries.read, CHANNEL_A, False),
        (StateVectorDict.read, [CHANNEL_A], True),
        (StateVector.read, CHANNEL_A, True),
    )
    for gap in (None, "ignore"):
        for reader, selector, is_statevector in families:
            kwargs = {"format": "gwf", "parallel": 2}
            if gap is not None:
                kwargs["gap"] = gap
            parallel = reader(sources, selector, **kwargs)
            kwargs["parallel"] = 1
            serial = reader(sources, selector, **kwargs)
            parallel_series = (
                parallel[CHANNEL_A]
                if isinstance(parallel, (TimeSeriesDict, StateVectorDict))
                else parallel
            )
            serial_series = (
                serial[CHANNEL_A]
                if isinstance(serial, (TimeSeriesDict, StateVectorDict))
                else serial
            )
            assert (
                parallel_series.value.tolist()
                == serial_series.value.tolist()
                == [10, 11]
            )
            assert (
                float(parallel_series.t0.value) == float(serial_series.t0.value) == 10
            )
            assert str(parallel_series.unit) == str(serial_series.unit)
            assert parallel_series.name == serial_series.name == f"{CHANNEL_A}-decoded"
            assert (
                parallel_series.channel.name == serial_series.channel.name == CHANNEL_A
            )
            if is_statevector:
                assert (
                    list(parallel_series.bits)
                    == list(serial_series.bits)
                    == ["decoded"]
                )


@pytest.mark.parametrize(
    ("reader", "selector"),
    [
        (TimeSeries.read, CHANNEL_A),
        (TimeSeriesDict.read, [CHANNEL_A]),
        (StateVector.read, CHANNEL_A),
        (StateVectorDict.read, [CHANNEL_A]),
    ],
)
def test_daemon_process_parallel_gwf_preflight_fails_before_backend(
    monkeypatch, reader, selector
) -> None:
    class DaemonProcess:
        daemon = True

    monkeypatch.setattr(gwf_io.multiprocessing, "current_process", DaemonProcess)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("backend I/O ran")
        ),
    )
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_statevectordict",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("backend I/O ran")
        ),
    )

    with pytest.raises(TypeError, match="not supported from a daemon process"):
        reader(
            [Path("K1-span-100-1.gwf"), Path("K1-span-200-1.gwf")],
            selector,
            format="gwf",
            parallel=2,
        )


@pytest.mark.parametrize("reader", [StateVector.read, StateVectorDict.read])
def test_statevector_parallel_requires_explicit_selector_before_backend(
    monkeypatch, reader
) -> None:
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_statevectordict",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("backend I/O ran")
        ),
    )

    with pytest.raises(TypeError, match="require a channel selector"):
        reader(
            [Path("K1-span-100-1.gwf"), Path("K1-span-200-1.gwf")],
            format="gwf",
            parallel=2,
        )


@pytest.mark.parametrize(
    ("reader", "selector"),
    [
        (TimeSeries.read, CHANNEL_A),
        (TimeSeriesDict.read, [CHANNEL_A]),
        (StateVector.read, CHANNEL_A),
        (StateVectorDict.read, [CHANNEL_A]),
    ],
)
@pytest.mark.parametrize(
    "invalid_source",
    [
        "https://frames.example.test/K1-test-0-1.gwf",
        "file:///frames/K1-test-0-1.gwf",
        "K1-test-0-1.gwf?cache=frames.cache",
        "frames.cache",
        BytesIO(b"not a frame path"),
    ],
)
def test_all_public_readers_reject_nonlocal_parallel_sources_before_backend(
    monkeypatch, reader, selector, invalid_source
) -> None:
    calls = []

    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("backend or source-segment I/O ran")

    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", unexpected)
    monkeypatch.setattr("gwpy.io.gwf.core.get_channel_names", unexpected)
    monkeypatch.setattr("gwpy.timeseries.io.gwf.core.read_timeseriesdict", unexpected)
    monkeypatch.setattr("gwpy.timeseries.io.gwf.core.read_statevectordict", unexpected)
    monkeypatch.setattr(StateVector.read.registry, "read", unexpected)

    with pytest.raises(TypeError, match="local GWF frame paths"):
        reader(
            [invalid_source, Path("K1-test-1-1.gwf")],
            selector,
            format="gwf",
            parallel=2,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("reader", "selector"),
    [
        (TimeSeries.read, CHANNEL_A),
        (TimeSeriesDict.read, [CHANNEL_A]),
        (StateVector.read, CHANNEL_A),
        (StateVectorDict.read, [CHANNEL_A]),
    ],
)
@pytest.mark.parametrize("option", ["parallel", "nproc"])
@pytest.mark.parametrize(
    "invalid_source",
    [
        "a.gwf+b.gwf",
        "a.gwf|b.gwf",
        "a.gwf;b.gwf",
        "a.gwf@b.gwf",
        "a.gwf b.gwf",
        "a.gwf%2Bb.gwf",
        "a.gwf%7Cb.gwf",
        "a.gwf%3Bb.gwf",
        "a.gwf%40b.gwf",
        "a.gwf%20b.gwf",
        ["a.gwf", "b.gwf"],
        ("a.gwf", "b.gwf"),
        {"cache": ["a.gwf", "b.gwf"]},
        _StringPath("a.gwf+b.gwf"),
    ],
)
def test_all_public_readers_reject_composite_parallel_sources_before_work(
    monkeypatch, reader, selector, option, invalid_source
) -> None:
    calls = []

    class ExplodingExecutor:
        def __init__(self, **kwargs):
            raise AssertionError("executor was created before source preflight")

    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("span resolver or backend I/O ran")

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", unexpected)
    monkeypatch.setattr("gwpy.io.gwf.core.get_channel_names", unexpected)
    monkeypatch.setattr("gwpy.timeseries.io.gwf.core.read_timeseriesdict", unexpected)
    monkeypatch.setattr("gwpy.timeseries.io.gwf.core.read_statevectordict", unexpected)
    monkeypatch.setattr(StateVector.read.registry, "read", unexpected)

    with pytest.raises(TypeError, match="local GWF frame paths"):
        reader(
            [invalid_source, Path("K1-test-1-1.gwf")],
            selector,
            format="gwf",
            **{option: 2},
        )
    assert calls == []


@pytest.mark.parametrize(
    "source",
    [
        "relative/K1-test-0-1.gwf",
        b"relative/K1-test-0-1.gwf",
        Path("relative/K1-test-0-1.gwf"),
        "relative/frame name.gwf",
        "relative/frame+tag.gwf",
        PureWindowsPath(r"C:\frames\K1-test-0-1.gwf"),
        PureWindowsPath(r"\\server\frames\K1-test-0-1.gwf"),
    ],
)
def test_parallel_source_preflight_accepts_structural_local_frame_paths(source) -> None:
    assert gwf_io._is_filesystem_path(source)


@pytest.mark.parametrize(
    ("reader", "selector", "worker_name"),
    [
        (TimeSeries.read, CHANNEL_A, "_read_gwf_timeseriesdict_worker"),
        (TimeSeriesDict.read, [CHANNEL_A], "_read_gwf_timeseriesdict_worker"),
        (StateVector.read, CHANNEL_A, "_read_gwf_statevectordict_worker"),
        (StateVectorDict.read, [CHANNEL_A], "_read_gwf_statevectordict_worker"),
    ],
)
@pytest.mark.parametrize(
    "source",
    [
        "relative/K1-test-0-1.gwf",
        b"relative/K1-test-0-1.gwf",
        Path("relative/K1-test-0-1.gwf"),
        PureWindowsPath(r"C:\frames\K1-test-0-1.gwf"),
        PureWindowsPath(r"\\server\frames\K1-test-0-1.gwf"),
    ],
)
def test_all_public_readers_accept_structural_local_parallel_paths(
    monkeypatch, reader, selector, worker_name, source
) -> None:
    _ImmediateExecutor.instances.clear()

    def timeseries_worker(source, channels, start, end, backend, read_kwargs):
        del source, start, end, backend, read_kwargs
        return GwpyTimeSeriesDict(
            {
                channel: GwpyTimeSeries(
                    [1.0], sample_rate=1, t0=1, unit="V", channel=channel
                )
                for channel in channels
            }
        )

    def statevector_worker(source, channels, start, end, backend, read_kwargs):
        del source, start, end, backend, read_kwargs
        return StateVectorDict(
            {
                channel: StateVector(
                    [1], bits=["ready"], sample_rate=1, t0=1, channel=channel
                )
                for channel in channels
            }
        )

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", lambda *args: (1, 2))
    monkeypatch.setattr(
        gwf_io,
        worker_name,
        statevector_worker
        if worker_name == "_read_gwf_statevectordict_worker"
        else timeseries_worker,
    )
    result = reader([source, source], selector, format="gwf", gap="ignore", parallel=2)
    assert len(result) > 0
    assert _ImmediateExecutor.instances[0].submit_calls


@pytest.mark.parametrize(
    ("reader", "selector", "worker_name"),
    [
        (TimeSeries.read, CHANNEL_A, "_read_gwf_timeseriesdict_worker"),
        (TimeSeriesDict.read, [CHANNEL_A], "_read_gwf_timeseriesdict_worker"),
        (StateVector.read, CHANNEL_A, "_read_gwf_statevectordict_worker"),
        (StateVectorDict.read, [CHANNEL_A], "_read_gwf_statevectordict_worker"),
    ],
)
def test_all_public_parallel_readers_preserve_worker_import_error_provenance(
    monkeypatch, reader, selector, worker_name
) -> None:
    _ImmediateExecutor.instances.clear()
    error = ImportError("worker backend import failure", "frame-extra")

    def failing_worker(*args):
        raise error

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(gwf_io, worker_name, failing_worker)

    with pytest.raises(ImportError) as raised:
        reader(
            [Path("early.gwf"), Path("late.gwf")],
            selector,
            format="gwf",
            parallel=2,
        )
    assert raised.value is error
    assert raised.value.args == ("worker backend import failure", "frame-extra")
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize("reader", [TimeSeries.read, TimeSeriesDict.read])
def test_timeseries_serial_import_error_keeps_existing_normalized_public_error(
    monkeypatch, reader
) -> None:
    error = ImportError("serial backend import failure")
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ImportError, match="Missing optional dependency") as raised:
        reader(Path("serial.gwf"), CHANNEL_A, format="gwf", parallel=1)
    assert raised.value is not error


@pytest.mark.parametrize("reader", [StateVector.read, StateVectorDict.read])
def test_statevector_serial_import_error_keeps_gwpy_connector_behavior(
    monkeypatch, reader
) -> None:
    error = ImportError("serial connector import failure")
    _patch_connector_open(monkeypatch)
    monkeypatch.setattr(
        StateVector.read.registry,
        "read",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ImportError) as raised:
        reader(Path("serial.gwf"), CHANNEL_A, format="gwf", parallel=1)
    assert raised.value is error


@pytest.mark.skipif(not FRAMEL_FIXTURE.exists(), reason="test.gwf fixture not found")
def test_segmentless_framel_filename_has_serial_parallel_parity_for_all_readers() -> (
    None
):
    pytest.importorskip("framel")
    readers = (
        (TimeSeries.read, FRAMEL_CHANNEL, False),
        (TimeSeriesDict.read, [FRAMEL_CHANNEL], False),
        (StateVector.read, FRAMEL_CHANNEL, True),
        (StateVectorDict.read, [FRAMEL_CHANNEL], True),
    )
    sources = [FRAMEL_FIXTURE, FRAMEL_FIXTURE]
    for reader, selector, is_statevector in readers:
        kwargs = {
            "format": "gwf",
            "backend": "framel",
            "gap": "ignore",
        }
        if is_statevector:
            kwargs["bits"] = ["quality"]
        serial = reader(
            sources,
            selector,
            parallel=1,
            **kwargs,
        )
        parallel = reader(
            sources,
            selector,
            parallel=2,
            **kwargs,
        )
        serial_series = (
            serial[FRAMEL_CHANNEL]
            if isinstance(serial, (TimeSeriesDict, StateVectorDict))
            else serial
        )
        parallel_series = (
            parallel[FRAMEL_CHANNEL]
            if isinstance(parallel, (TimeSeriesDict, StateVectorDict))
            else parallel
        )
        assert parallel_series.value.tolist() == serial_series.value.tolist()
        assert float(parallel_series.t0.value) == float(serial_series.t0.value)
        assert float(parallel_series.dt.value) == float(serial_series.dt.value)
        assert parallel_series.name == serial_series.name
        assert getattr(parallel_series.channel, "name", None) == getattr(
            serial_series.channel, "name", None
        )
        if is_statevector:
            assert list(parallel_series.bits) == list(serial_series.bits)


def test_scalar_timeseries_gwf_read_preserves_nested_public_metadata(
    monkeypatch,
) -> None:
    _ImmediateExecutor.instances.clear()
    source_payloads = []

    def metadata_read(source, channels, **kwargs):
        del kwargs
        start = _source_start(source)
        result = GwpyTimeSeriesDict()
        for channel in channels:
            result[channel] = GwpyTimeSeries(
                [start],
                sample_rate=1,
                t0=start,
                unit="V",
                channel=channel,
                name=f"{channel}-name",
            )
        result[CHANNEL_A].public_metadata = {
            "source": Path(source).name,
            "nested": {"labels": ["calibrated"]},
        }
        source_payloads.append(result)
        return result

    monkeypatch.setattr(gwf_io, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", _fake_span)
    monkeypatch.setattr(
        "gwpy.timeseries.io.gwf.core.read_timeseriesdict", metadata_read
    )
    sources = [Path("late.gwf"), Path("early.gwf")]
    serial = TimeSeries.read(sources, CHANNEL_A, format="gwf", gap="ignore", parallel=1)
    parallel = TimeSeries.read(
        sources, CHANNEL_A, format="gwf", gap="ignore", parallel=2
    )

    for result in (serial, parallel):
        assert result.public_metadata == {
            "source": "early.gwf",
            "nested": {"labels": ["calibrated"]},
        }
        assert str(result.unit) == "V"
        assert float(result.t0.value) == 1.0
        assert float(result.dt.value) == 1.0
        assert result.name == f"{CHANNEL_A}-name"
        assert result.channel.name == CHANNEL_A
    parallel.public_metadata["nested"]["labels"].append("mutated")
    assert serial.public_metadata["nested"]["labels"] == ["calibrated"]
    assert source_payloads[0][CHANNEL_A].public_metadata["nested"]["labels"] == [
        "calibrated"
    ]
