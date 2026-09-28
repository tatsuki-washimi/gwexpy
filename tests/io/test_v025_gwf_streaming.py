"""Bounded GWF reads must preserve the old-R public result and fault order."""

from __future__ import annotations

import logging
import os
import sys
import warnings
import weakref
from pathlib import Path

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

import gwexpy.timeseries._gwf_io as gwf_io
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

CHANNEL = "K1:V025-F4-STREAM"


def _frames(root: Path, count: int) -> list[Path]:
    """Write adjacent real GWF frames with distinct deterministic samples."""
    sources = []
    for index in range(count):
        path = root / f"part_{index:03d}.gwf"
        series = GwpyTimeSeries(
            np.arange(index * 8, (index + 1) * 8, dtype=np.float64),
            sample_rate=8,
            t0=1_000_000_000 + index,
            unit="m",
            channel=CHANNEL,
            name=CHANNEL,
        )
        GwpyTimeSeriesDict({CHANNEL: series}).write(path, format="gwf")
        sources.append(path)
    return sources


def test_large_sorted_serial_read_retains_bounded_parts(tmp_path, monkeypatch) -> None:
    """A qualifying successful read releases old parts as it advances."""
    sources = _frames(tmp_path, 16)
    original = gwf_io._coerce_gwf_timeseriesdict
    live = 0
    peak = 0

    def coerce_and_count(*args, **kwargs):
        nonlocal live, peak
        result = original(*args, **kwargs)
        live += 1
        peak = max(peak, live)

        def release() -> None:
            nonlocal live
            live -= 1

        weakref.finalize(result, release)
        return result

    monkeypatch.setattr(gwf_io, "_coerce_gwf_timeseriesdict", coerce_and_count)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)

    assert result[CHANNEL].value.tolist() == list(range(16 * 8))
    assert str(result[CHANNEL].unit) == "m"
    assert float(result[CHANNEL].t0.value) == 1_000_000_000.0
    assert float(result[CHANNEL].dt.value) == 0.125
    assert peak <= 4


def test_large_sorted_serial_places_values_without_repeated_append(
    tmp_path, monkeypatch
) -> None:
    """The contiguous fast route grows the result only once."""
    sources = _frames(tmp_path, 16)
    original = TimeSeriesDict.append
    original_copyto = gwf_io.np.copyto
    append_calls = 0
    placement_calls = 0
    initial_storage: list[bool] = []

    def counted_append(self, *args, **kwargs):
        nonlocal append_calls
        append_calls += 1
        result = original(self, *args, **kwargs)
        initial_storage.extend(
            [
                type(self[CHANNEL]) is TimeSeries,
                self[CHANNEL].flags.owndata,
                self[CHANNEL].flags.writeable,
                getattr(self[CHANNEL], "_xindex", None) is None,
                not np.shares_memory(self[CHANNEL].value, args[0][CHANNEL].value),
            ]
        )
        return result

    def counted_copyto(destination, values, *args, **kwargs):
        nonlocal placement_calls
        if kwargs.get("casting") == "no" and destination.shape == (8,):
            placement_calls += 1
        return original_copyto(destination, values, *args, **kwargs)

    monkeypatch.setattr(TimeSeriesDict, "append", counted_append)
    monkeypatch.setattr(gwf_io.np, "copyto", counted_copyto)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))
    assert append_calls == 1
    assert initial_storage == [True] * 5
    assert placement_calls == 15


def test_large_sorted_serial_preserves_public_result(tmp_path, monkeypatch) -> None:
    """Streaming carries exact values, order, time, units, and provenance."""
    sources = _frames(tmp_path, 16)
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 17)
    old = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 16)
    candidate = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)

    assert list(candidate) == list(old)
    np.testing.assert_array_equal(candidate[CHANNEL].value, old[CHANNEL].value)
    assert candidate[CHANNEL].dtype == old[CHANNEL].dtype
    assert candidate[CHANNEL].flags.owndata == old[CHANNEL].flags.owndata
    assert candidate[CHANNEL].flags.writeable == old[CHANNEL].flags.writeable
    assert candidate[CHANNEL].unit == old[CHANNEL].unit
    assert candidate[CHANNEL].name == old[CHANNEL].name
    assert candidate[CHANNEL].channel == old[CHANNEL].channel
    assert candidate[CHANNEL].t0 == old[CHANNEL].t0
    assert candidate[CHANNEL].dt == old[CHANNEL].dt
    assert candidate[CHANNEL].span == old[CHANNEL].span
    assert getattr(candidate, "_gwexpy_io", None) == getattr(old, "_gwexpy_io", None)
    assert getattr(candidate[CHANNEL], "_gwexpy_io", None) == getattr(
        old[CHANNEL], "_gwexpy_io", None
    )


def test_preallocated_result_avoids_final_full_array_copy(
    tmp_path, monkeypatch
) -> None:
    """An owned result needs one coercion per source and no final copy."""
    sources = _frames(tmp_path, 16)
    original = gwf_io._coerce_gwf_timeseriesdict
    coercions = 0

    def counted_coerce(*args, **kwargs):
        nonlocal coercions
        coercions += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gwf_io, "_coerce_gwf_timeseriesdict", counted_coerce)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))
    assert coercions == len(sources)


def test_large_sorted_serial_skips_backend_span_probe(tmp_path, monkeypatch) -> None:
    """A successful fast read derives spans from decoded parts only."""
    sources = _frames(tmp_path, 16)
    original = gwf_io._resolve_gwf_path_span
    span_calls = 0

    def counted_span(*args, **kwargs):
        nonlocal span_calls
        span_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gwf_io, "_resolve_gwf_path_span", counted_span)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))
    assert span_calls == 0


def test_adjacent_decoded_spans_skip_redundant_contiguity_view(
    tmp_path, monkeypatch
) -> None:
    """Already checked frame spans need no growing output slice per part."""
    sources = _frames(tmp_path, 16)

    def unexpected_contiguity(*args, **kwargs):
        raise AssertionError("decoded span adjacency already checked")

    monkeypatch.setattr(TimeSeries, "is_contiguous", unexpected_contiguity)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))


def test_later_dtype_mismatch_replays_old_merge(tmp_path, monkeypatch) -> None:
    """A part requiring GWpy's casting semantics returns to the old route."""
    sources = _frames(tmp_path, 16)
    original = gwf_io._read_gwf_timeseriesdict_serial
    calls: list[Path] = []

    def mixed_dtype_read(source, *args, **kwargs):
        calls.append(Path(source))
        part = original(source, *args, **kwargs)
        if Path(source) == sources[8]:
            part[CHANNEL] = part[CHANNEL].astype(np.float32)
        return part

    monkeypatch.setattr(gwf_io, "_read_gwf_timeseriesdict_serial", mixed_dtype_read)
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 17)
    old = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    calls.clear()
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 16)
    candidate = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)

    np.testing.assert_array_equal(candidate[CHANNEL].value, old[CHANNEL].value)
    assert candidate[CHANNEL].dtype == old[CHANNEL].dtype
    assert candidate[CHANNEL].unit == old[CHANNEL].unit
    assert candidate[CHANNEL].t0 == old[CHANNEL].t0
    assert candidate[CHANNEL].dt == old[CHANNEL].dt
    assert calls.count(sources[8]) == 2
    assert calls.count(sources[-1]) == 1


def test_speculative_diagnostics_are_not_published(
    tmp_path, monkeypatch, capfd, caplog
) -> None:
    """A noisy speculative decode replays old R without duplicate diagnostics."""
    sources = _frames(tmp_path, 16)
    original = gwf_io._read_gwf_timeseriesdict_serial
    emitted = False

    def noisy_read(source, *args, **kwargs):
        nonlocal emitted
        if Path(source) == sources[4] and not emitted:
            emitted = True
            warnings.warn("speculative warning", UserWarning, stacklevel=1)
            logging.getLogger("gwexpy.test.f4").warning("speculative log")
            print("speculative stderr", file=sys.stderr)
            os.write(2, b"speculative native stderr\n")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(gwf_io, "_read_gwf_timeseriesdict_serial", noisy_read)
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))
    assert "speculative stderr" not in capfd.readouterr().err
    assert "speculative log" not in caplog.text
    assert "speculative warning" not in [str(item.message) for item in caught]


def test_late_corruption_keeps_old_error_and_diagnostics(
    tmp_path, monkeypatch, capfd, caplog
) -> None:
    """A later read error must win over any speculative merge result."""
    sources = _frames(tmp_path, 17)
    original = gwf_io._read_gwf_timeseriesdict_serial

    def late_fault(source, *args, **kwargs):
        if Path(source) == sources[-1]:
            raise RuntimeError("late read fault")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(gwf_io, "_read_gwf_timeseriesdict_serial", late_fault)
    caplog.set_level(logging.WARNING)

    def capture() -> tuple[type[Exception], str, list[tuple[str, str]], str, str]:
        capfd.readouterr()
        caplog.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(Exception) as error:
                TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
        output = capfd.readouterr()
        return (
            type(error.value),
            str(error.value),
            [(item.category.__qualname__, str(item.message)) for item in caught],
            caplog.text,
            output.err,
        )

    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 18)
    old = capture()
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 16)
    candidate = capture()
    assert candidate == old


def test_read_warning_replays_once_from_old_route(tmp_path, monkeypatch) -> None:
    """A speculative read warning is discarded before the old-route retry."""
    sources = _frames(tmp_path, 16)
    original = gwf_io._read_gwf_timeseriesdict_serial

    def noisy_read(source, *args, **kwargs):
        if Path(source) == sources[4]:
            warnings.warn("source warning", UserWarning, stacklevel=1)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(gwf_io, "_read_gwf_timeseriesdict_serial", noisy_read)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))
    assert [
        (item.category.__qualname__, str(item.message))
        for item in caught
        if str(item.message) == "source warning"
    ] == [("UserWarning", "source warning")]


@pytest.mark.parametrize("order", ["reversed", "gap", "overlap"])
def test_nonadjacent_routes_keep_old_result(tmp_path, monkeypatch, order: str) -> None:
    """Unsorted, gap, and overlap inputs retain old read and merge behavior."""
    sources = _frames(tmp_path, 17)
    if order == "reversed":
        sources = list(reversed(sources))
    elif order == "gap":
        sources.pop(8)
    else:
        sources[8] = sources[7]

    def capture() -> tuple[str, str, list[float] | None]:
        try:
            result = TimeSeriesDict.read(
                sources, [CHANNEL], format="gwf", parallel=False
            )
        except Exception as error:
            return (type(error).__qualname__, str(error), None)
        return ("return", "", result[CHANNEL].value.tolist())

    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 18)
    old = capture()
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 16)
    assert capture() == old


def test_gap_pad_keeps_old_values(tmp_path, monkeypatch) -> None:
    """Padding options are delegated to the established merge path."""
    sources = _frames(tmp_path, 17)
    sources.pop(8)
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 18)
    old = TimeSeriesDict.read(
        sources, [CHANNEL], format="gwf", parallel=False, gap="pad"
    )
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 16)
    candidate = TimeSeriesDict.read(
        sources, [CHANNEL], format="gwf", parallel=False, gap="pad"
    )
    np.testing.assert_array_equal(candidate[CHANNEL].value, old[CHANNEL].value)


@pytest.mark.parametrize("shift", [-0.125, 0.125])
def test_one_sample_boundary_fault_replays_old_public_result(
    tmp_path, monkeypatch, shift: float
) -> None:
    """A one-sample overlap or gap preserves old outcome and warnings."""
    sources = _frames(tmp_path, 16)
    changed = GwpyTimeSeries(
        np.arange(8 * 8, 9 * 8, dtype=np.float64),
        sample_rate=8,
        t0=1_000_000_008 + shift,
        unit="m",
        channel=CHANNEL,
        name=CHANNEL,
    )
    sources[8].unlink()
    GwpyTimeSeriesDict({CHANNEL: changed}).write(sources[8], format="gwf")

    def capture():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = TimeSeriesDict.read(
                    sources, [CHANNEL], format="gwf", parallel=False
                )
            except Exception as error:
                outcome = ("error", type(error).__qualname__, str(error))
            else:
                series = result[CHANNEL]
                outcome = (
                    "return",
                    series.value.tolist(),
                    series.dtype.str,
                    str(series.unit),
                    float(series.t0.value),
                    float(series.dt.value),
                )
        return outcome, [
            (item.category.__qualname__, str(item.message)) for item in caught
        ]

    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 17)
    old = capture()
    monkeypatch.setattr(gwf_io, "_GWF_BOUNDED_MIN_SOURCES", 16)
    assert capture() == old


def test_decoded_span_disagreement_replays_old_route(tmp_path, monkeypatch) -> None:
    """A transient decoded gap cannot change the B1 merge result."""
    sources = _frames(tmp_path, 16)
    read_count = 0
    original_read = gwf_io._read_gwf_timeseriesdict_serial

    def shifted_read(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        part = original_read(*args, **kwargs)
        if read_count == 2:
            part[CHANNEL].t0 = float(part[CHANNEL].t0.value) + 1
        return part

    monkeypatch.setattr(gwf_io, "_read_gwf_timeseriesdict_serial", shifted_read)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))
    assert read_count == 18  # two speculative decodes, then all 16 old-R reads


def test_multithreaded_calls_use_old_route(tmp_path, monkeypatch) -> None:
    """fd 2 capture is never attempted while Python threads may compete."""
    sources = _frames(tmp_path, 16)
    monkeypatch.setattr(gwf_io.threading, "active_count", lambda: 2)

    def unexpected_capture():
        raise AssertionError("speculative diagnostics must be skipped")

    monkeypatch.setattr(gwf_io, "_capture_gwf_logs", unexpected_capture)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))


def test_log_capture_setup_failure_uses_old_route(tmp_path, monkeypatch) -> None:
    """An unavailable logging capture cannot change the public read."""
    sources = _frames(tmp_path, 16)
    root_handlers = tuple(logging.getLogger().handlers)
    monkeypatch.setattr(logging, "lastResort", None)

    def cannot_install_filter(*args, **kwargs):
        raise RuntimeError("filter setup unavailable")

    monkeypatch.setattr(logging.Handler, "addFilter", cannot_install_filter)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16 * 8))
    assert tuple(logging.getLogger().handlers) == root_handlers


def test_small_read_skips_speculative_diagnostic_setup(tmp_path, monkeypatch) -> None:
    """Two-frame reads pay no logging or descriptor-capture setup cost."""
    sources = _frames(tmp_path, 2)

    def unexpected_capture():
        raise AssertionError("small read must not enter speculative capture")

    monkeypatch.setattr(gwf_io, "_capture_gwf_logs", unexpected_capture)
    result = TimeSeriesDict.read(sources, [CHANNEL], format="gwf", parallel=False)
    assert result[CHANNEL].value.tolist() == list(range(16))
