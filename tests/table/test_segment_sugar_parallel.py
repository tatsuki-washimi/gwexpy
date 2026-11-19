"""Parallel eager crop/asd contracts (#407)."""

from __future__ import annotations

from threading import Barrier

import numpy as np
import pytest
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries

from gwexpy.table import CellStatus, SegmentCell, SegmentTable


def _table() -> SegmentTable:
    spans = [Segment(i * 4, (i + 1) * 4) for i in range(3)]
    table = SegmentTable.from_segments(spans)
    table.add_series_column(
        "raw",
        data=[
            TimeSeries(
                np.arange(64, dtype=float) + i, sample_rate=16, t0=i * 4, unit="V"
            )
            for i in range(3)
        ],
    )
    return table


@pytest.mark.parametrize("parallel", [False, True, 1, 2])
def test_crop_parallel_keeps_values_axes_and_row_order(parallel, monkeypatch) -> None:
    if parallel is True:
        monkeypatch.setattr("gwexpy.table.segment.stats.os.cpu_count", lambda: 2)
    table = _table()
    serial = table.crop("raw")
    result = table.crop("raw", parallel=parallel)
    for i in range(3):
        expected = serial.row(i)["raw_cropped"]
        actual = result.row(i)["raw_cropped"]
        np.testing.assert_array_equal(actual.value, expected.value)
        assert actual.t0 == expected.t0
        assert actual.dt == expected.dt
        assert actual.unit == expected.unit
    assert "raw_cropped" not in table.columns


@pytest.mark.parametrize("parallel", [False, True, 1, 2])
def test_asd_parallel_keeps_values_axes_and_row_order(parallel, monkeypatch) -> None:
    if parallel is True:
        monkeypatch.setattr("gwexpy.table.segment.stats.os.cpu_count", lambda: 2)
    table = _table()
    serial = table.asd("raw", fftlength=2)
    result = table.asd("raw", fftlength=2, parallel=parallel)
    for i in range(3):
        expected = serial.row(i)["raw_asd"]
        actual = result.row(i)["raw_asd"]
        np.testing.assert_array_equal(actual.value, expected.value)
        assert actual.f0 == expected.f0
        assert actual.df == expected.df
        assert actual.unit == expected.unit


@pytest.mark.parametrize("method", ["crop", "asd"])
def test_parallel_two_runs_rows_concurrently(method, monkeypatch) -> None:
    table = _table().select(mask=[True, True, False])
    barrier = Barrier(2)
    original = getattr(TimeSeries, method)

    def synchronized(self, *args, **kwargs):
        barrier.wait(timeout=3)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TimeSeries, method, synchronized)
    kwargs = {"fftlength": 2} if method == "asd" else {}
    result = getattr(table, method)("raw", parallel=2, **kwargs)
    assert len(result) == 2


def _failure_table() -> SegmentTable:
    table = _table()

    def fail():
        raise OSError("missing frame")

    table._payload["raw"][1] = SegmentCell(loader=fail)
    table._payload["raw"][2] = SegmentCell.missing()
    return table


@pytest.mark.parametrize("method", ["crop", "asd"])
@pytest.mark.parametrize("errors", ["record", "coerce", "skip"])
def test_parallel_error_policies_match_serial(method, errors) -> None:
    kwargs = {"fftlength": 2} if method == "asd" else {}
    serial = getattr(_failure_table(), method)("raw", errors=errors, **kwargs)
    parallel = getattr(_failure_table(), method)(
        "raw", errors=errors, parallel=2, **kwargs
    )
    output = f"raw_{'cropped' if method == 'crop' else 'asd'}"
    assert [cell.status for cell in parallel._payload[output]] == [
        cell.status for cell in serial._payload[output]
    ]
    assert [parallel.row(i)["span"] for i in range(len(parallel))] == [
        serial.row(i)["span"] for i in range(len(serial))
    ]
    if errors == "record":
        assert parallel._payload[output][1].status == CellStatus.FAILED
        assert parallel.row(1)[f"{output}_error_type"] == "OSError"


@pytest.mark.parametrize("method", ["crop", "asd"])
def test_invalid_parallel_worker_count_fails_even_for_empty_table(method) -> None:
    table = SegmentTable.from_segments([])
    table.add_series_column("raw", data=[])
    kwargs = {"fftlength": 2} if method == "asd" else {}
    with pytest.raises(ValueError, match="parallel"):
        getattr(table, method)("raw", parallel=0, **kwargs)


@pytest.mark.parametrize("method", ["crop", "asd"])
def test_parallel_sugar_loads_only_required_payload_column(method) -> None:
    table = _table()

    def unused_loader(segment):
        raise AssertionError("unrelated payload loaded")

    table.add_series_column("unused", loader=unused_loader)
    kwargs = {"fftlength": 2} if method == "asd" else {}
    getattr(table, method)("raw", parallel=2, **kwargs)
    assert all(cell.status == CellStatus.UNLOADED for cell in table._payload["unused"])


@pytest.mark.parametrize("method", ["crop", "asd"])
def test_parallel_raise_propagates_source_failure(method) -> None:
    kwargs = {"fftlength": 2} if method == "asd" else {}
    with pytest.raises(OSError, match="missing frame"):
        getattr(_failure_table(), method)("raw", parallel=2, **kwargs)


@pytest.mark.parametrize("method", ["crop", "asd"])
def test_parallel_inplace_returns_same_table(method) -> None:
    table = _table()
    kwargs = {"fftlength": 2} if method == "asd" else {}
    assert getattr(table, method)("raw", parallel=2, inplace=True, **kwargs) is table
