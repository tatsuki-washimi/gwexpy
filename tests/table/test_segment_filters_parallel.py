"""GWpy compatibility and row policies for SegmentTable filters."""

from threading import Barrier

import numpy as np
import pytest
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from gwexpy.table import CellStatus, SegmentCell, SegmentTable


def make_table(dictionary=False):
    table = SegmentTable.from_segments([Segment(0, 8), Segment(8, 16)])
    series = [
        TimeSeries(np.sin(np.arange(512) * 0.3) + i, sample_rate=64, t0=i * 8, unit="V")
        for i in range(2)
    ]
    if dictionary:
        table.add_series_column(
            "raw",
            data=[TimeSeriesDict(a=s, b=s * 2) for s in series],
            kind="timeseriesdict",
        )
    else:
        table.add_series_column("raw", data=series)
    return table


@pytest.mark.parametrize("method,args", [("bandpass", (4, 20)), ("resample", (32,))])
@pytest.mark.parametrize("parallel", [False, True, 2])
@pytest.mark.parametrize("dictionary", [False, True])
def test_matches_gwpy(method, args, parallel, dictionary, monkeypatch):
    monkeypatch.setattr("gwexpy.table.segment.stats.os.cpu_count", lambda: 2)
    table = make_table(dictionary)
    result = getattr(table, method)("raw", *args, parallel=parallel)
    for i in range(2):
        source = table.row(i)["raw"]
        actual = result.row(i)[f"raw_{method}"]
        keys = source.keys() if dictionary else [None]
        for key in keys:
            original = source[key] if dictionary else source
            output = actual[key] if dictionary else actual
            expected = getattr(original, method)(*args)
            np.testing.assert_array_equal(output.value, expected.value)
            assert (output.t0, output.dt, output.unit) == (
                expected.t0,
                expected.dt,
                expected.unit,
            )


@pytest.mark.parametrize("method,args", [("bandpass", (4, 20)), ("resample", (32,))])
@pytest.mark.parametrize("errors", ["record", "coerce", "skip"])
def test_failure_policy_and_inplace(method, args, errors):
    table = make_table()

    def fail():
        raise OSError("unavailable")

    table._payload["raw"][1] = SegmentCell(loader=fail)
    result = getattr(table, method)(
        "raw", *args, parallel=2, errors=errors, inplace=True
    )
    assert result is table
    if errors == "record":
        assert result._payload[f"raw_{method}"][1].status == CellStatus.FAILED
    elif errors == "skip":
        assert len(result) == 1
    else:
        assert result._payload[f"raw_{method}"][1].status == CellStatus.MISSING


@pytest.mark.parametrize("method,args", [("bandpass", (4, 20)), ("resample", (32,))])
def test_invalid_parallel_on_empty_table(method, args):
    table = SegmentTable.from_segments([])
    table.add_series_column("raw", data=[])
    with pytest.raises(ValueError, match="parallel"):
        getattr(table, method)("raw", *args, parallel=0)


@pytest.mark.parametrize("method,args", [("bandpass", (4, 20)), ("resample", (32,))])
def test_parallel_executes_concurrently_and_preserves_missing(
    method, args, monkeypatch
):
    table = make_table()
    barrier = Barrier(2)
    original = getattr(TimeSeries, method)

    def synchronized(self, *args, **kwargs):
        barrier.wait(timeout=3)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TimeSeries, method, synchronized)
    assert len(getattr(table, method)("raw", *args, parallel=2)) == 2

    table._payload["raw"][1] = SegmentCell.missing()
    monkeypatch.setattr(TimeSeries, method, original)
    result = getattr(table, method)("raw", *args, parallel=2)
    assert result._payload[f"raw_{method}"][1].status == CellStatus.MISSING
