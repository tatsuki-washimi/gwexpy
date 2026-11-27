"""Deferred SegmentTable transformations."""

from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries

from gwexpy.table import SegmentTable, col
from gwexpy.table.segment import CellStatus


def test_frame_defers_operations_and_keeps_source_unchanged():
    calls = []
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)], snr=[1, 4])
    table.add_series_column(
        "raw", loader=lambda segment: calls.append(segment) or TimeSeries([1])
    )
    frame = (
        table.frame().filter(col("snr") > 2, name="loud").define(double=col("snr") * 2)
    )
    assert calls == []
    result = frame.collect()
    assert result["double"].to_list() == [8]
    assert result["snr"].to_list() == [4]
    assert result._payload["raw"][0].status == CellStatus.UNLOADED
    assert "double" not in table.columns
    assert calls == []
    assert frame.report() == [{"name": "loud", "n_in": 2, "n_out": 1}]
    assert frame.count() == 1


def test_frame_branches_and_source_snapshot():
    table = SegmentTable.from_segments([Segment(0, 1)], snr=[1])
    frame = table.frame()
    branch = frame.define(value=col("snr") + 1)
    table._meta.loc[0, "snr"] = 10
    assert frame.collect()["snr"].to_list() == [1]
    assert "value" not in frame.collect().columns
    assert branch.collect()["value"].to_list() == [2]


def test_frame_actions_execute_each_time():
    calls = []
    table = SegmentTable.from_segments([Segment(0, 1)])

    def calculate(current):
        calls.append(len(current))
        return [7]

    frame = table.frame().define(value=calculate)
    assert calls == []
    assert frame.collect()["value"].to_list() == [7]
    assert frame.count() == 1
    assert calls == [1, 1]
