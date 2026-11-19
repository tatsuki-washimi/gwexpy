"""Channel reshape preserves segment rows, cell state, and explicit loading."""

import pytest
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from gwexpy.table import CellStatus, SegmentCell, SegmentTable


def test_explode_channels_preserves_rows_and_missing_cells():
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)], label=["a", "b"])
    table.add_series_column(
        "signals",
        kind="timeseriesdict",
        data=[
            TimeSeriesDict(H1=TimeSeries([1], unit="V"), L1=TimeSeries([2], unit="V")),
            SegmentCell.missing(),
        ],
    )
    result = table.explode_channels("signals")
    assert result["channel"].to_list()[:2] == ["H1", "L1"]
    assert result["label"].to_list() == ["a", "a", "b"]
    assert result.row(1)["signals"].value.tolist() == [2]
    assert result._payload["signals"][2].status == CellStatus.MISSING
    assert table.schema["signals"] == "timeseriesdict"
    assert result.schema["signals"] == "timeseries"


def test_explode_requires_explicit_load_and_preserves_source_state():
    calls = []
    table = SegmentTable.from_segments([Segment(0, 1)])
    table.add_series_column(
        "signals",
        kind="timeseriesdict",
        loader=lambda _: calls.append(1) or TimeSeriesDict(H1=TimeSeries([1])),
    )
    with pytest.raises(ValueError, match="load=True"):
        table.explode_channels("signals")
    assert calls == []
    result = table.explode_channels("signals", load=True)
    assert result["channel"].to_list() == ["H1"]
    assert calls == [1]
    assert table._payload["signals"][0].status == CellStatus.UNLOADED


def test_expand_bands_matches_gwpy_crop():
    table = SegmentTable.from_segments([Segment(0, 1)])
    spectrum = FrequencySeries([0, 1, 2, 3, 4, 5], f0=0, df=1, unit="V")
    table.add_series_column("spectrum", data=[spectrum], kind="frequencyseries")
    result = table.expand_bands("spectrum", {"low": (1, 3), "high": (3, 5)})
    assert result["band"].to_list() == ["low", "high"]
    for i, bounds in enumerate([(1, 3), (3, 5)]):
        expected = spectrum.crop(*bounds)
        actual = result.row(i)["spectrum"]
        assert actual.value.tolist() == expected.value.tolist()
        assert (actual.f0, actual.df, actual.unit) == (
            expected.f0,
            expected.df,
            expected.unit,
        )


def test_expand_bands_does_not_implicitly_load_or_erase_status():
    calls = []
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)])
    table.add_series_column(
        "spectrum",
        data=[
            SegmentCell(
                loader=lambda: calls.append(1) or FrequencySeries([1, 2, 3], f0=0, df=1)
            ),
            SegmentCell.masked(),
        ],
        kind="frequencyseries",
    )
    bands = {"low": (0, 2)}
    with pytest.raises(ValueError, match="load=True"):
        table.expand_bands("spectrum", bands)
    assert calls == []
    result = table.expand_bands("spectrum", bands, load=True)
    assert calls == [1]
    assert result._payload["spectrum"][1].status == CellStatus.MASKED
    assert table._payload["spectrum"][0].status == CellStatus.UNLOADED
