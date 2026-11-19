"""Eager grouped SegmentTable summaries (#592 S-8)."""

from __future__ import annotations

import json
import numbers

import matplotlib
import numpy as np
import pytest
from astropy import units as u
from gwpy.segments import Segment

from gwexpy.table import SegmentTable, SummaryTable
from gwexpy.table.segment import quantile

matplotlib.use("Agg")


def _table() -> SegmentTable:
    table = SegmentTable.from_segments(
        [Segment(i, i + 1) for i in range(8)],
        detector=["H1", "H1", "H1", "H1", "H1", "L1", "L1", "H1"],
        band=["low", "low", "low", "low", "low", "low", "low", "high"],
    )
    table.add_column(
        "rt60", [2.0, 4.0, np.nan, np.nan, np.nan, 8.0, 10.0, 12.0], unit=u.s
    )
    table.add_column(
        "rt60_status",
        ["valid", "valid", "missing", "failed", "masked", "valid", "valid", "valid"],
    )
    return table


def test_grouped_named_aggregates_status_counts_units_and_order() -> None:
    table = _table()
    result = table.groupby(["detector", "band"]).aggregate(
        n=("rt60", "count"),
        median=("rt60", "median"),
        q90=("rt60", quantile(0.9)),
        skipna=True,
        min_count=2,
    )
    assert isinstance(result, SummaryTable)
    assert result.units == {"median": u.s, "q90": u.s}
    frame = result.to_pandas()
    assert frame[["detector", "band"]].values.tolist() == [
        ["H1", "low"],
        ["L1", "low"],
        ["H1", "high"],
    ]
    assert frame[
        ["n_total", "n_valid", "n_missing", "n_failed", "n_masked"]
    ].values.tolist() == [[5, 2, 1, 1, 1], [2, 2, 0, 0, 0], [1, 1, 0, 0, 0]]
    assert frame["n"].tolist() == [2, 2, 1]
    assert frame["median"].iloc[:2].tolist() == [3.0, 9.0]
    assert frame["q90"].iloc[:2].tolist() == [3.8, 9.8]
    assert np.isnan(frame["median"].iloc[2])
    assert np.isnan(frame["q90"].iloc[2])
    frame.loc[0, "median"] = -1
    assert result.to_pandas().loc[0, "median"] == 3.0
    assert table._meta["rt60"].iloc[0] == 2.0


def test_groupby_rejects_ambiguous_or_invalid_requests() -> None:
    table = _table()
    with pytest.raises(ValueError, match="non-valid"):
        table.groupby("detector").aggregate(mean=("rt60", "mean"), skipna=False)
    with pytest.raises(ValueError, match="one source column"):
        table.groupby("detector").aggregate(a=("rt60", "mean"), b=("band", "count"))
    with pytest.raises(KeyError):
        table.groupby("unknown")
    with pytest.raises(ValueError, match="valid status"):
        table._meta.at[0, "detector"] = np.nan
        table.groupby("detector").aggregate(mean=("rt60", "mean"))
    with pytest.raises(ValueError, match="Unsupported"):
        _table().groupby("detector").aggregate(mean=("rt60", []))


def test_exact_large_integer_quantiles_survive_missing_rows_and_json(tmp_path) -> None:
    large = 2**63 - 1
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)], group=["a", "b"])
    table.add_column("value", [large, np.nan])
    result = table.groupby("group").aggregate(
        maximum=("value", "max"),
        median=("value", "median"),
        q90=("value", quantile(0.9)),
    )
    frame = result.to_pandas()
    assert frame.loc[0, "maximum"] == large
    assert frame.loc[0, "median"] == large
    assert frame.loc[0, "q90"] == large
    assert all(
        isinstance(frame.loc[0, name], numbers.Integral)
        for name in ("maximum", "median", "q90")
    )
    assert np.isnan(frame.loc[1, "maximum"])
    path = tmp_path / "summary.json"
    result.write(path)
    document = json.loads(path.read_text())
    assert document["rows"][0][-3:] == [large, large, large]
    assert document["rows"][1][-3:] == [None, None, None]


def test_summary_show_csv_and_plot(tmp_path) -> None:
    result = _table().groupby("detector").aggregate(mean=("rt60", "mean"))
    assert len(result.show(1)) == 1
    assert len(result) == 2
    csv_path = tmp_path / "summary.csv"
    result.write(csv_path)
    assert "n_total" in csv_path.read_text()
    ax = result.plot("detector", "mean")
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_quantity_source_normalizes_units_without_declared_column_unit() -> None:
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)], group=["a", "a"])
    table.add_column("distance", [1 * u.m, 100 * u.cm])
    result = table.groupby("group").aggregate(mean=("distance", "mean"))
    assert result.to_pandas().loc[0, "mean"] == 1
    assert result.units["mean"] == u.m


def test_payload_source_needs_extract_stat_first() -> None:
    table = SegmentTable.from_segments([Segment(0, 1)], group=["a"])
    table.add_series_column("signal", data=[np.array([1, 2, 3])])
    with pytest.raises(TypeError, match="extract_stat"):
        table.groupby("group").aggregate(mean=("signal", "mean"))


def test_group_key_with_failed_status_is_rejected() -> None:
    table = _table()
    table.add_column("detector_status", ["failed", *["valid"] * 7])
    with pytest.raises(ValueError, match="valid status"):
        table.groupby("detector").aggregate(mean=("rt60", "mean"))


def test_mixed_signed_unsigned_integers_keep_exact_extrema_quantiles_and_mean() -> None:
    maximum = 2**64 - 1
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)], group=["a", "a"])
    table.add_column("value", [-1, np.uint64(maximum)])
    result = table.groupby("group").aggregate(
        minimum=("value", "min"),
        maximum=("value", "max"),
        median=("value", "median"),
        q100=("value", quantile(1)),
        mean=("value", "mean"),
    )
    row = result.to_pandas().iloc[0]
    assert row["minimum"] == -1
    assert row["maximum"] == maximum
    assert row["q100"] == maximum
    assert row["median"] == 2**63 - 1
    assert row["mean"] == 2**63 - 1
    assert isinstance(row["maximum"], numbers.Integral)
    assert isinstance(row["q100"], numbers.Integral)


def test_float_quantile_interpolates_without_overflow() -> None:
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)], group=["a", "a"])
    table.add_column("value", [-1e308, 1e308])
    result = table.groupby("group").aggregate(
        q25=("value", quantile(0.25)), q75=("value", quantile(0.75))
    )
    row = result.to_pandas().iloc[0]
    assert row["q25"] == pytest.approx(-5e307)
    assert row["q75"] == pytest.approx(5e307)


def test_large_fractional_integer_mean_fails_instead_of_rounding() -> None:
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)], group=["a", "a"])
    table.add_column("value", [np.uint64(2**64 - 1), 0])
    with pytest.raises(ValueError, match="not exactly representable"):
        table.groupby("group").aggregate(mean=("value", "mean"))


@pytest.mark.parametrize("method", ["mean", "median", quantile(0.5), quantile(0.25)])
def test_boundary_fractional_integer_results_fail_instead_of_rounding(method) -> None:
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2)], group=["a", "a"])
    table.add_column("value", [2**53 - 1, 2**53])
    with pytest.raises(ValueError, match="not exactly representable"):
        table.groupby("group").aggregate(result=("value", method))
