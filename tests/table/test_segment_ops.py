"""Tests for S-4 row/column operations and segment-aware lookups."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries

from gwexpy.table import CellStatus, SegmentCell, SegmentTable, col


def _make_ops_table():
    segs = [
        Segment(10, 14),
        Segment(0, 4),
        Segment(20, 26),
        Segment(4, 12),
        Segment(30, 34),
    ]
    st = SegmentTable.from_segments(
        segs,
        snr=[8.0, 3.0, np.nan, 12.0, 8.0],
        label=["b", "a", "c", "a", "b"],
        chi2=[10.0, 20.0, 30.0, 40.0, 50.0],
        ndof=[5.0, 5.0, 5.0, 5.0, 5.0],
    )
    return st


class TestFilterAndRowOps:
    def test_assign_replaces_and_copies_column_units(self):
        table = _make_ops_table()
        table.add_column("distance", [1, 2, 3, 4, 5], unit=u.m)
        table.add_series_column(
            "signal", data=[TimeSeries(np.ones(4), unit="V")] * 5, unit=u.V
        )
        table.add_series_column("unitless", data=[TimeSeries(np.ones(4))] * 5)

        result = table.assign(distance_copy=col("distance"), signal=col("unitless"))
        assert result["distance_copy"].unit == u.m
        assert result["signal"].unit == u.dimensionless_unscaled
        assert table["signal"].unit == u.V
        assert table.assign(distance=0)["distance"].unit is None
        assert table.assign(distance=col("distance"))["distance"].unit == u.m

    def test_filter_expr_string_and_mask(self):
        st = _make_ops_table()
        # col() expr with name= provenance
        f1 = st.filter((col("snr") > 5) & (col("label") == "a"), name="high_snr_a")
        assert len(f1) == 1
        assert f1.row(0)["span"] == Segment(4, 12)
        assert f1._provenance[-1] == {
            "op": "filter",
            "name": "high_snr_a",
            "n_in": 5,
            "n_out": 1,
        }

        # string filter via parse_column_filters
        f2 = st.filter('snr >= 8 && label == "b"')
        assert [f2.row(i)["span"] for i in range(len(f2))] == [
            Segment(10, 14),
            Segment(30, 34),
        ]

        # boolean mask
        f3 = st.filter([True, False, False, False, True])
        assert len(f3) == 2

    def test_filter_isolates_cell_status_unlike_legacy_select(self):
        st = SegmentTable.from_segments([Segment(0, 4), Segment(4, 8)], snr=[1.0, 10.0])
        st.add_series_column(
            "ts",
            loader=[
                lambda: TimeSeries(np.ones(8), sample_rate=4),
                lambda: TimeSeries(np.ones(8) * 2, sample_rate=4),
            ],
            kind="timeseries",
        )
        filtered = st.filter(col("snr") > 5)
        assert filtered.row(0)["ts"].value[0] == 2.0
        assert filtered._payload["ts"][0].status == CellStatus.VALID
        assert st._payload["ts"][1].status == CellStatus.UNLOADED

    def test_head_tail_take_sample(self):
        st = _make_ops_table()
        assert len(st.head(2)) == 2
        assert st.head(2).row(0)["span"] == Segment(10, 14)
        assert len(st.tail(2)) == 2
        assert st.tail(2).row(1)["span"] == Segment(30, 34)

        taken = st.take([4, 1, -5])
        assert [taken.row(i)["span"] for i in range(3)] == [
            Segment(30, 34),
            Segment(0, 4),
            Segment(10, 14),
        ]
        with pytest.raises(IndexError):
            st.take([10])

        s1 = st.sample(3, random_state=42)
        s2 = st.sample(3, random_state=42)
        assert [s1.row(i)["span"] for i in range(3)] == [
            s2.row(i)["span"] for i in range(3)
        ]

    def test_sort_values_and_drop_duplicates(self):
        st = _make_ops_table()
        s_desc = st.sort_values(by=col("snr"), ascending=False, na_position="last")
        assert s_desc.row(0)["snr"] == 12.0
        assert np.isnan(s_desc.row(-1)["snr"])

        s_multi = st.sort_values(by=["label", "snr"], ascending=[True, False])
        assert [s_multi.row(i)["snr"] for i in range(2)] == [12.0, 3.0]

        s_span = st.sort_values(by=col("span").start, ascending=True)
        assert [s_span.row(i)["span"][0] for i in range(5)] == [0, 4, 10, 20, 30]

        dedup = st.drop_duplicates(on="label", keep="first")
        assert [dedup.row(i)["label"] for i in range(len(dedup))] == ["b", "a", "c"]


class TestColumnOps:
    def test_select_drop_rename_assign(self):
        st = _make_ops_table()
        sel = st.select_columns("snr")
        assert sel.columns == ["span", "snr"]

        dropped = st.drop_columns("chi2", "ndof")
        assert set(dropped.columns) == {"span", "snr", "label"}

        with pytest.raises(ValueError, match="span"):
            st.drop_columns("span")

        renamed = st.rename_columns(snr="snr_db", label="tag")
        assert "snr_db" in renamed.columns
        assert "snr" not in renamed.columns

        with pytest.raises(ValueError, match="span"):
            st.rename_columns(span="interval")

        assigned = st.assign(
            duration=col("span").duration,
            reduced_chi2=col("chi2") / col("ndof"),
            const=99,
        )
        assert assigned["duration"].to_list() == [4.0, 4.0, 6.0, 8.0, 4.0]
        assert assigned["reduced_chi2"].to_list() == [2.0, 4.0, 6.0, 8.0, 10.0]
        assert assigned["const"].to_list() == [99] * 5
        # Immutability check
        assert "duration" not in st.columns


class TestSegmentLookups:
    def test_overlapping_within_containing_before_after_nearest(self):
        st = _make_ops_table()
        # Spans: [10,14), [0,4), [20,26), [4,12), [30,34)

        ov = st.overlapping(Segment(3, 11))
        assert [ov.row(i)["span"] for i in range(len(ov))] == [
            Segment(10, 14),
            Segment(0, 4),
            Segment(4, 12),
        ]

        wi = st.within(Segment(0, 15))
        assert [wi.row(i)["span"] for i in range(len(wi))] == [
            Segment(10, 14),
            Segment(0, 4),
            Segment(4, 12),
        ]

        ct = st.containing(11.0)
        assert [ct.row(i)["span"] for i in range(len(ct))] == [
            Segment(10, 14),
            Segment(4, 12),
        ]
        # Half-open boundary: t=4 is in [4,12), not [0,4)
        ct_edge = st.containing(4.0)
        assert [ct_edge.row(i)["span"] for i in range(len(ct_edge))] == [Segment(4, 12)]

        bf = st.before(12.0)
        assert [bf.row(i)["span"] for i in range(len(bf))] == [
            Segment(0, 4),
            Segment(4, 12),
        ]

        af = st.after(20.0)
        assert [af.row(i)["span"] for i in range(len(af))] == [
            Segment(20, 26),
            Segment(30, 34),
        ]

        nr = st.nearest(17.0, n=2)
        # dists: [10,14)->3, [20,26)->3, [4,12)->5, [30,34)->13, [0,4)->13
        # tie between row 0 [10,14) and row 2 [20,26) broken by original index
        assert [nr.row(i)["span"] for i in range(2)] == [
            Segment(10, 14),
            Segment(20, 26),
        ]

        nr_inside = st.nearest(11.0, n=2)
        assert [nr_inside.row(i)["span"] for i in range(2)] == [
            Segment(10, 14),
            Segment(4, 12),
        ]
