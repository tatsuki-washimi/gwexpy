"""Tests for S-5 concat(), join(), and add_segment_id()."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries

from gwexpy.table import CellStatus, SegmentCell, SegmentTable


class TestConcat:
    def test_concat_strict_and_source_column(self):
        t1 = SegmentTable.from_segments([Segment(0, 4)], snr=[5.0])
        t1.add_series_column(
            "ts", data=[TimeSeries(np.ones(8), sample_rate=4)], kind="timeseries"
        )
        t2 = SegmentTable.from_segments([Segment(4, 8)], snr=[9.0])
        t2.add_series_column(
            "ts", data=[TimeSeries(np.ones(8) * 2, sample_rate=4)], kind="timeseries"
        )

        out = SegmentTable.concat({"run1": t1, "run2": t2}, source_column="run")
        assert len(out) == 2
        assert out["run"].to_list() == ["run1", "run2"]
        assert out["snr"].to_list() == [5.0, 9.0]
        assert out.row(1)["ts"].value[0] == 2.0

        # Strict rejects mismatched schema
        t3 = SegmentTable.from_segments([Segment(8, 12)], snr=[7.0], extra=["x"])
        with pytest.raises(ValueError, match="strict"):
            SegmentTable.concat([t1, t3], schema="strict")

    def test_concat_union_and_intersection(self):
        t1 = SegmentTable.from_segments([Segment(0, 4)], a=[1.0], shared=[10])
        t1.add_series_column("p1", data=[TimeSeries(np.ones(4), sample_rate=4)])
        t2 = SegmentTable.from_segments([Segment(4, 8)], b=[2.0], shared=[20])

        u = SegmentTable.concat([t1, t2], schema="union")
        assert set(u.columns) == {"span", "a", "b", "shared", "p1"}
        assert np.isnan(u.row(1)["a"])
        assert np.isnan(u.row(0)["b"])
        assert u._payload["p1"][0].status == CellStatus.VALID
        assert u._payload["p1"][1].status == CellStatus.MISSING

        inter = SegmentTable.concat([t1, t2], schema="intersection")
        assert inter.columns == ["span", "shared"]
        assert inter["shared"].to_list() == [10, 20]

    def test_concat_checks_segment_id_collisions(self):
        t1 = SegmentTable.from_segments(
            [Segment(0, 4)], detector=["H1"]
        ).add_segment_id()
        t2 = SegmentTable.from_segments(
            [Segment(0, 4)], detector=["H1"]
        ).add_segment_id()
        with pytest.raises(ValueError, match="segment_id"):
            SegmentTable.concat([t1, t2])

        t3 = SegmentTable.from_segments(
            [Segment(0, 4)], detector=["L1"]
        ).add_segment_id()
        combined = SegmentTable.concat([t1, t3])
        assert combined["segment_id"].to_list() == ["H1-0-4", "L1-0-4"]


class TestJoinAndSegmentId:
    def test_add_segment_id_custom_and_default_fmt(self):
        st = SegmentTable.from_segments(
            [Segment(100, 104), Segment(200, 204)], detector=["H1", "L1"]
        )
        with_id = st.add_segment_id(fmt="{detector}-{start:.0f}-{end:.0f}")
        assert with_id["segment_id"].to_list() == ["H1-100-104", "L1-200-204"]
        assert "segment_id" not in st.columns

        no_det = SegmentTable.from_segments([Segment(10, 15)]).add_segment_id()
        assert no_det["segment_id"].to_list() == ["10-15"]

    def test_join_requires_tolerance_on_span(self):
        left = SegmentTable.from_segments([Segment(0, 4), Segment(4, 8)], a=[1, 2])
        right = SegmentTable.from_segments([Segment(0.01, 4.01)], b=[10])

        with pytest.raises(ValueError, match="tolerance"):
            left.join(right, on="span")

        joined = left.join(right, on="span", how="left", tolerance=0.05)
        assert len(joined) == 2
        assert joined.row(0)["b"] == 10.0
        assert np.isnan(joined.row(1)["b"])

        inner = left.join(right, on="span", how="inner", tolerance=0.05)
        assert len(inner) == 1
        assert inner.row(0)["a"] == 1

    def test_join_on_keys_outer_suffixes_and_validate(self):
        left = SegmentTable.from_segments(
            [Segment(0, 4), Segment(4, 8)],
            detector=["H1", "L1"],
            snr=[5.0, 6.0],
        ).add_segment_id()
        right = SegmentTable.from_segments(
            [Segment(4, 8), Segment(8, 12)],
            detector=["L1", "V1"],
            snr=[7.0, 8.0],
        ).add_segment_id()
        right.add_series_column(
            "psd",
            data=[
                TimeSeries(np.ones(4), sample_rate=4),
                TimeSeries(np.ones(4) * 2, sample_rate=4),
            ],
            kind="timeseries",
        )

        outer = left.join(
            right,
            on="segment_id",
            how="outer",
            validate="1:1",
            suffixes=("_l", "_r"),
        )
        assert len(outer) == 3
        assert outer["segment_id"].to_list() == [
            "H1-0-4",
            "L1-4-8",
            "V1-8-12",
        ]
        assert outer._payload["psd"][0].status == CellStatus.MISSING
        assert outer._payload["psd"][1].status == CellStatus.VALID
        assert outer._payload["psd"][2].status == CellStatus.VALID
        assert outer.row(0)["snr_l"] == 5.0
        assert np.isnan(outer.row(0)["snr_r"])

        # validate failure on duplicate keys
        dup_right = SegmentTable.from_segments(
            [Segment(0, 4), Segment(0, 4)], key=["k", "k"], val=[1, 2]
        )
        single_left = SegmentTable.from_segments([Segment(0, 4)], key=["k"])
        with pytest.raises(ValueError, match="1:1"):
            single_left.join(dup_right, on="key", validate="1:1")
        with pytest.raises(ValueError, match="m:1"):
            single_left.join(dup_right, on="key", validate="m:1")
        # 1:m succeeds
        ok = single_left.join(dup_right, on="key", validate="1:m")
        assert len(ok) == 2
