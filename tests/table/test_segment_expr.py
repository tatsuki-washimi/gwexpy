"""Tests for S-3 col() AST expressions and SegmentColumn."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries

from gwexpy.table import CellStatus, SegmentCell, SegmentColumn, SegmentTable, col
from gwexpy.table.segment.expr import BinOp, Call, ColumnRef, Literal, UnaryOp


def _sample_table():
    segs = [Segment(0, 4), Segment(4, 10), Segment(10, 14), Segment(14, 20)]
    st = SegmentTable.from_segments(
        segs,
        snr=[3.0, 8.0, np.nan, 12.0],
        peak=[100.0, 35000.0, 200.0, 500.0],
        label=["noise", "glitch", "noise", None],
    )
    st.add_column("freq", [10.0, 20.0, 30.0, 40.0], unit=u.Hz)
    return st


class TestColExprAST:
    def test_ast_node_types_and_required_columns(self):
        c = col("snr")
        assert isinstance(c, ColumnRef)
        expr = (c > 5) & (col("peak") < 32700) & ~col("label").is_missing()
        assert isinstance(expr, BinOp)
        assert isinstance(Literal(5), Literal)
        assert isinstance(~col("label").is_missing(), UnaryOp)
        assert isinstance(col("label").is_missing(), Call)
        assert expr.required_columns() == frozenset({"snr", "peak", "label"})

    def test_bool_conversion_raises(self):
        with pytest.raises(TypeError, match="Cannot convert column expression"):
            bool(col("snr") > 5)

    def test_arithmetic_comparison_and_span_properties(self):
        st = _sample_table()
        mask = (col("snr") > 5) & (col("peak") < 32700)
        assert mask.evaluate(st).tolist() == [False, False, False, True]

        dur = col("span").duration.evaluate(st)
        assert dur.tolist() == [4.0, 6.0, 4.0, 6.0]
        assert col("span").start.evaluate(st).tolist() == [0.0, 4.0, 10.0, 14.0]
        assert col("span").end.evaluate(st).tolist() == [4.0, 10.0, 14.0, 20.0]

        ratio = (col("peak") / col("freq")).evaluate(st)
        assert ratio[0] == pytest.approx(10.0)
        assert ratio[3] == pytest.approx(12.5)

    def test_isin_between_contains(self):
        st = _sample_table()
        assert col("label").isin(["glitch"]).evaluate(st).tolist() == [
            False,
            True,
            False,
            False,
        ]
        assert col("snr").between(3.0, 10.0).evaluate(st).tolist() == [
            True,
            True,
            False,
            False,
        ]
        assert col("label").contains("gli").evaluate(st).tolist() == [
            False,
            True,
            False,
            False,
        ]

    def test_status_predicates_do_not_load_payload(self):
        calls = 0

        def _lazy():
            nonlocal calls
            calls += 1
            return TimeSeries(np.ones(16), sample_rate=16)

        segs = [Segment(i, i + 1) for i in range(5)]
        st = SegmentTable.from_segments(segs)
        st.add_series_column(
            "ts",
            data=[
                TimeSeries(np.ones(16), sample_rate=16),
                SegmentCell.missing(),
                SegmentCell.failed(RuntimeError("err")),
                SegmentCell(loader=_lazy),
                SegmentCell.masked(),
            ],
            kind="timeseries",
        )

        assert col("ts").is_valid().evaluate(st).tolist() == [
            True,
            False,
            False,
            False,
            False,
        ]
        assert col("ts").is_missing().evaluate(st).tolist() == [
            False,
            True,
            False,
            False,
            False,
        ]
        assert col("ts").is_failed().evaluate(st).tolist() == [
            False,
            False,
            True,
            False,
            False,
        ]
        assert col("ts").is_loaded().evaluate(st).tolist() == [
            True,
            False,
            False,
            False,
            False,
        ]
        assert col("ts").isna().evaluate(st).tolist() == [
            False,
            True,
            True,
            False,
            True,
        ]
        assert col("ts").notna().evaluate(st).tolist() == [
            True,
            False,
            False,
            True,
            False,
        ]
        assert calls == 0

    def test_payload_content_predicates_require_load_true(self):
        calls = 0

        def _lazy(val):
            def _load():
                nonlocal calls
                calls += 1
                return TimeSeries(np.array([1.0, val]), sample_rate=1.0)

            return _load

        st = SegmentTable.from_segments([Segment(0, 2), Segment(2, 4)])
        st.add_series_column(
            "ts",
            loader=[_lazy(5.0), _lazy(20.0)],
            kind="timeseries",
        )

        # Without load=True -> ValueError, no loader calls
        with pytest.raises(ValueError, match="load=True"):
            (col("ts").max() > 10.0).evaluate(st)
        with pytest.raises(ValueError, match="load=True"):
            col("ts").evaluate(st)
        assert calls == 0

        # With load=True -> evaluates properly
        res = (col("ts").max(load=True) > 10.0).evaluate(st)
        assert res.tolist() == [False, True]
        assert calls == 2


class TestSegmentColumn:
    def test_table_getitem_returns_segment_column(self):
        st = _sample_table()
        snr = st["snr"]
        assert isinstance(snr, SegmentColumn)
        assert snr.kind == "meta"
        assert st["freq"].unit == u.Hz
        assert snr.status.tolist() == [
            CellStatus.VALID,
            CellStatus.VALID,
            CellStatus.MISSING,
            CellStatus.VALID,
        ]
        assert snr.isna().tolist() == [False, False, True, False]
        assert snr.mean() == pytest.approx((3.0 + 8.0 + 12.0) / 3.0)
        assert snr.median() == pytest.approx(8.0)
        assert snr.quantile(0.5) == pytest.approx(8.0)
        assert np.isnan(snr.mean(skipna=False))
        assert snr.to_list()[0] == 3.0
        assert isinstance(snr.to_numpy(), np.ndarray)

    def test_table_getitem_rejects_expr_shortcut(self):
        st = _sample_table()
        with pytest.raises(TypeError, match="filter"):
            _ = st[col("snr") > 5]

    def test_payload_column_unit_and_no_unwanted_loading(self):
        calls = 0

        def _lazy():
            nonlocal calls
            calls += 1
            return TimeSeries([1.0, 2.0], sample_rate=1.0, unit="m")

        st = SegmentTable.from_segments([Segment(0, 2), Segment(2, 4)])
        st.add_series_column(
            "ts",
            data=[
                SegmentCell(value=TimeSeries([1.0, 2.0], sample_rate=1.0, unit="m")),
                SegmentCell(loader=_lazy),
            ],
            kind="timeseries",
        )
        c = st["ts"]
        assert c.unit == u.m
        assert c.status.tolist() == [CellStatus.VALID, CellStatus.UNLOADED]
        assert calls == 0

        with pytest.raises(ValueError, match="load=True"):
            c.to_numpy()
        with pytest.raises(ValueError, match="load=True"):
            c.to_list()
        assert calls == 0

        arr = c.to_numpy(load=True)
        assert len(arr) == 2
        assert calls == 1


class TestExprAndColumnCompleteness:
    def test_all_arithmetic_logical_and_unary_operators(self):
        st = _sample_table()
        assert ((col("freq") + 5.0) - 5.0).evaluate(st).tolist() == [
            10.0,
            20.0,
            30.0,
            40.0,
        ]
        assert ((5.0 + col("freq")) - (col("freq") * 1.0)).evaluate(st).tolist() == [
            5.0
        ] * 4
        assert (100.0 - col("freq")).evaluate(st).tolist() == [90.0, 80.0, 70.0, 60.0]
        assert (2.0 * col("freq")).evaluate(st).tolist() == [20.0, 40.0, 60.0, 80.0]
        assert (200.0 / col("freq")).evaluate(st).tolist() == [
            20.0,
            10.0,
            pytest.approx(200.0 / 30.0),
            5.0,
        ]
        assert (col("freq") // 15.0).evaluate(st).tolist() == [0.0, 1.0, 2.0, 2.0]
        assert (35.0 // col("freq")).evaluate(st).tolist() == [3.0, 1.0, 1.0, 0.0]
        assert (col("freq") % 15.0).evaluate(st).tolist() == [10.0, 5.0, 0.0, 10.0]
        assert (35.0 % col("freq")).evaluate(st).tolist() == [5.0, 15.0, 5.0, 35.0]
        assert (col("freq") ** 2).evaluate(st).tolist() == [100.0, 400.0, 900.0, 1600.0]
        assert (2.0 ** (col("freq") / 10.0)).evaluate(st).tolist() == [
            2.0,
            4.0,
            8.0,
            16.0,
        ]

        assert (-col("freq")).evaluate(st).tolist() == [-10.0, -20.0, -30.0, -40.0]
        assert (+col("freq")).evaluate(st).tolist() == [10.0, 20.0, 30.0, 40.0]
        assert abs(-col("freq")).evaluate(st).tolist() == [10.0, 20.0, 30.0, 40.0]
        assert col("freq").abs().evaluate(st).tolist() == [10.0, 20.0, 30.0, 40.0]

        assert ((col("freq") == 10.0) | (col("freq") >= 40.0)).evaluate(
            st
        ).tolist() == [
            True,
            False,
            False,
            True,
        ]
        assert (col("freq") <= 20.0).evaluate(st).tolist() == [True, True, False, False]
        assert (col("label") != "noise").evaluate(st).tolist() == [
            False,
            True,
            False,
            False,
        ]
        assert Literal(True).evaluate(st).tolist() == [True] * 4
        assert Literal("x").evaluate(st).tolist() == ["x"] * 4
        assert "col('freq')" in repr(col("freq"))

    def test_between_variants_status_on_meta_and_payload_reductions(self):
        st = _sample_table()
        assert col("freq").between(10.0, 30.0, inclusive="neither").evaluate(
            st
        ).tolist() == [
            False,
            True,
            False,
            False,
        ]
        assert col("freq").between(10.0, 30.0, inclusive="left").evaluate(
            st
        ).tolist() == [
            True,
            True,
            False,
            False,
        ]
        assert col("freq").between(10.0, 30.0, inclusive="right").evaluate(
            st
        ).tolist() == [
            False,
            True,
            True,
            False,
        ]
        with pytest.raises(ValueError, match="inclusive"):
            col("freq").between(10.0, 30.0, inclusive="bad").evaluate(st)

        # Status predicates on meta columns
        assert col("snr").is_valid().evaluate(st).tolist() == [True, True, False, True]
        assert col("snr").is_missing().evaluate(st).tolist() == [
            False,
            False,
            True,
            False,
        ]
        assert col("snr").is_failed().evaluate(st).tolist() == [False] * 4
        assert col("snr").is_loaded().evaluate(st).tolist() == [True] * 4
        assert col("snr").isna().evaluate(st).tolist() == [False, False, True, False]
        assert col("snr").notna().evaluate(st).tolist() == [True, True, False, True]

        # Payload reductions with load=True
        st.add_series_column(
            "ts",
            data=[
                TimeSeries([1.0, 3.0, 5.0], sample_rate=1.0),
                SegmentCell.missing(),
                TimeSeries([2.0, 4.0], sample_rate=1.0),
                TimeSeries([10.0, 20.0], sample_rate=1.0),
            ],
            kind="timeseries",
        )
        assert col("ts").min(load=True).evaluate(st)[0] == pytest.approx(1.0)
        assert col("ts").mean(load=True).evaluate(st)[0] == pytest.approx(3.0)
        assert col("ts").median(load=True).evaluate(st)[0] == pytest.approx(3.0)
        assert col("ts").std(load=True).evaluate(st)[2] == pytest.approx(1.0)
        assert col("ts").sum(load=True).evaluate(st)[3] == pytest.approx(30.0)
        assert col("ts").quantile(0.5, load=True).evaluate(st)[0] == pytest.approx(3.0)
        assert col("ts").apply(lambda ts: len(ts), load=True).evaluate(st)[0] == 3
        assert np.isnan(col("ts").mean(load=True).evaluate(st)[1])

    def test_segment_column_indexing_stats_and_empty(self):
        st = _sample_table()
        freq = st["freq"]
        assert freq.name == "freq"
        assert "SegmentColumn" in repr(freq)
        assert freq[-1] == 40.0
        assert freq[1:3] == [20.0, 30.0]
        assert list(freq) == [10.0, 20.0, 30.0, 40.0]
        assert freq.min() == 10.0
        assert freq.max() == 40.0
        assert freq.std() == pytest.approx(np.std([10.0, 20.0, 30.0, 40.0]))
        with pytest.raises(IndexError):
            _ = freq[99]
        with pytest.raises(TypeError, match="not numeric"):
            st["label"].mean()

        empty = SegmentTable.from_segments([], x=[])
        assert np.isnan(empty["x"].mean())
        assert np.isnan(empty["x"].median())
        assert np.isnan(empty["x"].quantile(0.5))
        assert np.isnan(empty["x"].min())
        assert np.isnan(empty["x"].max())
        assert np.isnan(empty["x"].std())
