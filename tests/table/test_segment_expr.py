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
