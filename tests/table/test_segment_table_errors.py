"""Tests for S-2 errors= policies and missing-data reporting/selection."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries

from gwexpy.table import CellStatus, SegmentCell, SegmentTable


def _make_ts_table():
    segs = [Segment(0, 4), Segment(4, 8), Segment(8, 12)]
    ts_ok1 = TimeSeries(np.ones(64), sample_rate=16, t0=0)
    ts_ok2 = TimeSeries(np.ones(64) * 2.0, sample_rate=16, t0=8)

    def _fail_loader():
        raise OSError("frame missing")

    st = SegmentTable.from_segments(segs, snr=[10.0, np.nan, 20.0])
    st.add_series_column(
        "raw",
        data=[
            SegmentCell(value=ts_ok1),
            SegmentCell(loader=_fail_loader),
            SegmentCell(value=ts_ok2),
        ],
        kind="timeseries",
    )
    return st


class TestApplyErrorsPolicy:
    def test_invalid_errors_raises_value_error(self):
        st = SegmentTable.from_segments([Segment(0, 4)])
        with pytest.raises(ValueError, match="Invalid errors policy"):
            st.apply(lambda r: {"x": 1}, errors="ignore")

    def test_apply_raise_default(self):
        st = SegmentTable.from_segments([Segment(0, 4), Segment(4, 8)], v=[1.0, 0.0])
        with pytest.raises(ZeroDivisionError):
            st.apply(lambda r: {"inv": 1.0 / float(r["v"])})

    def test_apply_record_emits_companion_columns_meta(self):
        st = SegmentTable.from_segments([Segment(0, 4), Segment(4, 8)], v=[2.0, 0.0])
        out = st.apply(lambda r: {"inv": 1.0 / float(r["v"])}, errors="record")

        assert len(out) == 2
        assert out.row(0)["inv"] == pytest.approx(0.5)
        assert np.isnan(out.row(1)["inv"])
        assert out.row(0)["inv_status"] == "valid"
        assert out.row(1)["inv_status"] == "failed"
        assert out.row(0)["inv_error_type"] == ""
        assert out.row(1)["inv_error_type"] == "ZeroDivisionError"
        assert out.row(0)["inv_error_message"] == ""
        assert "division by zero" in out.row(1)["inv_error_message"]
        assert len(out.dropna()) == 1
        assert len(out.select_valid()) == 1
        assert len(out.select_failed()) == 1
        assert len(out.select_missing()) == 0

    def test_apply_record_payload_column(self):
        st = _make_ts_table()
        out = st.apply(
            lambda r: {"scaled": r["raw"] * 2.0},
            out_cols=["scaled"],
            errors="record",
        )
        assert out._payload["scaled"][0].status == CellStatus.VALID
        assert out._payload["scaled"][1].status == CellStatus.FAILED
        assert isinstance(out._payload["scaled"][1].error, OSError)
        assert out._payload["scaled"][2].status == CellStatus.VALID
        assert out.row(1)["scaled_status"] == "failed"
        assert out.row(1)["scaled_error_type"] == "OSError"
        assert "frame missing" in out.row(1)["scaled_error_message"]

    def test_apply_coerce_marks_missing(self):
        st = _make_ts_table()
        out = st.apply(lambda r: {"scaled": r["raw"] * 2.0}, errors="coerce")
        assert out._payload["scaled"][0].status == CellStatus.VALID
        assert out._payload["scaled"][1].status == CellStatus.MISSING
        assert out._payload["scaled"][2].status == CellStatus.VALID

    def test_apply_skip_drops_failing_rows(self):
        st = _make_ts_table()
        out = st.apply(lambda r: {"scaled": r["raw"] * 2.0}, errors="skip")
        assert len(out) == 2
        assert [out.row(i)["span"] for i in range(2)] == [
            Segment(0, 4),
            Segment(8, 12),
        ]
        assert out._payload["scaled"][0].status == CellStatus.VALID
        assert out._payload["scaled"][1].status == CellStatus.VALID

    def test_apply_all_fail_with_out_cols_record(self):
        st = SegmentTable.from_segments([Segment(0, 4), Segment(4, 8)])

        def _boom(r):
            raise RuntimeError("always fails")

        out = st.apply(_boom, out_cols=["res"], errors="record")
        assert len(out) == 2
        assert all(c.status == CellStatus.FAILED for c in out._payload["res"])
        assert list(out._meta["res_status"]) == ["failed", "failed"]


class TestSugarAndFetchErrorsPolicy:
    def test_crop_errors_policies(self):
        st = _make_ts_table()
        rec = st.crop("raw", errors="record")
        assert rec._payload["raw_cropped"][1].status == CellStatus.FAILED
        assert rec.row(1)["raw_cropped_error_type"] == "OSError"

        coerced = st.crop("raw", errors="coerce")
        assert coerced._payload["raw_cropped"][1].status == CellStatus.MISSING

        skipped = st.crop("raw", errors="skip")
        assert len(skipped) == 2

    def test_asd_errors_policies(self):
        st = _make_ts_table()
        # Use fftlength=2.0 on 4s series
        rec = st.asd("raw", fftlength=2.0, errors="record")
        assert rec._payload["raw_asd"][0].status == CellStatus.VALID
        assert rec._payload["raw_asd"][1].status == CellStatus.FAILED
        assert rec.row(1)["raw_asd_error_type"] == "OSError"

        coerced = st.asd("raw", fftlength=2.0, errors="coerce")
        assert coerced._payload["raw_asd"][1].status == CellStatus.MISSING

        # Chaining crop(coerce) -> asd(raise) preserves MISSING cell without failing
        coerced_crop = st.crop("raw", errors="coerce")
        chained = coerced_crop.asd("raw_cropped", fftlength=2.0, errors="raise")
        assert chained._payload["raw_cropped_asd"][1].status == CellStatus.MISSING

        skipped = st.asd("raw", fftlength=2.0, errors="skip")
        assert len(skipped) == 2

    def test_fetch_and_materialize_errors_policies(self):
        st = _make_ts_table()
        # materialize(inplace=False, errors="record") leaves original untouched
        rec = st.materialize(inplace=False, errors="record")
        assert rec is not None
        assert st._payload["raw"][1].status == CellStatus.UNLOADED
        assert rec._payload["raw"][1].status == CellStatus.FAILED
        assert rec.row(1)["raw_error_type"] == "OSError"

        coerced = st.materialize(inplace=False, errors="coerce")
        assert coerced is not None
        assert coerced._payload["raw"][1].status == CellStatus.MISSING

        skipped = st.materialize(inplace=False, errors="skip")
        assert skipped is not None
        assert len(skipped) == 2


class TestMissingReportingAndSelection:
    def test_missing_report_does_not_load(self):
        calls = 0

        def _lazy():
            nonlocal calls
            calls += 1
            return TimeSeries(np.ones(16), sample_rate=16, t0=0)

        segs = [Segment(i * 4, i * 4 + 4) for i in range(5)]
        st = SegmentTable.from_segments(segs, snr=[1.0, np.nan, 3.0, 4.0, 5.0])
        st.add_series_column(
            "ts",
            data=[
                TimeSeries(np.ones(16), sample_rate=16, t0=0),
                SegmentCell.missing(),
                SegmentCell.failed(ValueError("bad")),
                SegmentCell(loader=_lazy),
                SegmentCell.masked(),
            ],
            kind="timeseries",
        )
        rep = st.missing_report()
        assert calls == 0
        ts_row = rep[rep["column"] == "ts"].iloc[0]
        assert ts_row["total"] == 5
        assert ts_row["valid"] == 1
        assert ts_row["missing"] == 1
        assert ts_row["failed"] == 1
        assert ts_row["unloaded"] == 1
        assert ts_row["masked"] == 1
        assert ts_row["not_applicable"] == 0

        snr_row = rep[rep["column"] == "snr"].iloc[0]
        assert snr_row["valid"] == 4
        assert snr_row["missing"] == 1

    def test_isna_shallow_and_deep(self):
        calls = 0

        def _lazy_nan():
            nonlocal calls
            calls += 1
            return TimeSeries([1.0, np.nan, 3.0], sample_rate=1.0, t0=0)

        segs = [Segment(0, 3), Segment(3, 6), Segment(6, 9), Segment(9, 12)]
        st = SegmentTable.from_segments(segs, snr=[1.0, np.nan, 3.0, 4.0])
        st.add_series_column(
            "ts",
            data=[
                SegmentCell(value=TimeSeries([1.0, 2.0], sample_rate=1.0)),
                SegmentCell.missing(),
                SegmentCell.failed(RuntimeError("e")),
                SegmentCell(loader=_lazy_nan),
            ],
            kind="timeseries",
        )
        na_shallow = st.isna()
        assert calls == 0
        assert na_shallow["snr"].tolist() == [False, True, False, False]
        assert na_shallow["ts"].tolist() == [False, True, True, False]

        na_no_failed = st.isna(include_failed=False)
        assert na_no_failed["ts"].tolist() == [False, True, False, False]

        na_deep = st.isna(columns=["ts"], deep=True)
        assert calls == 1
        assert na_deep["ts"].tolist() == [False, True, True, True]

    def test_select_valid_missing_failed_and_dropna(self):
        calls = 0

        def _lazy():
            nonlocal calls
            calls += 1
            return TimeSeries(np.ones(16), sample_rate=16)

        segs = [Segment(i, i + 1) for i in range(4)]
        st = SegmentTable.from_segments(segs, snr=[1.0, 2.0, np.nan, 4.0])
        st.add_series_column(
            "ts",
            data=[
                SegmentCell(value=1.0),
                SegmentCell.missing(),
                SegmentCell.failed(RuntimeError("fail")),
                SegmentCell(loader=_lazy),
            ],
            kind="object",
        )

        # UNLOADED is not VALID, so only row 0 is valid
        valid = st.select_valid()
        assert calls == 0
        assert len(valid) == 1
        assert valid.row(0)["span"] == Segment(0, 1)

        missing = st.select_missing()
        assert len(missing) == 2  # row 1 (ts missing) and row 2 (snr nan)

        failed = st.select_failed()
        assert len(failed) == 1
        assert failed.row(0)["span"] == Segment(2, 3)

        # dropna(load=False) keeps UNLOADED (row 3) and VALID (row 0)
        dropped = st.dropna(load=False)
        assert calls == 0
        assert [dropped.row(i)["span"] for i in range(len(dropped))] == [
            Segment(0, 1),
            Segment(3, 4),
        ]

        # dropna(how="all") keeps row 1 (snr valid) and drops row 2 (both snr and ts missing/failed)
        dropped_all = st.dropna(how="all", load=False)
        assert [dropped_all.row(i)["span"] for i in range(len(dropped_all))] == [
            Segment(0, 1),
            Segment(1, 2),
            Segment(3, 4),
        ]

        # dropna(load=True) resolves row 3's loader without mutating original table
        dropped_loaded = st.dropna(load=True)
        assert calls == 1
        assert st._payload["ts"][3].status == CellStatus.UNLOADED
        assert dropped_loaded._payload["ts"][1].status == CellStatus.VALID
