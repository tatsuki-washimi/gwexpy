"""Tests for S-1 explicit CellStatus state model and SegmentCell behavior."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.segments import Segment

from gwexpy.table import CellLoadError, CellStatus, SegmentCell, SegmentTable


class TestCellStatusModel:
    def test_valid_explicit_none_payload(self):
        cell = SegmentCell(value=None, status=CellStatus.VALID)
        assert cell.status == CellStatus.VALID
        assert cell.is_loaded()
        assert cell.get() is None
        assert cell.get_or_none() is None
        assert cell._summary() == "None"

        cell2 = SegmentCell.valid(None)
        assert cell2.status == CellStatus.VALID
        assert cell2.get() is None

    def test_loader_returning_none_transitions_to_valid(self):
        calls = 0

        def _load():
            nonlocal calls
            calls += 1
            return None

        cell = SegmentCell(loader=_load)
        assert cell.status == CellStatus.UNLOADED
        assert cell.get() is None
        assert cell.status == CellStatus.VALID
        assert cell.is_loaded()
        assert cell.get() is None
        assert calls == 1

    def test_legacy_value_none_remains_unloaded(self):
        cell = SegmentCell(value=None)
        assert cell.status == CellStatus.UNLOADED
        assert not cell.is_loaded()
        with pytest.raises(ValueError, match="no value and no loader"):
            cell.get()
        assert cell.get(default=123) == 123
        assert cell.get_or_none() is None
        assert cell._summary() == "<empty>"

    def test_missing_masked_not_applicable_states(self):
        for cell, tag in [
            (SegmentCell.missing(), "<missing>"),
            (SegmentCell.masked(), "<masked>"),
            (SegmentCell.not_applicable(), "<not_applicable>"),
        ]:
            assert not cell.is_loaded()
            assert cell.get() is None
            assert cell.get(default="fallback") == "fallback"
            assert cell.get_or_none() is None
            assert cell._summary() == tag
            cell.clear()
            assert cell._summary() == tag

    def test_failed_constructor_and_cell_load_error(self):
        orig = OSError("missing frame")
        cell = SegmentCell.failed(orig)
        assert cell.status == CellStatus.FAILED
        assert cell.error is orig
        assert not cell.is_loaded()
        assert cell._summary() == "<failed: OSError>"
        assert cell.get(default=0.0) == 0.0
        assert cell.get_or_none() is None

        with pytest.raises(CellLoadError) as excinfo:
            cell.get()
        assert excinfo.value.__cause__ is orig

    def test_loader_exception_transitions_to_failed_and_reraises(self):
        attempts = 0

        def _failing_loader():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("boom")
            return 42

        cell = SegmentCell(loader=_failing_loader)
        with pytest.raises(RuntimeError, match="boom"):
            cell.get()
        assert cell.status == CellStatus.FAILED
        assert isinstance(cell.error, RuntimeError)
        assert not cell.is_loaded()
        assert cell._summary() == "<failed: RuntimeError>"

        # Subsequent get() raises CellLoadError without re-running loader
        with pytest.raises(CellLoadError) as excinfo:
            cell.get()
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert attempts == 1
        assert cell.get(default=-1) == -1
        assert cell.get_or_none() is None

        # clear() resets to UNLOADED so loader can be retried
        cell.clear()
        assert cell.status == CellStatus.UNLOADED
        assert cell.error is None
        assert cell.get() == 42
        assert cell.status == CellStatus.VALID
        assert attempts == 2

    def test_unloaded_loader_with_default_catches_exception(self):
        def _boom():
            raise ValueError("bad data")

        cell = SegmentCell(loader=_boom)
        assert cell.get(default="fallback") == "fallback"
        assert cell.status == CellStatus.FAILED
        assert isinstance(cell.error, ValueError)

    def test_clone_isolates_status(self):
        cell = SegmentCell(loader=lambda: 7)
        cloned = cell.clone()
        assert cloned is not cell
        assert cloned.get() == 7
        assert cloned.status == CellStatus.VALID
        assert cell.status == CellStatus.UNLOADED


class TestToPandasAndCopyWithStatus:
    def test_to_pandas_missing_as_summary_and_nan(self):
        st = SegmentTable.from_segments(
            [Segment(0, 1), Segment(1, 2), Segment(2, 3), Segment(3, 4)]
        )
        lazy_called = False

        def _lazy():
            nonlocal lazy_called
            lazy_called = True
            return 99

        st.add_series_column(
            "val",
            data=[
                10.0,
                SegmentCell.missing(),
                SegmentCell.failed(RuntimeError("x")),
                SegmentCell(loader=_lazy),
            ],
            kind="object",
        )

        df_summary = st.to_pandas(meta_only=False, missing_as="summary")
        assert df_summary["val"].tolist() == [
            10.0,
            "<missing>",
            "<failed: RuntimeError>",
            "<lazy: object>",
        ]
        assert not lazy_called

        df_nan = st.to_pandas(meta_only=False, missing_as="nan")
        assert df_nan.loc[0, "val"] == 10.0
        assert np.isnan(df_nan.loc[1, "val"])
        assert np.isnan(df_nan.loc[2, "val"])
        assert np.isnan(df_nan.loc[3, "val"])
        assert not lazy_called

        with pytest.raises(ValueError, match="missing_as"):
            st.to_pandas(meta_only=False, missing_as="invalid")

    def test_copy_preserves_statuses_and_valid_none(self):
        st = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
        st.add_series_column(
            "val",
            data=[
                SegmentCell.valid(None),
                SegmentCell.missing(),
                SegmentCell.failed(ValueError("bad")),
            ],
            kind="object",
        )
        copied = st.copy(deep=True)
        assert copied._payload["val"][0].status == CellStatus.VALID
        assert copied._payload["val"][0].get() is None
        assert copied._payload["val"][1].status == CellStatus.MISSING
        assert copied._payload["val"][2].status == CellStatus.FAILED
        assert isinstance(copied._payload["val"][2].error, ValueError)
