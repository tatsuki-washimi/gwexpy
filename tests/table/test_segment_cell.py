"""Tests for SegmentCell."""

from __future__ import annotations

import pytest

from gwexpy.table.segment_cell import SegmentCell


class TestSegmentCellGet:
    def test_get_with_value(self):
        cell = SegmentCell(value=42)
        assert cell.get() == 42

    def test_get_with_loader(self):
        cell = SegmentCell(loader=lambda: 99)
        assert cell.get() == 99

    def test_get_caches_result(self):
        call_count = [0]

        def loader():
            call_count[0] += 1
            return "data"

        cell = SegmentCell(loader=loader, cacheable=True)
        assert cell.get() == "data"
        assert cell.get() == "data"
        assert call_count[0] == 1

    def test_get_no_cache(self):
        call_count = [0]

        def loader():
            call_count[0] += 1
            return "fresh"

        cell = SegmentCell(loader=loader, cacheable=False)
        cell.get()
        cell.get()
        assert call_count[0] == 2

    def test_get_empty_raises(self):
        cell = SegmentCell()
        with pytest.raises(ValueError, match="no value and no loader"):
            cell.get()

    def test_get_value_takes_priority(self):
        """If value is set, loader is never called."""
        loader_called = [False]

        def loader():
            loader_called[0] = True
            return "from_loader"

        cell = SegmentCell(value="from_value", loader=loader)
        assert cell.get() == "from_value"
        assert not loader_called[0]


class TestSegmentCellIsLoaded:
    def test_is_loaded_with_value(self):
        cell = SegmentCell(value=10)
        assert cell.is_loaded()

    def test_not_loaded_with_loader_only(self):
        cell = SegmentCell(loader=lambda: 0)
        assert not cell.is_loaded()

    def test_loaded_after_get(self):
        cell = SegmentCell(loader=lambda: "hello", cacheable=True)
        assert not cell.is_loaded()
        cell.get()
        assert cell.is_loaded()

    def test_not_loaded_after_get_no_cache(self):
        cell = SegmentCell(loader=lambda: "hi", cacheable=False)
        cell.get()
        assert not cell.is_loaded()


class TestSegmentCellClear:
    def test_clear_resets_cached_value(self):
        call_count = [0]

        def loader():
            call_count[0] += 1
            return "v"

        cell = SegmentCell(loader=loader, cacheable=True)
        cell.get()  # cache it
        assert call_count[0] == 1
        cell.clear()
        assert not cell.is_loaded()
        cell.get()  # should reload
        assert call_count[0] == 2

    def test_clear_noop_without_loader(self):
        cell = SegmentCell(value="permanent")
        cell.clear()  # should not remove value
        assert cell.get() == "permanent"


class TestSegmentCellSummary:
    def test_summary_lazy(self):
        cell = SegmentCell(loader=lambda: None)
        assert "lazy" in cell._summary("timeseries")

    def test_summary_empty(self):
        cell = SegmentCell()
        # _summary on empty cell
        assert "empty" in cell._summary()
