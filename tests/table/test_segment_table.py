"""Tests for SegmentTable (core + display + plot)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from gwexpy.table.segment_cell import SegmentCell
from gwexpy.table.segment_table import RowProxy, SegmentTable

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_segments(n: int = 4) -> list[Segment]:
    return [Segment(i * 4, i * 4 + 4) for i in range(n)]


def _make_simple_st(n: int = 4) -> SegmentTable:
    segs = _make_segments(n)
    return SegmentTable.from_segments(segs, label=[f"label_{i}" for i in range(n)])


def _make_ts(duration: float = 4.0, sample_rate: float = 16.0) -> TimeSeries:
    data = np.random.randn(int(duration * sample_rate))
    return TimeSeries(data, sample_rate=sample_rate)


def _make_fs(n_freqs: int = 64) -> FrequencySeries:
    data = np.abs(np.random.randn(n_freqs)) + 1e-20
    return FrequencySeries(data, df=1.0)


# ---------------------------------------------------------------------------
# TestSegmentTableInit
# ---------------------------------------------------------------------------


class TestSegmentTableInit:
    def test_init_basic(self):
        segs = _make_segments(3)
        df = pd.DataFrame({"span": segs})
        st = SegmentTable(df)
        assert len(st) == 3
        assert "span" in st.schema
        assert st.schema["span"] == "segment"

    def test_init_no_span_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="'span'"):
            SegmentTable(df)

    def test_init_bad_span_type_raises(self):
        df = pd.DataFrame({"span": [1, 2, 3]})
        with pytest.raises(TypeError, match="gwpy.segments.Segment"):
            SegmentTable(df)

    def test_init_resets_index(self):
        segs = _make_segments(4)
        df = pd.DataFrame({"span": segs}, index=[10, 20, 30, 40])
        st = SegmentTable(df)
        # Internal index should be 0-based
        assert st._meta.index.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# TestSegmentTableFactory
# ---------------------------------------------------------------------------


class TestSegmentTableFactory:
    def test_from_segments_basic(self):
        segs = _make_segments(3)
        st = SegmentTable.from_segments(segs, label=["a", "b", "c"])
        assert len(st) == 3
        assert "label" in st.columns

    def test_from_segments_empty(self):
        st = SegmentTable.from_segments([])
        assert len(st) == 0

    def test_from_segments_length_mismatch_raises(self):
        segs = _make_segments(3)
        with pytest.raises(ValueError, match="Length of 'label'"):
            SegmentTable.from_segments(segs, label=["x", "y"])

    def test_from_table_dataframe(self):
        segs = _make_segments(2)
        df = pd.DataFrame({"span": segs, "snr": [5.0, 6.0]})
        st = SegmentTable.from_table(df)
        assert len(st) == 2
        assert "snr" in st.columns

    def test_from_table_custom_span_name(self):
        segs = _make_segments(2)
        df = pd.DataFrame({"interval": segs})
        st = SegmentTable.from_table(df, span="interval")
        assert "span" in st.columns

    def test_from_table_missing_span_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Column 'span' not found"):
            SegmentTable.from_table(df)


# ---------------------------------------------------------------------------
# TestSegmentTableColumns
# ---------------------------------------------------------------------------


class TestSegmentTableColumns:
    def test_add_column_meta(self):
        st = _make_simple_st(3)
        st.add_column("snr", [1.0, 2.0, 3.0], kind="meta")
        assert "snr" in st.columns
        assert st.schema["snr"] == "meta"

    def test_add_column_duplicate_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(ValueError, match="already exists"):
            st.add_column("span", [None, None])

    def test_add_column_bad_kind_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(ValueError, match="add_column only accepts"):
            st.add_column("foo", [1, 2], kind="timeseries")

    def test_add_column_length_mismatch_raises(self):
        st = _make_simple_st(3)
        with pytest.raises(ValueError, match="Length of data"):
            st.add_column("x", [1, 2])

    def test_add_series_column_with_data(self):
        st = _make_simple_st(3)
        ts_list = [_make_ts() for _ in range(3)]
        st.add_series_column("raw", data=ts_list, kind="timeseries")
        assert "raw" in st.schema
        assert st.schema["raw"] == "timeseries"

    def test_add_series_column_with_loaders(self):
        st = _make_simple_st(2)
        loaders = [lambda: _make_ts(), lambda: _make_ts()]
        st.add_series_column("raw", loader=loaders, kind="timeseries")
        assert "raw" in st.schema

    def test_add_series_column_no_data_no_loader_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(ValueError, match="At least one of"):
            st.add_series_column("raw", kind="timeseries")

    def test_add_series_column_bad_kind_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(ValueError, match="Invalid payload kind"):
            st.add_series_column("raw", data=["a", "b"], kind="invalid")


# ---------------------------------------------------------------------------
# TestSegmentTableAccess
# ---------------------------------------------------------------------------


class TestSegmentTableAccess:
    def test_len(self):
        st = _make_simple_st(5)
        assert len(st) == 5

    def test_columns_order(self):
        st = _make_simple_st(2)
        cols = st.columns
        assert cols[0] == "span"
        assert "label" in cols

    def test_schema_keys(self):
        st = _make_simple_st(2)
        assert st.schema["span"] == "segment"
        assert st.schema["label"] == "meta"

    def test_row_meta_access(self):
        st = _make_simple_st(3)
        assert st.row(0)["span"] == Segment(0, 4)
        assert st.row(0)["label"] == "label_0"

    def test_row_payload_access(self):
        st = _make_simple_st(2)
        ts_list = [_make_ts() for _ in range(2)]
        st.add_series_column("raw", data=ts_list, kind="timeseries")
        result = st.row(0)["raw"]
        assert isinstance(result, TimeSeries)

    def test_row_out_of_range_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(IndexError):
            st.row(10)

    def test_row_unknown_key_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(KeyError):
            st.row(0)["nonexistent"]

    def test_row_negative_index(self):
        st = _make_simple_st(4)
        assert st.row(-1)["span"] == Segment(12, 16)


# ---------------------------------------------------------------------------
# TestSegmentTableApply
# ---------------------------------------------------------------------------


class TestSegmentTableApply:
    def test_apply_basic(self):
        st = _make_simple_st(3)

        def func(row):
            span = row["span"]
            return {"duration": span[1] - span[0]}

        st2 = st.apply(func)
        assert "duration" in st2.schema
        assert st2.row(0)["duration"] == 4

    def test_apply_returns_non_dict_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(TypeError, match="dict"):
            st.apply(lambda row: 42)

    def test_apply_out_cols_mismatch_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(ValueError, match="out_cols"):
            st.apply(lambda row: {"a": 1}, out_cols=["b"])

    def test_apply_inplace(self):
        st = _make_simple_st(2)
        result = st.apply(lambda row: {"x": 1}, inplace=True)
        assert result is st
        assert "x" in st.schema

    def test_apply_parallel_falls_back(self):
        """parallel=True should still produce correct results (sequential fallback)."""
        st = _make_simple_st(3)
        st2 = st.apply(lambda row: {"tag": row["label"]}, parallel=True)
        assert st2.row(0)["tag"] == "label_0"

    def test_map_basic(self):
        st = _make_simple_st(3)
        st2 = st.map("label", lambda x: x.upper())
        assert st2.row(0)["label_mapped"] == "LABEL_0"


# ---------------------------------------------------------------------------
# TestSegmentTableSugar
# ---------------------------------------------------------------------------


class TestSegmentTableSugar:
    def test_crop_timeseries(self):
        segs = [Segment(0, 4)]
        st = SegmentTable.from_segments(segs)
        ts = TimeSeries(np.random.randn(256), sample_rate=64, t0=0)
        st.add_series_column("raw", data=[ts], kind="timeseries")
        st2 = st.crop("raw", out_col="cropped")
        result = st2.row(0)["cropped"]
        assert isinstance(result, TimeSeries)

    def test_crop_invalid_kind_raises(self):
        st = _make_simple_st(2)
        with pytest.raises(TypeError, match="crop()"):
            st.crop("label")

    def test_crop_invalid_payload_type_raises(self):
        """crop() with timeseries kind but object that doesn't have .crop()."""
        segs = [Segment(0, 4)]
        st = SegmentTable.from_segments(segs)
        # Add an object that isn't a TimeSeries
        st._payload["raw"] = [SegmentCell(value="not_a_ts")]
        st._schema["raw"] = "timeseries"
        with pytest.raises(TypeError):
            st.crop("raw")


# ---------------------------------------------------------------------------
# TestSegmentTableSelect
# ---------------------------------------------------------------------------


class TestSegmentTableSelect:
    def test_select_mask(self):
        st = _make_simple_st(4)
        mask = [True, False, True, False]
        st2 = st.select(mask=mask)
        assert len(st2) == 2
        assert st2.row(0)["label"] == "label_0"
        assert st2.row(1)["label"] == "label_2"

    def test_select_condition(self):
        st = _make_simple_st(4)
        st2 = st.select(label="label_2")
        assert len(st2) == 1
        assert st2.row(0)["label"] == "label_2"

    def test_select_preserves_span(self):
        st = _make_simple_st(4)
        st2 = st.select(mask=[True, False, False, True])
        assert "span" in st2.schema

    def test_select_preserves_payload(self):
        st = _make_simple_st(4)
        ts_list = [_make_ts() for _ in range(4)]
        st.add_series_column("raw", data=ts_list, kind="timeseries")
        mask = [True, False, True, False]
        st2 = st.select(mask=mask)
        assert "raw" in st2.schema
        assert len(st2._payload["raw"]) == 2

    def test_select_mask_wrong_length_raises(self):
        st = _make_simple_st(4)
        with pytest.raises(ValueError, match="mask length"):
            st.select(mask=[True, False])

    def test_select_unknown_column_raises(self):
        st = _make_simple_st(4)
        with pytest.raises(KeyError):
            st.select(nonexistent="foo")


# ---------------------------------------------------------------------------
# TestSegmentTableConversion
# ---------------------------------------------------------------------------


class TestSegmentTableConversion:
    def test_to_pandas_meta_only(self):
        st = _make_simple_st(3)
        df = st.to_pandas(meta_only=True)
        assert isinstance(df, pd.DataFrame)
        assert "span" in df.columns

    def test_to_pandas_full(self):
        st = _make_simple_st(2)
        ts_list = [_make_ts() for _ in range(2)]
        st.add_series_column("raw", data=ts_list, kind="timeseries")
        df = st.to_pandas(meta_only=False)
        assert "raw" in df.columns

    def test_copy_shallow(self):
        st = _make_simple_st(3)
        ts_list = [_make_ts() for _ in range(3)]
        st.add_series_column("raw", data=ts_list, kind="timeseries")
        st2 = st.copy(deep=False)
        assert len(st2) == len(st)
        # Same cell object references
        assert st2._payload["raw"][0] is not st._payload["raw"][0]  # new SegmentCell wrapper
        assert st2._payload["raw"][0].value is st._payload["raw"][0].value  # same value ref

    def test_copy_deep(self):
        st = _make_simple_st(2)
        st2 = st.copy(deep=True)
        assert len(st2) == 2

    def test_fetch_loads_all_cells(self):
        st = _make_simple_st(3)
        loaders = [lambda: _make_ts() for _ in range(3)]
        st.add_series_column("raw", loader=loaders, kind="timeseries")
        assert not st._payload["raw"][0].is_loaded()
        st.fetch()
        assert st._payload["raw"][0].is_loaded()

    def test_materialize_inplace(self):
        st = _make_simple_st(2)
        loaders = [lambda: _make_ts(), lambda: _make_ts()]
        st.add_series_column("raw", loader=loaders, kind="timeseries")
        result = st.materialize(inplace=True)
        assert result is None
        assert st._payload["raw"][0].is_loaded()

    def test_materialize_returns_copy(self):
        st = _make_simple_st(2)
        loaders = [lambda: _make_ts(), lambda: _make_ts()]
        st.add_series_column("raw", loader=loaders, kind="timeseries")
        result = st.materialize(inplace=False)
        assert isinstance(result, SegmentTable)
        assert result is not st

    def test_clear_cache(self):
        st = _make_simple_st(2)
        loaders = [lambda: _make_ts(), lambda: _make_ts()]
        st.add_series_column("raw", loader=loaders, kind="timeseries")
        st.fetch()
        assert st._payload["raw"][0].is_loaded()
        st.clear_cache()
        assert not st._payload["raw"][0].is_loaded()


# ---------------------------------------------------------------------------
# TestSegmentTableDisplay
# ---------------------------------------------------------------------------


class TestSegmentTableDisplay:
    def test_repr(self):
        st = _make_simple_st(3)
        r = repr(st)
        assert "SegmentTable" in r
        assert "n_rows=3" in r

    def test_str(self):
        st = _make_simple_st(3)
        s = str(st)
        assert "span" in s

    def test_repr_html(self):
        st = _make_simple_st(3)
        html = st._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html.lower() or "<td" in html.lower()

    def test_display_returns_dataframe(self):
        st = _make_simple_st(3)
        result = st.display()
        assert isinstance(result, pd.DataFrame)

    def test_display_meta_only(self):
        st = _make_simple_st(2)
        ts_list = [_make_ts() for _ in range(2)]
        st.add_series_column("raw", data=ts_list, kind="timeseries")
        result = st.display(meta_only=True)
        assert "raw" not in result.columns

    def test_repr_payload_lazy(self):
        st = _make_simple_st(2)
        loaders = [lambda: _make_ts(), lambda: _make_ts()]
        st.add_series_column("raw", loader=loaders, kind="timeseries")
        s = str(st)
        assert "lazy" in s


# ---------------------------------------------------------------------------
# TestSegmentTablePlot
# ---------------------------------------------------------------------------


class TestSegmentTablePlot:
    """Tests for drawing API.

    We mock gwpy.plot.Plot so tests can run without a display.
    """

    @pytest.fixture(autouse=True)
    def matplotlib_backend(self):
        import matplotlib
        matplotlib.use("Agg")

    def _make_st_with_fs(self, n: int = 3) -> SegmentTable:
        segs = _make_segments(n)
        st = SegmentTable.from_segments(segs, snr=[float(i) for i in range(n)])
        fs_list = [_make_fs() for _ in range(n)]
        st.add_series_column("asd", data=fs_list, kind="frequencyseries")
        return st

    def _make_st_with_ts(self, n: int = 3) -> SegmentTable:
        segs = _make_segments(n)
        st = SegmentTable.from_segments(segs)
        ts_list = [_make_ts() for _ in range(n)]
        st.add_series_column("raw", data=ts_list, kind="timeseries")
        return st

    def test_scatter_returns_plot(self):
        st = _make_simple_st(4)
        st.add_column("snr", [1.0, 2.0, 3.0, 4.0])
        st.add_column("duration", [4.0, 4.0, 4.0, 4.0])
        plot = st.scatter("snr", "duration")
        assert plot is not None

    def test_hist_returns_plot(self):
        st = _make_simple_st(4)
        st.add_column("snr", [1.0, 2.0, 3.0, 4.0])
        plot = st.hist("snr", bins=5)
        assert plot is not None

    def test_segments_returns_plot(self):
        st = _make_simple_st(4)
        plot = st.segments()
        assert plot is not None

    def test_segments_with_color(self):
        st = _make_simple_st(4)
        plot = st.segments(color="label")
        assert plot is not None

    def test_overlay_spectra_returns_plot(self):
        st = self._make_st_with_fs(3)
        plot = st.overlay_spectra("asd")
        assert plot is not None

    def test_overlay_spectra_requires_channel_for_dict(self):
        from gwpy.frequencyseries import FrequencySeries

        segs = _make_segments(2)
        st = SegmentTable.from_segments(segs)

        # Build FrequencySeriesDict manually (TimeSeriesDict has no .asd())
        def _make_fsd():
            return {"H1": _make_fs(), "L1": _make_fs()}

        fsd_list = [_make_fsd() for _ in range(2)]
        st.add_series_column("asd", data=fsd_list, kind="frequencyseriesdict")
        with pytest.raises(ValueError, match="channel"):
            st.overlay_spectra("asd")  # channel=None for dict → error

    def test_overlay_spectra_invalid_kind_raises(self):
        st = _make_simple_st(3)
        with pytest.raises((KeyError, TypeError)):
            st.overlay_spectra("label")

    def test_overlay_spectra_color_by_row(self):
        st = self._make_st_with_fs(3)
        plot = st.overlay_spectra("asd", color_by="row")
        assert plot is not None

    def test_overlay_spectra_color_by_column(self):
        st = self._make_st_with_fs(3)
        plot = st.overlay_spectra("asd", color_by="snr")
        assert plot is not None

    def test_plot_methods_do_not_call_show(self):
        """None of the drawing methods should call plt.show()."""
        import matplotlib.pyplot as plt

        st = _make_simple_st(4)
        st.add_column("snr", [1.0, 2.0, 3.0, 4.0])
        st.add_column("duration", [4.0, 4.0, 4.0, 4.0])

        with patch.object(plt, "show") as mock_show:
            st.scatter("snr", "duration")
            st.hist("snr")
            st.segments()
            assert mock_show.call_count == 0

    def test_plot_column_row_required(self):
        st = _make_simple_st(3)
        with pytest.raises(ValueError, match="column.*row"):
            st.plot()  # neither column nor row specified
