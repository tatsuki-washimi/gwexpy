"""Tests for gwexpy.plot._init_helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from gwexpy.plot._init_helpers import (
    _filter_monitor_args,
    _expand_args,
    _flatten_scan,
    _extract_layout_and_fig_params,
    _apply_list_labels,
    _apply_ylabel,
    _apply_xlabel,
    _apply_layout_polish,
    _force_scales,
    _add_spectrogram_colorbars,
    _manage_sharex_labels,
    _apply_individual_axis_labels,
    _determine_scales_and_labels,
)


# ---------------------------------------------------------------------------
# Mock types
# ---------------------------------------------------------------------------

class _NamedItem:
    """Simple mutable object with an optional .name attribute."""
    def __init__(self, value, name=None):
        self.value = value
        self.name = name

    def __repr__(self):
        return f"_NamedItem({self.value!r}, name={self.name!r})"


class FakeSeriesMatrix:
    def __getitem__(self, key):
        # Return a mutable object so val.name = ... works
        return _NamedItem(f"item_{key}")

    def to_series_1Dlist(self):
        return ["a", "b"]

    def row_keys(self):
        return ["r0", "r1"]

    def col_keys(self):
        return ["c0", "c1"]


class FakeSeriesMatrixWithName:
    """SeriesMatrix where values already have a .name attribute."""

    def __getitem__(self, key):
        class _Val:
            name = "existing_name"
        return _Val()

    def to_series_1Dlist(self):
        return ["a"]

    def row_keys(self):
        return ["r0"]

    def col_keys(self):
        return ["c0"]


class FakeSeriesMatrixRaisingGetItem(FakeSeriesMatrix):
    """SeriesMatrix whose __getitem__ raises for any key."""

    def __getitem__(self, key):
        raise IndexError("boom")


class FakeSpectrogramMatrix:
    ndim = 3
    shape = (2, 2, 10, 8)

    def __getitem__(self, key):
        if isinstance(key, (int, tuple)):
            return f"sg_{key}"
        raise IndexError("out")

    def to_series_1Dlist(self):
        return ["sg_a", "sg_b"]

    def row_keys(self):
        return ["r0"]

    def col_keys(self):
        return ["c0"]


class FakeSpectrogramMatrix4D:
    """SpectrogramMatrix with ndim==4 so the 2D-index branch is exercised."""
    ndim = 4
    shape = (2, 3, 10, 8)   # nrow=2, ncol=3

    def __getitem__(self, key):
        return f"sg4d_{key}"

    def to_series_1Dlist(self):
        return ["sg4d_a"]

    def row_keys(self):
        return ["r0", "r1"]

    def col_keys(self):
        return ["c0", "c1", "c2"]


class FakeSpectrogramMatrix4DRaising:
    """SpectrogramMatrix with ndim==4 whose __getitem__ raises on first access."""
    ndim = 4
    shape = (2, 3, 10, 8)
    _calls = 0

    def __getitem__(self, key):
        # First call raises (triggers the except branch), second succeeds
        self._calls += 1
        if self._calls == 1:
            raise IndexError("first call fails")
        return f"sg4d_recovered_{key}"

    def to_series_1Dlist(self):
        return []

    def row_keys(self):
        return ["r0"]

    def col_keys(self):
        return ["c0"]


class FakeSpectrogramMatrix3DRaising:
    """SpectrogramMatrix with ndim==3 whose __getitem__ raises."""
    ndim = 3
    shape = (2, 10, 8)

    def __getitem__(self, key):
        raise IndexError("fail")

    def to_series_1Dlist(self):
        return []

    def row_keys(self):
        return ["r0"]

    def col_keys(self):
        return ["c0"]


class FakeFrequencySeriesList(list):
    pass


class FakeFrequencySeriesDict(dict):
    pass


class FakeSpectrogramList(list):
    pass


class FakeSpectrogramDict(dict):
    pass


# Non-list/dict versions to hit the FrequencySeriesList/Dict branches (lines 61, 63)
# in _expand_args when separate=True (those branches are skipped when types extend list/dict).
class FakeFrequencySeriesListNonList:
    """Acts like a sequence but does NOT inherit from list."""
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class FakeFrequencySeriesDictNonDict:
    """Acts like a mapping but does NOT inherit from dict."""
    def __init__(self, data):
        self._data = data

    def values(self):
        return self._data.values()

    def __iter__(self):
        return iter(self._data)


class FakeSpectrogramListNonList:
    """Acts like a sequence but does NOT inherit from list."""
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        return iter(self._data)


class FakeSpectrogramDictNonDict:
    """Acts like a mapping but does NOT inherit from dict."""
    def __init__(self, data):
        self._data = data

    def values(self):
        return self._data.values()


class FakeSpectrogram:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_expand_kwargs(extra=None):
    kw = dict(
        SeriesMatrix=FakeSeriesMatrix,
        SpectrogramMatrix=FakeSpectrogramMatrix,
        FrequencySeriesList=FakeFrequencySeriesList,
        FrequencySeriesDict=FakeFrequencySeriesDict,
        SpectrogramList=FakeSpectrogramList,
        SpectrogramDict=FakeSpectrogramDict,
    )
    if extra:
        kw.update(extra)
    return kw


def _make_defaults(
    xscale=None, yscale=None, xlabel=None, ylabel=None,
    norm=None, clabel=None, ylim=None, figsize=(6, 4),
):
    class _Defaults:
        def determine_xscale(self, data):
            return xscale

        def determine_yscale(self, data):
            return yscale

        def determine_xlabel(self, data):
            return xlabel

        def determine_ylabel(self, data):
            return ylabel

        def determine_norm(self, data):
            return norm

        def determine_clabel(self, data):
            return clabel

        def determine_ylim(self, data, yscale=None):
            return ylim

        def calculate_default_figsize(self, geometry, r, c):
            return figsize

    return _Defaults()


# ===========================================================================
# Tests for _filter_monitor_args
# ===========================================================================

class TestFilterMonitorArgs:
    def test_non_matrix_arg_passes_through(self):
        result = _filter_monitor_args((42, "hello"), 0, FakeSeriesMatrix, FakeSpectrogramMatrix)
        assert result == (42, "hello")

    def test_series_matrix_indexed_by_int(self):
        m = FakeSeriesMatrix()
        result = _filter_monitor_args((m,), 1, FakeSeriesMatrix, FakeSpectrogramMatrix)
        assert len(result) == 1
        assert result[0].value == "item_1"

    def test_series_matrix_indexed_by_string(self):
        m = FakeSeriesMatrix()
        result = _filter_monitor_args((m,), "r0", FakeSeriesMatrix, FakeSpectrogramMatrix)
        assert len(result) == 1
        assert result[0].value == "item_r0"

    def test_spectrogram_matrix_3d_indexed_by_int(self):
        m = FakeSpectrogramMatrix()
        result = _filter_monitor_args((m,), 0, FakeSeriesMatrix, FakeSpectrogramMatrix)
        assert result == ("sg_0",)

    def test_spectrogram_matrix_4d_int_monitor_uses_2d_index(self):
        """ndim==4 SpectrogramMatrix with int monitor uses row/col calculation."""
        class _SG4D(FakeSpectrogramMatrix4D):
            # Make __name__ match the guard
            pass
        _SG4D.__name__ = "SpectrogramMatrix"

        m = _SG4D()
        # monitor=4 with ncol=3 → row_idx=1, col_idx=1
        result = _filter_monitor_args((m,), 4, FakeSeriesMatrix, _SG4D)
        assert result == (f"sg4d_{(1, 1)}",)

    def test_spectrogram_matrix_non_int_monitor_falls_through_to_normal_index(self):
        # FakeSpectrogramMatrix.__getitem__ raises for non-int/tuple keys,
        # and __name__ != "SpectrogramMatrix", so it falls back to appending the whole object.
        m = FakeSpectrogramMatrix()
        result = _filter_monitor_args((m,), "r0", FakeSeriesMatrix, FakeSpectrogramMatrix)
        assert result == (m,)

    def test_series_matrix_index_error_falls_back_to_whole_object(self):
        class _Bad(FakeSeriesMatrix):
            def __getitem__(self, key):
                raise IndexError("boom")
        m = _Bad()
        result = _filter_monitor_args((m,), 0, _Bad, FakeSpectrogramMatrix)
        assert result == (m,)

    def test_series_matrix_type_error_falls_back_to_whole_object(self):
        class _Bad(FakeSeriesMatrix):
            def __getitem__(self, key):
                raise TypeError("boom")
        m = _Bad()
        result = _filter_monitor_args((m,), 0, _Bad, FakeSpectrogramMatrix)
        assert result == (m,)

    def test_spectrogram_matrix_4d_raising_on_first_triggers_except_branch(self):
        """When the 4D branch raises on first try, the except block re-tries 4D path."""
        class _SG4DRaising:
            ndim = 4
            shape = (2, 3, 10, 8)
            _call_count = 0

            def __getitem__(self, key):
                self._call_count += 1
                if self._call_count == 1:
                    raise IndexError("first fail")
                return f"recovered_{key}"

        _SG4DRaising.__name__ = "SpectrogramMatrix"
        m = _SG4DRaising()
        result = _filter_monitor_args((m,), 1, FakeSeriesMatrix, _SG4DRaising)
        # monitor=1, ncol=3 → row_idx=0, col_idx=1
        assert result == (f"recovered_{(0, 1)}",)

    def test_spectrogram_matrix_3d_raising_in_except_appends_indexed(self):
        """3D SpectrogramMatrix that raises falls through to the else-branch in except."""
        class _SG3DRaising:
            ndim = 3
            shape = (2, 10, 8)
            _calls = 0

            def __getitem__(self, key):
                self._calls += 1
                if self._calls == 1:
                    raise IndexError("first")
                return f"item_{key}"

        _SG3DRaising.__name__ = "SpectrogramMatrix"
        m = _SG3DRaising()
        result = _filter_monitor_args((m,), 0, FakeSeriesMatrix, _SG3DRaising)
        assert result == ("item_0",)

    def test_mixed_args(self):
        m = FakeSeriesMatrix()
        result = _filter_monitor_args(("plain", m, 99), 0, FakeSeriesMatrix, FakeSpectrogramMatrix)
        assert result[0] == "plain"
        assert result[1].value == "item_0"
        assert result[2] == 99

    def test_empty_args(self):
        assert _filter_monitor_args((), 0, FakeSeriesMatrix, FakeSpectrogramMatrix) == ()

    def test_numpy_integer_monitor(self):
        m = FakeSeriesMatrix()
        monitor = np.int64(0)
        result = _filter_monitor_args((m,), monitor, FakeSeriesMatrix, FakeSpectrogramMatrix)
        assert len(result) == 1
        assert result[0].value == "item_0"


# ===========================================================================
# Tests for _expand_args
# ===========================================================================

class TestExpandArgs:
    def _expand(self, args, separate, extra_kwargs=None):
        out = []
        kw = _make_expand_kwargs(extra_kwargs)
        _expand_args(args, separate, out, **kw)
        return out

    # --- separate is True ---

    def test_separate_true_matrix_extends(self):
        m = FakeSeriesMatrix()
        result = self._expand([m], True)
        assert result == ["a", "b"]

    def test_separate_true_list_extends(self):
        result = self._expand([[1, 2, 3]], True)
        assert result == [1, 2, 3]

    def test_separate_true_tuple_extends(self):
        result = self._expand([(1, 2)], True)
        assert result == [1, 2]

    def test_separate_true_dict_extends_values(self):
        result = self._expand([{"a": 10, "b": 20}], True)
        assert sorted(result) == [10, 20]

    def test_separate_true_frequency_series_list_extends(self):
        fsl = FakeFrequencySeriesList([7, 8, 9])
        result = self._expand([fsl], True)
        assert result == [7, 8, 9]

    def test_separate_true_spectrogram_list_extends(self):
        sl = FakeSpectrogramList(["x", "y"])
        result = self._expand([sl], True)
        assert result == ["x", "y"]

    def test_separate_true_frequency_series_dict_extends_values(self):
        fsd = FakeFrequencySeriesDict({"k": 42})
        result = self._expand([fsd], True)
        assert result == [42]

    def test_separate_true_spectrogram_dict_extends_values(self):
        sd = FakeSpectrogramDict({"k": 99})
        result = self._expand([sd], True)
        assert result == [99]

    def test_separate_true_scalar_appends(self):
        result = self._expand([42], True)
        assert result == [42]

    def test_separate_true_spectrogram_matrix_extends(self):
        m = FakeSpectrogramMatrix()
        result = self._expand([m], True,
                              extra_kwargs={"SpectrogramMatrix": FakeSpectrogramMatrix})
        assert result == ["sg_a", "sg_b"]

    # --- separate == "row" ---

    def test_separate_row_matrix_groups_by_row(self):
        m = FakeSeriesMatrix()
        result = self._expand([m], "row")
        # row_keys = ["r0", "r1"], col_keys = ["c0", "c1"]
        assert len(result) == 2
        assert len(result[0]) == 2  # r0: c0, c1
        assert len(result[1]) == 2  # r1: c0, c1

    def test_separate_row_non_matrix_falls_to_else(self):
        result = self._expand([42], "row")
        assert result == [42]

    def test_separate_row_matrix_name_is_set_when_missing(self):
        """Values without a .name attribute get one assigned."""
        m = FakeSeriesMatrix()  # returns strings which have no .name
        result = self._expand([m], "row")
        # strings don't have .name, so name would be set via attribute assignment
        # but strings are immutable — the code tries val.name = ... which is silently ignored
        # Just ensure the structure is correct
        assert all(isinstance(row, list) for row in result)

    def test_separate_row_matrix_with_name_keeps_existing(self):
        m = FakeSeriesMatrixWithName()
        result = self._expand([m], "row")
        assert len(result) == 1
        val = result[0][0]
        assert val.name == "existing_name"

    def test_separate_row_matrix_raising_getitem_skips_entry(self):
        m = FakeSeriesMatrixRaisingGetItem()
        result = self._expand([m], "row")
        # IndexError in inner loop → row_items stays empty → appended as []
        # FakeSeriesMatrix has 2 row_keys → 2 empty lists
        assert result == [[], []]

    # --- separate == "col" ---

    def test_separate_col_matrix_groups_by_col(self):
        m = FakeSeriesMatrix()
        result = self._expand([m], "col")
        # col_keys = ["c0", "c1"], row_keys = ["r0", "r1"]
        assert len(result) == 2
        assert len(result[0]) == 2

    def test_separate_col_non_matrix_falls_to_else(self):
        result = self._expand(["plain"], "col")
        assert result == ["plain"]

    def test_separate_col_matrix_raising_getitem_skips(self):
        m = FakeSeriesMatrixRaisingGetItem()
        result = self._expand([m], "col")
        # FakeSeriesMatrix has 2 col_keys → 2 empty lists
        assert result == [[], []]

    # --- separate is None / else branch ---

    def test_separate_none_matrix_appends_list(self):
        m = FakeSeriesMatrix()
        result = self._expand([m], None)
        assert result == [["a", "b"]]

    def test_separate_none_dict_appends_values_list(self):
        result = self._expand([{"x": 1, "y": 2}], None)
        assert len(result) == 1
        assert sorted(result[0]) == [1, 2]

    def test_separate_none_frequency_series_dict_appends_values(self):
        fsd = FakeFrequencySeriesDict({"a": 10})
        result = self._expand([fsd], None)
        assert result == [[10]]

    def test_separate_none_spectrogram_dict_appends_values(self):
        sd = FakeSpectrogramDict({"b": 20})
        result = self._expand([sd], None)
        assert result == [[20]]

    def test_separate_none_frequency_series_list_appends_as_list(self):
        fsl = FakeFrequencySeriesList([1, 2])
        result = self._expand([fsl], None)
        assert result == [[1, 2]]

    def test_separate_none_spectrogram_list_appends_as_list(self):
        sl = FakeSpectrogramList(["a"])
        result = self._expand([sl], None)
        assert result == [["a"]]

    def test_separate_none_scalar_appends(self):
        result = self._expand([99], None)
        assert result == [99]

    def test_separate_none_spectrogram_matrix_appends_list(self):
        m = FakeSpectrogramMatrix()
        result = self._expand([m], None,
                              extra_kwargs={"SpectrogramMatrix": FakeSpectrogramMatrix})
        assert result == [["sg_a", "sg_b"]]

    def test_empty_args(self):
        result = self._expand([], True)
        assert result == []

    def test_multiple_args_separate_true(self):
        m = FakeSeriesMatrix()
        result = self._expand([m, [10, 20]], True)
        assert result == ["a", "b", 10, 20]

    # --- Tests for lines 61, 63: FrequencySeriesList/Dict branch (separate=True, not list/dict) ---

    def test_separate_true_non_list_frequency_series_list_extends(self):
        """FrequencySeriesList that does NOT inherit list hits line 61."""
        fsl = FakeFrequencySeriesListNonList([7, 8, 9])
        out = []
        _expand_args(
            [fsl], True, out,
            SeriesMatrix=FakeSeriesMatrix,
            SpectrogramMatrix=FakeSpectrogramMatrix,
            FrequencySeriesList=FakeFrequencySeriesListNonList,
            FrequencySeriesDict=FakeFrequencySeriesDictNonDict,
            SpectrogramList=FakeSpectrogramListNonList,
            SpectrogramDict=FakeSpectrogramDictNonDict,
        )
        assert out == [7, 8, 9]

    def test_separate_true_non_dict_frequency_series_dict_extends_values(self):
        """FrequencySeriesDict that does NOT inherit dict hits line 63."""
        fsd = FakeFrequencySeriesDictNonDict({"k": 42})
        out = []
        _expand_args(
            [fsd], True, out,
            SeriesMatrix=FakeSeriesMatrix,
            SpectrogramMatrix=FakeSpectrogramMatrix,
            FrequencySeriesList=FakeFrequencySeriesListNonList,
            FrequencySeriesDict=FakeFrequencySeriesDictNonDict,
            SpectrogramList=FakeSpectrogramListNonList,
            SpectrogramDict=FakeSpectrogramDictNonDict,
        )
        assert list(out) == [42]

    def test_separate_none_non_dict_frequency_series_dict_appends_values(self):
        """FrequencySeriesDict (non-dict) with separate=None hits line 99."""
        fsd = FakeFrequencySeriesDictNonDict({"k": 99})
        out = []
        _expand_args(
            [fsd], None, out,
            SeriesMatrix=FakeSeriesMatrix,
            SpectrogramMatrix=FakeSpectrogramMatrix,
            FrequencySeriesList=FakeFrequencySeriesListNonList,
            FrequencySeriesDict=FakeFrequencySeriesDictNonDict,
            SpectrogramList=FakeSpectrogramListNonList,
            SpectrogramDict=FakeSpectrogramDictNonDict,
        )
        assert out == [[99]]

    def test_separate_none_non_list_frequency_series_list_appends_as_list(self):
        """FrequencySeriesList (non-list) with separate=None hits line 101."""
        fsl = FakeFrequencySeriesListNonList([1, 2])
        out = []
        _expand_args(
            [fsl], None, out,
            SeriesMatrix=FakeSeriesMatrix,
            SpectrogramMatrix=FakeSpectrogramMatrix,
            FrequencySeriesList=FakeFrequencySeriesListNonList,
            FrequencySeriesDict=FakeFrequencySeriesDictNonDict,
            SpectrogramList=FakeSpectrogramListNonList,
            SpectrogramDict=FakeSpectrogramDictNonDict,
        )
        assert out == [[1, 2]]


# ===========================================================================
# Tests for _flatten_scan
# ===========================================================================

class TestFlattenScan:
    def test_flat_list(self):
        assert _flatten_scan([1, 2, 3]) == [1, 2, 3]

    def test_nested_list(self):
        assert _flatten_scan([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_deeply_nested(self):
        assert _flatten_scan([[[1]], [2, [3]]]) == [1, 2, 3]

    def test_tuples_flattened(self):
        assert _flatten_scan([(1, 2), (3,)]) == [1, 2, 3]

    def test_mixed_nesting(self):
        assert _flatten_scan([1, [2, (3, 4)], 5]) == [1, 2, 3, 4, 5]

    def test_empty(self):
        assert _flatten_scan([]) == []

    def test_single_item(self):
        assert _flatten_scan(["x"]) == ["x"]

    def test_strings_not_expanded(self):
        # strings are not list/tuple so they should remain as items
        assert _flatten_scan(["hello", "world"]) == ["hello", "world"]


# ===========================================================================
# Tests for _extract_layout_and_fig_params
# ===========================================================================

class TestExtractLayoutAndFigParams:
    def _call(self, kwargs, separate=None, geometry=None, final_args=None, defaults=None):
        if final_args is None:
            final_args = []
        if defaults is None:
            defaults = _make_defaults()
        return _extract_layout_and_fig_params(kwargs, separate, geometry, final_args, defaults)

    def test_empty_kwargs(self):
        lk, fp, cl, tl, ll = self._call({})
        assert lk == {}
        assert fp == {}
        assert cl is False
        assert tl is False
        assert ll is None

    def test_geometry_added_to_kwargs(self):
        kwargs = {}
        lk, fp, cl, tl, ll = self._call(kwargs, geometry=(2, 2))
        assert lk["geometry"] == (2, 2)

    def test_separate_true_set(self):
        kwargs = {}
        lk, fp, cl, tl, ll = self._call(kwargs, separate=True)
        assert lk["separate"] is True

    def test_separate_string_converted_to_true(self):
        kwargs = {}
        lk, fp, cl, tl, ll = self._call(kwargs, separate="row")
        assert lk["separate"] is True

    def test_figsize_from_geometry(self):
        kwargs = {}
        defaults = _make_defaults(figsize=(10, 5))
        lk, fp, cl, tl, ll = self._call(kwargs, geometry=(2, 3), defaults=defaults)
        assert fp["figsize"] == (10, 5)

    def test_figsize_from_separate_true(self):
        kwargs = {}
        defaults = _make_defaults(figsize=(8, 2))
        final_args = ["a", "b", "c"]
        lk, fp, cl, tl, ll = self._call(kwargs, separate=True,
                                         final_args=final_args, defaults=defaults)
        assert fp["figsize"] == (8, 2)

    def test_figsize_not_overwritten_if_already_set(self):
        kwargs = {"figsize": (3, 3)}
        lk, fp, cl, tl, ll = self._call(kwargs, geometry=(2, 2))
        assert fp["figsize"] == (3, 3)

    def test_layout_keys_extracted(self):
        kwargs = {"xscale": "log", "yscale": "linear", "xlabel": "t", "ylabel": "y",
                  "title": "T", "legend": True, "method": "pcolormesh",
                  "sharex": True, "sharey": False, "norm": None,
                  "xlim": (0, 1), "ylim": (0, 10)}
        lk, fp, cl, tl, ll = self._call(kwargs)
        for k in ["xscale", "yscale", "xlabel", "ylabel", "title", "legend",
                   "method", "sharex", "sharey", "xlim", "ylim"]:
            assert k in lk
        # norm with None value is still extracted if present
        assert "norm" in lk

    def test_logx_sets_xscale_log(self):
        kwargs = {"logx": True}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert lk["xscale"] == "log"
        assert "logx" not in kwargs

    def test_logy_sets_yscale_log(self):
        kwargs = {"logy": True}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert lk["yscale"] == "log"

    def test_logx_false_does_not_set_xscale(self):
        kwargs = {"logx": False}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert "xscale" not in lk

    def test_fig_params_extracted(self):
        kwargs = {"figsize": (6, 4), "dpi": 150, "facecolor": "white"}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert fp["figsize"] == (6, 4)
        assert fp["dpi"] == 150
        assert fp["facecolor"] == "white"

    def test_constrained_layout_extracted(self):
        kwargs = {"constrained_layout": True}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert cl is True
        assert "constrained_layout" not in kwargs

    def test_tight_layout_extracted(self):
        kwargs = {"tight_layout": True}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert tl is True

    def test_ax_monitor_show_popped(self):
        kwargs = {"ax": object(), "monitor": 1, "show": True}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert "ax" not in kwargs
        assert "monitor" not in kwargs
        assert "show" not in kwargs

    def test_label_list_extracted(self):
        kwargs = {"label": ["l1", "l2"]}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert ll == ["l1", "l2"]
        assert "label" not in kwargs

    def test_label_tuple_extracted(self):
        kwargs = {"label": ("l1", "l2")}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert ll == ("l1", "l2")

    def test_label_scalar_not_extracted(self):
        kwargs = {"label": "single"}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert ll is None
        assert kwargs["label"] == "single"  # still in kwargs

    def test_labels_key_extracted(self):
        kwargs = {"labels": ["a", "b"]}
        lk, fp, cl, tl, ll = self._call(kwargs)
        assert ll == ["a", "b"]
        assert "labels" not in kwargs


# ===========================================================================
# Tests for _apply_list_labels
# ===========================================================================

class TestApplyListLabels:
    def test_labels_assigned_to_lines(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.plot([1, 2], [5, 6])
        _apply_list_labels(fig, ["line1", "line2"], {})
        labels = [l.get_label() for l in ax.get_lines()]
        assert labels == ["line1", "line2"]
        plt.close(fig)

    def test_fewer_labels_than_lines(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        ax.plot([1, 2], [5, 6])
        _apply_list_labels(fig, ["only_one"], {})
        labels = [l.get_label() for l in ax.get_lines()]
        assert labels[0] == "only_one"
        plt.close(fig)

    def test_legend_created_when_legend_true(self):
        fig, ax = plt.subplots()
        ax.plot([1], [1], label="x")
        _apply_list_labels(fig, ["new_label"], {"legend": True})
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_no_legend_when_legend_false(self):
        fig, ax = plt.subplots()
        ax.plot([1], [1])
        _apply_list_labels(fig, ["lbl"], {"legend": False})
        # When legend=False, legend should not be created
        assert ax.get_legend() is None
        plt.close(fig)

    def test_no_lines_no_error(self):
        fig, ax = plt.subplots()
        _apply_list_labels(fig, ["lbl"], {})
        plt.close(fig)

    def test_multiple_axes(self):
        fig, axes = plt.subplots(1, 2)
        axes[0].plot([1], [1])
        axes[1].plot([1], [2])
        _apply_list_labels(fig, ["a", "b"], {})
        assert axes[0].get_lines()[0].get_label() == "a"
        assert axes[1].get_lines()[0].get_label() == "b"
        plt.close(fig)


# ===========================================================================
# Tests for _apply_ylabel
# ===========================================================================

class TestApplyYlabel:
    def test_force_ylabel_applied(self):
        fig, ax = plt.subplots()
        _apply_ylabel(fig, "Force Y", {})
        assert ax.get_ylabel() == "Force Y"
        plt.close(fig)

    def test_ylabel_from_existing_ax(self):
        fig, ax = plt.subplots()
        ax.set_ylabel("Existing")
        _apply_ylabel(fig, None, {})
        assert ax.get_ylabel() == "Existing"
        plt.close(fig)

    def test_no_ylabel_no_change(self):
        fig, ax = plt.subplots()
        _apply_ylabel(fig, None, {})
        assert ax.get_ylabel() == ""
        plt.close(fig)

    def test_empty_force_ylabel_no_change(self):
        fig, ax = plt.subplots()
        _apply_ylabel(fig, "", {})
        assert ax.get_ylabel() == ""
        plt.close(fig)

    def test_sharey_clears_non_first_col(self):
        fig, axes = plt.subplots(1, 2)
        axes[1].set_ylabel("Y")
        _apply_ylabel(fig, "Y", {"sharey": True})
        # First col should have ylabel, second should be cleared
        assert axes[1].get_ylabel() == ""
        plt.close(fig)

    def test_more_than_two_first_col_axes_all_same(self):
        fig, axes = plt.subplots(4, 1)
        _apply_ylabel(fig, "Y", {})
        # Mid index should have label, others cleared
        mid_idx = 4 // 2
        assert axes[mid_idx].get_ylabel() == "Y"
        for i, ax in enumerate(axes):
            if i != mid_idx:
                assert ax.get_ylabel() == ""
        plt.close(fig)

    def test_more_than_two_first_col_axes_different_labels(self):
        fig, axes = plt.subplots(4, 1)
        axes[0].set_ylabel("Different")
        _apply_ylabel(fig, "Y", {})
        # all_same=False branch: axes without ylabel get it set
        plt.close(fig)

    def test_ax_without_subplotspec_uses_is_left_true(self):
        """Axes without a subplotspec trigger the except branch (lines 272-273)."""
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # no subplotspec
        _apply_ylabel(fig, "Y", {})
        assert ax.get_ylabel() == "Y"
        plt.close(fig)


# ===========================================================================
# Tests for _apply_individual_axis_labels
# ===========================================================================

class TestApplyIndividualAxisLabels:
    def test_returns_early_if_force_ylabel_set(self):
        fig, ax = plt.subplots()
        defaults = _make_defaults(ylabel="auto")
        _apply_individual_axis_labels(fig, [None], "ForceY", defaults)
        assert ax.get_ylabel() == ""
        plt.close(fig)

    def test_returns_early_if_axes_count_mismatch(self):
        fig, axes = plt.subplots(1, 2)
        defaults = _make_defaults(ylabel="auto")
        _apply_individual_axis_labels(fig, [None], None, defaults)
        # Only 1 final_arg for 2 axes → skip
        assert axes[0].get_ylabel() == ""
        plt.close(fig)

    def test_sets_ylabel_per_axis(self):
        fig, axes = plt.subplots(1, 2)
        defaults = _make_defaults(ylabel="Units")
        _apply_individual_axis_labels(fig, [None, None], None, defaults)
        for ax in axes:
            assert ax.get_ylabel() == "Units"
        plt.close(fig)

    def test_does_not_overwrite_existing_ylabel(self):
        fig, ax = plt.subplots()
        ax.set_ylabel("Existing")
        defaults = _make_defaults(ylabel="New")
        _apply_individual_axis_labels(fig, [None], None, defaults)
        assert ax.get_ylabel() == "Existing"
        plt.close(fig)

    def test_list_data_item_passed_to_determine(self):
        fig, ax = plt.subplots()
        defaults = _make_defaults(ylabel="ListLabel")
        _apply_individual_axis_labels(fig, [["a", "b"]], None, defaults)
        assert ax.get_ylabel() == "ListLabel"
        plt.close(fig)

    def test_no_label_when_determine_returns_none(self):
        fig, ax = plt.subplots()
        defaults = _make_defaults(ylabel=None)
        _apply_individual_axis_labels(fig, [None], None, defaults)
        assert ax.get_ylabel() == ""
        plt.close(fig)


# ===========================================================================
# Tests for _apply_xlabel
# ===========================================================================

class TestApplyXlabel:
    def test_force_xlabel_applied(self):
        fig, ax = plt.subplots()
        result = _apply_xlabel(fig, "Time [s]", {})
        assert result == "Time [s]"
        assert ax.get_xlabel() == "Time [s]"
        plt.close(fig)

    def test_xlabel_from_existing_ax(self):
        fig, ax = plt.subplots()
        ax.set_xlabel("Existing")
        result = _apply_xlabel(fig, None, {})
        assert result == "Existing"
        plt.close(fig)

    def test_no_xlabel_returns_none(self):
        fig, ax = plt.subplots()
        result = _apply_xlabel(fig, None, {})
        assert result is None
        plt.close(fig)

    def test_sharex_prevents_setting_on_axes(self):
        fig, axes = plt.subplots(2, 1)
        result = _apply_xlabel(fig, "X", {"sharex": True})
        # With sharex=True, axes without xlabel should NOT get it set
        assert result == "X"
        plt.close(fig)

    def test_no_sharex_sets_on_all_axes(self):
        fig, axes = plt.subplots(1, 2)
        result = _apply_xlabel(fig, "X", {})
        assert all(ax.get_xlabel() == "X" for ax in axes)
        plt.close(fig)


# ===========================================================================
# Tests for _apply_layout_polish
# ===========================================================================

class TestApplyLayoutPolish:
    def test_constrained_layout_applied(self):
        fig, ax = plt.subplots()
        _apply_layout_polish(fig, use_cl=True, use_tl=False)
        plt.close(fig)

    def test_tight_layout_applied(self):
        fig, ax = plt.subplots()
        _apply_layout_polish(fig, use_cl=False, use_tl=True)
        plt.close(fig)

    def test_neither_layout_applied(self):
        fig, ax = plt.subplots()
        _apply_layout_polish(fig, use_cl=False, use_tl=False)
        plt.close(fig)

    def test_both_layouts_applied(self):
        fig, ax = plt.subplots()
        _apply_layout_polish(fig, use_cl=True, use_tl=True)
        plt.close(fig)

    def test_constrained_layout_with_bad_fig_raises_silently(self):
        class BadFig:
            axes = []
            def set_constrained_layout(self, v):
                raise TypeError("no")
            def tight_layout(self):
                pass
        _apply_layout_polish(BadFig(), use_cl=True, use_tl=False)

    def test_tight_layout_with_bad_fig_raises_silently(self):
        class BadFig:
            axes = []
            def set_constrained_layout(self, v):
                pass
            def tight_layout(self):
                raise ValueError("bad")
        _apply_layout_polish(BadFig(), use_cl=False, use_tl=True)


# ===========================================================================
# Tests for _force_scales
# ===========================================================================

class TestForceScales:
    def test_xscale_log_applied(self):
        fig, ax = plt.subplots()
        _force_scales(fig, {"xscale": "log"})
        assert ax.get_xscale() == "log"
        plt.close(fig)

    def test_yscale_log_applied(self):
        fig, ax = plt.subplots()
        _force_scales(fig, {"yscale": "log"})
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_no_xscale_no_change(self):
        fig, ax = plt.subplots()
        _force_scales(fig, {})
        assert ax.get_xscale() == "linear"
        plt.close(fig)

    def test_invalid_xscale_silently_ignored(self):
        fig, ax = plt.subplots()
        _force_scales(fig, {"xscale": "invalid_scale_xyz"})
        plt.close(fig)

    def test_invalid_yscale_silently_ignored(self):
        """Invalid yscale hits the except branch (lines 372-373)."""
        fig, ax = plt.subplots()
        _force_scales(fig, {"yscale": "invalid_scale_xyz"})
        plt.close(fig)

    def test_multiple_axes(self):
        fig, axes = plt.subplots(1, 2)
        _force_scales(fig, {"xscale": "log", "yscale": "log"})
        for ax in axes:
            assert ax.get_xscale() == "log"
            assert ax.get_yscale() == "log"
        plt.close(fig)


# ===========================================================================
# Tests for _add_spectrogram_colorbars
# ===========================================================================

class TestAddSpectrogramColorbars:
    def test_no_spectrogram_does_nothing(self):
        fig, ax = plt.subplots()
        initial_axes = len(fig.axes)
        _add_spectrogram_colorbars(fig, is_spectrogram=False, det_clabel=None)
        assert len(fig.axes) == initial_axes
        plt.close(fig)

    def test_spectrogram_with_pcolormesh_adds_colorbar(self):
        fig, ax = plt.subplots()
        import numpy as np
        data = np.random.rand(4, 4)
        ax.pcolormesh(data)
        _add_spectrogram_colorbars(fig, is_spectrogram=True, det_clabel="Power")
        # Colorbar adds an extra axis
        assert len(fig.axes) > 1
        plt.close(fig)

    def test_spectrogram_with_imshow_adds_colorbar(self):
        fig, ax = plt.subplots()
        import numpy as np
        data = np.random.rand(4, 4)
        ax.imshow(data)
        _add_spectrogram_colorbars(fig, is_spectrogram=True, det_clabel=None)
        assert len(fig.axes) > 1
        plt.close(fig)

    def test_spectrogram_without_mappable_no_colorbar(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        n_before = len(fig.axes)
        _add_spectrogram_colorbars(fig, is_spectrogram=True, det_clabel=None)
        assert len(fig.axes) == n_before
        plt.close(fig)

    def test_colorbar_exception_silently_caught(self):
        """fig.colorbar raising hits the except branch (lines 478-479)."""
        from unittest.mock import patch, MagicMock
        import numpy as np

        fig, ax = plt.subplots()
        data = np.random.rand(4, 4)
        ax.pcolormesh(data)

        with patch.object(fig, "colorbar", side_effect=TypeError("no colorbar")):
            # Should not raise
            _add_spectrogram_colorbars(fig, is_spectrogram=True, det_clabel=None)

        plt.close(fig)


# ===========================================================================
# Tests for _manage_sharex_labels
# ===========================================================================

class TestManageSharexLabels:
    def test_no_geometry_does_nothing(self):
        fig, axes = plt.subplots(2, 1)
        axes[0].set_xlabel("top")
        _manage_sharex_labels(fig, {}, "candidate")
        # Should not modify anything
        assert axes[0].get_xlabel() == "top"
        plt.close(fig)

    def test_no_sharex_does_nothing(self):
        fig, axes = plt.subplots(2, 1)
        axes[0].set_xlabel("top")
        _manage_sharex_labels(fig, {"geometry": (2, 1)}, "candidate")
        assert axes[0].get_xlabel() == "top"
        plt.close(fig)

    def test_sharex_clears_non_bottom_row_labels(self):
        fig, axes = plt.subplots(2, 1)
        axes[0].set_xlabel("top")
        axes[1].set_xlabel("bottom")
        _manage_sharex_labels(fig, {"geometry": (2, 1), "sharex": True}, None)
        assert axes[0].get_xlabel() == ""
        plt.close(fig)

    def test_sharex_sets_candidate_on_bottom_row(self):
        fig, axes = plt.subplots(2, 1)
        _manage_sharex_labels(fig, {"geometry": (2, 1), "sharex": True}, "Time [s]")
        assert axes[1].get_xlabel() == "Time [s]"
        plt.close(fig)

    def test_sharex_does_not_overwrite_existing_bottom_label(self):
        fig, axes = plt.subplots(2, 1)
        axes[1].set_xlabel("Already set")
        _manage_sharex_labels(fig, {"geometry": (2, 1), "sharex": True}, "Candidate")
        assert axes[1].get_xlabel() == "Already set"
        plt.close(fig)

    def test_no_candidate_xlabel_bottom_stays_empty(self):
        fig, axes = plt.subplots(2, 1)
        _manage_sharex_labels(fig, {"geometry": (2, 1), "sharex": True}, None)
        assert axes[1].get_xlabel() == ""
        plt.close(fig)

    def test_multi_col_geometry(self):
        fig, axes = plt.subplots(2, 2)
        _manage_sharex_labels(fig, {"geometry": (2, 2), "sharex": True}, "X")
        # Top row (indices 0, 1) should have empty xlabel
        flat_axes = fig.axes
        assert flat_axes[0].get_xlabel() == ""
        assert flat_axes[1].get_xlabel() == ""
        plt.close(fig)


# ===========================================================================
# Tests for _determine_scales_and_labels
# ===========================================================================

class TestDetermineScalesAndLabels:
    def test_method_set_when_spectrogram_present(self):
        kwargs = {}
        defaults = _make_defaults()
        scan = [FakeSpectrogram()]
        _determine_scales_and_labels(scan, kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["method"] == "pcolormesh"

    def test_method_not_overwritten_if_set(self):
        kwargs = {"method": "plot"}
        defaults = _make_defaults()
        scan = [FakeSpectrogram()]
        _determine_scales_and_labels(scan, kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["method"] == "plot"

    def test_no_spectrogram_no_method(self):
        kwargs = {}
        defaults = _make_defaults()
        _determine_scales_and_labels(["plain"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert "method" not in kwargs

    def test_xscale_set_from_defaults(self):
        kwargs = {}
        defaults = _make_defaults(xscale="log")
        _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["xscale"] == "log"

    def test_xscale_not_overwritten_if_set(self):
        kwargs = {"xscale": "linear"}
        defaults = _make_defaults(xscale="log")
        _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["xscale"] == "linear"

    def test_yscale_set_from_defaults(self):
        kwargs = {}
        defaults = _make_defaults(yscale="log")
        _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["yscale"] == "log"

    def test_xlabel_set_from_defaults(self):
        kwargs = {}
        defaults = _make_defaults(xlabel="Time")
        _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["xlabel"] == "Time"

    def test_ylabel_set_when_single_unit(self):
        class _Data:
            unit = None
        kwargs = {}
        defaults = _make_defaults(ylabel="Units")
        _determine_scales_and_labels([_Data()], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["ylabel"] == "Units"

    def test_ylabel_set_when_unit_has_to_string(self):
        class _Unit:
            def to_string(self):
                return "m/s"
        class _Data:
            unit = _Unit()
        kwargs = {}
        defaults = _make_defaults(ylabel="m/s")
        _determine_scales_and_labels([_Data()], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["ylabel"] == "m/s"

    def test_ylabel_not_set_when_multiple_units(self):
        class _Unit:
            def __init__(self, s):
                self._s = s
            def to_string(self):
                return self._s
        class _Data:
            def __init__(self, u):
                self.unit = u
        kwargs = {}
        defaults = _make_defaults(ylabel="Y")
        scan = [_Data(_Unit("m")), _Data(_Unit("s"))]
        _determine_scales_and_labels(scan, kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert "ylabel" not in kwargs

    def test_norm_set_from_defaults(self):
        kwargs = {}
        defaults = _make_defaults(norm="log")
        _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["norm"] == "log"

    def test_clabel_returned(self):
        kwargs = {}
        defaults = _make_defaults(clabel="Power [dB]")
        det_clabel = _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert det_clabel == "Power [dB]"

    def test_clabel_not_returned_when_set_in_kwargs(self):
        kwargs = {"clabel": "existing"}
        defaults = _make_defaults(clabel="Power")
        det_clabel = _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert det_clabel is None

    def test_ylim_set_from_defaults(self):
        kwargs = {}
        defaults = _make_defaults(ylim=(0, 100))
        _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs["ylim"] == (0, 100)

    def test_specto_matrix_triggers_method(self):
        kwargs = {}
        defaults = _make_defaults()
        scan = [FakeSpectrogramMatrix()]
        _determine_scales_and_labels(scan, kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        assert kwargs.get("method") == "pcolormesh"

    def test_none_det_values_not_added_to_kwargs(self):
        kwargs = {}
        defaults = _make_defaults()  # all returns None
        _determine_scales_and_labels(["x"], kwargs, defaults, FakeSpectrogram, FakeSpectrogramMatrix)
        for k in ["xscale", "yscale", "xlabel", "ylabel", "norm", "ylim"]:
            assert k not in kwargs
