"""Tests for gwexpy/spectrogram/collections.py."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram
from gwexpy.spectrogram.collections import (
    SpectrogramDict,
    SpectrogramList,
    _resolve_crop_compat_args,
)


def _make_sg(n_times=10, n_freqs=8, t0=0.0, dt=1.0, df=1.0, name="sg"):
    """Create a test Spectrogram."""
    data = np.random.default_rng(42).normal(size=(n_times, n_freqs))
    times = np.arange(n_times) * dt + t0
    freqs = np.arange(n_freqs) * df
    return Spectrogram(
        data,
        times=times * u.s,
        frequencies=freqs * u.Hz,
        unit=u.m,
        name=name,
    )


# ---------------------------------------------------------------------------
# _resolve_crop_compat_args
# ---------------------------------------------------------------------------


class TestResolveCropCompatArgs:
    def test_positional_start_end(self):
        start, end, copy = _resolve_crop_compat_args(1.0, 5.0)
        assert start == 1.0
        assert end == 5.0
        assert copy is True

    def test_keyword_start_end(self):
        start, end, copy = _resolve_crop_compat_args(start=1.0, end=5.0)
        assert start == 1.0
        assert end == 5.0

    def test_positional_inplace_true(self):
        # Lines 58-84 — 3rd positional = legacy inplace
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            start, end, copy = _resolve_crop_compat_args(1.0, 5.0, True)
        assert copy is False

    def test_positional_inplace_false(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            start, end, copy = _resolve_crop_compat_args(1.0, 5.0, False)
        assert copy is True

    def test_too_many_positional_raises(self):
        # Line 47-50
        with pytest.raises(TypeError, match="at most 3"):
            _resolve_crop_compat_args(1.0, 5.0, True, "extra")

    def test_deprecated_t0_t1(self):
        # Lines 62-67 — t0/t1 deprecated
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            start, end, copy = _resolve_crop_compat_args(t0=1.0, t1=5.0)
        assert start == 1.0
        assert end == 5.0
        assert any("t0" in str(warning.message) for warning in w)

    def test_deprecated_inplace_kwarg(self):
        # Lines 69-77 — inplace kwarg deprecated
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            start, end, copy = _resolve_crop_compat_args(1.0, 5.0, inplace=True)
        assert copy is False
        assert any("inplace" in str(warning.message) for warning in w)

    def test_inplace_positional_and_kwarg_raises(self):
        # Lines 70-73 — duplicate inplace
        with pytest.raises(TypeError, match="both positional and keyword"):
            _resolve_crop_compat_args(1.0, 5.0, True, inplace=False)

    def test_copy_kwarg(self):
        # Lines 86-87
        start, end, copy = _resolve_crop_compat_args(1.0, 5.0, copy=False)
        assert copy is False

    def test_unexpected_kwargs_raises(self):
        # Lines 89-90
        with pytest.raises(TypeError, match="Unexpected keyword arguments"):
            _resolve_crop_compat_args(1.0, 5.0, foo=True, bar=False)


# ---------------------------------------------------------------------------
# SpectrogramList — construction and validation
# ---------------------------------------------------------------------------


class TestSpectrogramListConstruction:
    def test_empty(self):
        sl = SpectrogramList()
        assert len(sl) == 0

    def test_from_list(self):
        sg = _make_sg()
        sl = SpectrogramList([sg])
        assert len(sl) == 1

    def test_non_spectrogram_raises(self):
        # Lines 117-119 — TypeError for wrong type
        with pytest.raises(TypeError, match="Items must be of type Spectrogram"):
            SpectrogramList(["not_a_spectrogram"])

    def test_base_spectrogram_auto_convert(self):
        # Lines 113-115 — BaseSpectrogram → auto-convert to Spectrogram
        from gwpy.spectrogram import Spectrogram as BaseSpec
        data = np.random.randn(10, 8)
        times = np.arange(10) * u.s
        freqs = np.arange(8) * u.Hz
        base_sg = BaseSpec(data, times=times, frequencies=freqs, unit=u.m)
        sl = SpectrogramList([base_sg])
        assert isinstance(sl[0], Spectrogram)

    def test_setitem_valid(self):
        # Lines 121-127
        sl = SpectrogramList([_make_sg()])
        new_sg = _make_sg(name="new")
        sl[0] = new_sg
        assert sl[0].name == "new"

    def test_setitem_base_spectrogram_auto_convert(self):
        # Lines 122-124 — setitem with BaseSpectrogram
        from gwpy.spectrogram import Spectrogram as BaseSpec
        data = np.random.randn(10, 8)
        times = np.arange(10) * u.s
        freqs = np.arange(8) * u.Hz
        sl = SpectrogramList([_make_sg()])
        base_sg = BaseSpec(data, times=times, frequencies=freqs, unit=u.m)
        sl[0] = base_sg
        assert isinstance(sl[0], Spectrogram)

    def test_setitem_invalid_raises(self):
        sl = SpectrogramList([_make_sg()])
        with pytest.raises(TypeError, match="Value must be a Spectrogram"):
            sl[0] = "not_a_spectrogram"

    def test_append_valid(self):
        # Lines 129-135
        sl = SpectrogramList()
        sl.append(_make_sg())
        assert len(sl) == 1

    def test_append_base_spectrogram_auto_convert(self):
        # Lines 131-132 — append with BaseSpectrogram
        from gwpy.spectrogram import Spectrogram as BaseSpec
        data = np.random.randn(10, 8)
        times = np.arange(10) * u.s
        freqs = np.arange(8) * u.Hz
        sl = SpectrogramList()
        base_sg = BaseSpec(data, times=times, frequencies=freqs, unit=u.m)
        sl.append(base_sg)
        assert isinstance(sl[0], Spectrogram)

    def test_append_invalid_raises(self):
        sl = SpectrogramList()
        with pytest.raises(TypeError, match="Can only append Spectrogram"):
            sl.append("not_a_spectrogram")

    def test_extend(self):
        # Lines 137-139
        sl = SpectrogramList([_make_sg()])
        sl.extend([_make_sg(name="b")])
        assert len(sl) == 2


# ---------------------------------------------------------------------------
# SpectrogramList — crop
# ---------------------------------------------------------------------------


class TestSpectrogramListCrop:
    def test_crop_copy(self):
        # Lines 239-245
        sg = _make_sg()
        sl = SpectrogramList([sg])
        result = sl.crop(2.0, 7.0)
        assert isinstance(result, SpectrogramList)

    def test_crop_inplace(self):
        # Lines 246-249
        sg = _make_sg()
        sl = SpectrogramList([sg])
        result = sl.crop(2.0, 7.0, copy=False)
        assert result is sl


# ---------------------------------------------------------------------------
# SpectrogramList — write/reduce
# ---------------------------------------------------------------------------


class TestSpectrogramListReduce:
    def test_reduce_ex(self):
        # Line 187
        sl = SpectrogramList([_make_sg()])
        reduced = sl.__reduce_ex__(4)
        assert reduced[0] is list


class TestSpectrogramListWrite:
    def test_write_hdf5(self, tmp_path):
        # Lines 189-221
        sg = _make_sg()
        sl = SpectrogramList([sg])
        path = str(tmp_path / "test.hdf5")
        sl.write(path)

    def test_write_unsupported_format_raises(self, tmp_path):
        # Line 221
        sl = SpectrogramList([_make_sg()])
        with pytest.raises(NotImplementedError, match="Format csv"):
            sl.write(str(tmp_path / "test.csv"), format="csv")


# ---------------------------------------------------------------------------
# SpectrogramList — to_matrix
# ---------------------------------------------------------------------------


class TestSpectrogramListToMatrix:
    def test_to_matrix_basic(self):
        # Lines 337-420
        sg1 = _make_sg(name="a")
        sg2 = _make_sg(name="b")
        sl = SpectrogramList([sg1, sg2])
        matrix = sl.to_matrix()
        assert matrix.shape[0] == 2

    def test_to_matrix_empty(self):
        # Lines 368-369
        sl = SpectrogramList()
        matrix = sl.to_matrix()
        assert matrix.shape == (0, 0, 0)

    def test_to_matrix_shape_mismatch_raises(self):
        # Lines 384-389
        sg1 = _make_sg(n_times=10, n_freqs=8)
        sg2 = _make_sg(n_times=5, n_freqs=8)  # different shape
        sl = SpectrogramList([sg1, sg2])
        with pytest.raises(ValueError, match="mismatch"):
            sl.to_matrix()


# ---------------------------------------------------------------------------
# SpectrogramDict — construction
# ---------------------------------------------------------------------------


class TestSpectrogramDictConstruction:
    def test_empty(self):
        sd = SpectrogramDict()
        assert len(sd) == 0

    def test_from_dict(self):
        sg = _make_sg()
        sd = SpectrogramDict({"a": sg})
        assert "a" in sd

    def test_setitem_valid(self):
        # Lines 478-484
        sd = SpectrogramDict()
        sd["x"] = _make_sg()
        assert "x" in sd

    def test_setitem_invalid_raises(self):
        sd = SpectrogramDict()
        with pytest.raises(TypeError, match="Value must be a Spectrogram"):
            sd["x"] = "not_a_spectrogram"

    def test_update_dict(self):
        # Lines 486-498 — dict path
        sd = SpectrogramDict()
        sd.update({"a": _make_sg()})
        assert "a" in sd

    def test_update_keys(self):
        # Line 491-493 — hasattr(other, 'keys') path
        class FakeMapping:
            def keys(self):
                return ["b"]
            def __getitem__(self, k):
                return _make_sg(name=k)
        sd = SpectrogramDict()
        sd.update(FakeMapping())
        assert "b" in sd

    def test_update_iter(self):
        # Lines 494-496 — iterable of (k, v) pairs
        sd = SpectrogramDict()
        sd.update([("c", _make_sg())])
        assert "c" in sd

    def test_kwargs(self):
        sd = SpectrogramDict(d=_make_sg(name="d"))
        assert "d" in sd


# ---------------------------------------------------------------------------
# SpectrogramDict — crop
# ---------------------------------------------------------------------------


class TestSpectrogramDictCrop:
    def test_crop_copy(self):
        # Lines 603-609
        sg = _make_sg()
        sd = SpectrogramDict({"a": sg})
        result = sd.crop(2.0, 7.0)
        assert isinstance(result, SpectrogramDict)
        assert "a" in result

    def test_crop_inplace(self):
        # Lines 610-613
        sg = _make_sg()
        sd = SpectrogramDict({"a": sg})
        result = sd.crop(2.0, 7.0, copy=False)
        assert result is sd


# ---------------------------------------------------------------------------
# SpectrogramDict — write/reduce
# ---------------------------------------------------------------------------


class TestSpectrogramDictReduce:
    def test_reduce_ex(self):
        # Line 545
        sd = SpectrogramDict({"a": _make_sg()})
        reduced = sd.__reduce_ex__(4)
        assert reduced[0] is dict


class TestSpectrogramDictWrite:
    def test_write_hdf5(self, tmp_path):
        # Lines 547-581
        sg = _make_sg()
        sd = SpectrogramDict({"a": sg, "b": _make_sg(name="b")})
        path = str(tmp_path / "test.hdf5")
        sd.write(path)

    def test_write_unsupported_format_raises(self, tmp_path):
        # Line 581
        sd = SpectrogramDict({"a": _make_sg()})
        with pytest.raises(NotImplementedError, match="Format csv"):
            sd.write(str(tmp_path / "test.csv"), format="csv")


# ---------------------------------------------------------------------------
# SpectrogramList — radian / degree (PhaseMethodsMixin)
# ---------------------------------------------------------------------------


class TestSpectrogramListCropFrequencies:
    def test_crop_frequencies_copy(self):
        # Lines 251-274
        sg = _make_sg(n_freqs=16)
        sl = SpectrogramList([sg])
        result = sl.crop_frequencies(2.0, 10.0)
        assert isinstance(result, SpectrogramList)
        assert len(result) == 1

    def test_crop_frequencies_inplace(self):
        sg = _make_sg(n_freqs=16)
        sl = SpectrogramList([sg])
        result = sl.crop_frequencies(2.0, 10.0, inplace=True)
        assert result is sl


class TestSpectrogramListWriteGroupLayout:
    def test_write_hdf5_group_layout(self, tmp_path):
        # Lines 194-196 — LAYOUT_GROUP
        sg = _make_sg()
        sl = SpectrogramList([sg])
        path = str(tmp_path / "group.hdf5")
        sl.write(path, layout="group")


class TestSpectrogramDictCropFrequencies:
    def test_crop_frequencies_copy(self):
        # Lines 629-649
        sg = _make_sg(n_freqs=16)
        sd = SpectrogramDict({"a": sg})
        result = sd.crop_frequencies(2.0, 10.0)
        assert isinstance(result, SpectrogramDict)
        assert "a" in result

    def test_crop_frequencies_inplace(self):
        sg = _make_sg(n_freqs=16)
        sd = SpectrogramDict({"a": sg})
        result = sd.crop_frequencies(2.0, 10.0, inplace=True)
        assert result is sd


class TestSpectrogramDictSetitemBase:
    def test_setitem_base_spectrogram_auto_convert(self):
        # Lines 480-481 — BaseSpectrogram auto-convert in dict
        from gwpy.spectrogram import Spectrogram as BaseSpec
        data = np.random.randn(10, 8)
        times = np.arange(10) * u.s
        freqs = np.arange(8) * u.Hz
        sd = SpectrogramDict()
        base_sg = BaseSpec(data, times=times, frequencies=freqs, unit=u.m)
        sd["x"] = base_sg
        assert isinstance(sd["x"], Spectrogram)


class TestSpectrogramListPhase:
    def test_radian(self):
        # Line 453-455
        data = np.exp(1j * np.random.randn(10, 8))
        times = np.arange(10) * u.s
        freqs = np.arange(8) * u.Hz
        sg = Spectrogram(data, times=times, frequencies=freqs, unit=u.dimensionless_unscaled)
        sl = SpectrogramList([sg])
        result = sl.radian()
        assert isinstance(result, SpectrogramList)

    def test_degree(self):
        # Line 457-459
        data = np.exp(1j * np.random.randn(10, 8))
        times = np.arange(10) * u.s
        freqs = np.arange(8) * u.Hz
        sg = Spectrogram(data, times=times, frequencies=freqs, unit=u.dimensionless_unscaled)
        sl = SpectrogramList([sg])
        result = sl.degree()
        assert isinstance(result, SpectrogramList)
