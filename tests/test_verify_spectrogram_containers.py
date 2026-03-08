import warnings

import numpy as np
import pytest
from astropy import units as u
from gwpy.spectrogram import Spectrogram

from gwexpy.spectrogram import SpectrogramDict, SpectrogramList, SpectrogramMatrix


def create_mock_spectrogram(name="spec", shape=(10, 10)):
    # Create a 10x10 spectrogram
    # Time: 0 to 10s (dt=1), Freq: 0 to 100Hz (df=10)
    data = np.random.random(shape)
    spec = Spectrogram(data, t0=0, dt=1, f0=0, df=10, unit="strain", name=name)
    return spec


def test_spectrogram_list():
    s1 = create_mock_spectrogram("s1")
    s2 = create_mock_spectrogram("s2")

    sl = SpectrogramList([s1])
    sl.append(s2)
    assert len(sl) == 2
    assert sl[0].name == "s1"

    # Type check
    try:
        sl.append("invalid")
        raise AssertionError("Type check failed - allowed string append")
    except TypeError:
        pass

    # Crop
    sl_cropped = sl.crop(t0=2, t1=8)
    # t0=0, dt=1. Indices 2 to 8.
    # Result time axis should start >= 2
    assert sl_cropped[0].times[0].value >= 2
    assert len(sl_cropped) == 2

    # Crop frequencies
    # f0=0, df=10. 20Hz is index 2. 80Hz is index 8.
    sl_freq = sl.crop_frequencies(f0=20, f1=80)
    # Check freq axis
    assert sl_freq[0].frequencies[0].value >= 20
    assert sl_freq[0].frequencies[-1].value <= 80

    # to_matrix
    mat = sl.to_matrix()
    # Expect (2, 10, 10) - crop returns a copy, original is unchanged
    assert isinstance(mat, SpectrogramMatrix)
    assert mat.shape == (2, 10, 10)
    assert mat.times is not None
    assert len(mat.times) == 10
    assert mat.frequencies is not None
    assert len(mat.frequencies) == 10
    assert mat.unit == u.Unit("strain")

    # Plot check (dry run) - skip if matplotlib not available
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    p = sl.plot()
    p.close()

    # Shape mismatch for matrix
    s3 = create_mock_spectrogram("s3", shape=(5, 5))
    sl.append(s3)
    try:
        sl.to_matrix()
        raise AssertionError("Should fail with shape mismatch")
    except ValueError:
        pass


def test_spectrogram_dict():
    s1 = create_mock_spectrogram("s1")
    s2 = create_mock_spectrogram("s2")

    sd = SpectrogramDict({"a": s1})
    sd["b"] = s2

    assert len(sd) == 2

    # Type check
    try:
        sd["c"] = "invalid"
        raise AssertionError("Type check failed")
    except TypeError:
        pass

    # Crop
    sd_cropped = sd.crop(2, 8)
    assert sd_cropped["a"].times[0].value >= 2

    # Matrix - crop returns a copy, so sd is unchanged
    mat = sd.to_matrix()
    assert mat.shape == (2, 10, 10)


# ---------------------------------------------------------------------------
# crop compatibility tests
# ---------------------------------------------------------------------------


class TestCropCompat:
    """Tests for _resolve_crop_compat_args via SpectrogramList/Dict.crop."""

    def _make_list(self):
        s1 = create_mock_spectrogram("s1")
        s2 = create_mock_spectrogram("s2")
        return SpectrogramList([s1, s2])

    def _make_dict(self):
        s1 = create_mock_spectrogram("s1")
        s2 = create_mock_spectrogram("s2")
        return SpectrogramDict({"a": s1, "b": s2})

    # -- third positional inplace compatibility --

    def test_third_positional_inplace_list(self):
        """crop(start, end, True) treats 3rd arg as legacy inplace (True=inplace)."""
        sl = self._make_list()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sl.crop(2, 8, True)
            # inplace=True → copy=False → returns self
            assert result is sl
            assert any("positional inplace is deprecated" in str(x.message) for x in w)

    def test_third_positional_inplace_dict(self):
        """crop(start, end, True) treats 3rd arg as legacy inplace (True=inplace)."""
        sd = self._make_dict()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sd.crop(2, 8, True)
            assert result is sd
            assert any("positional inplace is deprecated" in str(x.message) for x in w)

    # -- deprecation warning emission --

    def test_deprecation_warning_t0_t1_list(self):
        """t0/t1 kwargs emit DeprecationWarning."""
        sl = self._make_list()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.crop(t0=2, t1=8)
            dep_msgs = [x for x in w if issubclass(x.category, DeprecationWarning)]
            texts = [str(x.message) for x in dep_msgs]
            assert any("t0 is deprecated" in t for t in texts)
            assert any("t1 is deprecated" in t for t in texts)

    def test_deprecation_warning_inplace_kwarg(self):
        """inplace kwarg emits DeprecationWarning."""
        sl = self._make_list()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.crop(2, 8, inplace=False)
            dep_msgs = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("inplace is deprecated" in str(x.message) for x in dep_msgs)

    # -- too many positional args --

    def test_too_many_positional_args_list(self):
        """crop(1, 2, 3, 4) raises TypeError."""
        sl = self._make_list()
        with pytest.raises(TypeError, match="at most 3 positional arguments"):
            sl.crop(1, 2, 3, 4)

    def test_too_many_positional_args_dict(self):
        """crop(1, 2, 3, 4) raises TypeError."""
        sd = self._make_dict()
        with pytest.raises(TypeError, match="at most 3 positional arguments"):
            sd.crop(1, 2, 3, 4)

    # -- duplicate legacy inplace rejection --

    def test_duplicate_legacy_inplace_list(self):
        """crop(start, end, True, inplace=True) raises TypeError."""
        sl = self._make_list()
        with pytest.raises(TypeError, match="both positional and keyword"):
            sl.crop(2, 8, True, inplace=True)

    def test_duplicate_legacy_inplace_dict(self):
        """crop(start, end, True, inplace=True) raises TypeError."""
        sd = self._make_dict()
        with pytest.raises(TypeError, match="both positional and keyword"):
            sd.crop(2, 8, True, inplace=True)

    # -- deterministic unexpected kwargs error --

    def test_deterministic_unexpected_kwargs_list(self):
        """Unexpected kwargs error uses sorted() for deterministic output."""
        sl = self._make_list()
        with pytest.raises(TypeError, match=r"\['bar', 'foo'\]"):
            sl.crop(2, 8, foo=1, bar=2)

    def test_deterministic_unexpected_kwargs_dict(self):
        """Unexpected kwargs error uses sorted() for deterministic output."""
        sd = self._make_dict()
        with pytest.raises(TypeError, match=r"\['bar', 'foo'\]"):
            sd.crop(2, 8, foo=1, bar=2)
