"""ROOT histogram read contracts exercised against ROOT itself."""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.histogram import Histogram

pytestmark = pytest.mark.root


@pytest.fixture(scope="module")
def ROOT():
    root = pytest.importorskip("ROOT")
    root.gROOT.SetBatch(True)
    return root


@pytest.mark.parametrize(
    ("suffix", "dtype"),
    [
        ("C", np.dtype("int8")),
        ("S", np.dtype("int16")),
        ("I", np.dtype("int32")),
        ("L", np.dtype("int64")),
        ("F", np.dtype("float32")),
        ("D", np.dtype("float64")),
    ],
)
def test_from_root_preserves_native_content_and_float64_sumw2(ROOT, suffix, dtype):
    hist = getattr(ROOT, f"TH1{suffix}")(f"native_{suffix}", "", 2, 0.0, 2.0)
    hist.Sumw2()
    for index, content, error in ((0, 1, 0.5), (1, 2, 1.5), (2, 3, 2.5), (3, 4, 3.5)):
        hist.SetBinContent(index, content)
        hist.SetBinError(index, error)

    result = Histogram.from_root(hist)

    assert result.values.value.dtype == dtype
    assert result.sumw2 is not None
    assert result.sumw2.value.dtype == np.dtype("float64")
    np.testing.assert_allclose(result.values.value, [2, 3])
    np.testing.assert_allclose(result.sumw2.value, [1.5**2, 2.5**2])
    assert result.underflow.value == 1
    assert result.overflow.value == 4
    assert result.underflow_sumw2 is not None
    assert result.overflow_sumw2 is not None
    assert result.underflow_sumw2.value == 0.5**2
    assert result.overflow_sumw2.value == 3.5**2


@pytest.mark.parametrize(
    "factory",
    [
        lambda root: root.TProfile("profile", "", 2, 0.0, 2.0),
        lambda root: root.TProfile2D("profile2d", "", 2, 0.0, 2.0, 2, 0.0, 2.0),
        lambda root: root.TProfile3D(
            "profile3d", "", 2, 0.0, 2.0, 2, 0.0, 2.0, 2, 0.0, 2.0
        ),
        lambda root: root.TProfile2Poly(),
        lambda root: root.TH2Poly(),
    ],
)
def test_from_root_rejects_profiles_and_polygon_histograms(ROOT, factory):
    with pytest.raises(TypeError, match="does not support"):
        Histogram.from_root(factory(ROOT))


def test_from_root_unknown_rectangular_th_uses_float64_fallback(ROOT):
    ROOT.gInterpreter.Declare(
        """
        class GWexpyUnknownTH1 final : public TH1D {
        public:
            GWexpyUnknownTH1() : TH1D("unknown_th", "", 2, 0., 2.) {}
        };
        """
    )
    hist = ROOT.GWexpyUnknownTH1()
    hist.SetBinContent(1, 1.25)
    hist.SetBinContent(2, 2.5)

    result = Histogram.from_root(hist)

    assert result.values.value.dtype == np.dtype("float64")
    np.testing.assert_allclose(result.values.value, [1.25, 2.5])
