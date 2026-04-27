import gwpy.segments as gwpy_segments
import numpy as np
import pytest
from astropy import units as u

import gwexpy.segments as gwexpy_segments
from gwexpy.histogram import Histogram, HistogramDict, HistogramList


def test_quantity_aliases_bin_geometry_and_passive_metadata_contract():
    channel = object()
    hist = Histogram(
        values=[2.0, 4.0, 6.0] * u.ct,
        edges=[0.0, 0.5, 2.0, 5.0] * u.s,
        name="strain-counts",
        channel=channel,
    )

    assert hist.values.unit == u.ct
    assert hist.edges.unit == u.s
    assert hist.unit == u.ct
    assert hist.xunit == u.s
    assert hist.value is hist.values
    assert hist.name == "strain-counts"
    assert hist.channel is channel

    np.testing.assert_allclose(hist.values.value, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(hist.edges.value, [0.0, 0.5, 2.0, 5.0])
    np.testing.assert_allclose(hist.bin_widths.value, [0.5, 1.5, 3.0])
    np.testing.assert_allclose(hist.xindex.value, [0.25, 1.25, 3.5])
    assert hist.bin_widths.unit == u.s
    assert hist.xindex.unit == u.s


def test_statistical_uncertainty_contract_prefers_covariance_diagonal():
    hist = Histogram(
        values=[10.0, 20.0, 30.0],
        edges=[0.0, 1.0, 2.0, 3.0],
        unit="ct",
        xunit="s",
        sumw2=[1.0, 4.0, 9.0],
        cov=np.diag([100.0, 400.0, 900.0]) * u.ct**2,
    )

    assert hist.sumw2 is not None
    assert hist.cov is not None
    assert hist.sumw2.shape == (3,)
    assert hist.cov.shape == (3, 3)
    assert hist.sumw2.unit == u.ct**2
    assert hist.cov.unit == u.ct**2
    assert hist.errors is not None
    assert hist.errors.unit == u.ct
    np.testing.assert_allclose(hist.errors.value, [10.0, 20.0, 30.0])

    sumw2_only = Histogram(
        values=[10.0, 20.0, 30.0],
        edges=[0.0, 1.0, 2.0, 3.0],
        unit="ct",
        sumw2=[1.0, 4.0, 9.0],
    )
    assert sumw2_only.errors is not None
    np.testing.assert_allclose(sumw2_only.errors.value, [1.0, 2.0, 3.0])


@pytest.mark.xfail(
    strict=True,
    reason="Histogram.copy(deep=True) currently passes Quantity underflow/overflow "
    "back through Histogram.__init__, which evaluates Quantity truthiness.",
)
def test_copy_deep_returns_independent_quantities_and_metadata_contract():
    hist = Histogram(
        values=[1.0, 2.0] * u.ct,
        edges=[0.0, 1.0, 2.0] * u.s,
        sumw2=[0.5, 1.5] * u.ct**2,
        name="source",
        channel="H1:TEST",
    )

    copied = hist.copy(deep=True)
    copied.values[0] = 99.0 * u.ct
    copied.edges[0] = -1.0 * u.s
    copied.sumw2[0] = 42.0 * u.ct**2

    np.testing.assert_allclose(hist.values.value, [1.0, 2.0])
    np.testing.assert_allclose(hist.edges.value, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(hist.sumw2.value, [0.5, 1.5])
    assert copied.name == hist.name
    assert copied.channel == hist.channel


def test_fill_returns_new_histogram_with_weighted_flow_bins_and_uncertainties():
    hist = Histogram(
        values=[1.0, 2.0],
        edges=[0.0, 1.0, 2.0],
        unit="ct",
        xunit="m",
        sumw2=[0.5, 1.5],
        underflow=3.0,
        overflow=4.0,
        name="fill-source",
        channel="H1:TEST",
    )

    filled = hist.fill(
        data=[-1.0, 0.25, 0.75, 1.5, 2.5] * u.m,
        weights=[2.0, 3.0, 4.0, 5.0, 6.0] * u.ct,
    )

    assert filled is not hist
    assert filled.unit == u.ct
    assert filled.xunit == u.m
    assert filled.name == hist.name
    assert filled.channel == hist.channel
    np.testing.assert_allclose(hist.values.value, [1.0, 2.0])
    np.testing.assert_allclose(hist.sumw2.value, [0.5, 1.5])
    assert hist.underflow == 3.0 * u.ct
    assert hist.overflow == 4.0 * u.ct

    np.testing.assert_allclose(filled.values.value, [8.0, 7.0])
    np.testing.assert_allclose(filled.sumw2.value, [25.5, 26.5])
    assert filled.underflow == 5.0 * u.ct
    assert filled.overflow == 10.0 * u.ct
    assert filled.underflow_sumw2 == 4.0 * u.ct**2
    assert filled.overflow_sumw2 == 36.0 * u.ct**2


def test_rebin_and_integral_use_total_bin_semantics_and_preserve_metadata():
    hist = Histogram(
        values=[10.0, 20.0, 30.0],
        edges=[0.0, 1.0, 2.0, 3.0],
        unit="ct",
        xunit="s",
        sumw2=[1.0, 4.0, 9.0],
        cov=np.diag([1.0, 4.0, 9.0]) * u.ct**2,
        name="rebinned-source",
        channel="L1:TEST",
    )

    rebinned = hist.rebin([0.5, 1.5, 2.5] * u.s)

    assert rebinned.unit == u.ct
    assert rebinned.xunit == u.s
    assert rebinned.name == hist.name
    assert rebinned.channel == hist.channel
    np.testing.assert_allclose(rebinned.edges.value, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(rebinned.values.value, [15.0, 25.0])
    np.testing.assert_allclose(rebinned.sumw2.value, [1.25, 3.25])
    np.testing.assert_allclose(rebinned.cov.value, [[1.25, 1.0], [1.0, 3.25]])
    assert rebinned.sumw2.unit == u.ct**2
    assert rebinned.cov.unit == u.ct**2

    partial = hist.integral(0.5 * u.s, 2.25 * u.s)
    assert partial.unit == u.ct
    assert partial.value == 32.5


def test_histogram_collections_preserve_container_type_and_order_on_rebin():
    h1 = Histogram([10.0, 20.0], [0.0, 1.0, 2.0], unit="ct", xunit="s", name="a")
    h2 = Histogram([30.0, 40.0], [0.0, 1.0, 2.0], unit="ct", xunit="s", name="b")

    hist_dict = HistogramDict([("first", h1), ("second", h2)])
    rebinned_dict = hist_dict.rebin([0.5, 1.5] * u.s)

    assert type(rebinned_dict) is HistogramDict
    assert list(rebinned_dict) == ["first", "second"]
    assert [hist.name for hist in rebinned_dict.values()] == ["a", "b"]
    np.testing.assert_allclose(rebinned_dict["first"].values.value, [15.0])
    np.testing.assert_allclose(rebinned_dict["second"].values.value, [35.0])

    hist_list = HistogramList([h1, h2])
    rebinned_list = hist_list.rebin([0.5, 1.5] * u.s)

    assert type(rebinned_list) is HistogramList
    assert [hist.name for hist in rebinned_list] == ["a", "b"]
    np.testing.assert_allclose(rebinned_list[0].values.value, [15.0])
    np.testing.assert_allclose(rebinned_list[1].values.value, [35.0])


def test_segments_public_module_proxies_representative_gwpy_symbols():
    names = ("Segment", "SegmentList", "DataQualityFlag")

    for name in names:
        assert getattr(gwexpy_segments, name) is getattr(gwpy_segments, name)
        assert name in gwexpy_segments.__all__
        assert name in dir(gwexpy_segments)
