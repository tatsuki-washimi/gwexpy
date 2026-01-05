import numpy as np
import pytest
from astropy import units as u
from gwpy.plot import Plot

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict, FrequencySeriesList


def _make_frequencyseries(scale: float = 1.0, name: str = "fs") -> FrequencySeries:
    freqs = np.arange(0, 11) * u.Hz
    data = (np.arange(11, dtype=float) * scale).copy()
    return FrequencySeries(data, frequencies=freqs, unit="1", name=name)


def test_frequencyseries_dictclass_binding():
    assert FrequencySeries.DictClass is FrequencySeriesDict


def test_frequencyseriesdict_type_constraint_setitem():
    d = FrequencySeriesDict()
    with pytest.raises(TypeError):
        d["a"] = object()


def test_frequencyseriesdict_type_constraint_update():
    d = FrequencySeriesDict()
    with pytest.raises(TypeError):
        d.update({"a": object()})


def test_frequencyseriesdict_type_constraint_setdefault_none_rejected():
    d = FrequencySeriesDict()
    with pytest.raises(TypeError):
        d.setdefault("a")


def test_frequencyseriesdict_copy_deepcopy_value_not_shared():
    d = FrequencySeriesDict({"a": _make_frequencyseries()})
    d2 = d.copy()
    assert isinstance(d2, FrequencySeriesDict)
    assert not np.shares_memory(d["a"].value, d2["a"].value)


def test_frequencyseriesdict_crop_maps_to_elements_and_returns_dict():
    d = FrequencySeriesDict({"a": _make_frequencyseries(), "b": _make_frequencyseries(2.0)})
    cropped = d.crop(2, 5)
    assert cropped is d  # in-place like GWpy
    assert len(d["a"]) == len(_make_frequencyseries().crop(2, 5))
    assert len(d["b"]) == len(_make_frequencyseries(2.0).crop(2, 5))

    with pytest.raises(TypeError):
        d.crop(0, 1, unexpected=True)


def test_frequencyseriesdict_plot_separate_axes_count_matches():
    d = FrequencySeriesDict({"a": _make_frequencyseries(), "b": _make_frequencyseries(2.0)})
    plot = d.plot(separate=True)
    try:
        assert isinstance(plot, Plot)
        assert len(plot.axes) == len(d)
    finally:
        plot.close()


def test_frequencyserieslist_type_constraint_append():
    l = FrequencySeriesList()
    with pytest.raises(TypeError):
        l.append(object())


def test_frequencyserieslist_slice_returns_same_type():
    l = FrequencySeriesList(_make_frequencyseries(), _make_frequencyseries(2.0))
    sliced = l[:1]
    assert isinstance(sliced, FrequencySeriesList)
    assert len(sliced) == 1


def test_frequencyserieslist_copy_deepcopy_value_not_shared():
    l = FrequencySeriesList(_make_frequencyseries(), _make_frequencyseries(2.0))
    l2 = l.copy()
    assert isinstance(l2, FrequencySeriesList)
    assert not np.shares_memory(l[0].value, l2[0].value)


def test_frequencyserieslist_plot_does_not_error():
    l = FrequencySeriesList(_make_frequencyseries(), _make_frequencyseries(2.0))
    plot = l.plot()
    try:
        assert isinstance(plot, Plot)
    finally:
        plot.close()


def test_frequencyseriesdict_span_uses_xspan():
    d = FrequencySeriesDict({"a": _make_frequencyseries(), "b": _make_frequencyseries(2.0)})
    span = d.span
    # xspan is Segment; span should also be Segment covering the union
    assert span.start == d["a"].xspan.start
    assert span.end == d["a"].xspan.end


def test_frequencyserieslist_segments_uses_xspan():
    l = FrequencySeriesList(_make_frequencyseries(), _make_frequencyseries(2.0))
    segments = l.segments
    assert len(segments) == 2
    assert segments[0].start == l[0].xspan.start
    assert segments[0].end == l[0].xspan.end
