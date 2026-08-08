"""List-operator compatibility for ``SpectrogramList``."""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList
from gwexpy.histogram import Histogram, HistogramList
from gwexpy.spectrogram import Spectrogram, SpectrogramList
from gwexpy.timeseries import TimeSeries, TimeSeriesList


def _spectrogram_list() -> SpectrogramList:
    return SpectrogramList([Spectrogram(np.ones((2, 2)), dt=1, f0=0, df=1)])


def _timeseries_list() -> TimeSeriesList:
    return TimeSeriesList(TimeSeries([1, 2], sample_rate=1))


def _frequencyseries_list() -> FrequencySeriesList:
    return FrequencySeriesList(FrequencySeries([1, 2], df=1))


@pytest.mark.parametrize(
    "collection_factory",
    [_timeseries_list, _frequencyseries_list, _spectrogram_list],
)
def test_series_list_binary_operators_return_native_lists(collection_factory):
    """All series-like lists share GWpy's binary-list operator semantics."""
    collection = collection_factory()

    for result in (
        [] + collection,
        collection + [],
        2 * collection,
        collection * 2,
    ):
        assert type(result) is list


def test_histogramlist_keeps_its_documented_safe_repetition_exception():
    """Measurements must not be duplicated by list repetition."""
    collection = HistogramList(Histogram([1, 2], [0, 1, 2]))

    for operation in (
        lambda: 2 * collection,
        lambda: collection * 2,
        lambda: collection.__imul__(2),
    ):
        with pytest.raises(TypeError, match="duplicate measurements"):
            operation()
