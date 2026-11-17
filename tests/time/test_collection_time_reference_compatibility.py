"""Collection-to-matrix conversions must not discard mixed timing states."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
)
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.time._reference import set_time_reference
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


@pytest.mark.parametrize(
    ("family", "container"),
    [
        ("timeseries", "list"),
        ("timeseries", "dict"),
        ("frequencyseries", "dict"),
        ("spectrogram", "list"),
        ("spectrogram", "dict"),
    ],
)
@pytest.mark.parametrize("relative_first", [True, False])
def test_collection_to_matrix_rejects_mixed_time_reference(
    family, relative_first, container
):
    if family == "timeseries":
        collection_type = TimeSeriesList if container == "list" else TimeSeriesDict
        relative = TimeSeries(np.arange(4), dt=0.25)
        absolute = TimeSeries(np.arange(4), dt=0.25, t0=0)
    elif family == "frequencyseries":
        collection_type = FrequencySeriesDict
        relative = FrequencySeries(np.arange(4), df=1)
        absolute = FrequencySeries(np.arange(4), df=1, epoch=0)
    else:
        collection_type = SpectrogramList if container == "list" else SpectrogramDict

        def make_spectrogram(*, epoch=None):
            return Spectrogram(
                np.ones((4, 2)),
                times=np.arange(4) * 0.25 * u.s,
                frequencies=np.arange(2) * u.Hz,
                epoch=epoch,
            )

        relative, absolute = make_spectrogram(), make_spectrogram(epoch=0)
    ordered = [relative, absolute] if relative_first else [absolute, relative]
    if container == "list":
        collection = collection_type(ordered)
    else:
        items = (
            {"relative": relative, "absolute": absolute}
            if relative_first
            else {"absolute": absolute, "relative": relative}
        )
        collection = collection_type(items)

    with pytest.raises(ValueError, match="time_reference"):
        collection.to_matrix()


@pytest.mark.parametrize("relative_first", [True, False])
def test_timeserieslist_join_rejects_mixed_time_reference(relative_first):
    relative = TimeSeries(np.arange(4.0), dt=0.25)
    absolute = TimeSeries(np.arange(4.0), dt=0.25, t0=1.0)
    ordered = [relative, absolute] if relative_first else [absolute, relative]

    with pytest.raises(ValueError, match="time_reference"):
        TimeSeriesList(ordered).join(pad=0.0, gap="pad")


def test_timeserieslist_join_preserves_uniform_time_reference():
    rel_a = TimeSeries(np.arange(4.0), dt=0.25)
    rel_b = TimeSeries(np.arange(4.0), dt=0.25)[4:]
    rel_joined = TimeSeriesList([rel_a, rel_b]).join(pad=0.0, gap="pad")
    assert rel_joined.time_reference == "relative"

    abs_a = TimeSeries(np.arange(4.0), dt=0.25, t0=1_400_000_000.0)
    abs_b = TimeSeries(np.arange(4.0), dt=0.25, t0=1_400_000_001.0)
    abs_joined = TimeSeriesList([abs_a, abs_b]).join(pad=0.0, gap="pad")
    assert abs_joined.time_reference == "absolute"
