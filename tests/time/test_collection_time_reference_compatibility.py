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

        def make_spectrogram():
            return Spectrogram(
                np.ones((4, 2)),
                times=np.arange(4) * 0.25 * u.s,
                frequencies=np.arange(2) * u.Hz,
            )

        relative, absolute = make_spectrogram(), make_spectrogram()
        set_time_reference(relative, "relative")
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
