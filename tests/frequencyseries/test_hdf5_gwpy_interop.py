from __future__ import annotations

import numpy as np
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)
from gwexpy.io.hdf5_collection import safe_hdf5_key


def test_frequencyseriesdict_hdf5_gwpy_read(tmp_path):
    freqs = np.arange(5.0)
    fs = FrequencySeries(np.arange(5.0), frequencies=freqs, unit="1")
    fsd = FrequencySeriesDict({"H1:ASD": fs})

    outfile = tmp_path / "fsd.h5"
    fsd.write(outfile, format="hdf5")

    path = safe_hdf5_key("H1:ASD")
    gw = GwpyFrequencySeries.read(outfile, format="hdf5", path=path)
    np.testing.assert_allclose(gw.value, fs.value)
    np.testing.assert_allclose(gw.frequencies.value, fs.frequencies.value)


def test_frequencyserieslist_hdf5_gwpy_read(tmp_path):
    freqs = np.arange(4.0)
    fs1 = FrequencySeries(np.arange(4.0), frequencies=freqs, unit="1")
    fs2 = FrequencySeries(np.arange(4.0) * 2, frequencies=freqs, unit="1")
    fsl = FrequencySeriesList([fs1, fs2])

    outfile = tmp_path / "fsl.h5"
    fsl.write(outfile, format="hdf5")

    gw = GwpyFrequencySeries.read(outfile, format="hdf5", path="0")
    np.testing.assert_allclose(gw.value, fs1.value)
    np.testing.assert_allclose(gw.frequencies.value, fs1.frequencies.value)
