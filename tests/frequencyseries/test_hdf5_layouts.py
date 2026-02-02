from __future__ import annotations

import numpy as np

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)


def test_frequencyseriesdict_group_layout_roundtrip(tmp_path):
    freqs = np.arange(3.0)
    fs = FrequencySeries(np.arange(3.0), frequencies=freqs, unit="1")
    fsd = FrequencySeriesDict({"H1:ASD": fs})

    outfile = tmp_path / "fsd_group.h5"
    fsd.write(outfile, format="hdf5", layout="group")

    fsd2 = FrequencySeriesDict.read(outfile, format="hdf5")
    assert list(fsd2.keys()) == list(fsd.keys())
    np.testing.assert_allclose(fsd2["H1:ASD"].value, fs.value)


def test_frequencyserieslist_group_layout_roundtrip(tmp_path):
    freqs = np.arange(3.0)
    fs1 = FrequencySeries(np.arange(3.0), frequencies=freqs, unit="1")
    fs2 = FrequencySeries(np.arange(3.0) * 2, frequencies=freqs, unit="1")
    fsl = FrequencySeriesList([fs1, fs2])

    outfile = tmp_path / "fsl_group.h5"
    fsl.write(outfile, format="hdf5", layout="group")

    fsl2 = FrequencySeriesList.read(outfile, format="hdf5")
    assert len(fsl2) == len(fsl)
    np.testing.assert_allclose(fsl2[0].value, fs1.value)
