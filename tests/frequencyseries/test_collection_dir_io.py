from __future__ import annotations

import numpy as np

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)


def test_frequencyseriesdict_csv_directory_roundtrip(tmp_path):
    freqs = np.arange(5.0)
    a = FrequencySeries(np.arange(5.0), frequencies=freqs, unit="1", name="a")
    b = FrequencySeries(np.arange(5.0) * 2, frequencies=freqs, unit="1", name="b")
    fsd = FrequencySeriesDict({"H1:ASD": a, "L1:ASD": b})

    outdir = tmp_path / "fsd_csv"
    fsd.write(outdir, format="csv")

    fsd2 = FrequencySeriesDict.read(outdir, format="csv")
    assert list(fsd2.keys()) == list(fsd.keys())
    for k in fsd:
        np.testing.assert_allclose(fsd2[k].value, fsd[k].value)
        np.testing.assert_allclose(fsd2[k].frequencies.value, fsd[k].frequencies.value)
        assert str(fsd2[k].unit) == str(fsd[k].unit)


def test_frequencyserieslist_txt_directory_roundtrip(tmp_path):
    freqs = np.arange(4.0)
    a = FrequencySeries(np.arange(4.0), frequencies=freqs, unit="m", name="x/1")
    b = FrequencySeries(np.arange(4.0) * 3, frequencies=freqs, unit="m", name="y:2")
    fsl = FrequencySeriesList([a, b])

    outdir = tmp_path / "fsl_txt"
    fsl.write(outdir, format="txt")

    fsl2 = FrequencySeriesList.read(outdir, format="txt")
    assert len(fsl2) == len(fsl)
    for i in range(len(fsl)):
        np.testing.assert_allclose(fsl2[i].value, fsl[i].value)
        np.testing.assert_allclose(fsl2[i].frequencies.value, fsl[i].frequencies.value)
        assert str(fsl2[i].unit) == str(fsl[i].unit)
