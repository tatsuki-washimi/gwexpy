import os

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.spectrogram import Spectrogram, SpectrogramDict
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


def get_root():
    try:
        import ROOT

        ROOT.gROOT.SetBatch(True)
        return ROOT
    except ImportError:
        return None


@pytest.fixture
def ROOT():
    r = get_root()
    if r is None:
        pytest.skip("ROOT (pyroot) not installed")
    return r


def test_spectrogram_root(ROOT):
    times = np.linspace(0, 1, 10)
    freqs = np.linspace(0, 100, 5)
    data = np.random.rand(len(times), len(freqs))
    spec = Spectrogram(data, times=times, frequencies=freqs, unit=u.V)

    # 1. To TH2D
    h = spec.to_th2d()
    assert isinstance(h, ROOT.TH2D)
    assert h.GetNbinsX() == 10
    assert h.GetNbinsY() == 5

    # 2. From TH2D
    spec2 = Spectrogram.from_root(h)
    assert np.allclose(spec.value, spec2.value)
    assert np.allclose(spec.times.value, spec2.times.value)

    # 3. With errors
    error_data = np.random.rand(len(times), len(freqs)) * 0.1
    spec_err = Spectrogram(error_data, times=times, frequencies=freqs, unit=u.V)
    h_err = spec.to_th2d(error=spec_err)

    spec3 = Spectrogram.from_root(h_err)
    # Check if value matches
    assert np.allclose(spec3.value, spec.value)


def test_timeseries_dict_root(ROOT, tmp_path):
    ts1 = TimeSeries([1, 2, 3], t0=0, dt=1, name="ch1")
    ts2 = TimeSeries([4, 5, 6], t0=0, dt=1, name="ch2")
    tsd = TimeSeriesDict({"H1": ts1, "L1": ts2})

    # TMultiGraph
    mg = tsd.to_tmultigraph(name="mydict")
    assert isinstance(mg, ROOT.TMultiGraph)
    assert mg.GetListOfGraphs().GetSize() == 2

    # Write
    filename = str(tmp_path / "test_tsd.root")
    tsd.write(filename)
    assert os.path.exists(filename)

    f = ROOT.TFile.Open(filename)
    assert isinstance(f.Get("H1"), ROOT.TGraph)
    assert isinstance(f.Get("L1"), ROOT.TGraph)
    f.Close()


def test_timeseries_list_root(ROOT, tmp_path):
    ts1 = TimeSeries([1, 2], name="ts1")
    tsl = TimeSeriesList(ts1)

    filename = str(tmp_path / "test_tsl.root")
    tsl.write(filename)
    assert os.path.exists(filename)
    f = ROOT.TFile.Open(filename)
    assert f.Get("ts1")
    f.Close()


def test_frequencyseries_dict_root(ROOT, tmp_path):
    fs1 = FrequencySeries([10, 20], frequencies=[1, 2], name="fs1")
    fsd = FrequencySeriesDict({"A": fs1})

    filename = str(tmp_path / "test_fsd.root")
    fsd.write(filename, format="root")
    assert os.path.exists(filename)
    f = ROOT.TFile.Open(filename)
    assert f.Get("A")
    f.Close()


def test_spectrogram_dict_root(ROOT, tmp_path):
    times = np.linspace(0, 1, 5)
    freqs = np.linspace(0, 10, 3)
    spec = Spectrogram(np.zeros((5, 3)), times=times, frequencies=freqs)
    sd = SpectrogramDict({"S1": spec})

    filename = str(tmp_path / "test_sd.root")
    sd.write(filename)
    assert os.path.exists(filename)
    f = ROOT.TFile.Open(filename)
    assert isinstance(f.Get("S1"), ROOT.TH2D)
    f.Close()
