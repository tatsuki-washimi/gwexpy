
import numpy as np
import os
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict, FrequencySeriesList
from gwexpy.spectrogram import Spectrogram, SpectrogramDict

# Try importing ROOT; skip tests if not available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False

pytestmark = pytest.mark.skipif(not HAS_ROOT, reason="ROOT (pyroot) not installed")

def test_spectrogram_root():
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

def test_timeseries_dict_root():
    ts1 = TimeSeries([1, 2, 3], t0=0, dt=1, name="ch1")
    ts2 = TimeSeries([4, 5, 6], t0=0, dt=1, name="ch2")
    tsd = TimeSeriesDict({"H1": ts1, "L1": ts2})
    
    # TMultiGraph
    mg = tsd.to_tmultigraph(name="mydict")
    assert isinstance(mg, ROOT.TMultiGraph)
    assert mg.GetListOfGraphs().GetSize() == 2
    
    # Write
    filename = "test_tsd.root"
    if os.path.exists(filename): os.remove(filename)
    tsd.write(filename)
    assert os.path.exists(filename)
    
    f = ROOT.TFile.Open(filename)
    assert isinstance(f.Get("H1"), ROOT.TGraph)
    assert isinstance(f.Get("L1"), ROOT.TGraph)
    f.Close()
    if os.path.exists(filename): os.remove(filename)

def test_timeseries_list_root():
    ts1 = TimeSeries([1, 2], name="ts1")
    tsl = TimeSeriesList(ts1)
    
    filename = "test_tsl.root"
    tsl.write(filename)
    assert os.path.exists(filename)
    f = ROOT.TFile.Open(filename)
    assert f.Get("ts1")
    f.Close()
    if os.path.exists(filename): os.remove(filename)

def test_frequencyseries_dict_root():
    fs1 = FrequencySeries([10, 20], frequencies=[1, 2], name="fs1")
    fsd = FrequencySeriesDict({"A": fs1})
    
    filename = "test_fsd.root"
    fsd.write(filename, format="root")
    assert os.path.exists(filename)
    f = ROOT.TFile.Open(filename)
    assert f.Get("A")
    f.Close()
    if os.path.exists(filename): os.remove(filename)

def test_spectrogram_dict_root():
    times = np.linspace(0, 1, 5)
    freqs = np.linspace(0, 10, 3)
    spec = Spectrogram(np.zeros((5, 3)), times=times, frequencies=freqs)
    sd = SpectrogramDict({"S1": spec})
    
    filename = "test_sd.root"
    sd.write(filename)
    assert os.path.exists(filename)
    f = ROOT.TFile.Open(filename)
    assert isinstance(f.Get("S1"), ROOT.TH2D)
    f.Close()
    if os.path.exists(filename): os.remove(filename)
