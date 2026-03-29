import os
import pytest
import numpy as np
from astropy import units as u
from gwexpy.histogram import Histogram, HistogramDict

try:
    import ROOT
except ImportError:
    ROOT = None

@pytest.mark.skipif(ROOT is None, reason="ROOT not installed")
def test_histogram_to_th1d():
    values = [10.0, 20.0, 30.0]
    edges = np.array([0.0, 1.0, 2.0, 5.0])
    h = Histogram(values, edges, unit="ct", xunit="m", name="test_hist")
    
    th1 = h.to_th1d()
    assert th1.GetName() == "test_hist"
    assert th1.GetNbinsX() == 3
    
    for i in range(3):
        assert np.isclose(th1.GetBinContent(i + 1), values[i])
        assert np.isclose(th1.GetXaxis().GetBinLowEdge(i + 1), edges[i])
    assert np.isclose(th1.GetXaxis().GetBinUpEdge(3), edges[3])

@pytest.mark.skipif(ROOT is None, reason="ROOT not installed")
def test_histogram_from_root():
    edges = np.array([0.0, 1.0, 3.0, 10.0])
    th1 = ROOT.TH1D("h_root", "h_root", 3, edges)
    th1.SetBinContent(1, 100)
    th1.SetBinContent(2, 200)
    th1.SetBinContent(3, 300)
    th1.SetBinError(1, 10)
    th1.SetBinError(2, 20)
    th1.SetBinError(3, 30)
    th1.GetXaxis().SetTitle("x [m]")
    th1.GetYaxis().SetTitle("y [ct]")
    
    h = Histogram.from_root(th1)
    assert h.nbins == 3
    assert np.allclose(h.values.value, [100, 200, 300])
    assert np.allclose(h.edges.value, edges)
    assert h.unit == u.Unit("ct")
    assert h.xunit == u.Unit("m")
    
    # Check Sumw2 (variance)
    assert np.allclose(h.sumw2.value, [100, 400, 900])

@pytest.mark.skipif(ROOT is None, reason="ROOT not installed")
def test_histogram_dict_write_root(tmp_path):
    h1 = Histogram([10, 20], [0, 1, 2], name="h1")
    hd = HistogramDict({"a": h1})
    
    fpath = str(tmp_path / "test.root")
    # write calls write_root_file
    hd.write(fpath, format="root")
    
    assert os.path.exists(fpath)
    f = ROOT.TFile.Open(fpath)
    h = f.Get("a")
    assert h.GetBinContent(1) == 10
    f.Close()
