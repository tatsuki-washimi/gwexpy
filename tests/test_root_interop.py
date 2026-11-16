import os

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.spectrogram import Spectrogram, SpectrogramDict
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList

pytestmark = pytest.mark.root


def test_root_histogram_dtype_mapping_is_explicit():
    from gwexpy.interop.root_ import _root_histogram_dtype

    assert _root_histogram_dtype("TH1C") == np.dtype("int8")
    assert _root_histogram_dtype("TH2S") == np.dtype("int16")
    assert _root_histogram_dtype("TH1I") == np.dtype("int32")
    assert _root_histogram_dtype("TH3L") == np.dtype("int64")
    assert _root_histogram_dtype("TH1F") == np.dtype("float32")
    assert _root_histogram_dtype("TH2D") == np.dtype("float64")


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
    assert isinstance(f.Get("H1"), ROOT.TH1D)
    assert isinstance(f.Get("L1"), ROOT.TH1D)
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


def test_from_root_tmultigraph_raises(ROOT):
    ts = TimeSeries(np.arange(10, dtype=float), t0=0, dt=1, unit=u.V, name="a")
    mg = TimeSeriesList([ts]).to_tmultigraph()
    with pytest.raises(TypeError, match="TMultiGraph"):
        TimeSeries.from_root(mg)


def test_from_root_tfile_raises(ROOT, tmp_path):
    ts = TimeSeries(np.arange(10, dtype=float), t0=0, dt=1, unit=u.V, name="a")
    tsd = TimeSeriesDict({"a": ts})
    filename = str(tmp_path / "test_tfile.root")
    tsd.write(filename)
    f = ROOT.TFile.Open(filename)
    try:
        with pytest.raises(TypeError, match="TFile"):
            TimeSeries.from_root(f)
    finally:
        f.Close()


@pytest.mark.parametrize(
    ("suffix", "series_cls", "axis_title", "axis_unit"),
    [
        ("D", TimeSeries, "Time [ms]", u.ms),
        ("F", TimeSeries, "Time [ms]", u.ms),
        ("D", FrequencySeries, "Frequency [kHz]", u.kHz),
        ("F", FrequencySeries, "Frequency [kHz]", u.kHz),
    ],
)
def test_from_root_th1_to_series_preserves_values_coordinates_and_units(
    ROOT, suffix, series_cls, axis_title, axis_unit
):
    edges = np.array([0.0, 1.0, 3.0, 7.0])
    hist = getattr(ROOT, f"TH1{suffix}")(f"series_{suffix}", "", len(edges) - 1, edges)
    hist.GetXaxis().SetTitle(axis_title)
    hist.GetYaxis().SetTitle("Amplitude [mV]")
    for index, value in enumerate((1.25, 2.5, 4.0), start=1):
        hist.SetBinContent(index, value)

    result = series_cls.from_root(hist)

    np.testing.assert_allclose(result.value, [1.25, 2.5, 4.0])
    np.testing.assert_allclose(result.xindex.value, [0.5, 2.0, 5.0])
    assert result.xunit == axis_unit
    assert result.unit == u.mV
    assert result.name == f"series_{suffix}"


def test_from_root_th1_series_returns_sumw2_errors(ROOT):
    hist = ROOT.TH1D("series_errors", "", 2, 0.0, 2.0)
    hist.Sumw2()
    hist.GetYaxis().SetTitle("Amplitude [mV]")
    hist.SetBinContent(1, 4.0)
    hist.SetBinContent(2, 9.0)
    hist.SetBinError(1, 0.5)
    hist.SetBinError(2, 2.0)

    result, error = TimeSeries.from_root(hist, return_error=True)

    np.testing.assert_allclose(result.value, [4.0, 9.0])
    np.testing.assert_allclose(error.value, [0.5, 2.0])
    assert error.unit == result.unit == u.mV
    assert error.name == "series_errors_error"
    np.testing.assert_allclose(error.xindex.value, result.xindex.value)


def test_from_root_th1_series_uses_root_default_bin_errors_without_sumw2(ROOT):
    hist = ROOT.TH1D("series_poisson_errors", "", 2, 0.0, 2.0)
    hist.SetBinContent(1, 4.0)
    hist.SetBinContent(2, 9.0)
    assert hist.GetSumw2N() == 0

    _, error = FrequencySeries.from_root(hist, return_error=True)

    np.testing.assert_allclose(error.value, [2.0, 3.0])


def test_from_root_th1_series_warns_when_flow_bins_are_discarded(ROOT):
    hist = ROOT.TH1D("series_flow", "", 2, 0.0, 2.0)
    hist.SetBinContent(0, 3.0)
    hist.SetBinContent(3, 4.0)

    with pytest.warns(UserWarning, match="underflow or overflow"):
        result = TimeSeries.from_root(hist)

    np.testing.assert_allclose(result.value, [0.0, 0.0])


def test_from_root_th1_series_rejects_profiles_explicitly(ROOT):
    profile = ROOT.TProfile("profile_series", "", 2, 0.0, 2.0)

    with pytest.raises(TypeError, match="does not support TProfile"):
        TimeSeries.from_root(profile)
