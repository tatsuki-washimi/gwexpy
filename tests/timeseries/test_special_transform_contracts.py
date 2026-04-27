"""Contract tests for gwexpy-owned special transform surfaces."""

import importlib

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.interop import _optional
from gwexpy.timeseries import TimeSeries
from gwexpy.types.time_plane_transform import LaplaceGram

pytest.importorskip("scipy", reason="scipy is required for special transforms")


def _regular_signal(*, n: int = 64, dt: u.Quantity = 0.01 * u.s) -> TimeSeries:
    dt_s = dt.to_value(u.s)
    t = np.arange(n) * dt_s
    data = np.sin(2 * np.pi * 10 * t) + 0.1 * np.cos(2 * np.pi * 5 * t)
    return TimeSeries(
        data,
        dt=dt,
        t0=123 * u.s,
        unit="V",
        name="sig",
        channel="H1:SIG",
    )


def _force_missing_optional(monkeypatch: pytest.MonkeyPatch, package_name: str) -> None:
    import_module = _optional.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == package_name:
            raise ImportError(f"missing {name}")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(_optional.importlib, "import_module", fake_import_module)


def test_dct_returns_frequency_series_with_dct_axis_and_metadata():
    ts = _regular_signal(n=16)

    dct = ts.dct(type=2, norm="ortho")

    assert isinstance(dct, FrequencySeries)
    assert dct.shape == ts.shape
    assert dct.unit == ts.unit
    assert dct.name == "sig_dct"
    assert dct.channel == ts.channel
    assert dct.epoch == ts.epoch
    assert dct.transform == "dct"
    assert dct.dct_type == 2
    assert dct.dct_norm == "ortho"
    assert dct.original_n == len(ts)
    assert dct.dt == ts.dt

    dt_s = ts.dt.to_value(u.s)
    expected_frequencies = np.arange(len(ts)) / (2 * len(ts) * dt_s)
    np.testing.assert_allclose(dct.frequencies.to_value(u.Hz), expected_frequencies)


def test_cepstrum_returns_dimensionless_quefrency_series_with_metadata():
    ts = _regular_signal(n=16)

    cepstrum = ts.cepstrum(kind="real", eps=1e-12)

    assert isinstance(cepstrum, FrequencySeries)
    assert cepstrum.shape == ts.shape
    assert cepstrum.unit == u.dimensionless_unscaled
    assert cepstrum.name == "sig_cepstrum"
    assert cepstrum.channel == ts.channel
    assert cepstrum.epoch == ts.epoch
    assert cepstrum.axis_type == "quefrency"
    assert cepstrum.transform == "cepstrum"
    assert cepstrum.cepstrum_kind == "real"
    assert cepstrum.original_n == len(ts)
    assert cepstrum.dt == ts.dt
    assert np.all(np.isfinite(cepstrum.value))

    dt_s = ts.dt.to_value(u.s)
    np.testing.assert_allclose(
        cepstrum.frequencies.to_value(u.s), np.arange(len(ts)) * dt_s
    )


def test_laplace_units_depend_on_normalization_and_preserve_axis_metadata():
    ts = _regular_signal(n=32)
    frequencies = [0, 10, 20] * u.Hz

    integral = ts.laplace(
        sigma=0.5 / u.s,
        frequencies=frequencies,
        normalize="integral",
    )
    mean = ts.laplace(
        sigma=0.5 / u.s,
        frequencies=frequencies,
        normalize="mean",
    )

    assert isinstance(integral, FrequencySeries)
    assert integral.unit == ts.unit * u.s
    assert integral.name == "sig_laplace"
    assert integral.channel == ts.channel
    assert integral.epoch == ts.epoch
    assert integral.laplace_sigma == pytest.approx(0.5)
    np.testing.assert_allclose(
        integral.frequencies.to_value(u.Hz), frequencies.to_value(u.Hz)
    )
    assert np.all(np.isfinite(integral.value))

    assert mean.unit == ts.unit
    assert mean.name == "sig_laplace"
    np.testing.assert_allclose(
        mean.frequencies.to_value(u.Hz), frequencies.to_value(u.Hz)
    )
    assert np.all(np.isfinite(mean.value))


def test_stlt_laplacegram_axes_units_metadata_and_time_centers():
    ts = _regular_signal(n=16)

    stlt = ts.stlt(stride="0.04s", window="0.08s", sigmas=[0, 1], scaling="dt")

    assert isinstance(stlt, LaplaceGram)
    assert stlt.kind == "stlt"
    assert stlt.shape == (3, 2, 5)
    assert stlt.unit == ts.unit * u.s
    assert [axis.name for axis in stlt.axes] == ["time", "sigma", "frequency"]
    np.testing.assert_allclose(stlt.times.to_value(u.s), [123.04, 123.08, 123.12])
    np.testing.assert_allclose(stlt.sigmas.to_value(1 / u.s), [0, 1])
    np.testing.assert_allclose(stlt.frequencies[:3].to_value(u.Hz), [0, 12.5, 25])
    assert stlt.meta["source"] == "sig"
    assert stlt.meta["window"] == pytest.approx(0.08)
    assert stlt.meta["stride"] == pytest.approx(0.04)
    assert stlt.meta["overlap"] == pytest.approx(0.04)

    sigma_plane = stlt.at_sigma(0 / u.s)
    assert sigma_plane.shape == (3, 5)
    assert sigma_plane.unit == stlt.unit


def test_stlt_explicit_quantity_frequencies_and_raw_scaling_unit():
    ts = _regular_signal(n=16)
    frequencies = [0, 12.5, 25] * u.Hz

    stlt = ts.stlt(
        stride="0.04s",
        window="0.08s",
        sigmas=[0, 1] / u.s,
        frequencies=frequencies,
        scaling="none",
    )

    assert stlt.shape == (3, 2, 3)
    assert stlt.unit == ts.unit
    np.testing.assert_allclose(stlt.sigmas.to_value(1 / u.s), [0, 1])
    np.testing.assert_allclose(
        stlt.frequencies.to_value(u.Hz), frequencies.to_value(u.Hz)
    )


def test_cwt_reports_analysis_extra_when_pywavelets_is_missing(monkeypatch):
    _force_missing_optional(monkeypatch, "pywt")

    with pytest.raises(ImportError) as excinfo:
        _regular_signal().cwt(widths=[1, 2], output="ndarray")

    message = str(excinfo.value)
    assert "pywt" in message
    assert "gwexpy[analysis]" in message


@pytest.mark.skipif(
    importlib.util.find_spec("pywt") is None,
    reason="pywt is required for installed CWT output contracts",
)
def test_cwt_width_output_contract_when_pywavelets_is_available():
    ts = _regular_signal(n=32)
    widths = np.arange(1, 5)

    coefficients, frequencies = ts.cwt(widths=widths, output="ndarray")

    assert coefficients.shape == (len(widths), len(ts))
    assert frequencies.unit == u.Hz
    assert np.all(np.isfinite(frequencies.to_value(u.Hz)))


def test_emd_and_hht_report_analysis_extra_when_pyemd_is_missing(monkeypatch):
    _force_missing_optional(monkeypatch, "PyEMD")
    ts = _regular_signal()

    with pytest.raises(ImportError) as emd_excinfo:
        ts.emd(method="emd")
    with pytest.raises(ImportError) as hht_excinfo:
        ts.hht(emd_method="emd")

    assert "PyEMD" in str(emd_excinfo.value)
    assert "gwexpy[analysis]" in str(emd_excinfo.value)
    assert "PyEMD" in str(hht_excinfo.value)
    assert "gwexpy[analysis]" in str(hht_excinfo.value)
