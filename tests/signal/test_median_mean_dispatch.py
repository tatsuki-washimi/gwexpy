"""Tests for GWexpy's GWpy-compatible median-mean PSD extension."""

import numpy as np
import pytest
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries as GWpyFrequencySeries
from gwpy.signal import spectral as gwpy_spectral

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.signal.spectral import median_mean
from gwexpy.timeseries import TimeSeries

PERIODIC_HANN_8 = np.array(
    [
        0.0,
        0.14644660940672627,
        0.5,
        0.8535533905932737,
        1.0,
        0.8535533905932737,
        0.5,
        0.14644660940672627,
    ],
)

# Frozen values from a direct lal.REAL8AverageSpectrumMedianMean call with
# the deterministic fixture below, segment length 8, stride 4, and the
# periodic Hann window above.  The wrapper registered by GWexpy is not used
# to produce this independent primary-backend oracle.
DIRECT_LAL_MEDIAN_MEAN_ORACLE = np.array(
    [
        5.2039104290153642e-04,
        4.5514643305049218e-01,
        1.3215889243883394e-01,
        1.4029171593092043e-03,
        2.2255780045281741e-05,
    ],
)


def _fixture_timeseries() -> TimeSeries:
    """Build a deterministic, exactly segmentable physical fixture."""
    samples = np.arange(60, dtype=float)
    data = np.sin(2 * np.pi * samples / 8) + 0.25 * np.cos(2 * np.pi * samples / 5)
    return TimeSeries(
        data,
        sample_rate=8,
        unit=u.V,
        t0=1234567890,
        name="median-mean-fixture",
        channel="H1:MEDIAN_MEAN",
    )


def test_median_mean_is_owned_and_registered_through_gwpy_surface():
    """Expose the GWexpy extension through GWpy's normal method lookup."""
    assert gwpy_spectral.get_method("median-mean") is median_mean


def test_median_mean_public_dispatch_contract_without_lal(monkeypatch):
    """Exercise PSD and ASD contracts through dispatch with a fake backend."""
    import gwexpy.signal.spectral._median_mean as median_mean_module

    backend_calls = []

    def fake_lal_median_mean(
        timeseries,
        segmentlength,
        noverlap=None,
        window=None,
        plan=None,
    ):
        backend_calls.append((segmentlength, noverlap, window, plan))
        return GWpyFrequencySeries(
            [1.0, 4.0, 9.0, 16.0, 25.0],
            f0=0,
            df=1,
            unit=timeseries.unit**2 / u.Hz,
            epoch=timeseries.epoch,
            name=timeseries.name,
        )

    monkeypatch.setattr(median_mean_module, "_lal_median_mean", fake_lal_median_mean)

    timeseries = _fixture_timeseries()
    kwargs = {"fftlength": 1, "overlap": 0.5, "window": "hann"}
    psd = timeseries.psd(method="median-mean", **kwargs)
    asd = timeseries.asd(method="median-mean", **kwargs)

    assert len(backend_calls) == 2
    for segmentlength, noverlap, window, plan in backend_calls:
        assert segmentlength == 8
        assert noverlap == 4
        assert isinstance(window, np.ndarray)
        np.testing.assert_allclose(window, PERIODIC_HANN_8, rtol=0, atol=0)
        assert plan is None
    assert isinstance(psd, FrequencySeries)
    assert isinstance(asd, FrequencySeries)
    assert psd.unit == u.V**2 / u.Hz
    assert asd.unit == u.V / u.Hz**0.5
    np.testing.assert_allclose(psd.frequencies.value, np.arange(5, dtype=float))
    np.testing.assert_allclose(asd.frequencies.value, psd.frequencies.value)
    np.testing.assert_allclose(asd.value, np.sqrt(psd.value))

    for result in (psd, asd):
        assert result.name == timeseries.name
        assert result.channel == timeseries.channel
        assert result.epoch == timeseries.epoch


def _direct_lal_median_mean(timeseries: TimeSeries) -> np.ndarray:
    """Evaluate the public LAL median-mean routine independently."""
    lal = pytest.importorskip("lal")
    lal_timeseries = timeseries.to_lal()
    sequence = lal.CreateREAL8Sequence(PERIODIC_HANN_8.size)
    sequence.data = PERIODIC_HANN_8.copy()
    window = lal.CreateREAL8WindowFromSequence(sequence)
    plan = lal.CreateForwardREAL8FFTPlan(8, 1)
    spectrum = lal.CreateREAL8FrequencySeries(
        timeseries.name or "",
        lal_timeseries.epoch,
        0.0,
        1.0 / 8,
        lal.StrainUnit,
        5,
    )
    assert (
        lal.REAL8AverageSpectrumMedianMean(
            spectrum,
            lal_timeseries,
            8,
            4,
            window,
            plan,
        )
        == 0
    )
    return np.asarray(spectrum.data.data).copy()


def test_median_mean_public_psd_matches_direct_lal_oracle():
    """Match an independent LAL calculation, including PSD metadata."""
    pytest.importorskip("lal")
    timeseries = _fixture_timeseries()
    psd = timeseries.psd(
        method="median-mean",
        fftlength=1,
        overlap=0.5,
        window="hann",
    )

    reference = _direct_lal_median_mean(timeseries)
    np.testing.assert_allclose(
        reference,
        DIRECT_LAL_MEDIAN_MEAN_ORACLE,
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_allclose(psd.value, reference, rtol=1e-6, atol=0)
    assert isinstance(psd, FrequencySeries)
    assert psd.unit == u.V**2 / u.Hz
    np.testing.assert_allclose(psd.frequencies.value, np.arange(5, dtype=float))
    assert psd.name == timeseries.name
    assert psd.channel == timeseries.channel
    assert psd.epoch == timeseries.epoch


def test_median_mean_public_psd_matches_direct_pycbc_oracle():
    """Match PyCBC's independent median-mean implementation when available."""
    pycbc_psd = pytest.importorskip("pycbc.psd")
    from pycbc.types import TimeSeries as PyCBCTimeSeries

    timeseries = _fixture_timeseries()
    psd = timeseries.psd(
        method="median-mean",
        fftlength=1,
        overlap=0.5,
        window="hann",
    )
    reference = pycbc_psd.welch(
        PyCBCTimeSeries(np.asarray(timeseries.value), delta_t=timeseries.dt.value),
        seg_len=8,
        seg_stride=4,
        window=PERIODIC_HANN_8,
        avg_method="median-mean",
        require_exact_data_fit=True,
    )
    reference_values = np.asarray(reference.numpy())
    np.testing.assert_allclose(
        reference_values,
        DIRECT_LAL_MEDIAN_MEAN_ORACLE,
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_allclose(psd.value, reference_values, rtol=1e-6, atol=0)


def test_median_mean_psd_and_asd_preserve_contract():
    """Preserve class, units, axes, metadata, and ASD/PSD numerical identity."""
    pytest.importorskip("lal")
    timeseries = _fixture_timeseries()
    kwargs = {"fftlength": 1, "overlap": 0.5, "window": "hann"}

    psd = timeseries.psd(method="median-mean", **kwargs)
    asd = timeseries.asd(method="median-mean", **kwargs)

    assert isinstance(psd, FrequencySeries)
    assert isinstance(asd, FrequencySeries)
    assert psd.unit == u.V**2 / u.Hz
    assert asd.unit == u.V / u.Hz**0.5
    np.testing.assert_allclose(psd.frequencies.value, np.arange(5, dtype=float))
    np.testing.assert_allclose(asd.frequencies.value, psd.frequencies.value)
    np.testing.assert_allclose(asd.value, np.sqrt(psd.value), rtol=1e-12, atol=1e-15)

    for result in (psd, asd):
        assert result.name == timeseries.name
        assert result.channel == timeseries.channel
        assert result.epoch == timeseries.epoch
