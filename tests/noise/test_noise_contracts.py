"""Contract tests for gwexpy.noise public helpers.

These tests intentionally record current behavior for issue #278 without
changing physics, unit, metadata, or optional-backend runtime semantics.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.noise.colored import pink_noise, power_law
from gwexpy.noise.gwinc_ import from_pygwinc
from gwexpy.noise.magnetic import schumann_resonance
from gwexpy.noise.non_gaussian import transient_gaussian_noise
from gwexpy.noise.obspy_ import from_obspy
from gwexpy.noise.peaks import gaussian_line, lorentzian_line, voigt_line
from gwexpy.noise.wave import colored as wave_noise
from gwexpy.noise.wave import from_asd
from gwexpy.noise.wave import gaussian as gaussian_noise
from gwexpy.noise.wave import uniform as uniform_noise
from gwexpy.timeseries import TimeSeries


def _channel_name(series: FrequencySeries | TimeSeries) -> str | None:
    channel = getattr(series, "channel", None)
    return getattr(channel, "name", channel)


def test_colored_asd_helpers_preserve_axis_units_metadata_and_psd_relation() -> None:
    freqs = np.array([1.0, 4.0, 16.0]) * u.Hz
    amp = 8.0 * u.Unit("m / Hz^(1/2)")

    asd = pink_noise(
        amp,
        f_ref=1.0 * u.Hz,
        frequencies=freqs,
        name="pink-contract",
        channel="X1:PINK",
    )

    assert isinstance(asd, FrequencySeries)
    np.testing.assert_allclose(asd.frequencies.to_value(u.Hz), [1.0, 4.0, 16.0])
    np.testing.assert_allclose(asd.value, [8.0, 4.0, 2.0])
    assert asd.unit == amp.unit
    assert asd.name == "pink-contract"
    assert _channel_name(asd) == "X1:PINK"

    psd = asd.value**2
    np.testing.assert_allclose(psd, [64.0, 16.0, 4.0])


def test_float_colored_amplitude_uses_explicit_unit_without_rescaling() -> None:
    freqs = np.array([1.0, 2.0, 4.0])

    asd = power_law(
        1.0,
        amplitude=5.0,
        f_ref=1.0,
        frequencies=freqs,
        unit="pT / Hz^(1/2)",
    )

    np.testing.assert_allclose(asd.value, [5.0, 2.5, 1.25])
    assert asd.unit == u.Unit("pT / Hz^(1/2)")


def test_peak_helpers_are_asd_peak_normalized_reference_signals() -> None:
    freqs = np.array([8.0, 10.0, 12.0])

    lorentzian = lorentzian_line(
        10.0,
        3.0 * u.Unit("m / Hz^(1/2)"),
        gamma=2.0,
        frequencies=freqs,
    )
    gaussian = gaussian_line(10.0, 3.0, sigma=2.0, frequencies=freqs)
    voigt = voigt_line(10.0, 3.0, sigma=2.0, gamma=1.0, frequencies=freqs)

    expected_lorentzian = 3.0 * 2.0 / np.sqrt((freqs - 10.0) ** 2 + 2.0**2)
    expected_gaussian = 3.0 * np.exp(-((freqs - 10.0) ** 2) / (2.0 * 2.0**2))

    np.testing.assert_allclose(lorentzian.value, expected_lorentzian)
    np.testing.assert_allclose(gaussian.value, expected_gaussian)
    assert lorentzian.unit == u.Unit("m / Hz^(1/2)")
    assert lorentzian.value[1] == pytest.approx(3.0)
    assert gaussian.value[1] == pytest.approx(3.0)
    assert voigt.value[1] == pytest.approx(3.0)
    assert voigt.value[0] == pytest.approx(voigt.value[2])


def test_schumann_resonance_sums_modes_in_psd_space() -> None:
    freqs = np.array([10.0])
    modes = [(10.0, 10.0, 2.0), (10.0, 10.0, 3.0)]

    asd = schumann_resonance(
        frequencies=freqs,
        modes=modes,
        unit="pT / Hz^(1/2)",
        name="schumann-contract",
        channel="M1:MAG",
    )

    assert isinstance(asd, FrequencySeries)
    np.testing.assert_allclose(asd.frequencies.to_value(u.Hz), freqs)
    assert asd.value[0] == pytest.approx(np.sqrt(2.0**2 + 3.0**2))
    assert asd.unit == u.Unit("pT / Hz^(1/2)")
    assert asd.name == "schumann-contract"
    assert _channel_name(asd) == "M1:MAG"


def test_wave_noise_seed_and_generator_reproducibility_contracts() -> None:
    first_seeded = gaussian_noise(duration=1.0, sample_rate=8.0, seed=123)
    second_seeded = gaussian_noise(duration=1.0, sample_rate=8.0, seed=123)
    np.testing.assert_allclose(first_seeded.value, second_seeded.value)

    rng = np.random.default_rng(321)
    first_from_rng = uniform_noise(duration=1.0, sample_rate=8.0, rng=rng)
    second_from_same_rng = uniform_noise(duration=1.0, sample_rate=8.0, rng=rng)
    first_from_matching_rng = uniform_noise(
        duration=1.0,
        sample_rate=8.0,
        rng=np.random.default_rng(321),
    )

    assert not np.allclose(first_from_rng.value, second_from_same_rng.value)
    np.testing.assert_allclose(first_from_rng.value, first_from_matching_rng.value)

    first_colored = wave_noise(duration=1.0, sample_rate=8.0, exponent=0.5, seed=456)
    second_colored = wave_noise(duration=1.0, sample_rate=8.0, exponent=0.5, seed=456)
    np.testing.assert_allclose(first_colored.value, second_colored.value)


def test_from_asd_seed_metadata_and_unit_contract() -> None:
    freqs = np.array([0.0, 1.0, 2.0, 4.0])
    asd = FrequencySeries(
        np.ones_like(freqs) * 2.0,
        frequencies=freqs,
        unit=u.m / (u.Hz**0.5),
        name="ASD_CONTRACT",
        channel="X1:ASD",
    )

    first = from_asd(asd, duration=2.0, sample_rate=8.0, t0=100.0, seed=789)
    second = from_asd(asd, duration=2.0, sample_rate=8.0, t0=100.0, seed=789)

    assert isinstance(first, TimeSeries)
    assert len(first) == 16
    np.testing.assert_allclose(first.value, second.value)
    assert first.sample_rate.value == pytest.approx(8.0)
    assert first.t0.value == pytest.approx(100.0)
    assert first.unit.is_equivalent(u.m)
    assert first.name == "ASD_CONTRACT"
    assert _channel_name(first) == "X1:ASD"


def test_non_gaussian_transient_psd_argument_is_currently_ignored() -> None:
    freqs = np.array([1.0, 2.0, 4.0])
    psd = FrequencySeries(
        np.array([1.0, 10.0, 100.0]),
        frequencies=freqs,
        unit="m2 / Hz",
    )

    rng_state = np.random.get_state()
    try:
        np.random.seed(20260428)
        without_psd = transient_gaussian_noise(
            duration=2.0,
            sample_rate=24.0,
            A1=0.5,
            psd=None,
            unit="m",
            name="transient",
        )

        np.random.seed(20260428)
        with_psd = transient_gaussian_noise(
            duration=2.0,
            sample_rate=24.0,
            A1=0.5,
            psd=psd,
            unit="m",
            name="transient",
        )
    finally:
        np.random.set_state(rng_state)

    np.testing.assert_allclose(with_psd.value, without_psd.value)
    assert with_psd.unit == without_psd.unit == u.m
    assert with_psd.name == without_psd.name == "transient"


def test_optional_backend_import_errors_include_install_hints() -> None:
    with patch.dict(sys.modules, {"gwinc": None}):
        with pytest.raises(ImportError, match="install pygwinc"):
            from_pygwinc("aLIGO")

    missing_obspy = {
        "obspy": None,
        "obspy.signal": None,
        "obspy.signal.spectral_estimation": None,
    }
    with patch.dict(sys.modules, missing_obspy):
        with pytest.raises(ImportError, match="install obspy"):
            from_obspy("NLNM")


def test_pygwinc_frequency_axis_quantity_units_and_metadata_with_stub() -> None:
    captured: dict[str, np.ndarray] = {}

    class FakeBudget:
        ifo = SimpleNamespace(Infrastructure=SimpleNamespace(Length=5.0))

        def run(self, freq: np.ndarray) -> SimpleNamespace:
            captured["freq"] = np.asarray(freq, dtype=float)
            return SimpleNamespace(psd=np.full(len(freq), 4.0))

    fake_gwinc = SimpleNamespace(load_budget=lambda model: FakeBudget())

    with patch.dict(sys.modules, {"gwinc": fake_gwinc}):
        asd = from_pygwinc(
            "A+",
            fmin=10.0,
            fmax=20.0,
            df=5.0,
            quantity="darm",
            channel="H1:DARM",
        )

    np.testing.assert_allclose(captured["freq"], [10.0, 15.0, 20.0])
    np.testing.assert_allclose(asd.frequencies.to_value(u.Hz), [10.0, 15.0, 20.0])
    np.testing.assert_allclose(asd.value, [10.0, 10.0, 10.0])
    assert asd.unit == u.Unit("m / Hz^(1/2)")
    assert asd.name == "Aplus"
    assert _channel_name(asd) == "H1:DARM"


def test_obspy_units_interpolation_conversion_and_metadata_with_stub() -> None:
    periods = np.array([1.0, 0.5])
    psd_db = np.array([-20.0, -20.0])
    fake_spec = SimpleNamespace(
        get_nhnm=lambda: (periods.copy(), psd_db.copy()),
        get_nlnm=lambda: (periods.copy(), psd_db.copy()),
        get_idc_infra_hi_noise=lambda: (periods.copy(), psd_db.copy()),
        get_idc_infra_low_noise=lambda: (periods.copy(), psd_db.copy()),
    )
    fake_signal = SimpleNamespace(spectral_estimation=fake_spec)
    fake_obspy = SimpleNamespace(signal=fake_signal)
    modules = {
        "obspy": fake_obspy,
        "obspy.signal": fake_signal,
        "obspy.signal.spectral_estimation": fake_spec,
    }

    freqs = np.array([0.0, 1.0, 2.0])
    with patch.dict(sys.modules, modules):
        acceleration = from_obspy(
            "nlnm",
            frequencies=freqs,
            quantity="acceleration",
            channel="X1:SEIS",
        )
        velocity = from_obspy("NLNM", frequencies=freqs, quantity="velocity")
        pressure = from_obspy("IDCH", frequencies=freqs, quantity="displacement")

    np.testing.assert_allclose(acceleration.frequencies.to_value(u.Hz), freqs)
    np.testing.assert_allclose(acceleration.value, [0.0, 0.1, 0.1])
    assert acceleration.unit == u.Unit("m / s^2 / Hz^(1/2)")
    assert acceleration.name == "NLNM"
    assert _channel_name(acceleration) == "X1:SEIS"

    assert np.isnan(velocity.value[0])
    np.testing.assert_allclose(
        velocity.value[1:], [0.1 / (2 * np.pi), 0.1 / (4 * np.pi)]
    )
    assert velocity.unit == u.Unit("m / s / Hz^(1/2)")
    assert velocity.name == "NLNM"

    np.testing.assert_allclose(pressure.value, [0.0, 0.1, 0.1])
    assert pressure.unit == u.Unit("Pa / Hz^(1/2)")
    assert pressure.name == "IDCH"
