"""Tests for #544 ScalarField.fft_space spacing_rtol, E-1/E-2/E-3 round-trip, and Parseval."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField


def _make_wave_field(x_vals: np.ndarray, *, k0: float = 2.0) -> ScalarField:
    x = x_vals * u.m
    data = np.cos(k0 * x_vals)[None, :, None, None] * np.ones((1, 1, 2, 2))
    return ScalarField(
        data,
        unit=u.V,
        axis0=np.array([0.0]) * u.s,
        axis1=x,
        axis2=np.array([0.0, 1.0]) * u.m,
        axis3=np.array([0.0, 1.0]) * u.m,
    )


def test_fft_space_spacing_rtol_opt_in_and_exact_roundtrip():
    # Construct linspace grid with ~1e-8 spacing perturbation (fails strict regular)
    n = 64
    base = np.arange(n, dtype=np.float64) * 0.25
    rng = np.random.default_rng(42)
    x_approx = base + rng.uniform(-1e-8, 1e-8, size=n) * 0.25
    sf = _make_wave_field(x_approx, k0=2.0)

    # Strict default rejects approximate grid
    with pytest.raises(ValueError, match="spacing_rtol"):
        sf.fft_space("x")

    # Negative / invalid spacing_rtol fails
    with pytest.raises(ValueError):
        sf.fft_space("x", spacing_rtol=-1e-5)

    # Opt-in spacing_rtol warns and succeeds, and ifft_space restores exact E-1 coords
    with pytest.warns(UserWarning, match="approximately uniform"):
        kf = sf.fft_space("x", spacing_rtol=1e-5)

    assert kf.space_domains["kx"] == "k"
    rt = kf.ifft_space("kx")
    np.testing.assert_allclose(rt.value, sf.value, atol=1e-12)
    np.testing.assert_array_equal(rt._axis1_index.value, x_approx)
    assert rt.space_domains["x"] == "real"


def test_fft_space_zero_pad_falls_back_to_e3_mean_delta():
    n = 30
    rng = np.random.default_rng(123)
    dx_true = 7.0 / n
    x_approx = (
        np.arange(n, dtype=np.float64) * dx_true
        + rng.uniform(-1e-8, 1e-8, size=n) * dx_true
    )
    sf = _make_wave_field(x_approx)
    with pytest.warns(UserWarning, match="approximately uniform"):
        kf = sf.fft_space("x", n=64, spacing_rtol=1e-5)
    rt = kf.ifft_space("kx")
    assert rt.shape[1] == 64
    dx = rt._axis1_index.value[1] - rt._axis1_index.value[0]
    assert dx == pytest.approx(dx_true, rel=1e-7)


def test_fft_space_parseval_and_scale_invariance():
    n = 128
    dx = 0.1
    x = np.arange(n, dtype=np.float64) * dx
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(n)
    sf = ScalarField(
        signal[None, :, None, None],
        unit=u.V,
        axis0=np.array([0.0]) * u.s,
        axis1=x * u.m,
        axis2=np.array([0.0]) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    kf = sf.fft_space("x")
    # Parseval for unnormalized 2-sided FFT with scipy default: sum(|f|^2) == sum(|F|^2)/N
    energy_x = np.sum(np.abs(sf.value) ** 2)
    energy_k = np.sum(np.abs(kf.value) ** 2) / n
    assert energy_k == pytest.approx(energy_x, rel=1e-12)

    # Scale invariance on tiny coordinates (X * 1e-20)
    x_tiny = x * 1e-20
    sf_tiny = ScalarField(
        signal[None, :, None, None],
        unit=u.V,
        axis0=np.array([0.0]) * u.s,
        axis1=x_tiny * u.m,
        axis2=np.array([0.0]) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    kf_tiny = sf_tiny.fft_space("x", spacing_rtol=1e-9)
    np.testing.assert_allclose(kf_tiny.value, kf.value, rtol=1e-12)
    rt_tiny = kf_tiny.ifft_space("kx")
    np.testing.assert_allclose(rt_tiny.value, sf_tiny.value, atol=1e-12)
    np.testing.assert_allclose(rt_tiny._axis1_index.value, x_tiny, rtol=1e-12)


def test_ifft_space_rejects_nonuniform_k_axis():
    k_nonuniform = np.array([0.0, 0.5, 1.5, 4.0]) * (1 / u.m)
    kf = ScalarField(
        np.ones((1, 4, 1, 1), dtype=complex),
        unit=u.V,
        axis0=np.array([0.0]) * u.s,
        axis1=k_nonuniform,
        axis2=np.array([0.0]) * u.m,
        axis3=np.array([0.0]) * u.m,
        axis_names=("t", "kx", "y", "z"),
        space_domain={"kx": "k", "y": "real", "z": "real"},
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        kf.ifft_space("kx")
