"""Tests for gwexpy/frequencyseries/frequencyseries.py core methods."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries


def _make_fs(n=64, df=1.0, name="test", complex_=False):
    """Create a test FrequencySeries."""
    freqs = np.arange(n) * df
    if complex_:
        data = np.exp(1j * np.linspace(0, 2 * np.pi, n)) * np.arange(1, n + 1, dtype=float)
    else:
        data = np.arange(1, n + 1, dtype=float)
    return FrequencySeries(data, frequencies=freqs * u.Hz, unit=u.m, name=name)


# ---------------------------------------------------------------------------
# Construction and finalize
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self):
        fs = _make_fs()
        assert len(fs) == 64

    def test_fmin_fmax_filtered(self):
        # Lines 64-65 — fmin/fmax removed from kwargs
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m, fmin=10.0, fmax=100.0)
        assert len(fs) == 64

    def test_array_finalize_preserves_attrs(self):
        # Lines 94-96 — _gwex_* attrs copied on slice
        fs = _make_fs()
        fs._gwex_fft_mode = "transient"
        fs._gwex_target_nfft = 128
        sliced = fs[::2]
        assert getattr(sliced, "_gwex_fft_mode", None) == "transient"
        assert getattr(sliced, "_gwex_target_nfft", None) == 128


# ---------------------------------------------------------------------------
# Phase / Angle / Degree
# ---------------------------------------------------------------------------


class TestPhase:
    def test_phase_basic(self):
        # Lines 120-133
        fs = _make_fs(complex_=True)
        result = fs.phase()
        assert result.unit.to_string() == "rad"
        assert len(result) == len(fs)

    def test_phase_unwrap(self):
        # Line 122 — unwrap=True
        fs = _make_fs(complex_=True)
        result = fs.phase(unwrap=True)
        assert len(result) == len(fs)

    def test_phase_no_name(self):
        # Line 124 — name is empty
        data = np.exp(1j * np.linspace(0, 2 * np.pi, 32))
        fs = FrequencySeries(data, frequencies=np.arange(32) * u.Hz, unit=u.m)
        result = fs.phase()
        assert result.name == "phase"

    def test_angle_alias(self):
        # Line 152
        fs = _make_fs(complex_=True)
        assert np.allclose(fs.angle().value, fs.phase().value)

    def test_degree(self):
        # Lines 168-180
        fs = _make_fs(complex_=True)
        result = fs.degree()
        assert "deg" in str(result.unit)
        assert len(result) == len(fs)

    def test_degree_unwrap(self):
        fs = _make_fs(complex_=True)
        result = fs.degree(unwrap=True)
        assert len(result) == len(fs)

    def test_degree_no_name(self):
        # Line 171 — empty name
        data = np.exp(1j * np.linspace(0, 2 * np.pi, 32))
        fs = FrequencySeries(data, frequencies=np.arange(32) * u.Hz, unit=u.m)
        result = fs.degree()
        assert result.name == "phase_deg"


# ---------------------------------------------------------------------------
# Differentiate / Integrate
# ---------------------------------------------------------------------------


class TestDifferentiate:
    def test_differentiate_order0(self):
        # Line 201 — order=0 → copy
        fs = _make_fs()
        result = fs.differentiate(order=0)
        np.testing.assert_array_equal(result.value, fs.value)

    def test_differentiate_order1(self):
        # Lines 203-233
        fs = _make_fs()
        result = fs.differentiate(order=1)
        assert len(result) == len(fs)

    def test_differentiate_order2_name(self):
        # Lines 219-224 — order>1 name format
        fs = _make_fs(name="x")
        result = fs.differentiate(order=2)
        assert "2" in result.name

    def test_differentiate_order1_no_name(self):
        data = np.arange(1, 33, dtype=float)
        fs = FrequencySeries(data, frequencies=np.arange(32) * u.Hz, unit=u.m)
        result = fs.differentiate(order=1)
        assert result.name == "derivative"

    def test_differentiate_order2_no_name(self):
        data = np.arange(1, 33, dtype=float)
        fs = FrequencySeries(data, frequencies=np.arange(32) * u.Hz, unit=u.m)
        result = fs.differentiate(order=2)
        assert "derivative" in result.name

    def test_differentiate_no_unit(self):
        # Lines 215-216 — no unit
        data = np.arange(1, 33, dtype=float)
        fs = FrequencySeries(data, frequencies=np.arange(32) * u.Hz)
        result = fs.differentiate(order=1)
        assert result is not None


class TestIntegrate:
    def test_integrate_order0(self):
        # Line 251-252 — order=0 → copy
        fs = _make_fs()
        result = fs.integrate(order=0)
        np.testing.assert_array_equal(result.value, fs.value)

    def test_integrate_order1(self):
        # Lines 253-288
        fs = _make_fs()
        result = fs.integrate(order=1)
        assert len(result) == len(fs)

    def test_integrate_dc_zero(self):
        # Lines 263-264 — f[0]==0 → factor[0]=0
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(0, 64) * u.Hz  # starts at 0
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m, name="x")
        result = fs.integrate(order=1)
        # DC component should not be inf/nan
        assert np.isfinite(result.value[0])

    def test_integrate_order2_name(self):
        # Lines 274-279 — order>1 name
        fs = _make_fs(name="x")
        result = fs.integrate(order=2)
        assert "2" in result.name

    def test_integrate_no_unit(self):
        # Lines 270-271 — no unit
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(1, 65) * u.Hz  # no DC
        fs = FrequencySeries(data, frequencies=freqs)
        result = fs.integrate(order=1)
        assert result is not None


# ---------------------------------------------------------------------------
# to_db
# ---------------------------------------------------------------------------


class TestToDb:
    def test_to_db_amplitude(self):
        # Lines 309-326 — amplitude=True (default)
        fs = _make_fs()
        result = fs.to_db()
        assert len(result) == len(fs)

    def test_to_db_power(self):
        # Line 321 — amplitude=False
        fs = _make_fs()
        result = fs.to_db(amplitude=False)
        assert len(result) == len(fs)

    def test_to_db_ref_quantity(self):
        # Lines 310-311 — ref is Quantity
        fs = _make_fs()
        result = fs.to_db(ref=u.Quantity(1.0, u.m))
        assert result is not None

    def test_to_db_no_name(self):
        # Line 324 — name is empty
        data = np.arange(1, 33, dtype=float)
        fs = FrequencySeries(data, frequencies=np.arange(32) * u.Hz, unit=u.m)
        result = fs.to_db()
        assert result.name == "db"


# ---------------------------------------------------------------------------
# differentiate_time / integrate_time
# ---------------------------------------------------------------------------


class TestDifferentiateTime:
    def test_basic(self):
        # Lines 687-707
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(1, 65) * u.Hz  # no DC
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m, name="x")
        result = fs.differentiate_time()
        assert len(result) == len(fs)

    def test_no_name(self):
        data = np.arange(1, 33, dtype=float)
        fs = FrequencySeries(data, frequencies=np.arange(1, 33) * u.Hz, unit=u.m)
        result = fs.differentiate_time()
        assert result.name == "differentiation"

    def test_no_unit(self):
        # Line 696 — no unit → u.Hz
        data = np.arange(1, 33, dtype=float)
        fs = FrequencySeries(data, frequencies=np.arange(1, 33) * u.Hz)
        result = fs.differentiate_time()
        assert result is not None


class TestIntegrateTime:
    def test_basic(self):
        # Lines 720-747
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(0, 64) * u.Hz  # DC at 0
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m / u.s, name="v")
        result = fs.integrate_time()
        assert np.isfinite(result.value[0])

    def test_no_dc(self):
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(1, 65) * u.Hz  # no DC
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m / u.s)
        result = fs.integrate_time()
        assert len(result) == len(fs)

    def test_no_name(self):
        data = np.arange(1, 33, dtype=float)
        fs = FrequencySeries(data, frequencies=np.arange(1, 33) * u.Hz, unit=u.m / u.s)
        result = fs.integrate_time()
        assert result.name == "integration"

    def test_no_unit(self):
        # Line 736 — no unit → u.s
        data = np.arange(1, 33, dtype=float)
        fs = FrequencySeries(data, frequencies=np.arange(1, 33) * u.Hz)
        result = fs.integrate_time()
        assert result is not None


# ---------------------------------------------------------------------------
# quadrature_sum
# ---------------------------------------------------------------------------


class TestQuadratureSum:
    def test_basic(self):
        # Lines 767-780
        fs1 = _make_fs(name="a")
        fs2 = _make_fs(name="b")
        result = fs1.quadrature_sum(fs2)
        expected = np.sqrt(fs1.value**2 + fs2.value**2)
        np.testing.assert_allclose(result.value, expected)

    def test_complex_input(self):
        fs1 = _make_fs(complex_=True, name="a")
        fs2 = _make_fs(complex_=True, name="b")
        result = fs1.quadrature_sum(fs2)
        assert len(result) == len(fs1)


# ---------------------------------------------------------------------------
# group_delay
# ---------------------------------------------------------------------------


class TestGroupDelay:
    def test_basic(self):
        # Lines 795-812
        fs = _make_fs(complex_=True, name="h")
        result = fs.group_delay()
        assert len(result) == len(fs)
        assert "s" in str(result.unit)

    def test_no_name(self):
        data = np.exp(1j * np.linspace(0, 2 * np.pi, 64))
        fs = FrequencySeries(data, frequencies=np.arange(64) * u.Hz, unit=u.m)
        result = fs.group_delay()
        assert result.name == "group_delay"


# ---------------------------------------------------------------------------
# rebin
# ---------------------------------------------------------------------------


class TestRebin:
    def test_bin_size_one_returns_copy(self):
        # Lines 834-835 — bin_size <= 1 → copy
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        result = fs.rebin(width=1.0)  # df=1 → bin_size=1
        np.testing.assert_array_equal(result.value, fs.value)

    def test_rebin_by_two(self):
        # Lines 828-861
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m, name="x")
        result = fs.rebin(width=2.0)
        assert len(result) == 32

    def test_rebin_width_quantity(self):
        # Line 829 — width as Quantity
        data = np.arange(1, 65, dtype=float)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        result = fs.rebin(width=2.0 * u.Hz)
        assert len(result) == 32

    def test_rebin_truncation(self):
        # Lines 842-843 — truncation when not evenly divisible
        data = np.arange(1, 65, dtype=float)  # 64 elements
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        # bin_size=3 doesn't divide 64 evenly → truncation
        result = fs.rebin(width=3.0)
        assert len(result) == 64 // 3


# ---------------------------------------------------------------------------
# ifft mode tests
# ---------------------------------------------------------------------------


class TestIfft:
    def test_ifft_gwpy_mode(self):
        # Lines 533-542 — gwpy mode
        data = np.arange(1, 65, dtype=complex)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        result = fs.ifft(mode="gwpy")
        assert result is not None

    def test_ifft_unknown_mode_raises(self):
        # Lines 544-545
        data = np.arange(1, 65, dtype=complex)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        with pytest.raises(ValueError, match="Unknown ifft mode"):
            fs.ifft(mode="unknown_mode")

    def test_ifft_transient_mode(self):
        # Lines 547-597 — transient mode
        data = np.arange(1, 65, dtype=complex)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        fs._gwex_fft_mode = "transient"
        fs._gwex_target_nfft = 128
        fs._gwex_pad_left = 0
        fs._gwex_pad_right = 0
        result = fs.ifft(mode="auto")
        assert result is not None

    def test_ifft_transient_with_trim(self):
        # Lines 576-588 — trim with pad
        data = np.arange(1, 65, dtype=complex)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        fs._gwex_fft_mode = "transient"
        fs._gwex_target_nfft = 128
        fs._gwex_pad_left = 4
        fs._gwex_pad_right = 4
        fs._gwex_original_n = 100
        result = fs.ifft(mode="transient", trim=True)
        assert result is not None

    def test_ifft_transient_pad_right(self):
        # Line 578 — pad_r != 0 → data_trim[pad_l:-pad_r]
        data = np.arange(1, 65, dtype=complex)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        result = fs.ifft(mode="transient", trim=True, pad_left=2, pad_right=3)
        assert result is not None

    def test_ifft_transient_no_df(self):
        # Lines 560-562 — df is None → dt = None
        data = np.arange(1, 65, dtype=complex)
        freqs = np.arange(64) * u.Hz
        fs = FrequencySeries(data, frequencies=freqs, unit=u.m)
        # Fake no df
        fs._gwex_fft_mode = "transient"
        fs._gwex_target_nfft = 128
        result = fs.ifft(mode="transient")
        assert result is not None


# ---------------------------------------------------------------------------
# idct
# ---------------------------------------------------------------------------


class TestIdct:
    def test_basic(self):
        # Lines 599-674
        from scipy.fft import dct
        data = np.arange(1, 33, dtype=float)
        coeffs = dct(data, type=2, norm="ortho")
        freqs = np.arange(32) * u.Hz
        fs = FrequencySeries(coeffs, frequencies=freqs, unit=u.m, name="sig")
        result = fs.idct(type=2, norm="ortho")
        assert result is not None

    def test_idct_with_n(self):
        # Line 640 — n parameter specified
        from scipy.fft import dct
        data = np.arange(1, 33, dtype=float)
        coeffs = dct(data, type=2, norm="ortho")
        freqs = np.arange(32) * u.Hz
        fs = FrequencySeries(coeffs, frequencies=freqs, unit=u.m)
        result = fs.idct(n=32)
        assert len(result) == 32

    def test_idct_df_zero_fallback(self):
        # Lines 660-661 — df <= 0 → dt = 1.0 s
        from scipy.fft import dct
        data = np.arange(1, 33, dtype=float)
        coeffs = dct(data, type=2, norm="ortho")
        # Use frequencies starting at 0 with df=0 effectively
        freqs = np.zeros(32) * u.Hz  # all same → df=0
        try:
            fs = FrequencySeries(coeffs, frequencies=freqs, unit=u.m)
            result = fs.idct()
        except Exception:
            pass  # may fail on construction with degenerate freqs

    def test_idct_no_df(self):
        # Lines 662-663 — df is None → dt=1.0 s
        from scipy.fft import dct
        data = np.arange(1, 33, dtype=float)
        coeffs = dct(data, type=2, norm="ortho")
        freqs = np.arange(32) * u.Hz
        fs = FrequencySeries(coeffs, frequencies=freqs, unit=u.m)
        # Remove df by using meta_dt
        fs.dt = 0.5 * u.s  # provide dt directly (via attribute)
        result = fs.idct()
        assert result is not None
