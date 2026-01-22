"""Tests for Field4D spatial FFT - two-sided signed FFT with angular wavenumber."""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwexpy.fields import ScalarField as Field4D


class TestField4DFftSpaceBasic:
    """Test basic fft_space functionality."""

    @pytest.fixture
    def real_space_field(self):
        """Create a real-space Field4D."""
        np.random.seed(42)
        data = np.random.randn(10, 16, 16, 16)
        t = np.arange(10) * 0.1 * u.s
        x = np.arange(16) * 0.5 * u.m
        y = np.arange(16) * 0.5 * u.m
        z = np.arange(16) * 0.5 * u.m
        return Field4D(
            data,
            unit=u.V,
            axis0=t,
            axis1=x,
            axis2=y,
            axis3=z,
            axis_names=["t", "x", "y", "z"],
            axis0_domain="time",
            space_domain="real",
        )

    def test_fft_space_basic(self, real_space_field):
        """Test basic fft_space execution."""
        result = real_space_field.fft_space()

        assert isinstance(result, Field4D)

    def test_fft_space_all_axes(self, real_space_field):
        """Test fft_space on all spatial axes."""
        result = real_space_field.fft_space()

        # All spatial axes should now be in k domain
        assert result.space_domains["kx"] == "k"
        assert result.space_domains["ky"] == "k"
        assert result.space_domains["kz"] == "k"

    def test_fft_space_single_axis(self, real_space_field):
        """Test fft_space on single axis."""
        result = real_space_field.fft_space(axes=["x"])

        # Only x should be transformed
        assert result.space_domains["kx"] == "k"
        assert result.space_domains["y"] == "real"
        assert result.space_domains["z"] == "real"

    def test_fft_space_two_axes(self, real_space_field):
        """Test fft_space on two axes."""
        result = real_space_field.fft_space(axes=["x", "z"])

        assert result.space_domains["kx"] == "k"
        assert result.space_domains["y"] == "real"
        assert result.space_domains["kz"] == "k"

    def test_fft_space_axis_names_change(self, real_space_field):
        """Test that axis names change x -> kx."""
        result = real_space_field.fft_space()

        assert result.axis_names == ("t", "kx", "ky", "kz")

    def test_fft_space_shape_preserved(self, real_space_field):
        """Test that shape is preserved by fft_space."""
        result = real_space_field.fft_space()

        assert result.shape == real_space_field.shape

    def test_fft_space_unit_preserved(self, real_space_field):
        """Test that data unit is preserved."""
        result = real_space_field.fft_space()
        assert result.unit == u.V

    def test_fft_space_time_domain_preserved(self, real_space_field):
        """Test that axis0_domain is preserved."""
        result = real_space_field.fft_space()
        assert result.axis0_domain == "time"


class TestField4DFftSpaceWavenumber:
    """Test angular wavenumber generation."""

    def test_k_axis_contains_negative_and_positive(self):
        """Test that k axis contains both negative and positive values."""
        data = np.random.randn(4, 16, 16, 16)
        field = Field4D(
            data,
            axis1=np.arange(16) * 0.5 * u.m,
            axis2=np.arange(16) * 0.5 * u.m,
            axis3=np.arange(16) * 0.5 * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )
        result = field.fft_space(axes=["x"])

        k_values = result._axis1_index.value
        assert np.any(k_values < 0), "k axis should contain negative values"
        assert np.any(k_values > 0), "k axis should contain positive values"

    def test_k_axis_is_angular_wavenumber(self):
        """Test k = 2π * fftfreq (angular wavenumber)."""
        n = 16
        dx = 0.5  # m
        data = np.random.randn(4, n, n, n)
        field = Field4D(
            data,
            axis1=np.arange(n) * dx * u.m,
            axis2=np.arange(n) * dx * u.m,
            axis3=np.arange(n) * dx * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )
        result = field.fft_space(axes=["x"])

        # Expected: k = 2π * fftfreq(n, d=dx)
        expected_k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        assert_allclose(result._axis1_index.value, expected_k)

    def test_k_axis_unit(self):
        """Test k axis unit is 1/m (rad/m)."""
        data = np.random.randn(4, 8, 8, 8)
        field = Field4D(
            data,
            axis1=np.arange(8) * 1.0 * u.m,
            axis2=np.arange(8) * 1.0 * u.m,
            axis3=np.arange(8) * 1.0 * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )
        result = field.fft_space(axes=["x"])

        assert result._axis1_index.unit == (1 / u.m)


class TestField4DFftSpaceErrors:
    """Test fft_space error conditions."""

    def test_fft_space_on_k_domain_raises(self):
        """Test fft_space raises if axis already in k domain."""
        data = np.random.randn(4, 8, 8, 8)
        field = Field4D(
            data,
            axis_names=["t", "kx", "y", "z"],
            space_domain={"kx": "k", "y": "real", "z": "real"},
        )

        with pytest.raises(ValueError, match="not in 'real' domain"):
            field.fft_space(axes=["kx"])

    def test_fft_space_on_axis0_raises(self):
        """Test fft_space raises if axis 0 specified."""
        field = Field4D(np.zeros((10, 4, 4, 4)), axis_names=["t", "x", "y", "z"])

        with pytest.raises(ValueError, match="Cannot use fft_space on axis 0"):
            field.fft_space(axes=["t"])

    def test_fft_space_non_uniform_raises(self):
        """Test fft_space raises if axis is not uniformly spaced."""
        # Non-uniform x axis
        x = np.array([0, 1, 3, 6, 10, 15, 21, 28]) * u.m  # non-uniform
        data = np.random.randn(4, 8, 4, 4)
        field = Field4D(
            data,
            axis1=x,
            axis2=np.arange(4) * 1.0 * u.m,
            axis3=np.arange(4) * 1.0 * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )

        with pytest.raises(ValueError, match="not uniformly spaced"):
            field.fft_space(axes=["x"])

    def test_fft_space_no_axes_raises(self):
        """Test fft_space raises if no axes to transform."""
        field = Field4D(
            np.zeros((4, 8, 8, 8)),
            axis_names=["t", "kx", "ky", "kz"],
            space_domain="k",
        )

        with pytest.raises(ValueError, match="No axes specified"):
            field.fft_space()


class TestField4DIfftSpaceBasic:
    """Test basic ifft_space functionality."""

    @pytest.fixture
    def k_space_field(self):
        """Create a k-space Field4D."""
        data = np.random.randn(4, 16, 16, 16)
        field = Field4D(
            data,
            axis1=np.arange(16) * 0.5 * u.m,
            axis2=np.arange(16) * 0.5 * u.m,
            axis3=np.arange(16) * 0.5 * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )
        return field.fft_space()

    def test_ifft_space_basic(self, k_space_field):
        """Test basic ifft_space execution."""
        result = k_space_field.ifft_space()

        assert isinstance(result, Field4D)

    def test_ifft_space_domain_transition(self, k_space_field):
        """Test ifft_space changes domain k -> real."""
        result = k_space_field.ifft_space()

        assert result.space_domains["x"] == "real"
        assert result.space_domains["y"] == "real"
        assert result.space_domains["z"] == "real"

    def test_ifft_space_axis_names(self, k_space_field):
        """Test ifft_space changes axis names kx -> x."""
        result = k_space_field.ifft_space()

        assert result.axis_names == ("t", "x", "y", "z")


class TestField4DFftIfftSpaceReversibility:
    """Test spatial FFT/IFFT reversibility."""

    def test_ifft_fft_space_reversible(self):
        """Test ifft_space(fft_space(field)) recovers original."""
        np.random.seed(456)
        data = np.random.randn(4, 16, 16, 16)
        x = np.arange(16) * 0.5 * u.m

        original = Field4D(
            data,
            unit=u.V,
            axis1=x,
            axis2=x.copy(),
            axis3=x.copy(),
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )

        k_space = original.fft_space()
        recovered = k_space.ifft_space()

        assert_allclose(recovered.value, original.value, rtol=1e-10)
        assert recovered.space_domains == {"x": "real", "y": "real", "z": "real"}

    def test_partial_fft_ifft_reversible(self):
        """Test partial FFT/IFFT reversibility."""
        np.random.seed(789)
        data = np.random.randn(4, 8, 8, 8)

        original = Field4D(
            data,
            axis1=np.arange(8) * 1.0 * u.m,
            axis2=np.arange(8) * 1.0 * u.m,
            axis3=np.arange(8) * 1.0 * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )

        # Transform only x and z
        k_space = original.fft_space(axes=["x", "z"])
        recovered = k_space.ifft_space(axes=["kx", "kz"])

        assert_allclose(recovered.value, original.value, rtol=1e-10)


class TestField4DIfftSpaceErrors:
    """Test ifft_space error conditions."""

    def test_ifft_space_on_real_domain_raises(self):
        """Test ifft_space raises if axis is in real domain."""
        field = Field4D(
            np.zeros((4, 8, 8, 8)),
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )

        with pytest.raises(ValueError, match="not in 'k' domain"):
            field.ifft_space(axes=["x"])


class TestField4DWavelength:
    """Test wavelength computation from k-space."""

    def test_wavelength_basic(self):
        """Test wavelength = 2π / |k|."""
        data = np.zeros((4, 8, 8, 8))
        field = Field4D(
            data,
            axis1=np.arange(8) * 1.0 * u.m,
            axis2=np.arange(8) * 1.0 * u.m,
            axis3=np.arange(8) * 1.0 * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )
        k_field = field.fft_space(axes=["x"])

        wavelength = k_field.wavelength("kx")

        # For non-zero k, λ = 2π/|k|
        k_values = k_field._axis1_index.value
        expected_lambda = 2 * np.pi / np.abs(k_values)
        assert_allclose(wavelength.value, expected_lambda)

    def test_wavelength_zero_k_is_inf(self):
        """Test wavelength is inf for k=0."""
        data = np.zeros((4, 8, 8, 8))
        field = Field4D(
            data,
            axis1=np.arange(8) * 1.0 * u.m,
            axis2=np.arange(8) * 1.0 * u.m,
            axis3=np.arange(8) * 1.0 * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )
        k_field = field.fft_space(axes=["x"])

        wavelength = k_field.wavelength("kx")

        # DC component (k=0) should have inf wavelength
        k_values = k_field._axis1_index.value
        zero_idx = np.where(k_values == 0)[0]
        if len(zero_idx) > 0:
            assert np.isinf(wavelength.value[zero_idx[0]])

    def test_wavelength_unit(self):
        """Test wavelength unit is correct."""
        data = np.zeros((4, 8, 8, 8))
        field = Field4D(
            data,
            axis1=np.arange(8) * 1.0 * u.m,
            axis2=np.arange(8) * 1.0 * u.m,
            axis3=np.arange(8) * 1.0 * u.m,
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )
        k_field = field.fft_space(axes=["x"])

        wavelength = k_field.wavelength("kx")

        # λ unit should be m (inverse of k unit which is 1/m)
        assert wavelength.unit == u.m

    def test_wavelength_real_domain_raises(self):
        """Test wavelength raises if axis is in real domain."""
        field = Field4D(
            np.zeros((4, 8, 8, 8)),
            axis_names=["t", "x", "y", "z"],
            space_domain="real",
        )

        with pytest.raises(ValueError, match="not in 'k' domain"):
            field.wavelength("x")
