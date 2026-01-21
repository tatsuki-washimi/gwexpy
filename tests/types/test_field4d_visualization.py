"""Tests for Field4D extraction and visualization methods (Phase T1-T2)."""

import numpy as np
import pytest
from astropy import units as u


class TestField4DExtractPoints:
    """Tests for Field4D.extract_points method."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample Field4D for testing."""
        from gwexpy.types import Field4D

        # Create 4D test data: (t=10, x=5, y=5, z=5)
        np.random.seed(42)
        data = np.random.randn(10, 5, 5, 5)

        times = np.arange(10) * 0.1 * u.s
        x = np.arange(5) * 1.0 * u.m
        y = np.arange(5) * 1.0 * u.m
        z = np.arange(5) * 1.0 * u.m

        return Field4D(
            data,
            unit=u.V,
            axis0=times,
            axis1=x,
            axis2=y,
            axis3=z,
            axis_names=["t", "x", "y", "z"],
        )

    def test_extract_single_point(self, sample_field):
        """Test extracting a single point returns TimeSeriesList of length 1."""
        from gwexpy.timeseries import TimeSeriesList

        points = [(2.0 * u.m, 2.0 * u.m, 2.0 * u.m)]
        result = sample_field.extract_points(points)

        assert isinstance(result, TimeSeriesList)
        assert len(result) == 1
        assert len(result[0]) == 10  # Same as time axis length

    def test_extract_multiple_points(self, sample_field):
        """Test extracting multiple points."""
        points = [
            (1.0 * u.m, 1.0 * u.m, 1.0 * u.m),
            (3.0 * u.m, 3.0 * u.m, 3.0 * u.m),
        ]
        result = sample_field.extract_points(points)

        assert len(result) == 2
        for ts in result:
            assert len(ts) == 10

    def test_extract_points_preserves_unit(self, sample_field):
        """Test that extracted time series has correct unit."""
        points = [(2.0 * u.m, 2.0 * u.m, 2.0 * u.m)]
        result = sample_field.extract_points(points)

        assert result[0].unit == u.V

    def test_extract_points_has_label(self, sample_field):
        """Test that extracted time series has coordinate label."""
        points = [(2.0 * u.m, 2.0 * u.m, 2.0 * u.m)]
        result = sample_field.extract_points(points)

        assert result[0].name is not None
        assert "x=" in result[0].name
        assert "y=" in result[0].name
        assert "z=" in result[0].name


class TestField4DSliceMap2D:
    """Tests for Field4D.slice_map2d method."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample Field4D for testing."""
        from gwexpy.types import Field4D

        data = np.random.randn(10, 8, 8, 8)
        times = np.arange(10) * 0.1 * u.s
        x = np.arange(8) * 1.0 * u.m
        y = np.arange(8) * 1.0 * u.m
        z = np.arange(8) * 1.0 * u.m

        return Field4D(
            data,
            unit=u.V,
            axis0=times,
            axis1=x,
            axis2=y,
            axis3=z,
            axis_names=["t", "x", "y", "z"],
        )

    def test_slice_xy_plane(self, sample_field):
        """Test slicing XY plane keeps x and y axes."""
        sliced = sample_field.slice_map2d(
            "xy", at={"t": 0.5 * u.s, "z": 3.0 * u.m}
        )

        # t and z should be length 1
        assert sliced.shape[0] == 1  # t
        assert sliced.shape[1] == 8  # x (kept)
        assert sliced.shape[2] == 8  # y (kept)
        assert sliced.shape[3] == 1  # z

    def test_slice_xz_plane(self, sample_field):
        """Test slicing XZ plane keeps x and z axes."""
        sliced = sample_field.slice_map2d(
            "xz", at={"t": 0.5 * u.s, "y": 3.0 * u.m}
        )

        assert sliced.shape[0] == 1  # t
        assert sliced.shape[1] == 8  # x (kept)
        assert sliced.shape[2] == 1  # y
        assert sliced.shape[3] == 8  # z (kept)

    def test_slice_tx_plane(self, sample_field):
        """Test slicing TX (time-space) plane."""
        sliced = sample_field.slice_map2d(
            "tx", at={"y": 3.0 * u.m, "z": 3.0 * u.m}
        )

        assert sliced.shape[0] == 10  # t (kept)
        assert sliced.shape[1] == 8   # x (kept)
        assert sliced.shape[2] == 1   # y
        assert sliced.shape[3] == 1   # z

    def test_slice_preserves_unit(self, sample_field):
        """Test that sliced Field4D preserves unit."""
        sliced = sample_field.slice_map2d(
            "xy", at={"t": 0.5 * u.s, "z": 3.0 * u.m}
        )
        assert sliced.unit == u.V

    def test_slice_auto_length1(self, sample_field):
        """Test that length-1 axes are used automatically when at=None."""
        # First reduce to have length-1 axes
        reduced = sample_field[:, :, :, 3:4]  # z is now length 1
        sliced = reduced.slice_map2d("xy", at={"t": 0.5 * u.s})

        assert sliced.shape[0] == 1
        assert sliced.shape[3] == 1


class TestField4DPlotMap2D:
    """Tests for Field4D.plot_map2d method."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample Field4D for testing."""
        from gwexpy.types import Field4D

        data = np.random.randn(10, 8, 8, 8)
        times = np.arange(10) * 0.1 * u.s
        x = np.arange(8) * 1.0 * u.m
        y = np.arange(8) * 1.0 * u.m
        z = np.arange(8) * 1.0 * u.m

        return Field4D(
            data,
            unit=u.V,
            axis0=times,
            axis1=x,
            axis2=y,
            axis3=z,
            axis_names=["t", "x", "y", "z"],
        )

    def test_plot_returns_fig_ax(self, sample_field):
        """Test that plot_map2d returns (fig, ax) tuple."""
        import matplotlib.pyplot as plt

        fig, ax = sample_field.plot_map2d(
            "xy", at={"t": 0.5 * u.s, "z": 3.0 * u.m}
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_has_colorbar(self, sample_field):
        """Test that colorbar is added when add_colorbar=True."""
        import matplotlib.pyplot as plt

        fig, ax = sample_field.plot_map2d(
            "xy", at={"t": 0.5 * u.s, "z": 3.0 * u.m}, add_colorbar=True
        )

        # Figure should have more than 1 axes (main + colorbar)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_no_colorbar(self, sample_field):
        """Test that colorbar is not added when add_colorbar=False."""
        import matplotlib.pyplot as plt

        fig, ax = sample_field.plot_map2d(
            "xy", at={"t": 0.5 * u.s, "z": 3.0 * u.m}, add_colorbar=False
        )

        # Figure should have only 1 axes
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_axes_have_labels_with_units(self, sample_field):
        """Test that axes have labels with units."""
        import matplotlib.pyplot as plt

        fig, ax = sample_field.plot_map2d(
            "xy", at={"t": 0.5 * u.s, "z": 3.0 * u.m}
        )

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        # Should contain unit indicators
        assert "[" in xlabel and "]" in xlabel
        assert "[" in ylabel and "]" in ylabel
        plt.close(fig)


class TestField4DPlotTimeseriesPoints:
    """Tests for Field4D.plot_timeseries_points method."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample Field4D for testing."""
        from gwexpy.types import Field4D

        data = np.random.randn(10, 5, 5, 5)
        times = np.arange(10) * 0.1 * u.s
        x = np.arange(5) * 1.0 * u.m
        y = np.arange(5) * 1.0 * u.m
        z = np.arange(5) * 1.0 * u.m

        return Field4D(
            data,
            unit=u.V,
            axis0=times,
            axis1=x,
            axis2=y,
            axis3=z,
            axis_names=["t", "x", "y", "z"],
        )

    def test_plot_single_point(self, sample_field):
        """Test plotting single point."""
        import matplotlib.pyplot as plt

        points = [(2.0 * u.m, 2.0 * u.m, 2.0 * u.m)]
        fig, ax = sample_field.plot_timeseries_points(points)

        # Should have 1 line
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_plot_multiple_points(self, sample_field):
        """Test plotting multiple points creates multiple lines."""
        import matplotlib.pyplot as plt

        points = [
            (1.0 * u.m, 1.0 * u.m, 1.0 * u.m),
            (2.0 * u.m, 2.0 * u.m, 2.0 * u.m),
            (3.0 * u.m, 3.0 * u.m, 3.0 * u.m),
        ]
        fig, ax = sample_field.plot_timeseries_points(points)

        # Should have 3 lines
        assert len(ax.lines) == 3
        plt.close(fig)


class TestField4DPlotProfile:
    """Tests for Field4D.plot_profile method."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample Field4D for testing."""
        from gwexpy.types import Field4D

        data = np.random.randn(10, 8, 8, 8)
        times = np.arange(10) * 0.1 * u.s
        x = np.arange(8) * 1.0 * u.m
        y = np.arange(8) * 1.0 * u.m
        z = np.arange(8) * 1.0 * u.m

        return Field4D(
            data,
            unit=u.V,
            axis0=times,
            axis1=x,
            axis2=y,
            axis3=z,
            axis_names=["t", "x", "y", "z"],
        )

    def test_plot_x_profile(self, sample_field):
        """Test plotting profile along x axis."""
        import matplotlib.pyplot as plt

        fig, ax = sample_field.plot_profile(
            "x", at={"t": 0.5 * u.s, "y": 3.0 * u.m, "z": 3.0 * u.m}
        )

        # Should have 1 line
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_profile_has_correct_length(self, sample_field):
        """Test that profile line has correct number of points."""
        import matplotlib.pyplot as plt

        fig, ax = sample_field.plot_profile(
            "x", at={"t": 0.5 * u.s, "y": 3.0 * u.m, "z": 3.0 * u.m}
        )

        # Line should have 8 points (x-axis length)
        line = ax.lines[0]
        assert len(line.get_xdata()) == 8
        plt.close(fig)


# =============================================================================
# Phase T3: Comparison Methods Tests
# =============================================================================


class TestField4DDiff:
    """Tests for Field4D.diff method."""

    @pytest.fixture
    def sample_fields(self):
        """Create two sample Field4D objects for comparison."""
        from gwexpy.types import Field4D

        np.random.seed(42)
        data1 = np.ones((10, 5, 5, 5)) * 10.0
        data2 = np.ones((10, 5, 5, 5)) * 5.0

        times = np.arange(10) * 0.1 * u.s
        x = np.arange(5) * 1.0 * u.m
        y = np.arange(5) * 1.0 * u.m
        z = np.arange(5) * 1.0 * u.m

        field1 = Field4D(
            data1, unit=u.V, axis0=times, axis1=x, axis2=y, axis3=z,
            axis_names=["t", "x", "y", "z"]
        )
        field2 = Field4D(
            data2, unit=u.V, axis0=times, axis1=x, axis2=y, axis3=z,
            axis_names=["t", "x", "y", "z"]
        )
        return field1, field2

    def test_diff_mode(self, sample_fields):
        """Test diff mode computes difference correctly."""
        field1, field2 = sample_fields
        result = field1.diff(field2, mode="diff")

        np.testing.assert_array_almost_equal(result.value, 5.0)
        assert result.unit == u.V

    def test_ratio_mode(self, sample_fields):
        """Test ratio mode computes ratio correctly."""
        field1, field2 = sample_fields
        result = field1.diff(field2, mode="ratio")

        np.testing.assert_array_almost_equal(result.value, 2.0)
        assert result.unit == u.dimensionless_unscaled

    def test_percent_mode(self, sample_fields):
        """Test percent mode computes percentage correctly."""
        field1, field2 = sample_fields
        result = field1.diff(field2, mode="percent")

        np.testing.assert_array_almost_equal(result.value, 100.0)
        assert result.unit == u.percent

    def test_shape_mismatch_raises(self, sample_fields):
        """Test that shape mismatch raises ValueError."""
        from gwexpy.types import Field4D

        field1, _ = sample_fields
        # Create field with different shape
        field_small = Field4D(
            np.ones((5, 3, 3, 3)),
            axis0=np.arange(5) * 0.1 * u.s,
            axis1=np.arange(3) * u.m,
            axis2=np.arange(3) * u.m,
            axis3=np.arange(3) * u.m,
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            field1.diff(field_small)


class TestField4DZscore:
    """Tests for Field4D.zscore method."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample Field4D with known statistics."""
        from gwexpy.types import Field4D

        # Create data with known mean=5, std=2 pattern
        np.random.seed(42)
        data = np.random.randn(100, 4, 4, 4) * 2.0 + 5.0

        times = np.arange(100) * 0.01 * u.s
        x = np.arange(4) * 1.0 * u.m
        y = np.arange(4) * 1.0 * u.m
        z = np.arange(4) * 1.0 * u.m

        return Field4D(
            data, unit=u.V, axis0=times, axis1=x, axis2=y, axis3=z,
            axis_names=["t", "x", "y", "z"]
        )

    def test_zscore_is_dimensionless(self, sample_field):
        """Test that zscore result is dimensionless."""
        result = sample_field.zscore()
        assert result.unit == u.dimensionless_unscaled

    def test_zscore_with_baseline(self, sample_field):
        """Test zscore with specific baseline range."""
        result = sample_field.zscore(baseline_t=(0.0 * u.s, 0.5 * u.s))
        assert result.shape == sample_field.shape

    def test_zscore_mean_near_zero(self, sample_field):
        """Test that zscore of full data has mean near zero."""
        result = sample_field.zscore()
        # Mean along time axis should be near 0
        mean_values = np.mean(result.value, axis=0)
        np.testing.assert_array_almost_equal(mean_values, 0.0, decimal=1)


# =============================================================================
# Phase T4: Time Summary Methods Tests
# =============================================================================


class TestField4DTimeStatMap:
    """Tests for Field4D.time_stat_map method."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample Field4D for testing."""
        from gwexpy.types import Field4D

        # Create data where mean along time = 5
        data = np.ones((10, 4, 4, 4)) * 5.0

        times = np.arange(10) * 0.1 * u.s
        x = np.arange(4) * 1.0 * u.m
        y = np.arange(4) * 1.0 * u.m
        z = np.arange(4) * 1.0 * u.m

        return Field4D(
            data, unit=u.V, axis0=times, axis1=x, axis2=y, axis3=z,
            axis_names=["t", "x", "y", "z"]
        )

    def test_mean_stat(self, sample_field):
        """Test mean statistic computation."""
        result = sample_field.time_stat_map(stat="mean")

        assert result.shape[0] == 1  # Time reduced to 1
        np.testing.assert_array_almost_equal(result.value, 5.0)

    def test_std_stat(self, sample_field):
        """Test std statistic computation (should be 0 for constant data)."""
        result = sample_field.time_stat_map(stat="std")
        np.testing.assert_array_almost_equal(result.value, 0.0)

    def test_t_range_selection(self, sample_field):
        """Test that t_range properly selects time subset."""
        result = sample_field.time_stat_map(
            stat="mean", t_range=(0.0 * u.s, 0.5 * u.s)
        )
        # Should still work and have valid result
        assert result.shape[0] == 1


class TestField4DTimeSpaceMap:
    """Tests for Field4D.time_space_map method."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample Field4D for testing."""
        from gwexpy.types import Field4D

        data = np.random.randn(10, 8, 4, 4)

        times = np.arange(10) * 0.1 * u.s
        x = np.arange(8) * 1.0 * u.m
        y = np.arange(4) * 1.0 * u.m
        z = np.arange(4) * 1.0 * u.m

        return Field4D(
            data, unit=u.V, axis0=times, axis1=x, axis2=y, axis3=z,
            axis_names=["t", "x", "y", "z"]
        )

    def test_time_space_map_shape(self, sample_field):
        """Test that time_space_map returns correct shape."""
        t_axis, x_axis, data = sample_field.time_space_map(
            "x", at={"y": 2.0 * u.m, "z": 2.0 * u.m}
        )

        assert len(t_axis) == 10  # time axis length
        assert len(x_axis) == 8   # x axis length
        assert data.shape == (10, 8)  # (t, x)

