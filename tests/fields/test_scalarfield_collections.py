"""Tests for FieldList and FieldDict collections."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, FieldList, ScalarField


class TestFieldListBasic:
    """Test FieldList basic functionality."""

    @pytest.fixture
    def field_list(self):
        """Create a list of ScalarField objects."""
        fields = []
        for i in range(3):
            np.random.seed(i)
            data = np.random.randn(16, 4, 4, 4)
            f = ScalarField(
                data,
                unit=u.V,
                axis0=np.arange(16) * 0.01 * u.s,
                axis1=np.arange(4) * 1.0 * u.m,
                axis2=np.arange(4) * 1.0 * u.m,
                axis3=np.arange(4) * 1.0 * u.m,
                axis_names=["t", "x", "y", "z"],
                axis0_domain="time",
                space_domain="real",
            )
            fields.append(f)
        return FieldList(fields)

    def test_scalarfield_list_is_list(self, field_list):
        """Test that FieldList is a list."""
        assert isinstance(field_list, list)

    def test_scalarfield_list_length(self, field_list):
        """Test list length."""
        assert len(field_list) == 3

    def test_scalarfield_list_indexing(self, field_list):
        """Test list indexing."""
        assert isinstance(field_list[0], ScalarField)

    def test_scalarfield_list_append(self, field_list):
        """Test appending to list."""
        new_field = ScalarField(np.zeros((16, 4, 4, 4)))
        field_list.append(new_field)
        assert len(field_list) == 4


class TestFieldListFftOperations:
    """Test FieldList batch FFT operations."""

    @pytest.fixture
    def time_domain_list(self):
        """Create a list of time-domain ScalarField objects."""
        fields = []
        for i in range(3):
            np.random.seed(i)
            data = np.random.randn(32, 4, 4, 4)
            f = ScalarField(
                data,
                axis0=np.arange(32) * 0.01 * u.s,
                axis1=np.arange(4) * 1.0 * u.m,
                axis2=np.arange(4) * 1.0 * u.m,
                axis3=np.arange(4) * 1.0 * u.m,
                axis_names=["t", "x", "y", "z"],
                axis0_domain="time",
                space_domain="real",
            )
            fields.append(f)
        return FieldList(fields)

    def test_fft_time_all(self, time_domain_list):
        """Test fft_time_all applies to all fields."""
        result = time_domain_list.fft_time_all()

        assert isinstance(result, FieldList)
        assert len(result) == 3
        for f in result:
            assert isinstance(f, ScalarField)
            assert f.axis0_domain == "frequency"

    def test_ifft_time_all(self, time_domain_list):
        """Test ifft_time_all after fft_time_all."""
        freq_list = time_domain_list.fft_time_all()
        result = freq_list.ifft_time_all()

        assert isinstance(result, FieldList)
        assert len(result) == 3
        for f in result:
            assert f.axis0_domain == "time"

    def test_fft_space_all(self, time_domain_list):
        """Test fft_space_all applies to all fields."""
        result = time_domain_list.fft_space_all()

        assert isinstance(result, FieldList)
        assert len(result) == 3
        for f in result:
            assert f.space_domains["kx"] == "k"
            assert f.space_domains["ky"] == "k"
            assert f.space_domains["kz"] == "k"

    def test_ifft_space_all(self, time_domain_list):
        """Test ifft_space_all after fft_space_all."""
        k_list = time_domain_list.fft_space_all()
        result = k_list.ifft_space_all()

        assert isinstance(result, FieldList)
        for f in result:
            assert f.space_domains["x"] == "real"


class TestFieldListValidation:
    """Test FieldList validation."""

    def test_validate_consistent_fields(self):
        """Test validation passes for consistent fields."""
        fields = [
            ScalarField(
                np.zeros((10, 4, 4, 4)), unit=u.V, axis_names=["t", "x", "y", "z"]
            )
            for _ in range(3)
        ]
        # Should not raise
        flist = FieldList(fields, validate=True)
        assert len(flist) == 3

    def test_validate_inconsistent_units_raises(self):
        """Test validation raises for inconsistent units."""
        fields = [
            ScalarField(np.zeros((10, 4, 4, 4)), unit=u.V),
            ScalarField(np.zeros((10, 4, 4, 4)), unit=u.A),
        ]

        with pytest.raises(ValueError, match="Inconsistent unit"):
            FieldList(fields, validate=True)

    def test_validate_inconsistent_axis_names_raises(self):
        """Test validation raises for inconsistent axis names."""
        fields = [
            ScalarField(np.zeros((10, 4, 4, 4)), axis_names=["t", "x", "y", "z"]),
            ScalarField(np.zeros((10, 4, 4, 4)), axis_names=["t", "a", "b", "c"]),
        ]

        with pytest.raises(ValueError, match="Inconsistent axis_names"):
            FieldList(fields, validate=True)

    def test_validate_inconsistent_domain_raises(self):
        """Test validation raises for inconsistent domains."""
        # Use same axis_names so domain validation is reached
        fields = [
            ScalarField(
                np.zeros((10, 4, 4, 4)),
                axis_names=["t", "x", "y", "z"],
                axis0_domain="time",
            ),
            ScalarField(
                np.zeros((10, 4, 4, 4)),
                axis_names=["t", "x", "y", "z"],
                axis0_domain="frequency",
            ),
        ]

        with pytest.raises(ValueError, match="Inconsistent axis0_domain"):
            FieldList(fields, validate=True)

    def test_validate_non_scalarfield_raises(self):
        """Test validation raises for non-ScalarField items."""
        items = [ScalarField(np.zeros((10, 4, 4, 4))), "not a field"]

        with pytest.raises(TypeError, match="Expected ScalarField"):
            FieldList(items, validate=True)


class TestFieldDictBasic:
    """Test FieldDict basic functionality."""

    @pytest.fixture
    def field_dict(self):
        """Create a dict of ScalarField objects."""
        fields = {}
        for name in ["Ex", "Ey", "Ez"]:
            np.random.seed(hash(name) % 1000)
            data = np.random.randn(16, 4, 4, 4)
            f = ScalarField(
                data,
                unit=u.V / u.m,
                axis0=np.arange(16) * 0.01 * u.s,
                axis1=np.arange(4) * 1.0 * u.m,
                axis2=np.arange(4) * 1.0 * u.m,
                axis3=np.arange(4) * 1.0 * u.m,
                axis_names=["t", "x", "y", "z"],
                axis0_domain="time",
                space_domain="real",
            )
            fields[name] = f
        return FieldDict(fields)

    def test_scalarfield_dict_is_dict(self, field_dict):
        """Test that FieldDict is a dict."""
        assert isinstance(field_dict, dict)

    def test_scalarfield_dict_length(self, field_dict):
        """Test dict length."""
        assert len(field_dict) == 3

    def test_scalarfield_dict_keys(self, field_dict):
        """Test dict keys."""
        assert set(field_dict.keys()) == {"Ex", "Ey", "Ez"}

    def test_scalarfield_dict_indexing(self, field_dict):
        """Test dict indexing."""
        assert isinstance(field_dict["Ex"], ScalarField)


class TestFieldDictFftOperations:
    """Test FieldDict batch FFT operations."""

    @pytest.fixture
    def time_domain_dict(self):
        """Create a dict of time-domain ScalarField objects."""
        fields = {}
        for name in ["Ex", "Ey"]:
            np.random.seed(hash(name) % 1000)
            data = np.random.randn(32, 4, 4, 4)
            f = ScalarField(
                data,
                axis0=np.arange(32) * 0.01 * u.s,
                axis1=np.arange(4) * 1.0 * u.m,
                axis2=np.arange(4) * 1.0 * u.m,
                axis3=np.arange(4) * 1.0 * u.m,
                axis_names=["t", "x", "y", "z"],
                axis0_domain="time",
                space_domain="real",
            )
            fields[name] = f
        return FieldDict(fields)

    def test_fft_time_all(self, time_domain_dict):
        """Test fft_time_all applies to all fields."""
        result = time_domain_dict.fft_time_all()

        assert isinstance(result, FieldDict)
        assert set(result.keys()) == {"Ex", "Ey"}
        for f in result.values():
            assert isinstance(f, ScalarField)
            assert f.axis0_domain == "frequency"

    def test_ifft_time_all(self, time_domain_dict):
        """Test ifft_time_all after fft_time_all."""
        freq_dict = time_domain_dict.fft_time_all()
        result = freq_dict.ifft_time_all()

        assert isinstance(result, FieldDict)
        for f in result.values():
            assert f.axis0_domain == "time"

    def test_fft_space_all(self, time_domain_dict):
        """Test fft_space_all applies to all fields."""
        result = time_domain_dict.fft_space_all()

        assert isinstance(result, FieldDict)
        for f in result.values():
            assert f.space_domains["kx"] == "k"

    def test_ifft_space_all(self, time_domain_dict):
        """Test ifft_space_all after fft_space_all."""
        k_dict = time_domain_dict.fft_space_all()
        result = k_dict.ifft_space_all()

        assert isinstance(result, FieldDict)
        for f in result.values():
            assert f.space_domains["x"] == "real"


class TestFieldDictValidation:
    """Test FieldDict validation."""

    def test_validate_consistent_fields(self):
        """Test validation passes for consistent fields."""
        fields = {
            "a": ScalarField(np.zeros((10, 4, 4, 4)), unit=u.V),
            "b": ScalarField(np.zeros((10, 4, 4, 4)), unit=u.V),
        }
        # Should not raise
        fdict = FieldDict(fields, validate=True)
        assert len(fdict) == 2

    def test_validate_inconsistent_units_raises(self):
        """Test validation raises for inconsistent units."""
        fields = {
            "a": ScalarField(np.zeros((10, 4, 4, 4)), unit=u.V),
            "b": ScalarField(np.zeros((10, 4, 4, 4)), unit=u.A),
        }

        with pytest.raises(ValueError, match="Inconsistent unit"):
            FieldDict(fields, validate=True)

    def test_validate_non_scalarfield_raises(self):
        """Test validation raises for non-ScalarField values."""
        items = {"a": ScalarField(np.zeros((10, 4, 4, 4))), "b": "not a field"}

        with pytest.raises(TypeError, match="Expected ScalarField"):
            FieldDict(items, validate=True)


# ---------------------------------------------------------------------------
# FieldList — validate error paths and batch methods
# ---------------------------------------------------------------------------

class TestFieldListValidation:
    def _make_field(self, unit=u.V, axis0_domain="time"):
        return ScalarField(
            np.ones((8, 2, 2, 2)),
            unit=unit,
            axis0=np.arange(8) * 0.01 * u.s,
            axis1=np.arange(2) * 1.0 * u.m,
            axis2=np.arange(2) * 1.0 * u.m,
            axis3=np.arange(2) * 1.0 * u.m,
            axis_names=["t", "x", "y", "z"],
            axis0_domain=axis0_domain,
            space_domain="real",
        )

    def test_validate_empty_list(self):
        fl = FieldList([], validate=True)
        assert len(fl) == 0

    def test_validate_non_scalarfield_raises(self):
        with pytest.raises(TypeError, match="Expected ScalarField"):
            FieldList(["not_a_field"], validate=True)

    def test_validate_inconsistent_axis_names_raises(self):
        f1 = self._make_field()
        f2 = ScalarField(
            np.ones((8, 2, 2, 2)),
            unit=u.V,
            axis0=np.arange(8) * 0.01 * u.s,
            axis_names=["t", "a", "b", "c"],
            axis0_domain="time",
        )
        with pytest.raises(ValueError, match="axis_names"):
            FieldList([f1, f2], validate=True)

    def test_validate_space_domains_mismatch_raises(self):
        f1 = self._make_field()
        # f2 with k-domain has different axis_names (kx,ky,kz), caught before space_domains check
        f2 = ScalarField(
            np.ones((8, 2, 2, 2)),
            unit=u.V,
            axis0=np.arange(8) * 0.01 * u.s,
            axis1=np.arange(2) * 1.0 / u.m,
            axis2=np.arange(2) * 1.0 / u.m,
            axis3=np.arange(2) * 1.0 / u.m,
            axis0_domain="time",
            space_domain="k",
        )
        with pytest.raises(ValueError):
            FieldList([f1, f2], validate=True)

    def test_validate_axis_shape_mismatch_raises(self):
        f1 = self._make_field()
        f2 = ScalarField(
            np.ones((8, 3, 2, 2)),
            unit=u.V,
            axis0=np.arange(8) * 0.01 * u.s,
            axis1=np.arange(3) * 1.0 * u.m,
            axis2=np.arange(2) * 1.0 * u.m,
            axis3=np.arange(2) * 1.0 * u.m,
            axis0_domain="time",
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            FieldList([f1, f2], validate=True)

    def test_validate_axis_coordinate_mismatch_raises(self):
        f1 = self._make_field()
        f2 = ScalarField(
            np.ones((8, 2, 2, 2)),
            unit=u.V,
            axis0=np.arange(8) * 0.02 * u.s,  # different dt
            axis1=np.arange(2) * 1.0 * u.m,
            axis2=np.arange(2) * 1.0 * u.m,
            axis3=np.arange(2) * 1.0 * u.m,
            axis0_domain="time",
        )
        with pytest.raises(ValueError, match="coordinate mismatch"):
            FieldList([f1, f2], validate=True)

    def test_batch_methods(self):
        f1 = self._make_field()
        f2 = self._make_field()
        fl = FieldList([f1, f2])
        # fft_time_all
        result = fl.fft_time_all()
        assert len(result) == 2
        # ifft_time_all
        result2 = result.ifft_time_all()
        assert len(result2) == 2
        # fft_space_all
        result3 = fl.fft_space_all()
        assert len(result3) == 2
        # ifft_space_all
        result4 = result3.ifft_space_all()
        assert len(result4) == 2
        # isel_all
        result5 = fl.isel_all(t=slice(0, 4))
        assert len(result5) == 2
        # sel_all
        result6 = fl.sel_all(x=0.0 * u.m)
        assert len(result6) == 2


# ---------------------------------------------------------------------------
# FieldDict — arithmetic and validate error paths
# ---------------------------------------------------------------------------

class TestFieldDictExtra:
    def _make_fd(self):
        f = ScalarField(np.ones((8, 2, 2, 2)), unit=u.dimensionless_unscaled, axis0_domain="time")
        return FieldDict({"a": f, "b": f.copy()})

    def test_mul_scalar(self):
        fd = self._make_fd()
        result = fd * 2.0
        assert isinstance(result, FieldDict)

    def test_rmul_scalar(self):
        fd = self._make_fd()
        result = 2.0 * fd
        assert isinstance(result, FieldDict)

    def test_mul_non_scalar_returns_not_implemented(self):
        fd = self._make_fd()
        result = fd.__mul__(object())
        assert result is NotImplemented

    def test_add_scalar(self):
        fd = self._make_fd()
        result = fd.__add__(1.0)
        assert isinstance(result, FieldDict)

    def test_radd_scalar(self):
        fd = self._make_fd()
        result = fd.__radd__(1.0)
        assert isinstance(result, FieldDict)

    def test_add_non_scalar_returns_not_implemented(self):
        fd = self._make_fd()
        result = fd.__add__(object())
        assert result is NotImplemented

    def test_sub_scalar(self):
        fd = self._make_fd()
        result = fd.__sub__(1.0)
        assert isinstance(result, FieldDict)

    def test_rsub_scalar(self):
        fd = self._make_fd()
        result = fd.__rsub__(5.0)
        assert isinstance(result, FieldDict)

    def test_rsub_non_scalar_returns_not_implemented(self):
        fd = self._make_fd()
        result = fd.__rsub__(object())
        assert result is NotImplemented

    def test_validate_inconsistent_axis_names_raises(self):
        f1 = ScalarField(np.ones((8, 2, 2, 2)), unit=u.V, axis0_domain="time")
        f2 = ScalarField(
            np.ones((8, 2, 2, 2)),
            unit=u.V,
            axis_names=["t", "a", "b", "c"],
            axis0_domain="time",
        )
        with pytest.raises(ValueError, match="axis_names"):
            FieldDict({"a": f1, "b": f2}, validate=True)

    def test_validate_space_domains_mismatch_raises(self):
        f1 = ScalarField(np.ones((8, 2, 2, 2)), unit=u.V, axis0_domain="time")
        f2 = ScalarField(
            np.ones((8, 2, 2, 2)),
            unit=u.V,
            axis0=np.arange(8) * 0.01 * u.s,
            axis1=np.arange(2) * 1.0 / u.m,
            axis2=np.arange(2) * 1.0 / u.m,
            axis3=np.arange(2) * 1.0 / u.m,
            axis0_domain="time",
            space_domain="k",
        )
        # k-domain field has different axis_names (kx,ky,kz), caught before space_domains check
        with pytest.raises(ValueError):
            FieldDict({"a": f1, "b": f2}, validate=True)

    def test_validate_axis_shape_mismatch_raises(self):
        f1 = ScalarField(np.ones((8, 2, 2, 2)), unit=u.V)
        f2 = ScalarField(np.ones((8, 3, 2, 2)), unit=u.V)
        with pytest.raises(ValueError, match="shape mismatch"):
            FieldDict({"a": f1, "b": f2}, validate=True)

    def test_validate_axis_coordinate_mismatch_raises(self):
        f1 = ScalarField(np.ones((8, 2, 2, 2)), unit=u.V,
                         axis0=np.arange(8) * 0.01 * u.s)
        f2 = ScalarField(np.ones((8, 2, 2, 2)), unit=u.V,
                         axis0=np.arange(8) * 0.02 * u.s)
        with pytest.raises(ValueError, match="coordinate mismatch"):
            FieldDict({"a": f1, "b": f2}, validate=True)

    def test_batch_sel_isel(self):
        f = ScalarField(
            np.ones((8, 4, 2, 2)),
            unit=u.V,
            axis0=np.arange(8) * 0.01 * u.s,
            axis1=np.arange(4) * 1.0 * u.m,
            axis2=np.arange(2) * 1.0 * u.m,
            axis3=np.arange(2) * 1.0 * u.m,
            axis_names=["t", "x", "y", "z"],
            axis0_domain="time",
        )
        fd = FieldDict({"a": f, "b": f.copy()})
        result_sel = fd.sel_all(x=0.0 * u.m)
        assert len(result_sel) == 2
        result_isel = fd.isel_all(t=slice(0, 4))
        assert len(result_isel) == 2
