"""Tests for Field4DList and Field4DDict collections."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict as Field4DDict
from gwexpy.fields import FieldList as Field4DList
from gwexpy.fields import ScalarField as Field4D


class TestField4DListBasic:
    """Test Field4DList basic functionality."""

    @pytest.fixture
    def field_list(self):
        """Create a list of Field4D objects."""
        fields = []
        for i in range(3):
            np.random.seed(i)
            data = np.random.randn(16, 4, 4, 4)
            f = Field4D(
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
        return Field4DList(fields)

    def test_field4d_list_is_list(self, field_list):
        """Test that Field4DList is a list."""
        assert isinstance(field_list, list)

    def test_field4d_list_length(self, field_list):
        """Test list length."""
        assert len(field_list) == 3

    def test_field4d_list_indexing(self, field_list):
        """Test list indexing."""
        assert isinstance(field_list[0], Field4D)

    def test_field4d_list_append(self, field_list):
        """Test appending to list."""
        new_field = Field4D(np.zeros((16, 4, 4, 4)))
        field_list.append(new_field)
        assert len(field_list) == 4


class TestField4DListFftOperations:
    """Test Field4DList batch FFT operations."""

    @pytest.fixture
    def time_domain_list(self):
        """Create a list of time-domain Field4D objects."""
        fields = []
        for i in range(3):
            np.random.seed(i)
            data = np.random.randn(32, 4, 4, 4)
            f = Field4D(
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
        return Field4DList(fields)

    def test_fft_time_all(self, time_domain_list):
        """Test fft_time_all applies to all fields."""
        result = time_domain_list.fft_time_all()

        assert isinstance(result, Field4DList)
        assert len(result) == 3
        for f in result:
            assert isinstance(f, Field4D)
            assert f.axis0_domain == "frequency"

    def test_ifft_time_all(self, time_domain_list):
        """Test ifft_time_all after fft_time_all."""
        freq_list = time_domain_list.fft_time_all()
        result = freq_list.ifft_time_all()

        assert isinstance(result, Field4DList)
        assert len(result) == 3
        for f in result:
            assert f.axis0_domain == "time"

    def test_fft_space_all(self, time_domain_list):
        """Test fft_space_all applies to all fields."""
        result = time_domain_list.fft_space_all()

        assert isinstance(result, Field4DList)
        assert len(result) == 3
        for f in result:
            assert f.space_domains["kx"] == "k"
            assert f.space_domains["ky"] == "k"
            assert f.space_domains["kz"] == "k"

    def test_ifft_space_all(self, time_domain_list):
        """Test ifft_space_all after fft_space_all."""
        k_list = time_domain_list.fft_space_all()
        result = k_list.ifft_space_all()

        assert isinstance(result, Field4DList)
        for f in result:
            assert f.space_domains["x"] == "real"


class TestField4DListValidation:
    """Test Field4DList validation."""

    def test_validate_consistent_fields(self):
        """Test validation passes for consistent fields."""
        fields = [
            Field4D(np.zeros((10, 4, 4, 4)), unit=u.V, axis_names=["t", "x", "y", "z"])
            for _ in range(3)
        ]
        # Should not raise
        flist = Field4DList(fields, validate=True)
        assert len(flist) == 3

    def test_validate_inconsistent_units_raises(self):
        """Test validation raises for inconsistent units."""
        fields = [
            Field4D(np.zeros((10, 4, 4, 4)), unit=u.V),
            Field4D(np.zeros((10, 4, 4, 4)), unit=u.A),
        ]

        with pytest.raises(ValueError, match="Inconsistent unit"):
            Field4DList(fields, validate=True)

    def test_validate_inconsistent_axis_names_raises(self):
        """Test validation raises for inconsistent axis names."""
        fields = [
            Field4D(np.zeros((10, 4, 4, 4)), axis_names=["t", "x", "y", "z"]),
            Field4D(np.zeros((10, 4, 4, 4)), axis_names=["t", "a", "b", "c"]),
        ]

        with pytest.raises(ValueError, match="Inconsistent axis_names"):
            Field4DList(fields, validate=True)

    def test_validate_inconsistent_domain_raises(self):
        """Test validation raises for inconsistent domains."""
        # Use same axis_names so domain validation is reached
        fields = [
            Field4D(
                np.zeros((10, 4, 4, 4)),
                axis_names=["t", "x", "y", "z"],
                axis0_domain="time",
            ),
            Field4D(
                np.zeros((10, 4, 4, 4)),
                axis_names=["t", "x", "y", "z"],
                axis0_domain="frequency",
            ),
        ]

        with pytest.raises(ValueError, match="Inconsistent axis0_domain"):
            Field4DList(fields, validate=True)

    def test_validate_non_field4d_raises(self):
        """Test validation raises for non-Field4D items."""
        items = [Field4D(np.zeros((10, 4, 4, 4))), "not a field"]

        with pytest.raises(TypeError, match="Expected Field4D"):
            Field4DList(items, validate=True)


class TestField4DDictBasic:
    """Test Field4DDict basic functionality."""

    @pytest.fixture
    def field_dict(self):
        """Create a dict of Field4D objects."""
        fields = {}
        for name in ["Ex", "Ey", "Ez"]:
            np.random.seed(hash(name) % 1000)
            data = np.random.randn(16, 4, 4, 4)
            f = Field4D(
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
        return Field4DDict(fields)

    def test_field4d_dict_is_dict(self, field_dict):
        """Test that Field4DDict is a dict."""
        assert isinstance(field_dict, dict)

    def test_field4d_dict_length(self, field_dict):
        """Test dict length."""
        assert len(field_dict) == 3

    def test_field4d_dict_keys(self, field_dict):
        """Test dict keys."""
        assert set(field_dict.keys()) == {"Ex", "Ey", "Ez"}

    def test_field4d_dict_indexing(self, field_dict):
        """Test dict indexing."""
        assert isinstance(field_dict["Ex"], Field4D)


class TestField4DDictFftOperations:
    """Test Field4DDict batch FFT operations."""

    @pytest.fixture
    def time_domain_dict(self):
        """Create a dict of time-domain Field4D objects."""
        fields = {}
        for name in ["Ex", "Ey"]:
            np.random.seed(hash(name) % 1000)
            data = np.random.randn(32, 4, 4, 4)
            f = Field4D(
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
        return Field4DDict(fields)

    def test_fft_time_all(self, time_domain_dict):
        """Test fft_time_all applies to all fields."""
        result = time_domain_dict.fft_time_all()

        assert isinstance(result, Field4DDict)
        assert set(result.keys()) == {"Ex", "Ey"}
        for f in result.values():
            assert isinstance(f, Field4D)
            assert f.axis0_domain == "frequency"

    def test_ifft_time_all(self, time_domain_dict):
        """Test ifft_time_all after fft_time_all."""
        freq_dict = time_domain_dict.fft_time_all()
        result = freq_dict.ifft_time_all()

        assert isinstance(result, Field4DDict)
        for f in result.values():
            assert f.axis0_domain == "time"

    def test_fft_space_all(self, time_domain_dict):
        """Test fft_space_all applies to all fields."""
        result = time_domain_dict.fft_space_all()

        assert isinstance(result, Field4DDict)
        for f in result.values():
            assert f.space_domains["kx"] == "k"

    def test_ifft_space_all(self, time_domain_dict):
        """Test ifft_space_all after fft_space_all."""
        k_dict = time_domain_dict.fft_space_all()
        result = k_dict.ifft_space_all()

        assert isinstance(result, Field4DDict)
        for f in result.values():
            assert f.space_domains["x"] == "real"


class TestField4DDictValidation:
    """Test Field4DDict validation."""

    def test_validate_consistent_fields(self):
        """Test validation passes for consistent fields."""
        fields = {
            "a": Field4D(np.zeros((10, 4, 4, 4)), unit=u.V),
            "b": Field4D(np.zeros((10, 4, 4, 4)), unit=u.V),
        }
        # Should not raise
        fdict = Field4DDict(fields, validate=True)
        assert len(fdict) == 2

    def test_validate_inconsistent_units_raises(self):
        """Test validation raises for inconsistent units."""
        fields = {
            "a": Field4D(np.zeros((10, 4, 4, 4)), unit=u.V),
            "b": Field4D(np.zeros((10, 4, 4, 4)), unit=u.A),
        }

        with pytest.raises(ValueError, match="Inconsistent unit"):
            Field4DDict(fields, validate=True)

    def test_validate_non_field4d_raises(self):
        """Test validation raises for non-Field4D values."""
        items = {"a": Field4D(np.zeros((10, 4, 4, 4))), "b": "not a field"}

        with pytest.raises(TypeError, match="Expected Field4D"):
            Field4DDict(items, validate=True)
