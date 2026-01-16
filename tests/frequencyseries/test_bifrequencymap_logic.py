import pytest
import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries, BifrequencyMap

class TestBifrequencyMapLogic:
    """Test A2-c logic in BifrequencyMap."""

    @pytest.fixture
    def simple_map(self):
        # 2x2 map: Output = 2 * Input
        data = np.array([[2, 0], [0, 2]])
        return BifrequencyMap(data, xindex=[10, 20], yindex=[10, 20], unit='m/V')

    def test_propagate_exact_match(self, simple_map):
        input_fs = FrequencySeries([1, 3], frequencies=[10, 20], unit='V')
        output_fs = simple_map.propagate(input_fs)
        
        np.testing.assert_array_equal(output_fs.value, [2, 6])
        assert output_fs.unit == u.m
        np.testing.assert_array_equal(output_fs.frequencies.value, simple_map.yindex.value)

    def test_propagate_interpolation(self, simple_map):
        # Input grid differs from map xindex
        input_fs = FrequencySeries([1, 1, 1], frequencies=[10, 15, 20], unit='V')
        # Should interpolate input_fs onto map.xindex (10, 20) -> vals 1, 1
        output_fs = simple_map.propagate(input_fs, interpolate=True)
        np.testing.assert_allclose(output_fs.value, [2, 2])

    def test_diagonal_logic(self):
        data = np.eye(3)
        bfm = BifrequencyMap(data, xindex=[0, 1, 2], yindex=[0, 1, 2])
        diag = bfm.diagonal()
        assert np.sum(diag.value) > 0
