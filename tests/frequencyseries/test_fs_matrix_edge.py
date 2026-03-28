"""
Tests for FrequencySeriesMatrix specific edge cases.

Covers:
- Parameter priority for frequencies vs df/f0
- Empty data initialization and its frequencies/shape
- Conversion methods (to_list, to_dict) return types, element types, and key shapes.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix
from gwexpy.frequencyseries.collections import FrequencySeriesDict, FrequencySeriesList


class TestFSMatrixFreqParamPriority:
    def test_frequencies_wins_over_contradicting_df(self):
        # Case: frequencies=([10, 20, 30] Hz) implies df=10, f0=10.
        # But we pass contradicting df=50, f0=100.
        # Explicit 'frequencies' parameter should win (via xindex mapping in __new__).
        data = np.random.rand(1, 1, 3)
        freqs = [10, 20, 30] * u.Hz
        fsm = FrequencySeriesMatrix(
            data,
            frequencies=freqs,
            df=50.0 * u.Hz,
            f0=100.0 * u.Hz
        )
        
        # Verify result is based on freqs
        np.testing.assert_allclose(fsm.frequencies.value, [10, 20, 30])
        assert fsm.f0 == 10.0 * u.Hz
        assert fsm.df == 10.0 * u.Hz


class TestFSMatrixEmptyInit:
    def test_data_none_init(self):
        # FrequencySeriesMatrix(data=None)
        fsm = FrequencySeriesMatrix(data=None)
        
        # Verify shape
        assert fsm.shape == (0, 0, 0)
        
        # Verify frequencies (xindex)
        assert len(fsm.frequencies) == 0
        assert isinstance(fsm.frequencies, np.ndarray)


class TestFSMatrixConversionMethods:
    @pytest.fixture
    def fsm_2x2(self):
        # 2x2 matrix with 3 frequency bins
        data = np.random.rand(2, 2, 3)
        freqs = [0, 10, 20] * u.Hz
        return FrequencySeriesMatrix(
            data, 
            frequencies=freqs, 
            rows=["R1", "R2"], 
            cols=["C1", "C2"]
        )

    def test_to_list_type_and_element(self, fsm_2x2):
        # to_list() should return FrequencySeriesList
        fs_list = fsm_2x2.to_list()
        
        assert isinstance(fs_list, FrequencySeriesList)
        assert len(fs_list) == 4  # N*M elements (flattened list)
        
        # Check first element
        first = fs_list[0]
        assert isinstance(first, FrequencySeries)
        # Verify frequencies match source matrix
        np.testing.assert_allclose(first.frequencies.value, fsm_2x2.frequencies.value)
        assert first.frequencies.unit == fsm_2x2.frequencies.unit

    def test_to_dict_type_and_multi_column_keys(self, fsm_2x2):
        # to_dict() should return FrequencySeriesDict
        fs_dict = fsm_2x2.to_dict()
        
        assert isinstance(fs_dict, FrequencySeriesDict)
        assert len(fs_dict) == 4
        
        # Multi-column key check: (row, col) key format for SeriesMatrixBase
        expected_keys = [
            ("R1", "C1"), ("R1", "C2"),
            ("R2", "C1"), ("R2", "C2")
        ]
        assert list(fs_dict.keys()) == expected_keys
        
        # Check first element
        first = fs_dict[("R1", "C1")]
        assert isinstance(first, FrequencySeries)
        # Verify frequencies match source matrix
        np.testing.assert_allclose(first.frequencies.value, fsm_2x2.frequencies.value)


class TestFSMatrixF0Fallback:
    def test_default_f0_to_zero(self):
        # f0 should default to 0 if only df is specified
        data = np.random.rand(1, 1, 10)
        fsm = FrequencySeriesMatrix(data, df=1.0 * u.Hz)
        assert fsm.f0 == 0 * u.Hz
        assert fsm.df == 1.0 * u.Hz


class TestFSMatrixChannelNamesReshaping:
    def test_reshape_matching_total_size(self):
        # cn.size == N * M
        data = np.random.rand(2, 3, 5)
        names = [f"ch{i}" for i in range(6)]
        fsm = FrequencySeriesMatrix(data, channel_names=names, df=1.0)
        assert fsm.meta.names.shape == (2, 3)
        assert fsm.meta[0, 1].name == "ch1"
        assert fsm.meta[1, 0].name == "ch3"

    def test_reshape_matching_rows(self):
        # cn.size == N
        data = np.random.rand(2, 3, 5)
        names = ["Row1", "Row2"]
        fsm = FrequencySeriesMatrix(data, channel_names=names, df=1.0)
        assert fsm.meta.names.shape == (2, 3)
        assert fsm.meta[0, 0].name == "Row1"
        assert fsm.meta[1, 2].name == "Row2"

    def test_no_reshape_other_size(self):
        # cn.size != N*M and cn.size != N
        data = np.random.rand(2, 3, 5)
        names = ["A", "B", "C"]  # size 3 == M
        fsm = FrequencySeriesMatrix(data, channel_names=names, df=1.0)
        # Should broadcast to (2, 3)
        assert fsm.meta.names.shape == (2, 3)
        assert fsm.meta[0, 0].name == "A"
        assert fsm.meta[1, 0].name == "A"

    def test_handle_scalar_data_no_reshape(self):
        # Case where len(dshape) < 2 (scalar data)
        # Line 97: kwargs["names"] = cn
        fsm = FrequencySeriesMatrix(1.0, channel_names=["A"], frequencies=[0])
        assert fsm.meta.names.shape == (1, 1)
        assert fsm.meta[0, 0].name == "A"

    def test_handle_exception_reshape_fallback(self):
        # Trigger the 'except' block at Line 98 using a custom object with None shape
        class FaultyData:
            shape = None  # len(None) raises TypeError
        
        data = FaultyData()
        names = ["ch1", "ch2", "ch3"]
        
        with patch("gwexpy.frequencyseries.matrix.SeriesMatrix.__new__") as mock_new:
            mock_new.return_value = MagicMock()
            fsm = FrequencySeriesMatrix(data, channel_names=names, frequencies=[0, 1, 2])
            
            # Check what was passed to names (it should be reshaped)
            args, kwargs = mock_new.call_args
            assert kwargs["names"].shape == (3, 1)
            assert kwargs["names"][0, 0] == "ch1"

    def test_handle_exception_multidim_names_no_reshape(self):
        # Trigger the 'except' block but cn.ndim != 1
        class FaultyData:
            shape = None
        
        data = FaultyData()
        names = np.array([["A"]])
        
        with patch("gwexpy.frequencyseries.matrix.SeriesMatrix.__new__") as mock_new:
            mock_new.return_value = MagicMock()
            fsm = FrequencySeriesMatrix(data, channel_names=names, frequencies=[0])
            
            args, kwargs = mock_new.call_args
            # Hits Line 102: kwargs["names"] = cn
            assert kwargs["names"].shape == (1, 1)
            assert kwargs["names"][0, 0] == "A"
