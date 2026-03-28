"""
Refined ops and edge case tests for SpectrogramMatrix.

Covers:
- Label-based indexing (string lists)
- 4D -> 3D reduction Case B (Col scalar)
- KeyError for unknown labels (row/col)
- to_series_1Dlist() for 3D/4D and ndim < 3 error
- _all_element_units_equivalent() for None mix and semantic match (m/cm)
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import SpectrogramMatrix
from gwexpy.spectrogram.spectrogram import Spectrogram
from gwexpy.types.metadata import MetaData, MetaDataMatrix


class TestSgmGetitemStringList:
    def test_batch_selection_by_string_list(self):
        # Indexing with ['A', 'C'] on 3D (Batch, Time, Freq)
        data = np.random.rand(3, 10, 5)
        sgm = SpectrogramMatrix(data, rows=["A", "B", "C"], times=np.arange(10), frequencies=np.arange(5))
        
        subset = sgm[["A", "C"]]
        assert subset.shape == (2, 10, 5)
        assert list(subset.rows.keys()) == ["A", "C"]
        np.testing.assert_allclose(subset[0].value, sgm[0].value)
        np.testing.assert_allclose(subset[1].value, sgm[2].value)

    def test_invalid_label_keyerror(self):
        data = np.random.rand(2, 5, 5)
        sgm = SpectrogramMatrix(data, rows=["R1", "R2"], cols=["C1"], times=np.arange(5))
        
        # Unknown row
        with pytest.raises(KeyError, match="Invalid row key: Unknown"):
            _ = sgm["Unknown"]
        
        # Unknown col on 3D matrix. 
        # Current implementation passes the string through if not ndim==4, 
        # leading to IndexError in numpy.
        with pytest.raises((KeyError, IndexError)):
            _ = sgm[0, "UnknownCol"]


class TestSgm4dToLowerDim:
    def test_4d_to_3d_case_b_col_scalar(self):
        # (Row, Col, Time, Freq) -> (Row, Time, Freq)
        # Case B: row is slice, col is scalar -> result Batch is Row
        data = np.random.rand(2, 3, 10, 5)
        # Create metadata matrix 2x3
        meta_arr = np.empty((2, 3), dtype=object)
        for i in range(2):
            for j in range(3):
                meta_arr[i, j] = MetaData(unit=u.m, name=f"R{i}C{j}")
        meta = MetaDataMatrix(meta_arr)
        
        sgm = SpectrogramMatrix(
            data, 
            rows=["R0", "R1"], 
            cols=["C0", "C1", "C2"], 
            meta=meta,
            times=np.arange(10),
            frequencies=np.arange(5)
        )
        
        # Slice: all Rows, Col 1
        subset = sgm[:, 1]
        
        # Verify 3D shape (N_row, T, F)
        assert subset.ndim == 3
        # In Case B, Batch axis should inherited from Row
        assert list(subset.rows.keys()) == ["R0", "R1"]
        assert subset.shape == (2, 10, 5)
        
        # Verify metadata (N_row, 1)
        assert subset.meta.shape == (2, 1)
        assert subset.meta[0, 0].name == "R0C1"
        assert subset.meta[1, 0].name == "R1C1"


class TestSgmToSeries1DList:
    def test_3d_to_list(self):
        data = np.random.rand(2, 5, 5)
        # N=2. Time=5. Freq=5.
        sgm = SpectrogramMatrix(data, rows=["A", "B"], times=np.arange(5))
        # Sync meta names with rows to ensure __getitem__ picks them up
        sgm.meta.names = ["A", "B"]
        
        lst = sgm.to_series_1Dlist()
        assert len(lst) == 2
        assert isinstance(lst[0], Spectrogram)
        assert lst[0].name == "A"

    def test_4d_to_list(self):
        data = np.random.rand(2, 2, 5, 5)
        sgm = SpectrogramMatrix(
            data, 
            rows=["R1", "R2"], 
            cols=["C1", "C2"], 
            times=np.arange(5)
        )
        lst = sgm.to_series_1Dlist()
        assert len(lst) == 4
        assert isinstance(lst[0], Spectrogram)

    def test_ndim_less_than_3_value_error(self):
        # Create a 2D view
        data = np.random.rand(5, 5)
        sgm = data.view(SpectrogramMatrix)
        # ndim is 2
        with pytest.raises(ValueError, match="Unsupported SpectrogramMatrix dimension: 2"):
            sgm.to_series_1Dlist()


class TestSgmAllElementUnits:
    def test_meta_none_returns_true(self):
        data = np.random.rand(1, 5, 5)
        sgm = SpectrogramMatrix(data, unit=u.m, times=np.arange(5))
        sgm.meta = None
        eq, unit = sgm._all_element_units_equivalent()
        assert eq is True
        assert unit == u.m

    def test_semantic_equivalence_m_cm(self):
        # m and cm ARE equivalent
        data = np.random.rand(2, 1, 5, 5)
        meta = MetaDataMatrix([
            [MetaData(unit=u.m)],
            [MetaData(unit=u.cm)]
        ])
        sgm = SpectrogramMatrix(data, meta=meta, times=np.arange(5))
        eq, unit = sgm._all_element_units_equivalent()
        assert eq is True
        assert unit == u.m

    def test_unit_none_mix_is_equivalent(self):
        # None should be skipped
        data = np.random.rand(2, 1, 5, 5)
        m1 = MetaData(unit=u.m)
        m2 = MetaData()
        # Bypassing setter to force None
        m2["unit"] = None
        
        meta = MetaDataMatrix([[m1], [m2]])
        sgm = SpectrogramMatrix(data, meta=meta, times=np.arange(5))
        
        # Verify m2.unit is indeed None
        assert sgm.meta[1, 0].unit is None
        
        eq, unit = sgm._all_element_units_equivalent()
        assert eq is True
        assert unit == u.m

    def test_non_equivalent_m_s(self):
        # m and s are NOT equivalent
        data = np.random.rand(2, 1, 5, 5)
        meta = MetaDataMatrix([
            [MetaData(unit=u.m)],
            [MetaData(unit=u.s)]
        ])
        sgm = SpectrogramMatrix(data, meta=meta, times=np.arange(5))
        eq, unit = sgm._all_element_units_equivalent()
        assert eq is False
        assert unit == u.m
