import pickle
import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList, SpectrogramMatrix


@pytest.fixture
def sgm_4d():
    # Shape: (2, 3, 10, 5) -> (Row, Col, Time, Freq)
    data = np.random.rand(2, 3, 10, 5)
    times = np.arange(10)
    freqs = np.arange(5)
    return SpectrogramMatrix(
        data, 
        times=times, 
        frequencies=freqs, 
        rows=["R1", "R2"], 
        cols=["C1", "C2", "C3"],
        name="Matrix4D",
        unit=u.V
    )


def test_sgm_attribute_propagation(sgm_4d):
    sgm_4d._gwex_custom = "test_value"
    
    # Slice Row 0
    sub = sgm_4d[0]
    assert sub._gwex_custom == "test_value"
    assert sub.ndim == 3
    assert sub.shape == (3, 10, 5) # (Col, Time, Freq)
    
    # Sourced View
    view = sgm_4d.view(SpectrogramMatrix)
    assert view._gwex_custom == "test_value"


def test_sgm_pickle_roundtrip(sgm_4d):
    sgm_4d._gwex_extra = "pickle_me"
    
    # Pickle and Unpickle
    pdata = pickle.dumps(sgm_4d)
    restored = pickle.loads(pdata)
    
    assert isinstance(restored, SpectrogramMatrix)
    assert restored._gwex_extra == "pickle_me"
    assert np.allclose(restored.value, sgm_4d.value)
    assert np.allclose(restored.times, sgm_4d.times)
    assert np.allclose(restored.frequencies, sgm_4d.frequencies)
    assert list(restored.rows.keys()) == ["R1", "R2"]
    assert list(restored.cols.keys()) == ["C1", "C2", "C3"]


def test_sgm_conversion_methods(sgm_4d):
    # to_list
    sgl = sgm_4d.to_list()
    assert isinstance(sgl, SpectrogramList)
    assert len(sgl) == 2 * 3 # N * M
    assert isinstance(sgl[0], Spectrogram)
    
    # to_dict
    sgd = sgm_4d.to_dict()
    assert isinstance(sgd, SpectrogramDict)
    # Check key format for 4D matrix: (row_key, col_key)
    assert ("R1", "C2") in sgd
    assert isinstance(sgd[("R1", "C2")], Spectrogram)
    
    # to_series_2Dlist
    list2d = sgm_4d.to_series_2Dlist()
    assert len(list2d) == 2 # Rows
    assert len(list2d[0]) == 3 # Cols
    assert isinstance(list2d[0][0], Spectrogram)


def test_sgm_4d_metadata_slicing(sgm_4d):
    # Slice a sub-matrix (Row 0, Col 1:3)
    # Result shape: (2, 10, 5) if row is scalar, or (1, 2, 10, 5) if row is slice
    sub = sgm_4d[0, 1:3]
    assert sub.ndim == 3
    assert sub.shape == (2, 10, 5)
    # Metadata should follow: rows=['C2', 'C3'] (since it's now Batch=Col)
    # Actually, the slicing logic in SpectrogramMatrix.__getitem__ for 4D:
    # if Row is scalar, it reduces to (Col, T, F).
    assert list(sub.row_keys()) == ["C2", "C3"]
    assert sub.meta.shape == (2, 1)


def test_sgm_getitem_single_element(sgm_4d):
    # Access single element by label
    spec = sgm_4d["R2", "C1"]
    assert isinstance(spec, Spectrogram)
    assert spec.shape == (10, 5)
    # Check that unit and name propagated correctly from meta if possible
    # (By default it uses global unit/name if meta not set per-element)
    assert spec.unit == u.V
    assert spec.name == "Matrix4D"
