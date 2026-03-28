import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeriesMatrix


def test_fs_matrix_new_parameters():
    # Test f0 and df
    data = np.random.rand(2, 5)
    fsm = FrequencySeriesMatrix(data, f0=10 * u.Hz, df=2 * u.Hz)
    assert fsm.f0 == 10 * u.Hz
    assert fsm.df == 2 * u.Hz
    assert np.array_equal(fsm.frequencies.value, [10, 12, 14, 16, 18])

    # Test frequencies explicitly
    freqs = [10, 20, 30, 40, 50] * u.Hz
    fsm_f = FrequencySeriesMatrix(data, frequencies=freqs)
    assert np.array_equal(fsm_f.frequencies, freqs)
    assert fsm_f.f0 == 10 * u.Hz
    assert fsm_f.df == 10 * u.Hz


def test_fs_matrix_channel_names_reshaping():
    # 2D data (N, Freq) -> N channels
    data_2d = np.random.rand(3, 10)
    names = ["A", "B", "C"]
    fsm_2d = FrequencySeriesMatrix(data_2d, channel_names=names, df=1)
    assert list(fsm_2d.channel_names) == names

    # 3D data (N, M, Freq) -> N*M channels
    data_3d = np.random.rand(2, 2, 10)
    names_4 = ["A1", "A2", "B1", "B2"]
    fsm_3d = FrequencySeriesMatrix(data_3d, channel_names=names_4, df=1)
    # SeriesMatrix handles names as (N, M) if size matches N*M
    assert fsm_3d.names.shape == (2, 2)
    assert fsm_3d.names[0, 1] == "A2"


def test_fs_matrix_metadata_init():
    data = np.random.rand(2, 3, 10)
    fsm = FrequencySeriesMatrix(
        data, 
        df=1, 
        rows=["R1", "R2"], 
        cols=["C1", "C2", "C3"],
        name="TestMatrix"
    )
    assert fsm.name == "TestMatrix"
    assert list(fsm.rows.keys()) == ["R1", "R2"]
    assert list(fsm.cols.keys()) == ["C1", "C2", "C3"]
    assert fsm.meta.shape == (2, 3)


def test_fs_matrix_attribute_propagation():
    data = np.random.rand(2, 5)
    fsm = FrequencySeriesMatrix(data, df=1)
    fsm._gwex_custom = "propagate_me"
    
    # Slicing
    sub = fsm[0:1, :]
    assert sub._gwex_custom == "propagate_me"
    
    # View casting
    view = fsm.view(FrequencySeriesMatrix)
    assert view._gwex_custom == "propagate_me"
