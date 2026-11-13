"""HDF5 round trip of the ``SpectrogramMatrix`` frequency axis (#883)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import SpectrogramMatrix
from gwexpy.timeseries import TimeSeriesMatrix

TIMES = np.array([100.0, 101.0, 103.0])
IRREGULAR_FREQS = np.array([10.0, 20.0, 45.0, 100.0])
REGULAR_FREQS = np.array([0.0, 8.0, 16.0, 24.0])
DATA_3D = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4) * 1e-21
DATA_4D = np.arange(2 * 2 * 3 * 4, dtype=float).reshape(2, 2, 3, 4) * 1e-21


def _roundtrip(matrix, tmp_path):
    target = tmp_path / "spectrogram_matrix.h5"
    matrix.write(target, format="hdf5")
    return SpectrogramMatrix.read(target, format="hdf5"), target


@pytest.mark.parametrize(
    "frequencies", [IRREGULAR_FREQS, REGULAR_FREQS], ids=["irregular", "regular"]
)
@pytest.mark.parametrize("data", [DATA_3D, DATA_4D], ids=["3d", "4d"])
def test_frequency_axis_roundtrip(tmp_path, data, frequencies):
    matrix = SpectrogramMatrix(data, times=TIMES, frequencies=frequencies, name="tf")

    restored, _ = _roundtrip(matrix, tmp_path)

    assert restored.frequencies is not None
    np.testing.assert_array_equal(np.asarray(restored.frequencies), frequencies)
    np.testing.assert_array_equal(np.asarray(restored.times), TIMES)
    np.testing.assert_array_equal(restored.value, matrix.value)
    assert restored.shape == matrix.shape
    assert restored.name == "tf"


@pytest.mark.parametrize("unit", [u.Hz, u.kHz, u.mHz])
def test_frequency_axis_unit_roundtrip(tmp_path, unit):
    frequencies = u.Quantity(IRREGULAR_FREQS, unit)
    matrix = SpectrogramMatrix(DATA_3D, times=TIMES, frequencies=frequencies)

    restored, _ = _roundtrip(matrix, tmp_path)

    assert isinstance(restored.frequencies, u.Quantity)
    assert restored.frequencies.unit == unit
    np.testing.assert_array_equal(restored.frequencies.value, IRREGULAR_FREQS)


def test_time_axis_unit_still_roundtrips(tmp_path):
    matrix = SpectrogramMatrix(
        DATA_3D,
        times=u.Quantity(TIMES, u.s),
        frequencies=u.Quantity(IRREGULAR_FREQS, u.Hz),
    )

    restored, _ = _roundtrip(matrix, tmp_path)

    assert restored.times.unit == u.s
    np.testing.assert_array_equal(restored.times.value, TIMES)


def test_unitless_frequency_axis_stays_unitless(tmp_path):
    matrix = SpectrogramMatrix(DATA_3D, times=TIMES, frequencies=IRREGULAR_FREQS)

    restored, target = _roundtrip(matrix, tmp_path)

    assert not isinstance(restored.frequencies, u.Quantity)
    with h5py.File(target, "r") as h5f:
        assert "unit" not in h5f["frequencies"].attrs
        np.testing.assert_array_equal(h5f["frequencies/value"][:], IRREGULAR_FREQS)


def test_missing_frequency_axis_is_not_invented(tmp_path):
    matrix = SpectrogramMatrix(DATA_3D, times=TIMES)
    assert matrix.frequencies is None

    restored, target = _roundtrip(matrix, tmp_path)

    assert restored.frequencies is None
    with h5py.File(target, "r") as h5f:
        assert "frequencies" not in h5f


def test_file_without_frequency_group_still_reads(tmp_path):
    """Files written before #883 carry no ``frequencies`` group."""
    matrix = SpectrogramMatrix(DATA_3D, times=TIMES, frequencies=IRREGULAR_FREQS)
    target = tmp_path / "legacy.h5"
    matrix.write(target, format="hdf5")
    with h5py.File(target, "a") as h5f:
        del h5f["frequencies"]

    restored = SpectrogramMatrix.read(target, format="hdf5")

    assert restored.frequencies is None
    np.testing.assert_array_equal(restored.value, matrix.value)
    np.testing.assert_array_equal(np.asarray(restored.times), TIMES)


def test_frequency_axis_length_mismatch_in_file_is_rejected(tmp_path):
    matrix = SpectrogramMatrix(DATA_3D, times=TIMES, frequencies=IRREGULAR_FREQS)
    target = tmp_path / "corrupt.h5"
    matrix.write(target, format="hdf5")
    with h5py.File(target, "a") as h5f:
        del h5f["frequencies/value"]
        h5f["frequencies"].create_dataset("value", data=IRREGULAR_FREQS[:-1])

    with pytest.raises(ValueError, match="frequencies length mismatch"):
        SpectrogramMatrix.read(target, format="hdf5")


def test_single_axis_matrix_layout_has_no_frequency_group(tmp_path):
    """Matrices with one sample axis keep their existing HDF5 layout."""
    matrix = TimeSeriesMatrix(np.arange(8.0).reshape(2, 1, 4), dt=0.5, t0=0)
    target = tmp_path / "tsm.h5"

    matrix.write(target, format="hdf5")

    with h5py.File(target, "r") as h5f:
        assert set(h5f) == {"data", "xindex", "meta", "rows", "cols"}
    restored = TimeSeriesMatrix.read(target, format="hdf5")
    np.testing.assert_array_equal(restored.value, matrix.value)
