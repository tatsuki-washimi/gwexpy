from __future__ import annotations

import copy

import numpy as np
import pytest
from astropy import units as u

import gwexpy  # noqa: F401 - register constructors used by as_series
from gwexpy.types import Array3D, Array4D, as_series
from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix


def test_axis_api_rename_axes_copy_rejects_duplicates_and_preserves_source():
    arr = Array4D(
        np.zeros((2, 3, 4, 5)),
        axis_names=("time", "x", "y", "frequency"),
    )

    renamed = arr.rename_axes({"x": "distance"})

    assert renamed.axis_names == ("time", "distance", "y", "frequency")
    assert arr.axis_names == ("time", "x", "y", "frequency")

    with pytest.raises(ValueError, match="Duplicate axis names"):
        arr.rename_axes({"x": "time"})


def test_axis_api_isel_preserves_axis_names_units_and_sliced_coordinates():
    arr = Array3D(
        np.zeros((3, 4, 5)),
        axis_names=("time", "distance", "frequency"),
        axis0=np.array([10.0, 11.0, 12.0]) * u.s,
        axis1=np.array([1.0, 2.0, 4.0, 8.0]) * u.m,
        axis2=np.array([20.0, 30.0, 40.0, 50.0, 60.0]) * u.Hz,
    )

    selected = arr.isel({"distance": slice(1, 3), "frequency": slice(2, 5)})

    assert isinstance(selected, Array3D)
    assert selected.axis_names == ("time", "distance", "frequency")
    assert np.all(selected.axis("distance").index == [2.0, 4.0] * u.m)
    assert np.all(selected.axis("frequency").index == [40.0, 50.0, 60.0] * u.Hz)


def test_metadata_dict_and_matrix_deepcopy_isolates_entries():
    md = MetaData(name="strain", channel="H1:STRAIN", unit=u.m)
    metadata = MetaDataDict({"h1": md})
    matrix = MetaDataMatrix([[md]])

    metadata_copy = copy.deepcopy(metadata)
    matrix_copy = copy.deepcopy(matrix)

    metadata_copy["h1"].name = "copy"
    matrix_copy[0, 0].unit = u.s

    assert metadata["h1"].name == "strain"
    assert matrix[0, 0].unit == u.m
    assert metadata_copy["h1"] is not metadata["h1"]
    assert matrix_copy[0, 0] is not matrix[0, 0]


def test_metadata_matrix_shape_default_and_fill_create_independent_cells():
    matrix = MetaDataMatrix(shape=(2, 2), default={"name": "base", "unit": u.m})
    matrix[0, 0].name = "changed"

    assert matrix[0, 1].name == "base"
    assert matrix[1, 1].unit == u.m

    matrix.fill(MetaData(name="filled", channel="L1:TEST", unit=u.s))
    matrix[0, 0].channel = "H1:TEST"

    assert str(matrix[0, 0].channel) == "H1:TEST"
    assert str(matrix[0, 1].channel) == "L1:TEST"
    assert matrix[0, 0] is not matrix[0, 1]


def test_metadata_csv_roundtrip_preserves_channels_units_and_names(tmp_path):
    metadata = MetaDataDict(
        {
            "row0": MetaData(name="left", channel="H1:LOW", unit=u.m),
            "row1": MetaData(name="right", channel="L1:LOW", unit=u.s),
        }
    )
    matrix = MetaDataMatrix(
        [
            [
                MetaData(name="a", channel="H1:A", unit=u.Hz),
                MetaData(name="b", channel="L1:B", unit=u.m),
            ]
        ]
    )

    metadata_path = tmp_path / "metadata.csv"
    matrix_path = tmp_path / "matrix.csv"
    metadata.write(metadata_path)
    matrix.write(matrix_path)

    metadata2 = MetaDataDict.read(metadata_path)
    matrix2 = MetaDataMatrix.read(matrix_path)

    assert list(metadata2.keys()) == ["row0", "row1"]
    assert metadata2["row0"].name == "left"
    assert str(metadata2["row0"].channel) == "H1:LOW"
    assert metadata2["row1"].unit == u.s

    assert matrix2.shape == (1, 2)
    assert matrix2[0, 0].name == "a"
    assert str(matrix2[0, 0].channel) == "H1:A"
    assert matrix2[0, 0].unit == u.Hz
    assert str(matrix2[0, 1].channel) == "L1:B"
    assert matrix2[0, 1].unit == u.m


@pytest.mark.xfail(
    strict=True,
    reason="MetaDataMatrix CSV missing channel/unit cells currently reload as pandas NaN",
)
def test_metadata_matrix_csv_roundtrip_normalizes_missing_values(tmp_path):
    matrix = MetaDataMatrix(
        [[MetaData(name="empty", channel="", unit=u.dimensionless_unscaled)]]
    )
    matrix_path = tmp_path / "matrix-missing.csv"

    matrix.write(matrix_path)
    matrix2 = MetaDataMatrix.read(matrix_path)

    assert matrix2[0, 0].name == "empty"
    assert str(matrix2[0, 0].channel) == ""
    assert matrix2[0, 0].unit == u.dimensionless_unscaled


def test_as_series_angular_frequency_default_keeps_values_angular_and_axis_hz():
    angular_axis = (2 * np.pi * np.array([1.0, 2.0, 4.0])) * (u.rad / u.s)

    series = as_series(angular_axis)

    assert series.unit == u.rad / u.s
    assert series.frequencies.unit == u.Hz
    assert np.allclose(series.value, angular_axis.value)
    assert np.allclose(series.frequencies.value, [1.0, 2.0, 4.0])


def test_as_series_angular_frequency_can_emit_hz_values_on_hz_axis():
    angular_axis = (2 * np.pi * np.array([1.0, 2.0, 4.0])) * (u.rad / u.s)

    series = as_series(angular_axis, unit=u.Hz)

    assert series.unit == u.Hz
    assert series.frequencies.unit == u.Hz
    assert np.allclose(series.value, [1.0, 2.0, 4.0])
    assert np.allclose(series.frequencies.value, [1.0, 2.0, 4.0])
