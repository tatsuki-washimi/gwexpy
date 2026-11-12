"""``MetaData`` must not turn an invalid unit into dimensionless (#776)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix

INVALID_UNITS = ["garbage-unit-xyz", "m/", "not a unit", object(), [1, 2]]
INVALID_IDS = ["unknown-name", "syntax", "words", "object", "list"]


@pytest.mark.parametrize("unit", INVALID_UNITS, ids=INVALID_IDS)
def test_constructor_rejects_invalid_unit(unit):
    with pytest.raises(ValueError, match="[Ii]nvalid unit"):
        MetaData(name="x", unit=unit)


@pytest.mark.parametrize("unit", INVALID_UNITS, ids=INVALID_IDS)
def test_setter_rejects_invalid_unit_and_keeps_previous(unit):
    meta = MetaData(name="x", unit="m")

    with pytest.raises(ValueError, match="[Ii]nvalid unit"):
        meta.unit = unit

    assert meta.unit == u.m


def test_invalid_unit_error_names_the_input_and_chains_the_cause():
    with pytest.raises(ValueError, match="garbage-unit-xyz") as excinfo:
        MetaData(unit="garbage-unit-xyz")

    assert excinfo.value.__cause__ is not None


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("m", u.m),
        ("m / s", u.m / u.s),
        ("1e-21 m", u.Unit("1e-21 m")),
        (u.Hz, u.Hz),
        (u.m**2 / u.Hz, u.m**2 / u.Hz),
        (u.dimensionless_unscaled, u.dimensionless_unscaled),
        ("", u.dimensionless_unscaled),
        # Astropy itself parses a blank string as dimensionless.
        ("  ", u.dimensionless_unscaled),
        ("1", u.dimensionless_unscaled),
        (1, u.dimensionless_unscaled),
        (None, u.dimensionless_unscaled),
    ],
    ids=[
        "m",
        "m/s",
        "scaled",
        "Unit",
        "composite",
        "dimless",
        "empty",
        "blank",
        "one-str",
        "one",
        "None",
    ],
)
def test_valid_units_are_preserved(unit, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        built = MetaData(unit=unit)
        assigned = MetaData(unit="m")
        assigned.unit = unit

    assert built.unit == expected
    assert assigned.unit == expected


def test_omitted_unit_is_dimensionless():
    assert MetaData().unit == u.dimensionless_unscaled
    assert MetaData(name="x").unit == u.dimensionless_unscaled


def test_metadatadict_rejects_invalid_unit_entry():
    with pytest.raises(ValueError, match="[Ii]nvalid unit"):
        MetaDataDict({"a": {"name": "a", "unit": "garbage-unit-xyz"}})


def test_metadatamatrix_units_setter_rejects_invalid_unit():
    matrix = MetaDataMatrix(shape=(1, 2))

    with pytest.raises(ValueError, match="[Ii]nvalid unit"):
        matrix.units = [["m", "garbage-unit-xyz"]]


def test_series_matrix_rejects_invalid_unit_instead_of_dropping_it():
    data = np.arange(8.0).reshape(2, 1, 4)

    with pytest.raises(ValueError, match="[Ii]nvalid unit"):
        TimeSeriesMatrix(data, dt=1.0, units=[["garbage-unit-xyz"], ["m"]])


def test_series_matrix_valid_units_still_propagate():
    data = np.arange(8.0).reshape(2, 1, 4)

    matrix = TimeSeriesMatrix(data, dt=1.0, units=[["m"], ["s"]])

    assert matrix.meta[0, 0].unit == u.m
    assert matrix.meta[1, 0].unit == u.s
