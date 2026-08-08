"""Regression contracts for fail-closed Field collection arithmetic (#578)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, FieldList, ScalarField, TensorField, VectorField


def _vector_field() -> VectorField:
    axes = (
        np.arange(2) * u.s,
        np.arange(2) * u.m,
        np.arange(2) * u.m,
        np.arange(2) * u.m,
    )
    return VectorField(
        {
            "x": ScalarField(
                np.ones((2, 2, 2, 2)),
                unit=u.V,
                axis0=axes[0],
                axis1=axes[1],
                axis2=axes[2],
                axis3=axes[3],
                name="electric-x",
            ),
            "y": ScalarField(
                np.full((2, 2, 2, 2), 2.0),
                unit=u.V,
                axis0=axes[0],
                axis1=axes[1],
                axis2=axes[2],
                axis3=axes[3],
                name="electric-y",
            ),
        },
        basis="custom",
    )


def test_quantity_left_multiply_preserves_vectorfield_and_component_metadata():
    vector = _vector_field()

    result = (2 * u.m) * vector

    assert isinstance(result, VectorField)
    assert result.basis == "custom"
    assert list(result) == ["x", "y"]
    assert result["x"].name == "electric-x"
    assert result["x"].unit == u.V * u.m
    np.testing.assert_allclose(result["x"].value, 2.0)
    np.testing.assert_allclose(result["x"]._axis1_index.to_value(u.m), [0.0, 1.0])


@pytest.mark.parametrize(
    ("operation", "unit"),
    [
        (lambda fields: fields * u.m, u.V * u.m),
        (lambda fields: u.m * fields, u.V * u.m),
        (lambda fields: fields / u.m, u.V / u.m),
        (lambda fields: u.m / fields, u.m / u.V),
    ],
)
def test_unit_multiply_and_divide_preserve_vectorfield(operation, unit):
    result = operation(_vector_field())

    assert isinstance(result, VectorField)
    assert result.basis == "custom"
    assert list(result) == ["x", "y"]
    assert result["x"].unit == unit


def test_fieldlist_scalar_multiplication_is_not_list_repetition():
    result = 2 * FieldList([_vector_field()["x"]])

    assert isinstance(result, FieldList)
    assert len(result) == 1
    assert result[0].unit == u.V
    np.testing.assert_allclose(result[0].value, 2.0)


def test_fieldlist_preserves_the_native_left_list_concatenation_limit():
    field_list = FieldList([_vector_field()["x"]])

    with pytest.raises(TypeError):
        field_list + []

    # ``FieldList`` remains a ``list`` subclass for GWpy compatibility.  The
    # left-hand built-in list implementation runs first and cannot be
    # intercepted by a subclass, so it deliberately has normal list semantics.
    result = [] + field_list
    assert type(result) is list
    assert result == list(field_list)


@pytest.mark.parametrize("fields", [FieldDict(), FieldList()])
def test_direct_ufunc_is_fail_closed(fields):
    with pytest.raises(TypeError, match="ufunc"):
        np.add(fields, 1)


def test_unitful_fields_reject_bare_scalar_addition():
    with pytest.raises(TypeError, match="dimensionless"):
        _vector_field() + 1


def test_quantity_addition_converts_compatible_units_without_losing_container():
    result = _vector_field() + (1000 * u.mV)

    assert isinstance(result, VectorField)
    assert result["x"].unit == u.V
    np.testing.assert_allclose(result["x"].value, 2.0)


def _snapshot_vector(vector: VectorField):
    return {
        "type": type(vector),
        "keys": list(vector),
        "basis": vector.basis,
        "units": {key: value.unit for key, value in vector.items()},
        "values": {key: value.value.copy() for key, value in vector.items()},
        "axes": {
            key: tuple(
                index.copy()
                for index in (
                    value._axis0_index,
                    value._axis1_index,
                    value._axis2_index,
                    value._axis3_index,
                )
            )
            for key, value in vector.items()
        },
        "names": {key: value.name for key, value in vector.items()},
    }


def _assert_vector_snapshot(vector: VectorField, snapshot) -> None:
    assert type(vector) is snapshot["type"]
    assert list(vector) == snapshot["keys"]
    assert vector.basis == snapshot["basis"]
    for key, value in vector.items():
        assert value.unit == snapshot["units"][key]
        assert value.name == snapshot["names"][key]
        np.testing.assert_array_equal(value.value, snapshot["values"][key])
        for actual, expected in zip(
            (
                value._axis0_index,
                value._axis1_index,
                value._axis2_index,
                value._axis3_index,
            ),
            snapshot["axes"][key],
        ):
            assert actual.unit == expected.unit
            np.testing.assert_array_equal(actual.value, expected.value)


def test_inplace_incompatible_quantity_is_atomic():
    vector = _vector_field()
    snapshot = _snapshot_vector(vector)

    with pytest.raises(u.UnitConversionError):
        vector += 1 * u.s

    _assert_vector_snapshot(vector, snapshot)


def test_inplace_quantity_scale_updates_the_existing_vectorfield():
    vector = _vector_field()
    alias = vector

    vector *= 2 * u.m

    assert vector is alias
    assert vector["x"].unit == u.V * u.m
    np.testing.assert_allclose(vector["x"].value, 2.0)


def test_scaled_component_axes_do_not_alias_the_source_metadata():
    source = _vector_field()
    result = 2 * source

    result["x"]._axis1_index[0] = 99 * u.m

    assert source["x"]._axis1_index[0] == 0 * u.m


def test_reflected_division_preserves_tensorfield_rank_and_key_order():
    vector = _vector_field()
    tensor = TensorField({(0, 0): vector["x"], (1, 1): vector["y"]}, rank=2)

    result = u.m / tensor

    assert isinstance(result, TensorField)
    assert result.rank == 2
    assert list(result) == [(0, 0), (1, 1)]
    assert result[(0, 0)].unit == u.m / u.V
