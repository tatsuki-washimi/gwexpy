"""Contract coverage for ScalarField space transforms and collection wrappers."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwexpy.fields import FieldDict, FieldList, ScalarField, TensorField, VectorField


def _make_scalar_field(
    *,
    name: str | None = None,
    channel: str | None = None,
    epoch: float | None = 1234567890.0,
    offset: float = 0.0,
) -> ScalarField:
    data = np.arange(4 * 4 * 4 * 4, dtype=float).reshape(4, 4, 4, 4) + offset
    field = ScalarField(
        data,
        unit=u.V,
        name=name,
        channel=channel,
        epoch=epoch,
        axis0=np.arange(4) * 0.1 * u.s,
        axis1=np.arange(4) * 1.0 * u.m,
        axis2=np.arange(4) * 1.0 * u.m,
        axis3=np.arange(4) * 1.0 * u.m,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
        space_domain="real",
    )
    return field


def _channel_name(channel):
    return getattr(channel, "name", channel)


def _field_snapshot(field: ScalarField) -> dict[str, object]:
    return {
        "value": field.value.copy(),
        "unit": field.unit,
        "axis_names": field.axis_names,
        "axis0_domain": field.axis0_domain,
        "space_domains": field.space_domains,
        "axes": {name: field.axis(name).index.copy() for name in field.axis_names},
        "name": field.name,
        "channel": _channel_name(field.channel),
        "epoch": field.epoch,
    }


def _assert_quantity_matches(actual, expected):
    assert actual.unit == expected.unit
    assert_allclose(actual.value, expected.value)


def _assert_field_matches_snapshot(field: ScalarField, snapshot: dict[str, object]):
    assert_allclose(field.value, snapshot["value"])
    assert field.unit == snapshot["unit"]
    assert field.axis_names == snapshot["axis_names"]
    assert field.axis0_domain == snapshot["axis0_domain"]
    assert field.space_domains == snapshot["space_domains"]
    assert field.name == snapshot["name"]
    assert _channel_name(field.channel) == snapshot["channel"]
    assert field.epoch == snapshot["epoch"]
    for name, expected_axis in snapshot["axes"].items():
        _assert_quantity_matches(field.axis(name).index, expected_axis)


@pytest.mark.parametrize(
    ("axes", "expected_axis_names", "expected_space_domains"),
    [
        (["x"], ("t", "kx", "y", "z"), {"y": "real", "z": "real", "kx": "k"}),
        (["x", "z"], ("t", "kx", "y", "kz"), {"y": "real", "kx": "k", "kz": "k"}),
    ],
)
def test_scalarfield_fft_space_partial_axes_updates_axis_names_and_domains(
    axes, expected_axis_names, expected_space_domains
):
    field = _make_scalar_field()
    source_snapshot = _field_snapshot(field)

    result = field.fft_space(axes=axes)

    assert result.axis_names == expected_axis_names
    assert result.space_domains == expected_space_domains
    _assert_field_matches_snapshot(field, source_snapshot)


def test_scalarfield_fft_space_k_axis_units_and_wavelength_units():
    field = _make_scalar_field()
    result = field.fft_space(axes=["x"])

    expected_k = 2 * np.pi * np.fft.fftfreq(4, d=1.0)
    k_axis = result.axis("kx").index

    assert k_axis.unit == 1 / u.m
    assert_allclose(k_axis.value, expected_k)

    wavelength = result.wavelength("kx")
    assert wavelength.unit == u.m
    assert np.isinf(wavelength.value[0])
    assert_allclose(wavelength.value[1:], [4.0, 2.0, 4.0])


def test_scalarfield_ifft_space_round_trip_supported_axes():
    field = _make_scalar_field()
    source_snapshot = _field_snapshot(field)
    transformed = field.fft_space(axes=["x", "z"])
    transformed_snapshot = _field_snapshot(transformed)

    round_tripped = transformed.ifft_space(axes=["kx", "kz"])

    assert round_tripped.axis_names == field.axis_names
    assert round_tripped.space_domains == field.space_domains
    assert_allclose(round_tripped.value, field.value)
    _assert_field_matches_snapshot(field, source_snapshot)
    _assert_field_matches_snapshot(transformed, transformed_snapshot)


@pytest.mark.parametrize(
    ("transform_name", "transform_kwargs"),
    [
        ("fft_space", {"axes": ["x"]}),
        ("ifft_space", {"axes": ["kx"]}),
    ],
)
def test_scalarfield_space_transforms_drop_name_channel_and_keep_source_metadata(
    transform_name, transform_kwargs
):
    field = _make_scalar_field(name="MyField", channel="L1:TEST-CHANNEL")
    if transform_name == "ifft_space":
        field = field.fft_space(axes=["x"])
        field.name = "MyField"
        field.channel = "L1:TEST-CHANNEL"
    source_snapshot = _field_snapshot(field)

    result = getattr(field, transform_name)(**transform_kwargs)

    assert result.name is None
    assert result.channel is None
    _assert_field_matches_snapshot(field, source_snapshot)


def test_fieldlist_fft_space_all_preserves_order_and_axis_passthrough():
    left = _make_scalar_field(offset=0.0)
    right = _make_scalar_field(offset=1000.0)
    field_list = FieldList([left, right])
    source_snapshots = [_field_snapshot(field) for field in field_list]

    result = field_list.fft_space_all(axes=["x"])

    assert isinstance(result, FieldList)
    assert_allclose(result[0].value, left.fft_space(axes=["x"]).value)
    assert_allclose(result[1].value, right.fft_space(axes=["x"]).value)
    assert all(field.axis_names == ("t", "kx", "y", "z") for field in result)
    assert all(
        field.space_domains == {"y": "real", "z": "real", "kx": "k"} for field in result
    )
    for field, snapshot in zip(field_list, source_snapshots):
        _assert_field_matches_snapshot(field, snapshot)


def test_fieldlist_ifft_space_all_preserves_order_and_axis_passthrough():
    real_fields = [
        _make_scalar_field(offset=0.0),
        _make_scalar_field(offset=1000.0),
    ]
    field_list = FieldList([field.fft_space(axes=["x"]) for field in real_fields])
    source_snapshots = [_field_snapshot(field) for field in field_list]

    result = field_list.ifft_space_all(axes=["kx"])

    assert isinstance(result, FieldList)
    assert [field.axis_names for field in result] == [
        ("t", "x", "y", "z"),
        ("t", "x", "y", "z"),
    ]
    assert all(
        field.space_domains == {"y": "real", "z": "real", "x": "real"}
        for field in result
    )
    assert_allclose(result[0].value, real_fields[0].value)
    assert_allclose(result[1].value, real_fields[1].value)
    for field, snapshot in zip(field_list, source_snapshots):
        _assert_field_matches_snapshot(field, snapshot)


def test_fielddict_fft_space_all_preserves_keys_and_axis_passthrough():
    field_dict = FieldDict(
        {
            "Ex": _make_scalar_field(offset=0.0),
            "Ey": _make_scalar_field(offset=1000.0),
            "Ez": _make_scalar_field(offset=2000.0),
        }
    )
    source_snapshots = {
        key: _field_snapshot(field) for key, field in field_dict.items()
    }

    result = field_dict.fft_space_all(axes=["x"])

    assert isinstance(result, FieldDict)
    assert list(result.keys()) == ["Ex", "Ey", "Ez"]
    assert_allclose(result["Ex"].value, field_dict["Ex"].fft_space(axes=["x"]).value)
    assert_allclose(result["Ey"].value, field_dict["Ey"].fft_space(axes=["x"]).value)
    assert_allclose(result["Ez"].value, field_dict["Ez"].fft_space(axes=["x"]).value)
    assert all(field.axis_names == ("t", "kx", "y", "z") for field in result.values())
    assert all(
        field.space_domains == {"y": "real", "z": "real", "kx": "k"}
        for field in result.values()
    )
    for key, snapshot in source_snapshots.items():
        _assert_field_matches_snapshot(field_dict[key], snapshot)


def test_fielddict_ifft_space_all_preserves_keys_and_axis_passthrough():
    real_fields = {
        "Ex": _make_scalar_field(offset=0.0),
        "Ey": _make_scalar_field(offset=1000.0),
        "Ez": _make_scalar_field(offset=2000.0),
    }
    field_dict = FieldDict(
        {key: field.fft_space(axes=["x"]) for key, field in real_fields.items()}
    )
    source_snapshots = {
        key: _field_snapshot(field) for key, field in field_dict.items()
    }

    result = field_dict.ifft_space_all(axes=["kx"])

    assert isinstance(result, FieldDict)
    assert list(result.keys()) == ["Ex", "Ey", "Ez"]
    assert all(field.axis_names == ("t", "x", "y", "z") for field in result.values())
    assert all(
        field.space_domains == {"y": "real", "z": "real", "x": "real"}
        for field in result.values()
    )
    for key, field in result.items():
        assert_allclose(field.value, real_fields[key].value)
    for key, snapshot in source_snapshots.items():
        _assert_field_matches_snapshot(field_dict[key], snapshot)


def test_vectorfield_fft_space_all_resets_basis_to_cartesian():
    vector = VectorField(
        {
            "x": _make_scalar_field(offset=0.0),
            "y": _make_scalar_field(offset=1000.0),
        },
        basis="custom",
        validate=False,
    )
    source_snapshots = {key: _field_snapshot(field) for key, field in vector.items()}

    result = vector.fft_space_all(axes=["x"])

    assert isinstance(result, VectorField)
    assert vector.basis == "custom"
    assert result.basis == "cartesian"
    assert list(result.keys()) == ["x", "y"]
    assert result["x"].axis_names == ("t", "kx", "y", "z")
    for key, snapshot in source_snapshots.items():
        _assert_field_matches_snapshot(vector[key], snapshot)


def test_vectorfield_ifft_space_all_resets_basis_to_cartesian():
    real_components = {
        "x": _make_scalar_field(offset=0.0),
        "y": _make_scalar_field(offset=1000.0),
    }
    vector = VectorField(
        {key: field.fft_space(axes=["x"]) for key, field in real_components.items()},
        basis="custom",
        validate=False,
    )
    source_snapshots = {key: _field_snapshot(field) for key, field in vector.items()}

    result = vector.ifft_space_all(axes=["kx"])

    assert isinstance(result, VectorField)
    assert vector.basis == "custom"
    assert result.basis == "cartesian"
    assert list(result.keys()) == ["x", "y"]
    assert result["x"].axis_names == ("t", "x", "y", "z")
    assert result["x"].space_domains == {"y": "real", "z": "real", "x": "real"}
    for key, field in result.items():
        assert_allclose(field.value, real_components[key].value)
    for key, snapshot in source_snapshots.items():
        _assert_field_matches_snapshot(vector[key], snapshot)


def test_tensorfield_fft_space_all_reconstructs_with_inferred_tuple_key_rank():
    tensor = TensorField(
        {
            (0, 0): _make_scalar_field(offset=0.0),
            (0, 1): _make_scalar_field(offset=1000.0),
            (1, 0): _make_scalar_field(offset=2000.0),
            (1, 1): _make_scalar_field(offset=3000.0),
        },
        rank=4,
        validate=False,
    )
    source_snapshots = {key: _field_snapshot(field) for key, field in tensor.items()}

    result = tensor.fft_space_all(axes=["x"])

    assert isinstance(result, TensorField)
    assert tensor.rank == 4
    assert result.rank == 2
    assert list(result.keys()) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert result[(0, 0)].axis_names == ("t", "kx", "y", "z")
    for key, snapshot in source_snapshots.items():
        _assert_field_matches_snapshot(tensor[key], snapshot)


def test_tensorfield_ifft_space_all_reconstructs_with_inferred_tuple_key_rank():
    real_components = {
        (0, 0): _make_scalar_field(offset=0.0),
        (0, 1): _make_scalar_field(offset=1000.0),
        (1, 0): _make_scalar_field(offset=2000.0),
        (1, 1): _make_scalar_field(offset=3000.0),
    }
    tensor = TensorField(
        {key: field.fft_space(axes=["x"]) for key, field in real_components.items()},
        rank=4,
        validate=False,
    )
    source_snapshots = {key: _field_snapshot(field) for key, field in tensor.items()}

    result = tensor.ifft_space_all(axes=["kx"])

    assert isinstance(result, TensorField)
    assert tensor.rank == 4
    assert result.rank == 2
    assert list(result.keys()) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert result[(0, 0)].axis_names == ("t", "x", "y", "z")
    assert result[(0, 0)].space_domains == {
        "y": "real",
        "z": "real",
        "x": "real",
    }
    for key, field in result.items():
        assert_allclose(field.value, real_components[key].value)
    for key, snapshot in source_snapshots.items():
        _assert_field_matches_snapshot(tensor[key], snapshot)
