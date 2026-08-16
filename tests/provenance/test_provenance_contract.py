from __future__ import annotations

import copy
import json
import pickle

import astropy.units as u
import numpy as np
import pytest
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.spectrogram import Spectrogram as GwpySpectrogram
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.provenance import (
    build_provenance,
    dumps_json,
    loads_json,
    normalize_json,
)
from gwexpy.spectrogram import Spectrogram
from gwexpy.statistics.gauch import compute_gauch
from gwexpy.statistics.rayleigh_test import rayleigh_pvalue
from gwexpy.statistics.student_t_indicator import compute_student_t_nu
from gwexpy.timeseries import TimeSeries


def _assert_schema(provenance: dict, *, operation: bool = False) -> None:
    expected = {
        "schema",
        "version",
        "algorithm",
        "parameters",
        "rng",
        "software",
    }
    if operation:
        expected.add("inputs")
    assert set(provenance) == expected
    assert provenance["schema"] == "gwexpy.provenance"
    assert provenance["version"] == 1
    json.dumps(provenance, sort_keys=True, allow_nan=False)


def _analysis_ts() -> TimeSeries:
    return TimeSeries(np.random.default_rng(0).standard_normal(512), sample_rate=256)


def test_build_provenance_normalizes_json_values_and_rng_modes() -> None:
    provenance = build_provenance(
        "example",
        {"unit": u.m, "values": (np.int64(2), np.float64(3.5))},
        seed=7,
    )
    _assert_schema(provenance)
    assert provenance["parameters"] == {
        "unit": {"__gwexpy_type__": "astropy.unit", "value": "m"},
        "values": [2, 3.5],
    }
    assert provenance["rng"] == {
        "method": "seeded_generator",
        "bit_generator": "PCG64",
        "seed": 7,
    }

    legacy = build_provenance(
        "legacy",
        {},
    )
    assert legacy["rng"] == {
        "method": "legacy_global",
        "bit_generator": "MT19937",
        "seed": None,
    }

    external = build_provenance("external", {}, rng=np.random.default_rng(4))
    assert external["rng"] == {
        "method": "caller_managed",
        "bit_generator": "PCG64",
        "seed": None,
    }


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string key"},
        float("nan"),
        float("inf"),
        np.array([1, 2]),
        object(),
    ],
)
def test_build_provenance_rejects_non_json_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_provenance("invalid", {"value": value})


def test_build_provenance_rejects_cycles() -> None:
    value: list[object] = []
    value.append(value)
    with pytest.raises(ValueError, match="cycle"):
        build_provenance("cycle", {"value": value})


def test_json_helpers_are_deterministic_and_decode_units() -> None:
    payload = {"duration": 2 * u.s, "unit": u.m, "values": [np.int64(2), 3.5]}
    encoded = dumps_json(payload)
    assert encoded == (
        '{"duration":{"__gwexpy_type__":"astropy.quantity",'
        '"unit":{"__gwexpy_type__":"astropy.unit","value":"s"},'
        '"value":2.0},"unit":{"__gwexpy_type__":"astropy.unit","value":"m"},'
        '"values":[2,3.5]}'
    )
    decoded = loads_json(encoded)
    assert decoded == {"duration": 2 * u.s, "unit": u.m, "values": [2, 3.5]}
    assert decoded["duration"] == 2 * u.s
    assert decoded["unit"] == u.m


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (
            '{"__gwexpy_type__":"astropy.unit","value":"m"}',
            u.m,
        ),
        (
            '{"__gwexpy_type__":"astropy.quantity",'
            '"unit":{"__gwexpy_type__":"astropy.unit","value":"m"},'
            '"value":2.0}',
            2 * u.m,
        ),
    ],
)
def test_json_helpers_round_trip_exact_astropy_tagged_forms(
    encoded: str, expected: object
) -> None:
    assert dumps_json(loads_json(encoded)) == encoded
    assert loads_json(encoded) == expected


@pytest.mark.parametrize(
    ("encoded", "duplicate_key"),
    [
        (
            '{"__gwexpy_type__":"astropy.unit","value":"m","value":"s"}',
            "value",
        ),
        (
            '{"__gwexpy_type__":"astropy.quantity",'
            '"unit":{"__gwexpy_type__":"astropy.unit","value":"m"},'
            '"unit":{"__gwexpy_type__":"astropy.unit","value":"s"},'
            '"value":2.0}',
            "unit",
        ),
        (
            '{"__gwexpy_type__":"astropy.quantity",'
            '"unit":{"__gwexpy_type__":"astropy.unit","value":"m"},'
            '"value":2.0,"value":3.0}',
            "value",
        ),
        (
            '{"__gwexpy_type__":"astropy.unit",'
            '"__gwexpy_type__":"astropy.quantity","value":"m"}',
            "__gwexpy_type__",
        ),
        ('{"key":1,"key":2}', "key"),
        ('{"outer":{"key":1,"key":2}}', "key"),
    ],
)
def test_json_loads_rejects_duplicate_object_members(
    encoded: str, duplicate_key: str
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^duplicate JSON object member: {duplicate_key!r}$",
    ):
        loads_json(encoded)


@pytest.mark.parametrize(
    "encoded", ['{"value": NaN}', '{"value": Infinity}', '{"value": 1e999}']
)
def test_json_loads_rejects_non_finite_constants(encoded: str) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        loads_json(encoded)


def test_json_loads_rejects_malformed_unit_tags() -> None:
    with pytest.raises(ValueError, match="astropy.unit"):
        loads_json('{"__gwexpy_type__":"astropy.unit","value":1}')


@pytest.mark.parametrize(
    "encoded",
    [
        '{"__gwexpy_type__":null}',
        '{"outer":{"__gwexpy_type__":null}}',
        '{"outer":{"__gwexpy_type__":"unknown","value":"m"}}',
        '{"outer":{"__gwexpy_type__":"astropy.unit","value":"m","extra":true}}',
    ],
)
def test_json_loads_rejects_reserved_type_keys_unless_valid_tags(
    encoded: str,
) -> None:
    with pytest.raises(ValueError):
        loads_json(encoded)


@pytest.mark.parametrize(
    "invalid_tag",
    [
        {"__gwexpy_type__": None},
        {"__gwexpy_type__": 1},
        {"__gwexpy_type__": "unknown", "value": "m"},
        {"__gwexpy_type__": "astropy.unit"},
        {"__gwexpy_type__": "astropy.unit", "value": 1},
        {"__gwexpy_type__": "astropy.unit", "value": "not a unit"},
        {
            "__gwexpy_type__": "astropy.unit",
            "value": "m",
            "extra": True,
        },
        {"__gwexpy_type__": "astropy.quantity", "unit": {"value": "m"}},
        {
            "__gwexpy_type__": "astropy.quantity",
            "unit": {"__gwexpy_type__": "astropy.unit", "value": "m"},
        },
        {
            "__gwexpy_type__": "astropy.quantity",
            "unit": "m",
            "value": 2,
        },
        {
            "__gwexpy_type__": "astropy.quantity",
            "unit": {
                "__gwexpy_type__": "astropy.unit",
                "value": "not a unit",
            },
            "value": 2,
        },
        {
            "__gwexpy_type__": "astropy.quantity",
            "unit": {"__gwexpy_type__": "astropy.unit", "value": "m"},
            "value": [2],
        },
        {
            "__gwexpy_type__": "astropy.quantity",
            "unit": {"__gwexpy_type__": "astropy.unit", "value": "m"},
            "value": 2,
            "extra": True,
        },
    ],
)
def test_provenance_helpers_reject_reserved_tags_at_every_depth(
    invalid_tag: dict[str, object],
) -> None:
    for value in (invalid_tag, [invalid_tag], {"nested": invalid_tag}):
        with pytest.raises((TypeError, ValueError)):
            normalize_json(value)
        with pytest.raises((TypeError, ValueError)):
            dumps_json(value)
        with pytest.raises((TypeError, ValueError)):
            build_provenance("invalid-tag", {"value": value})


def test_normalization_canonicalizes_user_supplied_astropy_tags() -> None:
    value = {
        "unit": {"__gwexpy_type__": "astropy.unit", "value": "meter"},
        "quantity": {
            "__gwexpy_type__": "astropy.quantity",
            "unit": {"__gwexpy_type__": "astropy.unit", "value": "meter"},
            "value": 2,
        },
    }

    normalized = normalize_json(value)

    assert normalized == {
        "unit": {"__gwexpy_type__": "astropy.unit", "value": "m"},
        "quantity": {
            "__gwexpy_type__": "astropy.quantity",
            "unit": {"__gwexpy_type__": "astropy.unit", "value": "m"},
            "value": 2.0,
        },
    }
    assert dumps_json(loads_json(dumps_json(value))) == dumps_json(value)


def test_accepted_provenance_values_round_trip_through_strict_loader() -> None:
    values = [
        {"unit": u.m, "quantity": 2 * u.s},
        {
            "unit": {"__gwexpy_type__": "astropy.unit", "value": "meter"},
            "quantity": {
                "__gwexpy_type__": "astropy.quantity",
                "unit": {"__gwexpy_type__": "astropy.unit", "value": "s"},
                "value": 2,
            },
        },
    ]

    for value in values:
        encoded = dumps_json(value)
        assert dumps_json(loads_json(encoded)) == encoded

    built = build_provenance("round-trip", {"values": values})
    encoded = dumps_json(built)
    assert dumps_json(loads_json(encoded)) == encoded


def test_reserved_tag_cycle_is_rejected() -> None:
    value: dict[str, object] = {
        "__gwexpy_type__": "astropy.quantity",
        "unit": None,
        "value": 2,
    }
    value["unit"] = value

    with pytest.raises(ValueError, match="cycle"):
        normalize_json(value)


def test_statistical_outputs_have_structured_and_legacy_provenance() -> None:
    ts = _analysis_ts()
    gauch = compute_gauch(ts, fftlength=0.25, window=4, n_monte_carlo=12, seed=7)
    _assert_schema(gauch.provenance)
    assert type(gauch.provenance) is dict
    assert type(gauch.metadata) is dict
    assert gauch.metadata is gauch.provenance
    assert gauch.provenance["parameters"]["window"] == 4
    assert gauch.provenance["rng"]["seed"] == 7
    assert gauch.pvalue_map.provenance == gauch.provenance
    assert gauch.statistic_map.provenance == gauch.provenance
    assert gauch.pvalue_map.provenance is not gauch.provenance
    assert gauch.statistic_map.provenance is not gauch.provenance
    assert (
        gauch.pvalue_map.provenance["parameters"] is not gauch.provenance["parameters"]
    )
    assert gauch.n_monte_carlo == 12
    assert gauch.seed == 7
    assert gauch.fftlength == 0.25
    assert gauch.stride == 0.25
    assert "fftlength" not in gauch.metadata
    assert "stride" not in gauch.metadata
    assert "n_monte_carlo" not in gauch.provenance
    assert gauch.provenance["parameters"]["legacy"] == {
        "fftlength": 0.25,
        "n_monte_carlo": 12,
        "seed": 7,
        "stride": 0.25,
    }
    assert json.loads(json.dumps(gauch.provenance)) == gauch.provenance
    restored_gauch = pickle.loads(pickle.dumps(gauch))
    assert type(restored_gauch.provenance) is dict
    assert type(restored_gauch.metadata) is dict
    assert restored_gauch.metadata is restored_gauch.provenance
    assert restored_gauch.provenance == gauch.provenance
    assert restored_gauch.provenance is not gauch.provenance
    assert "n_monte_carlo" not in restored_gauch.provenance
    assert restored_gauch.fftlength == 0.25
    assert restored_gauch.stride == 0.25

    with pytest.warns(UserWarning, match="seed is ignored"):
        gauch_rng = compute_gauch(
            ts,
            fftlength=0.25,
            window=4,
            n_monte_carlo=12,
            rng=np.random.default_rng(7),
            seed=11,
        )
    assert gauch_rng.fftlength == 0.25
    assert gauch_rng.stride == 0.25
    assert gauch_rng.n_monte_carlo == 12
    assert gauch_rng.rng_provided is True
    assert gauch_rng.seed_unused is True
    assert gauch_rng.provenance["parameters"]["legacy"] == {
        "fftlength": 0.25,
        "n_monte_carlo": 12,
        "rng_provided": True,
        "seed_unused": True,
        "stride": 0.25,
    }

    rayleigh = rayleigh_pvalue(
        Spectrogram(np.ones((2, 3)), t0=0, dt=1, f0=1, df=1),
        n_samples=4,
        n_monte_carlo=12,
        seed=7,
    )
    _assert_schema(rayleigh.provenance)
    assert rayleigh.provenance["parameters"]["n_samples"] == 4
    assert rayleigh.n_monte_carlo == 12
    assert rayleigh.seed == 7

    student = compute_student_t_nu(ts, fftlength=0.25, window=2)
    _assert_schema(student.provenance)
    assert student.provenance["rng"] == {
        "method": "none",
        "bit_generator": None,
        "seed": None,
    }


def test_spectrogram_provenance_propagates_without_aliases() -> None:
    source = Spectrogram(np.ones((3, 2)), t0=0, dt=1, f0=1, df=1)
    source.provenance = build_provenance("source", {"nested": {"value": 1}})

    copied = source.copy()
    sliced = source[:2, :]
    unary = -source
    for result in (copied, sliced, unary):
        assert result.provenance == source.provenance
        assert result.provenance is not source.provenance
        assert result.provenance["parameters"] is not source.provenance["parameters"]


def test_spectrogram_dimension_reducing_slices_copy_provenance_without_type_changes() -> (
    None
):
    source = Spectrogram(np.arange(12.0).reshape(3, 4), t0=0, dt=1, f0=1, df=1)
    source.provenance = build_provenance(
        "source", {"nested": {"value": 1, "items": ["original"]}}
    )

    row = source[0, :]
    column = source[:, 0]
    normal = source[1:, 1:]
    scalar = source[0, 0]

    assert type(row) is GwpyFrequencySeries
    assert type(column) is GwpyTimeSeries
    assert isinstance(normal, Spectrogram)
    assert isinstance(scalar, u.Quantity)
    assert not hasattr(scalar, "provenance")

    for result in (row, column, normal):
        assert result.provenance == source.provenance
        assert result.provenance is not source.provenance
        assert result.provenance["parameters"] is not source.provenance["parameters"]

    row.provenance["parameters"]["nested"]["items"].append("row")
    column.provenance["parameters"]["nested"]["items"].append("column")
    normal.provenance["parameters"]["nested"]["items"].append("normal")
    normal.provenance["parameters"]["nested"]["value"] = 2

    assert source.provenance["parameters"] == {
        "nested": {"value": 1, "items": ["original"]}
    }
    assert row.provenance["parameters"]["nested"]["items"] == ["original", "row"]
    assert column.provenance["parameters"]["nested"]["items"] == [
        "original",
        "column",
    ]
    assert normal.provenance["parameters"]["nested"]["items"] == [
        "original",
        "normal",
    ]


@pytest.mark.parametrize("axis", [0, 1])
def test_spectrogram_reduction_1d_slices_copy_provenance_without_type_changes(
    axis: int,
) -> None:
    source = Spectrogram(np.arange(6.0).reshape(2, 3), t0=10, dt=2, f0=20, df=0.5)
    source.provenance = build_provenance("source", {"nested": {"items": ["original"]}})

    reduced = np.add.reduce(source, axis=axis)
    sliced = reduced[1:]
    scalar = reduced[0]

    assert type(sliced) is type(reduced)
    assert sliced.shape == (reduced.shape[0] - 1,)
    assert sliced.x0 == reduced.x0 + reduced.dx
    assert sliced.dx == reduced.dx
    assert sliced.provenance == reduced.provenance
    assert sliced.provenance is not reduced.provenance
    assert (
        sliced.provenance["inputs"]["left"] is not reduced.provenance["inputs"]["left"]
    )
    assert isinstance(scalar, u.Quantity)
    assert not hasattr(scalar, "provenance")

    sliced.provenance["inputs"]["left"]["parameters"]["nested"]["items"].append(
        "sliced"
    )
    assert reduced.provenance["inputs"]["left"]["parameters"]["nested"]["items"] == [
        "original"
    ]
    assert source.provenance["parameters"]["nested"]["items"] == ["original"]


def test_spectrogram_binary_provenance_is_deterministic_operation_tree() -> None:
    left = Spectrogram(np.ones((2, 2)), t0=0, dt=1, f0=1, df=1)
    right = Spectrogram(np.full((2, 2), 2.0), t0=0, dt=1, f0=1, df=1)
    left.provenance = build_provenance("left", {"nested": {"value": 1}})
    right.provenance = build_provenance("right", {"nested": {"value": 2}})

    result = left + right
    _assert_schema(result.provenance, operation=True)
    assert result.provenance["algorithm"] == "numpy.add"
    assert result.provenance["parameters"] == {}
    inputs = result.provenance["inputs"]
    assert inputs["left"] == left.provenance
    assert inputs["right"] == right.provenance
    assert inputs["left"] is not left.provenance
    assert inputs["right"] is not right.provenance

    scalar_result = left * 2
    assert scalar_result.provenance["inputs"]["left"] == left.provenance
    assert scalar_result.provenance["inputs"]["right"] is None

    plain = Spectrogram(np.ones((2, 2)), t0=0, dt=1, f0=1, df=1)
    assert not hasattr((plain + plain), "provenance")


def test_spectrogram_at_rejects_before_mutating_values_or_provenance() -> None:
    source = Spectrogram(np.arange(6.0).reshape(2, 3), t0=0, dt=1, f0=1, df=1)
    source.provenance = build_provenance("source", {"nested": {"value": 1}})
    values_before = source.value.copy()
    provenance_before = copy.deepcopy(source.provenance)

    with pytest.raises(TypeError, match="at"):
        np.add.at(source, (0, 1), 100)

    np.testing.assert_array_equal(source.value, values_before)
    assert source.provenance == provenance_before


@pytest.mark.parametrize(
    ("method", "expected_values", "expected_shape"),
    [
        ("reduce", np.array([3.0, 5.0, 7.0]), (3,)),
        ("accumulate", np.array([[0.0, 1.0, 2.0], [3.0, 5.0, 7.0]]), (2, 3)),
        ("reduceat", np.array([[3.0, 5.0, 7.0]]), (1, 3)),
    ],
)
def test_spectrogram_reduction_ufuncs_record_method_operations(
    method: str, expected_values: np.ndarray, expected_shape: tuple[int, ...]
) -> None:
    source = Spectrogram(np.arange(6.0).reshape(2, 3), t0=0, dt=1, f0=1, df=1)
    source.provenance = build_provenance("source", {"nested": {"value": 1}})

    if method == "reduce":
        result = np.add.reduce(source, axis=0)
    elif method == "accumulate":
        result = np.add.accumulate(source, axis=0)
    else:
        result = np.add.reduceat(source, [0], axis=0)

    assert type(result) is type(source)
    assert result.shape == expected_shape
    np.testing.assert_array_equal(result.value, expected_values)
    assert result.provenance["algorithm"] == f"numpy.add.{method}"
    assert result.provenance["inputs"]["left"] == source.provenance
    assert result.provenance["inputs"]["left"] is not source.provenance
    assert (
        result.provenance["inputs"]["left"]["parameters"]
        is not source.provenance["parameters"]
    )


def test_spectrogram_outer_records_independent_operand_snapshots() -> None:
    left = Spectrogram(np.ones((2, 2)), t0=0, dt=1, f0=1, df=1)
    right = Spectrogram(np.full((2, 2), 2.0), t0=0, dt=1, f0=1, df=1)
    left.provenance = build_provenance("left", {"nested": {"value": 1}})
    right.provenance = build_provenance("right", {"nested": {"value": 2}})

    result = np.add.outer(left, right)

    assert type(result) is type(left)
    assert result.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(result.value, 3.0)
    assert result.provenance["algorithm"] == "numpy.add.outer"
    inputs = result.provenance["inputs"]
    assert inputs["left"] == left.provenance
    assert inputs["right"] == right.provenance
    assert inputs["left"] is not inputs["right"]
    assert inputs["left"]["parameters"] is not left.provenance["parameters"]
    assert inputs["right"]["parameters"] is not right.provenance["parameters"]


def test_spectrogram_ufunc_out_uses_predelegation_snapshot_and_independent_tree() -> (
    None
):
    source = Spectrogram(np.ones((2, 2)), t0=0, dt=1, f0=1, df=1)
    right = Spectrogram(np.full((2, 2), 2.0), t0=0, dt=1, f0=1, df=1)
    source.provenance = build_provenance("source", {"nested": {"value": 1}})
    right.provenance = build_provenance("right", {"nested": {"value": 2}})
    source_provenance_before = copy.deepcopy(source.provenance)

    returned = np.add(source, right, out=(source,))

    assert returned is source
    np.testing.assert_array_equal(source.value, 3.0)
    assert source.provenance["algorithm"] == "numpy.add"
    inputs = source.provenance["inputs"]
    assert inputs["left"] == source_provenance_before
    assert inputs["right"] == right.provenance
    assert inputs["left"] is not source.provenance
    assert inputs["right"] is not right.provenance
    assert inputs["left"]["parameters"] is not source.provenance["parameters"]
    assert inputs["right"]["parameters"] is not source.provenance["parameters"]


def _assert_spectrogram_ufunc_output(
    output: Spectrogram,
    expected: np.ndarray,
    source: Spectrogram,
) -> None:
    assert type(output) is type(source)
    np.testing.assert_array_equal(output.value, expected)
    assert output.unit == u.dimensionless_unscaled
    assert output.name == source.name
    assert output.channel == source.channel
    np.testing.assert_array_equal(output.xindex, source.xindex)
    np.testing.assert_array_equal(output.yindex, source.yindex)
    assert output.xindex is not source.xindex
    assert output.yindex is not source.yindex


@pytest.mark.parametrize(
    ("function", "expected_function"),
    [(np.modf, np.modf), (np.frexp, np.frexp)],
)
def test_spectrogram_multi_output_ufuncs_preserve_typed_outputs_and_provenance(
    function, expected_function
) -> None:
    source = Spectrogram(
        np.array([[1.5, -2.25], [3.0, 4.5]]),
        t0=10,
        dt=2,
        f0=20,
        df=0.5,
        name="multi-output",
        channel="H1:TEST",
    )
    source.provenance = build_provenance(
        "source", {"nested": {"items": ["original"], "value": 1}}
    )

    result = function(source)
    expected = expected_function(source.value)

    assert type(result) is tuple
    assert len(result) == 2
    for item, expected_item in zip(result, expected):
        _assert_spectrogram_ufunc_output(item, expected_item, source)
        assert item.provenance == source.provenance
        assert item.provenance is not source.provenance
        assert item.provenance["parameters"] is not source.provenance["parameters"]
    assert result[0].provenance is not result[1].provenance
    assert result[0].provenance["parameters"] is not result[1].provenance["parameters"]

    result[0].provenance["parameters"]["nested"]["items"].append("output-0")
    assert source.provenance["parameters"]["nested"]["items"] == ["original"]
    assert result[1].provenance["parameters"]["nested"]["items"] == ["original"]


@pytest.mark.parametrize("function", [np.modf, np.frexp])
def test_spectrogram_multi_output_unit_failure_precedes_out_mutation(function) -> None:
    source = Spectrogram(
        np.array([[1.5, -2.25], [3.0, 4.5]]),
        t0=10,
        dt=2,
        f0=20,
        df=0.5,
        unit=u.m,
    )
    outputs = (source.copy(), source.copy())
    source_values = source.value.copy()
    output_values = tuple(output.value.copy() for output in outputs)

    with pytest.raises(u.UnitTypeError):
        function(source, out=outputs)

    np.testing.assert_array_equal(source.value, source_values)
    for output, expected in zip(outputs, output_values):
        np.testing.assert_array_equal(output.value, expected)


@pytest.mark.parametrize("function", [np.modf, np.frexp])
def test_spectrogram_multi_output_ufunc_rejects_partial_where(function) -> None:
    source = Spectrogram(np.array([[1.5, -2.25], [3.0, 4.5]]), t0=0, dt=1, f0=1, df=1)

    with pytest.raises(TypeError, match="where"):
        function(source, where=np.array([[True, False], [True, True]]))


@pytest.mark.parametrize("function", [np.modf, np.frexp])
def test_spectrogram_multi_output_out_updates_each_output_independently(
    function,
) -> None:
    source = Spectrogram(np.array([[1.5, -2.25], [3.0, 4.5]]), t0=0, dt=1, f0=1, df=1)
    source.provenance = build_provenance("source", {"nested": {"value": 1}})
    outputs = (source.copy(), source.copy())

    returned = function(source, out=outputs)

    assert type(returned) is tuple
    assert returned[0] is outputs[0]
    assert returned[1] is outputs[1]
    expected = function(source.value)
    for item, expected_item in zip(returned, expected):
        _assert_spectrogram_ufunc_output(item, expected_item, source)
        assert item.provenance == source.provenance
        assert item.provenance is not source.provenance
    assert returned[0].provenance is not returned[1].provenance
    assert (
        returned[0].provenance["parameters"] is not returned[1].provenance["parameters"]
    )


def test_spectrogram_pickle_allowlists_provenance_and_copies_it() -> None:
    source = Spectrogram(np.ones((2, 2)), t0=0, dt=1, f0=1, df=1)
    source.provenance = build_provenance("source", {"value": {"nested": True}})
    source.arbitrary_attribute = "drop me"

    restored = pickle.loads(pickle.dumps(source))
    assert isinstance(restored, GwpySpectrogram)
    assert restored.provenance == source.provenance
    assert restored.provenance is not source.provenance
    assert restored.provenance["parameters"] is not source.provenance["parameters"]
    assert not hasattr(restored, "arbitrary_attribute")


def _assert_spectrogram_copy_contract(source: Spectrogram, copied: Spectrogram) -> None:
    assert type(copied) is type(source)
    np.testing.assert_array_equal(copied.value, source.value)
    assert not np.shares_memory(copied.value, source.value)
    assert copied.unit == source.unit
    assert copied.name == source.name
    assert copied.channel == source.channel
    np.testing.assert_array_equal(copied.xindex, source.xindex)
    np.testing.assert_array_equal(copied.yindex, source.yindex)
    assert not np.shares_memory(copied.xindex.value, source.xindex.value)
    assert not np.shares_memory(copied.yindex.value, source.yindex.value)
    _assert_schema(copied.provenance)
    assert copied.provenance == source.provenance
    assert copied.provenance is not source.provenance
    assert copied.provenance["parameters"] is not source.provenance["parameters"]
    assert (
        copied.provenance["parameters"]["nested"]
        is not source.provenance["parameters"]["nested"]
    )


def _spectrogram_with_provenance() -> Spectrogram:
    source = Spectrogram(
        np.arange(6.0).reshape(2, 3),
        t0=1,
        dt=2,
        f0=10,
        df=0.5,
        unit="strain",
        name="copy-contract",
        channel="H1:TEST",
    )
    source.provenance = build_provenance("source", {"nested": {"value": 1}})
    return source


def test_stdlib_copy_copy_delegates_to_spectrogram_copy_contract() -> None:
    source = _spectrogram_with_provenance()

    copied = copy.copy(source)

    _assert_spectrogram_copy_contract(source, copied)
    copied.value[0, 0] = 99
    copied.provenance["parameters"]["nested"]["value"] = 2
    assert source.value[0, 0] == 0
    assert source.provenance["parameters"]["nested"]["value"] == 1


def test_stdlib_copy_deepcopy_preserves_contract_and_memoizes() -> None:
    source = _spectrogram_with_provenance()
    memo: dict[int, object] = {}

    copied = copy.deepcopy(source, memo)

    _assert_spectrogram_copy_contract(source, copied)
    assert memo[id(source)] is copied
    shared = copy.deepcopy([source, source])
    assert shared[0] is shared[1]
