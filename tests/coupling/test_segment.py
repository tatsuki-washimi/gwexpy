from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.table import Table, vstack

from gwexpy.frequencyseries import FrequencySeries

REQUIRED_COLUMNS = {
    "start_gps_ns",
    "duration_ns",
    "source_channel",
    "response_channel",
    "frequency_hz",
    "coupling_factor",
    "coupling_factor_unit",
}


def _table(**overrides: object) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "start_gps_ns": [123],
        "duration_ns": [10],
        "source_channel": ["source"],
        "response_channel": ["response"],
        "frequency_hz": [10.0],
        "coupling_factor": [0.25],
        "coupling_factor_unit": ["m / V"],
    }
    values.update({name: [value] for name, value in overrides.items()})
    return pd.DataFrame(values)


def _result() -> SimpleNamespace:
    frequencies = np.array([10.0, 20.0, 30.0, 40.0]) * u.Hz
    return SimpleNamespace(
        cf=FrequencySeries(
            [0.1, np.nan, -1.0, np.nan], frequencies=frequencies, unit=u.m / u.V
        ),
        cf_ul=FrequencySeries(
            [0.2, 0.3, np.nan, np.inf], frequencies=frequencies, unit=u.m / u.V
        ),
        valid_mask=np.array([True, False, False, True]),
        witness_name="source",
        target_name="response",
        _significance_array=lambda: np.array([2.0, 3.0, np.nan, 5.0]),
    )


def _assert_nullable_confidence(
    series: pd.Series, expected: list[float | None]
) -> None:
    """Assert v1's nullable-binary64 pandas representation semantically."""
    assert str(series.dtype) == "Float64"
    assert len(series) == len(expected)
    for actual, wanted in zip(series.tolist(), expected, strict=True):
        if wanted is None:
            assert actual is pd.NA
        else:
            assert actual == wanted


def test_validate_accepts_v1_pandas_table_without_mutation() -> None:
    from gwexpy.coupling.segment import SCHEMA_NAME, validate

    table = _table()
    table.attrs["origin"] = {"run": "A"}
    before = table.copy(deep=True)

    assert SCHEMA_NAME == "gwexpy.coupling.segment.v1"
    assert validate(table) is table
    pd.testing.assert_frame_equal(table, before, check_dtype=True, check_exact=True)
    assert table.attrs == before.attrs


def test_validate_preserves_astropy_metadata_units_and_column_settings() -> None:
    from gwexpy.coupling.segment import validate

    table = Table({name: values for name, values in _table().items()})
    table.meta["origin"] = {"run": "A"}
    table["frequency_hz"].unit = u.Hz
    table["frequency_hz"].description = "injection frequency"
    table["frequency_hz"].format = ".2f"
    before = table.copy(copy_data=True)

    assert validate(table) is table
    assert table.meta == before.meta
    for name in table.colnames:
        assert table[name].unit == before[name].unit
        assert table[name].description == before[name].description
        assert table[name].format == before[name].format
        np.testing.assert_array_equal(table[name], before[name])


@pytest.mark.parametrize(
    ("table", "match"),
    [
        (_table().drop(columns=["frequency_hz"]), "required"),
        (_table(extra=[1]), "unknown"),
        (_table(start_gps_ns=True), "start_gps_ns"),
        (_table(duration_ns=0), "duration_ns"),
        (_table(start_gps_ns=2**63), "start_gps_ns"),
        (_table(start_gps_ns=2**63 - 1, duration_ns=1), "endpoint"),
        (_table(source_channel=""), "source_channel"),
        (_table(frequency_hz=np.nan), "frequency_hz"),
        (_table(coupling_factor=-0.1), "coupling_factor"),
        (_table(coupling_factor_unit=" "), "coupling_factor_unit"),
        (_table(coupling_factor_unit=u.m / u.V), "coupling_factor_unit"),
        (_table(estimate_kind="upper_limit"), "limit_method"),
        (_table(limit_method="threshold"), "limit_method"),
        (_table(significance=2.0), "unknown"),
    ],
)
def test_validate_fails_closed_for_invalid_rows(
    table: pd.DataFrame, match: str
) -> None:
    from gwexpy.coupling.segment import validate

    with pytest.raises((TypeError, ValueError), match=match):
        validate(table)


def test_validate_enforces_astropy_frequency_unit_and_does_not_mutate_on_failure() -> (
    None
):
    from gwexpy.coupling.segment import validate

    table = Table({name: values for name, values in _table().items()})
    table.meta["origin"] = "test"
    table["frequency_hz"].unit = u.kHz
    before = table.copy(copy_data=True)

    with pytest.raises(ValueError, match="frequency_hz.*Hz"):
        validate(table)

    assert table.meta == before.meta
    assert table["frequency_hz"].unit == before["frequency_hz"].unit
    np.testing.assert_array_equal(table["frequency_hz"], before["frequency_hz"])


def test_validate_upper_limit_optional_values_are_non_null_and_bounded() -> None:
    from gwexpy.coupling.segment import validate

    valid = _table(
        estimate_kind="upper_limit",
        limit_method="threshold",
        confidence_level=0.95,
    )
    assert validate(valid) is valid

    for confidence in [0.0, 1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="confidence_level"):
            validate(
                _table(
                    estimate_kind="upper_limit",
                    limit_method="threshold",
                    confidence_level=confidence,
                )
            )


def test_from_result_constructs_json_safe_valid_measurements_and_limits() -> None:
    from gwexpy.coupling.segment import from_result, validate

    table = from_result(
        _result(),
        start_gps_ns=123,
        duration_ns=10,
        limit_method="threshold",
        confidence_level=0.95,
    )

    assert set(REQUIRED_COLUMNS).issubset(table.columns)
    assert table["frequency_hz"].tolist() == [10.0, 20.0]
    assert table["estimate_kind"].tolist() == ["measurement", "upper_limit"]
    assert table["limit_method"].tolist() == [None, "threshold"]
    _assert_nullable_confidence(table["confidence_level"], [None, 0.95])
    assert "significance" not in table
    assert not any(
        isinstance(value, float) and np.isnan(value)
        for name in ("limit_method", "confidence_level")
        for value in table[name]
    )
    json.dumps(table.to_dict(orient="records"))
    assert validate(table) is table


def test_from_result_preserves_inputs_and_converts_equivalent_units() -> None:
    from gwexpy.coupling.segment import from_result

    result = _result()
    result.cf_ul = FrequencySeries(
        [20.0, 30.0, np.nan, np.inf],
        frequencies=np.array([0.01, 0.02, 0.03, 0.04]) * u.kHz,
        unit=u.cm / u.V,
    )
    cf_before = result.cf.copy()
    ul_before = result.cf_ul.copy()
    mask_before = result.valid_mask.copy()

    table = from_result(result, start_gps_ns=123, duration_ns=10, limit_method="x")

    assert table["frequency_hz"].tolist() == [10.0, 20.0]
    assert table["coupling_factor"].tolist() == [0.1, 0.3]
    assert table["coupling_factor_unit"].tolist() == ["m / V", "m / V"]
    np.testing.assert_array_equal(result.cf.value, cf_before.value)
    np.testing.assert_array_equal(result.cf.xindex, cf_before.xindex)
    np.testing.assert_array_equal(result.cf_ul.value, ul_before.value)
    np.testing.assert_array_equal(result.cf_ul.xindex, ul_before.xindex)
    np.testing.assert_array_equal(result.valid_mask, mask_before)


@pytest.mark.parametrize(
    "invalid_mask",
    [[True, "no", False, True], [True, None, False, True], np.array([1, 0, 1, 0])],
)
def test_from_result_fails_closed_for_non_boolean_valid_mask(
    invalid_mask: object,
) -> None:
    from gwexpy.coupling.segment import from_result

    result = _result()
    result.valid_mask = invalid_mask

    with pytest.raises(TypeError, match="valid_mask"):
        from_result(result, start_gps_ns=123, duration_ns=10)


def test_from_result_rejects_incompatible_upper_limit_grid_or_unit() -> None:
    from gwexpy.coupling.segment import from_result

    result = _result()
    result.cf_ul = FrequencySeries(
        [0.2, 0.3, np.nan, np.inf],
        frequencies=np.array([10.0, 20.0, 31.0, 40.0]) * u.Hz,
        unit=u.Hz,
    )

    with pytest.raises(ValueError, match="frequency grid"):
        from_result(result, start_gps_ns=123, duration_ns=10)

    result.cf_ul = FrequencySeries(
        [0.2, 0.3, np.nan, np.inf], frequencies=result.cf.xindex, unit=u.Hz
    )
    with pytest.raises(ValueError, match="coupling-factor unit"):
        from_result(result, start_gps_ns=123, duration_ns=10)


def test_from_result_empty_schema_round_trip_is_independent() -> None:
    from gwexpy.coupling.segment import from_result, validate

    result = _result()
    result.cf = FrequencySeries(
        [np.nan], frequencies=np.array([1.0]) * u.Hz, unit=u.dimensionless_unscaled
    )
    result.cf_ul = None
    result.valid_mask = np.array([False])

    table = from_result(result, start_gps_ns=0, duration_ns=1)
    clone = table.copy(deep=True)
    clone.loc[:, "source_channel"] = "changed"

    assert len(table) == 0
    assert table["coupling_factor_unit"].dtype == object
    assert table is not clone
    assert list(table["source_channel"]) == []
    assert validate(table) is table


def test_validate_uses_row_unit_string_as_the_single_unit_authority() -> None:
    from gwexpy.coupling.segment import validate

    table = Table({name: values for name, values in _table().items()})
    table["frequency_hz"].unit = u.Hz
    table["coupling_factor"].unit = u.m / u.V
    assert validate(table) is table

    round_tripped = table.to_pandas()
    assert validate(round_tripped) is round_tripped

    table["coupling_factor"].unit = u.Hz
    with pytest.raises(ValueError, match="coupling_factor_unit"):
        validate(table)


def test_validate_requires_canonical_or_unitless_astropy_time_units() -> None:
    from gwexpy.coupling.segment import validate

    table = Table({name: values for name, values in _table().items()})
    table.meta["origin"] = "time-units"
    before = table.copy(copy_data=True)

    assert validate(table) is table
    assert table.meta == before.meta

    table["start_gps_ns"].unit = u.ns
    table["duration_ns"].unit = u.ns
    assert validate(table) is table

    for name, unit, expected_unit in (
        ("start_gps_ns", u.s, "ns"),
        ("duration_ns", u.day, "ns"),
        ("start_gps_ns", u.Hz, "ns"),
        ("duration_ns", u.m, "ns"),
        ("frequency_hz", u.m, "Hz"),
    ):
        invalid = table.copy(copy_data=True)
        invalid[name].unit = unit
        invalid_before = invalid.copy(copy_data=True)

        with pytest.raises(ValueError, match=rf"{name}.*{expected_unit}"):
            validate(invalid)

        assert invalid.meta == invalid_before.meta
        assert invalid[name].unit == invalid_before[name].unit
        np.testing.assert_array_equal(invalid[name], invalid_before[name])


def test_schema_aware_pandas_astropy_adapters_preserve_absence_units_and_metadata() -> (
    None
):
    from gwexpy.coupling.segment import to_astropy, to_pandas, validate

    source = Table(
        {
            "start_gps_ns": [123, 123],
            "duration_ns": [10, 10],
            "source_channel": ["source", "source"],
            "response_channel": ["response", "response"],
            "frequency_hz": [10.0, 20.0],
            "coupling_factor": [0.25, 0.5],
            "coupling_factor_unit": ["m / V", "m / V"],
            "estimate_kind": ["measurement", "upper_limit"],
            "limit_method": ["", "threshold"],
            "confidence_level": [0.0, 0.95],
        },
        masked=True,
    )
    for name, unit in (
        ("start_gps_ns", u.ns),
        ("duration_ns", u.ns),
        ("frequency_hz", u.Hz),
        ("coupling_factor", u.m / u.V),
    ):
        source[name].unit = unit
    source["limit_method"].mask = [True, False]
    source["confidence_level"].mask = [True, False]
    source["frequency_hz"].description = "bin frequency"
    source["frequency_hz"].format = ".2f"
    source.meta["origin"] = {"run": "adapter"}
    source_before = source.copy(copy_data=True)

    pandas_table = to_pandas(source)
    assert pandas_table["limit_method"].tolist() == [None, "threshold"]
    _assert_nullable_confidence(pandas_table["confidence_level"], [None, 0.95])
    assert pandas_table.attrs["gwexpy.coupling.segment.v1.astropy_metadata"][
        "table_meta"
    ] == {"origin": {"run": "adapter"}}
    assert validate(pandas_table) is pandas_table
    assert source.meta == source_before.meta
    assert source["frequency_hz"].unit == source_before["frequency_hz"].unit
    np.testing.assert_array_equal(
        source["limit_method"].mask, source_before["limit_method"].mask
    )

    restored = to_astropy(pandas_table)
    assert restored.meta == source.meta
    assert restored["start_gps_ns"].unit == u.ns
    assert restored["duration_ns"].unit == u.ns
    assert restored["frequency_hz"].unit == u.Hz
    assert restored["coupling_factor"].unit == u.m / u.V
    assert restored["frequency_hz"].description == "bin frequency"
    assert restored["frequency_hz"].format == ".2f"
    assert restored["limit_method"].mask.tolist() == [True, False]
    assert restored["confidence_level"].mask.tolist() == [True, False]
    assert validate(restored) is restored

    native_pandas = source.to_pandas()
    with pytest.raises(ValueError, match="limit_method"):
        validate(native_pandas)

    native_astropy = Table.from_pandas(_table())
    assert native_astropy["frequency_hz"].unit is None
    assert validate(native_astropy) is native_astropy


@pytest.mark.parametrize("confidence_level", [None, 0.95])
def test_from_result_mixed_optionals_round_trip_through_public_adapters_and_json(
    confidence_level: float | None,
) -> None:
    from gwexpy.coupling.segment import (
        from_json_envelope,
        from_result,
        to_astropy,
        to_json_envelope,
        to_pandas,
        validate,
    )

    result = _result()
    cf_before = result.cf.copy()
    ul_before = result.cf_ul.copy()
    table = from_result(
        result,
        start_gps_ns=123,
        duration_ns=10,
        limit_method="threshold",
        confidence_level=confidence_level,
    )

    assert table["limit_method"].tolist() == [None, "threshold"]
    if confidence_level is None:
        assert "confidence_level" not in table
    else:
        _assert_nullable_confidence(table["confidence_level"], [None, 0.95])

    pandas_table = to_pandas(table)
    astropy_table = to_astropy(pandas_table)
    assert astropy_table["limit_method"].mask.tolist() == [True, False]
    if confidence_level is not None:
        assert astropy_table["confidence_level"].mask.tolist() == [True, False]
        assert np.issubdtype(astropy_table["confidence_level"].dtype, np.floating)
        assert astropy_table["confidence_level"][1] == 0.95

    restored = from_json_envelope(json.loads(json.dumps(to_json_envelope(table))))
    assert restored["limit_method"].tolist() == [None, "threshold"]
    if confidence_level is not None:
        _assert_nullable_confidence(restored["confidence_level"], [None, 0.95])
    assert validate(to_astropy(restored))
    np.testing.assert_array_equal(result.cf.value, cf_before.value)
    np.testing.assert_array_equal(result.cf_ul.value, ul_before.value)


@pytest.mark.parametrize("absence", [None, pd.NA, ""])
def test_public_adapters_canonicalize_allowed_measurement_optional_absence(
    absence: object,
) -> None:
    from gwexpy.coupling.segment import (
        to_astropy,
        to_json_envelope,
        to_pandas,
        validate,
    )

    table = pd.concat(
        [
            _table(
                estimate_kind="measurement",
                limit_method=absence,
                confidence_level=absence,
            ),
            _table(
                estimate_kind="upper_limit",
                limit_method="threshold",
                confidence_level=0.95,
            ),
        ],
        ignore_index=True,
    )
    assert validate(table) is table
    before = table.copy(deep=True)

    pandas_table = to_pandas(table)
    assert pandas_table["limit_method"].tolist() == [None, "threshold"]
    _assert_nullable_confidence(pandas_table["confidence_level"], [None, 0.95])
    astropy_table = to_astropy(table)
    assert astropy_table["limit_method"].mask.tolist() == [True, False]
    assert astropy_table["confidence_level"].mask.tolist() == [True, False]
    assert np.issubdtype(astropy_table["confidence_level"].dtype, np.floating)
    envelope = to_json_envelope(table)
    assert envelope["rows"][0][-1] is None
    pd.testing.assert_frame_equal(table, before)


def test_public_adapters_preserve_masked_optional_absence_as_numeric_mask() -> None:
    from gwexpy.coupling.segment import to_astropy, to_pandas

    source = Table(
        {
            "start_gps_ns": [123, 123],
            "duration_ns": [10, 10],
            "source_channel": ["source", "source"],
            "response_channel": ["response", "response"],
            "frequency_hz": [10.0, 20.0],
            "coupling_factor": [0.25, 0.5],
            "coupling_factor_unit": ["m / V", "m / V"],
            "estimate_kind": ["measurement", "upper_limit"],
            "limit_method": ["", "threshold"],
            "confidence_level": [0.0, 0.95],
        },
        masked=True,
    )
    source["limit_method"].mask = [True, False]
    source["confidence_level"].mask = [True, False]

    restored = to_astropy(to_pandas(source))
    assert restored["limit_method"].mask.tolist() == [True, False]
    assert restored["confidence_level"].mask.tolist() == [True, False]
    assert np.issubdtype(restored["confidence_level"].dtype, np.floating)


def test_json_round_trip_canonicalizes_legacy_measurement_optionals_without_estimate_kind() -> (
    None
):
    from gwexpy.coupling.segment import (
        from_json_envelope,
        to_json_envelope,
        validate,
    )

    table = _table(limit_method="", confidence_level="")
    envelope = to_json_envelope(table)
    rows = [
        dict(zip(envelope["columns"], row, strict=True)) for row in envelope["rows"]
    ]

    assert rows == [
        {**table.iloc[0].to_dict(), "limit_method": None, "confidence_level": None}
    ]
    restored = from_json_envelope(envelope)
    assert restored["limit_method"].tolist() == [None]
    _assert_nullable_confidence(restored["confidence_level"], [None])
    assert validate(restored) is restored


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (
            {
                "estimate_kind": "upper_limit",
                "limit_method": "",
                "confidence_level": 0.95,
            },
            "limit_method",
        ),
        (
            {
                "estimate_kind": "upper_limit",
                "limit_method": "threshold",
                "confidence_level": "",
            },
            "confidence_level",
        ),
        ({"source_channel": ""}, "source_channel"),
    ],
)
def test_validate_rejects_empty_strings_outside_allowed_measurement_optionals(
    values: dict[str, object], match: str
) -> None:
    from gwexpy.coupling.segment import validate

    with pytest.raises((TypeError, ValueError), match=match):
        validate(_table(**values))


def test_json_envelope_normalizes_supported_numeric_scalars_or_rejects_them() -> None:
    from gwexpy.coupling.segment import to_json_envelope, validate

    table = _table(
        start_gps_ns=np.int64(123),
        frequency_hz=np.longdouble("10.5"),
        coupling_factor=Fraction(1, 3),
        estimate_kind="upper_limit",
        limit_method="threshold",
        confidence_level=Fraction(19, 20),
    )

    assert validate(table) is table
    envelope = to_json_envelope(table)
    assert json.dumps(envelope)
    row = dict(zip(envelope["columns"], envelope["rows"][0], strict=True))
    assert isinstance(row["start_gps_ns"], int)
    assert isinstance(row["frequency_hz"], float)
    assert isinstance(row["coupling_factor"], float)
    assert isinstance(row["confidence_level"], float)

    with pytest.raises(TypeError, match="coupling_factor"):
        validate(_table(coupling_factor=Decimal("0.25")))


def _advance_binary64(value: float, steps: int, direction: float = np.inf) -> float:
    for _ in range(steps):
        value = float(np.nextafter(value, direction))
    return value


@pytest.mark.parametrize(("steps", "matches"), [(32, True), (33, False), (40, False)])
def test_frequency_grid_uses_true_32_ulp_tolerance_near_zero(
    steps: int, matches: bool
) -> None:
    from gwexpy.coupling.segment import _frequency_grids_match

    minimum = float(np.nextafter(0.0, np.inf))
    reference = np.array([0.0, minimum])
    candidate = reference.copy()
    candidate[0] = _advance_binary64(reference[0], steps)

    assert _frequency_grids_match(reference, candidate) is matches


@pytest.mark.parametrize("reference", [0.0, 1.0, 32.0])
@pytest.mark.parametrize("direction", [np.inf, -np.inf])
@pytest.mark.parametrize(("steps", "matches"), [(32, True), (33, False)])
def test_frequency_grid_limits_nextafter_steps_in_each_direction(
    reference: float, direction: float, steps: int, matches: bool
) -> None:
    from gwexpy.coupling.segment import _frequency_grids_match

    candidate = _advance_binary64(reference, steps, direction)

    assert (
        _frequency_grids_match(np.array([reference]), np.array([candidate])) is matches
    )


def test_schema_adapters_preserve_nested_column_metadata_independently() -> None:
    from gwexpy.coupling.segment import to_astropy, to_pandas

    source = Table({name: values for name, values in _table().items()}, masked=True)
    for name, unit in (
        ("start_gps_ns", u.ns),
        ("duration_ns", u.ns),
        ("frequency_hz", u.Hz),
        ("coupling_factor", u.m / u.V),
    ):
        source[name].unit = unit
    for name in source.colnames:
        source[name].meta["calibration"] = {
            "revision": 3,
            "coefficients": [1.0, {"offset": 0.25}],
        }
        source[name].meta["provenance"] = {"steps": ["inject", "estimate"]}
        source[name].description = f"{name} calibration data"
        source[name].format = (
            ".6g"
            if name
            in {"start_gps_ns", "duration_ns", "frequency_hz", "coupling_factor"}
            else "{:s}"
        )
    source.meta["provenance"] = {"run": {"id": "nested", "stages": [1, 2]}}

    pandas_table = to_pandas(source)
    restored = to_astropy(pandas_table)

    for name in source.colnames:
        assert restored[name].meta == source[name].meta
        assert restored[name].description == source[name].description
        assert restored[name].format == source[name].format
    assert restored.meta == source.meta

    carrier = pandas_table.attrs["gwexpy.coupling.segment.v1.astropy_metadata"]
    carrier["columns"]["frequency_hz"]["meta"]["calibration"]["coefficients"][1][
        "offset"
    ] = 99.0
    restored["duration_ns"].meta["provenance"]["steps"].append("restored")

    assert (
        source["frequency_hz"].meta["calibration"]["coefficients"][1]["offset"] == 0.25
    )
    assert (
        restored["frequency_hz"].meta["calibration"]["coefficients"][1]["offset"]
        == 0.25
    )
    assert source["duration_ns"].meta["provenance"]["steps"] == [
        "inject",
        "estimate",
    ]
    assert carrier["columns"]["duration_ns"]["meta"]["provenance"]["steps"] == [
        "inject",
        "estimate",
    ]


def test_schema_adapters_fail_closed_for_malformed_metadata_carriers() -> None:
    from gwexpy.coupling.segment import to_astropy, to_pandas

    table = _table()
    table.attrs["gwexpy.coupling.segment.v1.astropy_metadata"] = {
        "schema": "gwexpy.coupling.segment.v1",
        "table_meta": {},
        "columns": {},
    }

    for adapter in (to_pandas, to_astropy):
        with pytest.raises(ValueError, match="adapter metadata"):
            adapter(table)


def test_schema_adapters_reject_malformed_nested_metadata_carriers() -> None:
    from gwexpy.coupling.segment import to_astropy, to_pandas

    source = Table({name: values for name, values in _table().items()})
    pandas_table = to_pandas(source)
    carrier = pandas_table.attrs["gwexpy.coupling.segment.v1.astropy_metadata"]
    carrier["columns"]["frequency_hz"]["meta"] = []

    for adapter in (to_pandas, to_astropy):
        with pytest.raises(ValueError, match="adapter metadata"):
            adapter(pandas_table)


@pytest.mark.parametrize(
    "value",
    [
        np.nextafter(np.longdouble(0), np.longdouble(1)),
        -np.nextafter(np.longdouble(0), np.longdouble(1)),
        Fraction(1, 10**400),
        Fraction(10**400, 1),
        np.longdouble("1e4900"),
    ],
)
def test_validate_rejects_nonrepresentable_binary64_physical_values(
    value: object,
) -> None:
    from gwexpy.coupling.segment import validate

    values = {name: list(column) for name, column in _table().items()}
    values["coupling_factor"] = [value]
    table = Table(values)
    with pytest.raises((TypeError, ValueError), match="coupling_factor"):
        validate(table)


def test_validate_and_json_preserve_representable_binary64_subnormal_values() -> None:
    from gwexpy.coupling.segment import to_json_envelope, validate

    smallest = float(np.nextafter(0.0, np.inf))
    table = _table(frequency_hz=smallest, coupling_factor=smallest)

    assert validate(table) is table
    envelope = to_json_envelope(table)
    row = dict(zip(envelope["columns"], envelope["rows"][0], strict=True))
    assert row["frequency_hz"] == smallest
    assert row["coupling_factor"] == smallest
    assert json.dumps(envelope)


def test_from_result_rejects_a_duck_result_without_a_coupling_unit() -> None:
    from gwexpy.coupling.segment import from_result

    result = SimpleNamespace(
        cf=SimpleNamespace(value=np.array([1.0]), xindex=np.array([1.0]) * u.Hz),
        valid_mask=np.array([True]),
        witness_name="source",
        target_name="response",
    )

    with pytest.raises(TypeError, match="coupling-factor unit"):
        from_result(result, start_gps_ns=0, duration_ns=1)


def test_from_result_accepts_hz_khz_frequency_roundoff() -> None:
    from gwexpy.coupling.segment import from_result

    result = _result()
    result.cf_ul = FrequencySeries(
        [0.2, 0.3, np.nan, np.inf],
        frequencies=np.array([0.010000000000000002, 0.02, 0.03, 0.04]) * u.kHz,
        unit=u.m / u.V,
    )

    table = from_result(result, start_gps_ns=0, duration_ns=1, limit_method="x")

    assert table["frequency_hz"].tolist() == [10.0, 20.0]


def test_from_result_rejects_a_real_upper_limit_bin_mismatch() -> None:
    from gwexpy.coupling.segment import from_result

    result = _result()
    result.cf_ul = FrequencySeries(
        [0.2, 0.3, np.nan, np.inf],
        frequencies=np.array([0.0101, 0.02, 0.03, 0.04]) * u.kHz,
        unit=u.m / u.V,
    )

    with pytest.raises(ValueError, match="frequency grid"):
        from_result(result, start_gps_ns=0, duration_ns=1, limit_method="x")


def test_from_result_rejects_confidence_without_an_upper_limit_method() -> None:
    from gwexpy.coupling.segment import from_result

    with pytest.raises(ValueError, match="limit_method"):
        from_result(_result(), start_gps_ns=0, duration_ns=1, confidence_level=0.95)


def test_from_result_rejects_segment_endpoint_overflow() -> None:
    from gwexpy.coupling.segment import from_result

    with pytest.raises(ValueError, match="endpoint"):
        from_result(_result(), start_gps_ns=2**63 - 1, duration_ns=1)


@pytest.mark.parametrize(
    ("limit_method", "confidence_level", "optional_columns"),
    [
        ("threshold", 0.95, ["limit_method", "confidence_level"]),
        ("threshold", None, ["limit_method"]),
        (None, None, []),
    ],
)
def test_measurement_only_from_result_uses_the_requested_optional_shape(
    limit_method: str | None,
    confidence_level: float | None,
    optional_columns: list[str],
) -> None:
    from gwexpy.coupling.segment import (
        from_json_envelope,
        from_result,
        to_astropy,
        to_json_envelope,
        to_pandas,
        validate,
    )

    result = _result()
    result.cf_ul = None
    cf_before = result.cf.copy()
    mask_before = result.valid_mask.copy()
    table = from_result(
        result,
        start_gps_ns=0,
        duration_ns=1,
        limit_method=limit_method,
        confidence_level=confidence_level,
    )

    assert list(table.columns) == [
        *_table().columns,
        "estimate_kind",
        *optional_columns,
    ]
    assert table["estimate_kind"].tolist() == ["measurement"]
    for name in optional_columns:
        if name == "confidence_level":
            _assert_nullable_confidence(table[name], [None])
        else:
            assert table[name].tolist() == [None]

    pandas_table = to_pandas(table)
    astropy_table = to_astropy(pandas_table)
    assert list(astropy_table.colnames) == list(table.columns)
    for name in optional_columns:
        assert astropy_table[name].mask.tolist() == [True]
    restored = from_json_envelope(
        json.loads(json.dumps(to_json_envelope(astropy_table)))
    )
    assert list(restored.columns) == list(table.columns)
    for name in optional_columns:
        if name == "confidence_level":
            _assert_nullable_confidence(restored[name], [None])
        else:
            assert restored[name].tolist() == [None]
    assert validate(restored) is restored
    np.testing.assert_array_equal(result.cf.value, cf_before.value)
    np.testing.assert_array_equal(result.valid_mask, mask_before)


@pytest.mark.parametrize(
    ("limit_method", "confidence_level", "optional_columns"),
    [
        ("threshold", 0.95, ["limit_method", "confidence_level"]),
        ("threshold", None, ["limit_method"]),
        (None, None, []),
    ],
)
def test_empty_from_results_uses_the_requested_optional_shape(
    limit_method: str | None,
    confidence_level: float | None,
    optional_columns: list[str],
) -> None:
    from gwexpy.coupling.segment import (
        from_json_envelope,
        from_results,
        to_astropy,
        to_json_envelope,
    )

    table = from_results(
        {},
        start_gps_ns=0,
        duration_ns=1,
        limit_method=limit_method,
        confidence_level=confidence_level,
    )

    assert table.empty
    assert list(table.columns) == [
        *_table().columns,
        "estimate_kind",
        *optional_columns,
    ]
    assert list(to_astropy(table).colnames) == list(table.columns)
    restored = from_json_envelope(json.loads(json.dumps(to_json_envelope(table))))
    assert restored.empty
    assert list(restored.columns) == list(table.columns)


_EMPTY_SCHEMA_PANDAS_DTYPES = {
    "start_gps_ns": "int64",
    "duration_ns": "int64",
    "source_channel": "object",
    "response_channel": "object",
    "frequency_hz": "float64",
    "coupling_factor": "float64",
    "coupling_factor_unit": "object",
    "estimate_kind": "object",
    "limit_method": "object",
    "confidence_level": "Float64",
}


@pytest.mark.parametrize(
    ("limit_method", "confidence_level", "optional_columns"),
    [
        (None, None, []),
        ("method-with-an-arbitrarily-long-name", None, ["limit_method"]),
        (
            "method-with-an-arbitrarily-long-name",
            0.95,
            ["limit_method", "confidence_level"],
        ),
    ],
)
def test_empty_factory_schema_dtypes_survive_all_public_adapters(
    limit_method: str | None,
    confidence_level: float | None,
    optional_columns: list[str],
) -> None:
    from gwexpy.coupling.segment import (
        from_json_envelope,
        from_result,
        from_results,
        to_astropy,
        to_json_envelope,
        to_pandas,
        validate,
    )

    columns = [*_table().columns, "estimate_kind", *optional_columns]
    empty = from_results(
        {},
        start_gps_ns=0,
        duration_ns=1,
        limit_method=limit_method,
        confidence_level=confidence_level,
    )

    assert empty.empty
    assert list(empty.columns) == columns
    assert {name: str(dtype) for name, dtype in empty.dtypes.items()} == {
        name: _EMPTY_SCHEMA_PANDAS_DTYPES[name] for name in columns
    }

    astropy_empty = to_astropy(empty)
    assert list(astropy_empty.colnames) == columns
    for name in columns:
        expected = (
            np.dtype(np.int64)
            if name in {"start_gps_ns", "duration_ns"}
            else np.dtype(np.float64)
            if name in {"frequency_hz", "coupling_factor", "confidence_level"}
            else np.dtype(object)
        )
        assert astropy_empty[name].dtype == expected

    long_channel = "source-" + "x" * 300
    long_response = "response-" + "y" * 300
    long_unit = "m / V" + " " * 300
    row = {
        "start_gps_ns": 0,
        "duration_ns": 1,
        "source_channel": long_channel,
        "response_channel": long_response,
        "frequency_hz": 10.0,
        "coupling_factor": 0.25,
        "coupling_factor_unit": long_unit,
        "estimate_kind": "measurement" if not optional_columns else "upper_limit",
    }
    if "limit_method" in optional_columns:
        row["limit_method"] = limit_method
    if "confidence_level" in optional_columns:
        row["confidence_level"] = confidence_level
    astropy_empty.add_row(row)

    assert astropy_empty["source_channel"][0] == long_channel
    assert astropy_empty["response_channel"][0] == long_response
    assert astropy_empty["coupling_factor_unit"][0] == long_unit
    assert validate(astropy_empty) is astropy_empty

    pandas_round_trip = to_pandas(astropy_empty)
    assert {name: str(dtype) for name, dtype in pandas_round_trip.dtypes.items()} == {
        name: _EMPTY_SCHEMA_PANDAS_DTYPES[name] for name in columns
    }
    json_round_trip = from_json_envelope(
        json.loads(json.dumps(to_json_envelope(empty)))
    )
    assert list(json_round_trip.columns) == columns
    assert json_round_trip.empty
    assert {name: str(dtype) for name, dtype in json_round_trip.dtypes.items()} == {
        name: _EMPTY_SCHEMA_PANDAS_DTYPES[name] for name in columns
    }

    populated_result = _result()
    if not optional_columns:
        populated_result.cf_ul = None
    populated = from_result(
        populated_result,
        start_gps_ns=0,
        duration_ns=1,
        limit_method=limit_method,
        confidence_level=confidence_level,
    )
    concatenated = pd.concat([empty, populated], ignore_index=True)
    assert {name: str(dtype) for name, dtype in concatenated.dtypes.items()} == {
        name: _EMPTY_SCHEMA_PANDAS_DTYPES[name] for name in columns
    }
    stacked = vstack([to_astropy(empty), to_astropy(populated)])
    assert validate(stacked) is stacked
    for name in columns:
        assert stacked[name].dtype == to_astropy(empty)[name].dtype


def test_empty_adapter_dtype_contract_preserves_masks_and_metadata_carrier() -> None:
    from gwexpy.coupling.segment import from_results, to_astropy, to_pandas

    empty = from_results(
        {},
        start_gps_ns=0,
        duration_ns=1,
        limit_method="threshold",
        confidence_level=0.95,
    )
    astropy_empty = to_astropy(empty)
    astropy_empty.meta["provenance"] = {"calibration": ["empty"]}
    astropy_empty["limit_method"].meta["calibration"] = {"revision": 3}
    astropy_empty["confidence_level"].meta["calibration"] = {"revision": 4}

    pandas_empty = to_pandas(astropy_empty)
    restored = to_astropy(pandas_empty)

    assert restored.meta == astropy_empty.meta
    assert restored["limit_method"].meta == astropy_empty["limit_method"].meta
    assert restored["confidence_level"].meta == astropy_empty["confidence_level"].meta
    assert restored["limit_method"].mask.dtype == np.dtype(bool)
    assert restored["confidence_level"].mask.dtype == np.dtype(bool)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("start_gps_ns", 2**63, "start_gps_ns"),
        ("duration_ns", 1.0, "duration_ns"),
        ("frequency_hz", "10", "frequency_hz"),
        ("confidence_level", "0.95", "confidence_level"),
    ],
)
def test_empty_schema_adapter_fails_closed_for_incompatible_later_values(
    column: str, value: object, match: str
) -> None:
    from gwexpy.coupling.segment import from_results, to_astropy, validate

    table = from_results(
        {},
        start_gps_ns=0,
        duration_ns=1,
        limit_method="threshold",
        confidence_level=0.95,
    )
    row = {
        "start_gps_ns": 0,
        "duration_ns": 1,
        "source_channel": "source",
        "response_channel": "response",
        "frequency_hz": 10.0,
        "coupling_factor": 0.25,
        "coupling_factor_unit": "m / V",
        "estimate_kind": "upper_limit",
        "limit_method": "threshold",
        "confidence_level": 0.95,
    }
    row[column] = value
    with pytest.raises((TypeError, ValueError), match=match):
        to_astropy(pd.DataFrame([row], columns=table.columns))
    with pytest.raises((TypeError, ValueError), match=match):
        validate(pd.DataFrame([row], columns=table.columns))


def test_json_restoration_validates_before_schema_dtype_normalization() -> None:
    from gwexpy.coupling.segment import from_json_envelope, to_json_envelope

    envelope = to_json_envelope(
        _table(
            estimate_kind="upper_limit",
            limit_method="threshold",
            confidence_level=0.95,
        )
    )
    confidence_index = envelope["columns"].index("confidence_level")
    envelope["rows"][0][confidence_index] = "0.95"

    with pytest.raises(TypeError, match="confidence_level"):
        from_json_envelope(envelope)


def test_from_results_adapts_empty_and_multi_target_mappings() -> None:
    from gwexpy.coupling.segment import from_result, from_results, validate

    empty = from_results({}, start_gps_ns=0, duration_ns=1)
    assert empty.empty
    assert set(REQUIRED_COLUMNS).issubset(empty.columns)
    assert validate(empty) is empty

    combined = from_results(
        {"first": _result(), "second": _result()}, start_gps_ns=0, duration_ns=1
    )
    assert len(combined) == 2
    assert set(combined["response_channel"]) == {"response"}

    with pytest.raises(TypeError, match="mapping"):
        from_result({}, start_gps_ns=0, duration_ns=1)

    with pytest.raises(ValueError, match="limit_method"):
        from_results({}, start_gps_ns=0, duration_ns=1, confidence_level=0.95)

    with pytest.raises(ValueError, match="endpoint"):
        from_results({}, start_gps_ns=2**63 - 1, duration_ns=1)


def test_from_results_normalizes_heterogeneous_optional_columns_deterministically() -> (
    None
):
    from gwexpy.coupling.segment import (
        from_json_envelope,
        from_results,
        to_astropy,
        to_json_envelope,
        to_pandas,
        validate,
    )

    measurement_only = _result()
    measurement_only.target_name = "measurement"
    measurement_only.cf_ul = None
    upper_limits = _result()
    upper_limits.target_name = "upper"
    measurement_before = measurement_only.cf.copy()
    upper_before = upper_limits.cf.copy()
    upper_limit_before = upper_limits.cf_ul.copy()

    combined = from_results(
        {"upper": upper_limits, "measurement": measurement_only},
        start_gps_ns=0,
        duration_ns=1,
        limit_method="threshold",
        confidence_level=0.95,
    )
    reversed_combined = from_results(
        {"measurement": measurement_only, "upper": upper_limits},
        start_gps_ns=0,
        duration_ns=1,
        limit_method="threshold",
        confidence_level=0.95,
    )

    assert combined["response_channel"].tolist() == ["measurement", "upper", "upper"]
    assert combined["estimate_kind"].tolist() == [
        "measurement",
        "measurement",
        "upper_limit",
    ]
    assert combined["limit_method"].tolist() == [None, None, "threshold"]
    _assert_nullable_confidence(combined["confidence_level"], [None, None, 0.95])
    assert not any(
        isinstance(value, float) and np.isnan(value)
        for name in ("limit_method", "confidence_level")
        for value in combined[name]
    )
    assert validate(combined) is combined
    pd.testing.assert_frame_equal(combined, reversed_combined)
    pandas_combined = to_pandas(combined)
    astropy_combined = to_astropy(pandas_combined)
    assert astropy_combined["limit_method"].mask.tolist() == [True, True, False]
    assert astropy_combined["confidence_level"].mask.tolist() == [True, True, False]
    assert np.issubdtype(astropy_combined["confidence_level"].dtype, np.floating)
    restored = from_json_envelope(
        json.loads(json.dumps(to_json_envelope(astropy_combined)))
    )
    pd.testing.assert_frame_equal(restored, pandas_combined)
    np.testing.assert_array_equal(measurement_only.cf.value, measurement_before.value)
    np.testing.assert_array_equal(upper_limits.cf.value, upper_before.value)
    np.testing.assert_array_equal(upper_limits.cf_ul.value, upper_limit_before.value)


@pytest.mark.parametrize("absence", [None, pd.NA])
def test_validate_and_json_round_trip_preserve_null_measurement_optionals(
    absence: object,
) -> None:
    from gwexpy.coupling.segment import (
        from_json_envelope,
        to_json_envelope,
        validate,
    )

    pandas_table = pd.concat(
        [
            _table(
                estimate_kind="measurement",
                limit_method=absence,
                confidence_level=absence,
            ),
            _table(
                estimate_kind="upper_limit",
                limit_method="threshold",
                confidence_level=0.95,
            ),
        ],
        ignore_index=True,
    )
    pandas_table.attrs["origin"] = {"run": "mixed"}
    assert validate(pandas_table) is pandas_table
    assert pandas_table.attrs == {"origin": {"run": "mixed"}}

    astropy_table = Table(
        {name: values for name, values in pandas_table.items()}, masked=True
    )
    astropy_table.meta["origin"] = {"run": "mixed"}
    astropy_table["frequency_hz"].unit = u.Hz
    assert validate(astropy_table) is astropy_table
    assert astropy_table.meta == {"origin": {"run": "mixed"}}

    restored = from_json_envelope(
        json.loads(json.dumps(to_json_envelope(astropy_table)))
    )
    assert restored["limit_method"].tolist() == [None, "threshold"]
    _assert_nullable_confidence(restored["confidence_level"], [None, 0.95])
    assert validate(restored) is restored


@pytest.mark.parametrize("column", ["limit_method", "confidence_level"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_validate_fails_closed_for_non_null_optional_measurement_values(
    column: str,
    invalid: float,
) -> None:
    from gwexpy.coupling.segment import validate

    with pytest.raises((TypeError, ValueError), match=column):
        validate(_table(estimate_kind="measurement", **{column: invalid}))


def test_json_envelope_preserves_the_v1_empty_table_schema() -> None:
    from gwexpy.coupling.segment import (
        SCHEMA_NAME,
        from_json_envelope,
        from_result,
        to_json_envelope,
    )

    result = _result()
    result.cf = FrequencySeries(
        [np.nan], frequencies=np.array([1.0]) * u.Hz, unit=u.m / u.V
    )
    result.cf_ul = None
    result.valid_mask = np.array([False])
    empty = from_result(result, start_gps_ns=0, duration_ns=1)

    envelope = to_json_envelope(empty)
    assert envelope["schema"] == SCHEMA_NAME
    assert envelope["rows"] == []
    restored = from_json_envelope(json.loads(json.dumps(envelope)))
    assert restored.empty
    assert list(restored.columns) == list(empty.columns)


def test_json_envelope_rejects_unknown_envelope_fields() -> None:
    from gwexpy.coupling.segment import from_json_envelope, to_json_envelope

    envelope = to_json_envelope(_table())
    envelope["unexpected"] = True

    with pytest.raises(ValueError, match="only schema, columns, and rows"):
        from_json_envelope(envelope)
