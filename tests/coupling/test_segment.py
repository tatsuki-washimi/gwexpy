from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.table import Table

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
    assert table["limit_method"].tolist() == ["", "threshold"]
    assert table["confidence_level"].tolist() == ["", 0.95]
    assert "significance" not in table
    assert not table.isna().any().any()
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


def test_from_result_accepts_resolution_aware_frequency_roundoff() -> None:
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
    from gwexpy.coupling.segment import from_results, validate

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
    assert combined["confidence_level"].tolist() == [None, None, 0.95]
    assert not any(
        isinstance(value, float) and np.isnan(value)
        for name in ("limit_method", "confidence_level")
        for value in combined[name]
    )
    assert validate(combined) is combined
    pd.testing.assert_frame_equal(combined, reversed_combined)
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
    assert restored["confidence_level"].tolist() == [None, 0.95]
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
