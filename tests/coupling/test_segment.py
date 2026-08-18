from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.table import Table

from gwexpy.coupling.segment import SCHEMA_NAME, from_result, validate
from gwexpy.frequencyseries import FrequencySeries

REQUIRED = [
    "start_gps_ns",
    "duration_ns",
    "source_channel",
    "response_channel",
    "frequency_hz",
    "coupling_factor",
    "coupling_factor_unit",
]


def _table(**overrides: object) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "start_gps_ns": [123],
        "duration_ns": [10],
        "source_channel": ["WIT"],
        "response_channel": ["TGT"],
        "frequency_hz": [10.0],
        "coupling_factor": [0.25],
        "coupling_factor_unit": ["m/V"],
    }
    values.update({key: [value] for key, value in overrides.items()})
    return pd.DataFrame(values)


def test_validate_returns_original_pandas_table_without_mutation() -> None:
    table = _table()
    table.attrs["metadata"] = {"source": "test"}
    before = table.copy(deep=True)

    assert SCHEMA_NAME == "gwexpy.coupling.segment.v1"
    assert validate(table) is table
    pd.testing.assert_frame_equal(table, before, check_dtype=True, check_exact=True)
    assert set(table.columns) == set(REQUIRED)


def test_validate_accepts_astropy_table_without_mutation() -> None:
    table = Table({key: values for key, values in _table().items()})
    table.meta["metadata"] = {"source": "test"}
    table["frequency_hz"].unit = u.Hz
    table["frequency_hz"].description = "frequency"
    table["frequency_hz"].format = ".2f"
    before = table.copy(copy_data=True)

    assert validate(table) is table
    assert table.colnames == before.colnames
    assert table.meta == before.meta
    for name in table.colnames:
        assert table[name].dtype == before[name].dtype
        assert table[name].unit == before[name].unit
        assert table[name].description == before[name].description
        assert table[name].format == before[name].format
        np.testing.assert_array_equal(table[name], before[name])


def test_validate_requires_astropy_frequency_column_to_be_in_hz() -> None:
    table = Table({key: values for key, values in _table().items()})
    table["frequency_hz"].unit = u.kHz

    with pytest.raises(ValueError, match="frequency_hz.*Hz"):
        validate(table)


def test_validate_rejects_astropy_unit_objects_in_unit_cells() -> None:
    table = _table(coupling_factor_unit=u.m / u.V)

    with pytest.raises(TypeError, match="coupling_factor_unit"):
        validate(table)


@pytest.mark.parametrize("table_kind", ["pandas", "astropy"])
def test_validate_failure_does_not_mutate_after_unit_parsing(table_kind: str) -> None:
    pandas_table = _table(estimate_kind="not-an-estimate")
    pandas_table.attrs["metadata"] = {"source": "test"}
    table: pd.DataFrame | Table
    if table_kind == "pandas":
        table = pandas_table
    else:
        table = Table({key: values for key, values in pandas_table.items()})
    before: pd.DataFrame | Table
    if isinstance(table, Table):
        table.meta["metadata"] = {"source": "test"}
        table["frequency_hz"].unit = u.Hz
        before = table.copy(copy_data=True)
    else:
        before = table.copy(deep=True)

    with pytest.raises(ValueError, match="estimate_kind"):
        validate(table)

    if isinstance(table, pd.DataFrame):
        pd.testing.assert_frame_equal(table, before, check_dtype=True, check_exact=True)
    else:
        assert table.colnames == before.colnames
        assert table.meta == before.meta
        for name in table.colnames:
            assert table[name].dtype == before[name].dtype
            np.testing.assert_array_equal(table[name], before[name])


def test_validate_rejects_unknown_and_missing_columns() -> None:
    with pytest.raises(ValueError, match="unknown"):
        validate(_table(extra=[1]))

    missing = _table().drop(columns=["frequency_hz"])
    with pytest.raises(ValueError, match="required"):
        validate(missing)


@pytest.mark.parametrize(
    "column,value",
    [
        ("start_gps_ns", True),
        ("duration_ns", np.bool_(True)),
        ("start_gps_ns", 1.5),
        ("duration_ns", "10"),
    ],
)
def test_validate_rejects_non_signed_integer_time_fields(
    column: str, value: object
) -> None:
    with pytest.raises(TypeError):
        validate(_table(**{column: value}))


@pytest.mark.parametrize(
    "column,value",
    [
        ("start_gps_ns", -1),
        ("duration_ns", 0),
        ("start_gps_ns", 2**63),
        ("duration_ns", 2**63),
    ],
)
def test_validate_rejects_invalid_or_overflowed_time_fields(
    column: str, value: int
) -> None:
    with pytest.raises(ValueError):
        validate(_table(**{column: value}))


def test_validate_rejects_nulls_empty_strings_and_bad_numeric_values() -> None:
    for column, value in [
        ("start_gps_ns", None),
        ("source_channel", ""),
        ("coupling_factor_unit", " "),
        ("frequency_hz", np.nan),
        ("coupling_factor", -0.1),
        ("coupling_factor", np.inf),
    ]:
        with pytest.raises((TypeError, ValueError)):
            validate(_table(**{column: value}))


def test_validate_enforces_estimate_kind_and_limit_contract() -> None:
    with pytest.raises(ValueError, match="estimate_kind"):
        validate(_table(estimate_kind="not-an-estimate"))

    with pytest.raises(ValueError, match="limit_method"):
        validate(_table(estimate_kind="upper_limit"))

    with pytest.raises(ValueError, match="limit_method"):
        validate(_table(limit_method="threshold"))


@pytest.mark.parametrize("confidence", [0.0, 1.0, np.nan, np.inf])
def test_validate_confidence_level_is_upper_limit_only(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        validate(
            _table(
                estimate_kind="upper_limit",
                limit_method="threshold",
                confidence_level=confidence,
            )
        )

    with pytest.raises(ValueError, match="confidence_level"):
        validate(_table(confidence_level=confidence))


def test_validate_requires_finite_dimensionless_significance() -> None:
    with pytest.raises(ValueError, match="significance"):
        validate(_table(significance=np.inf))

    with pytest.raises(ValueError, match="significance"):
        validate(_table(significance=1.0 * u.Hz))


def _result() -> SimpleNamespace:
    freqs = np.array([10.0, 20.0, 30.0, 40.0])
    cf = FrequencySeries(
        [0.1, np.nan, -1.0, np.nan],
        frequencies=freqs * u.Hz,
        unit=u.m / u.V,
        name="CF",
    )
    cf_ul = FrequencySeries(
        [0.2, 0.3, np.nan, np.inf],
        frequencies=freqs * u.Hz,
        unit=u.m / u.V,
        name="CF UL",
    )
    return SimpleNamespace(
        cf=cf,
        cf_ul=cf_ul,
        valid_mask=np.array([True, False, False, True]),
        witness_name="WIT",
        target_name="TGT",
        _significance_array=lambda: np.array([2.0, 3.0, np.nan, 5.0]),
    )


def test_from_result_emits_measurements_and_finite_upper_limits_without_nulls() -> None:
    table = from_result(
        _result(),
        start_gps_ns=123,
        duration_ns=10,
        limit_method="threshold",
        confidence_level=0.95,
    )

    assert list(table["frequency_hz"]) == [10.0, 20.0]
    assert list(table["estimate_kind"]) == ["measurement", "upper_limit"]
    assert table["coupling_factor"].tolist() == [0.1, 0.3]
    assert table["limit_method"].tolist() == ["", "threshold"]
    assert table["confidence_level"].tolist() == ["", 0.95]
    assert table["significance"].tolist() == [2.0, 3.0]
    assert not table.isna().any().any()
    validate(table)


def test_from_result_normalizes_frequency_axis_to_numeric_hz() -> None:
    result = _result()
    result.cf = FrequencySeries(
        [0.1, np.nan, -1.0, np.nan],
        frequencies=np.array([1.0, 2.0, 3.0, 4.0]) * u.kHz,
        unit=u.m / u.V,
    )
    result.cf_ul = None

    table = from_result(result, start_gps_ns=123, duration_ns=10)

    assert table["frequency_hz"].tolist() == [1000.0]


def test_from_result_converts_frequency_axis_to_hz() -> None:
    result = SimpleNamespace(
        cf=FrequencySeries(
            [0.1, 0.2],
            frequencies=np.array([1.0, 2.0]) * u.kHz,
            unit=u.m / u.V,
        ),
        valid_mask=np.array([True, True]),
        witness_name="WIT",
        target_name="TGT",
    )

    table = from_result(result, start_gps_ns=123, duration_ns=10)

    assert table["frequency_hz"].tolist() == [1000.0, 2000.0]


@pytest.mark.parametrize(
    "valid_mask",
    [
        [True, "no", False, True],
        [True, np.nan, False, True],
        [True, None, False, True],
        np.array([1, 0, 1, 0]),
    ],
)
def test_from_result_rejects_non_boolean_masks(valid_mask: object) -> None:
    result = _result()
    result.valid_mask = valid_mask

    with pytest.raises(TypeError, match="valid_mask"):
        from_result(result, start_gps_ns=123, duration_ns=10)


def test_from_result_rejects_mismatched_upper_limit_frequency_grid() -> None:
    result = _result()
    result.cf_ul = FrequencySeries(
        [0.2, 0.3, np.nan, np.inf],
        frequencies=np.array([10.0, 20.0, 31.0, 40.0]) * u.Hz,
        unit=u.m / u.V,
    )

    with pytest.raises(ValueError, match="frequency grid"):
        from_result(result, start_gps_ns=123, duration_ns=10)


def test_from_result_accepts_equivalent_upper_limit_frequency_units() -> None:
    result = _result()
    result.cf_ul = FrequencySeries(
        [0.2, 0.3, np.nan, np.inf],
        frequencies=np.array([0.01, 0.02, 0.03, 0.04]) * u.kHz,
        unit=u.m / u.V,
    )

    table = from_result(
        result, start_gps_ns=123, duration_ns=10, limit_method="threshold"
    )

    assert table["frequency_hz"].tolist() == [10.0, 20.0]


def test_from_result_converts_equivalent_upper_limit_coupling_units() -> None:
    result = _result()
    result.cf_ul = FrequencySeries(
        [20.0, 30.0, np.nan, np.inf],
        frequencies=result.cf.xindex,
        unit=u.cm / u.V,
    )

    table = from_result(
        result, start_gps_ns=123, duration_ns=10, limit_method="threshold"
    )

    assert table["coupling_factor"].tolist() == [0.1, 0.3]
    assert table["coupling_factor_unit"].tolist() == ["m / V", "m / V"]


def test_from_result_does_not_mutate_result_inputs() -> None:
    result = _result()
    cf_before = result.cf.copy()
    cf_ul_before = result.cf_ul.copy()
    mask_before = result.valid_mask.copy()

    from_result(result, start_gps_ns=123, duration_ns=10, limit_method="threshold")

    np.testing.assert_array_equal(result.cf.value, cf_before.value)
    np.testing.assert_array_equal(result.cf.xindex, cf_before.xindex)
    np.testing.assert_array_equal(result.cf_ul.value, cf_ul_before.value)
    np.testing.assert_array_equal(result.cf_ul.xindex, cf_ul_before.xindex)
    np.testing.assert_array_equal(result.valid_mask, mask_before)
    assert result.cf.unit == cf_before.unit
    assert result.cf_ul.unit == cf_ul_before.unit


def test_from_result_rejects_incompatible_upper_limit_coupling_units() -> None:
    result = _result()
    result.cf_ul = FrequencySeries(
        [0.2, 0.3, np.nan, np.inf],
        frequencies=result.cf.xindex,
        unit=u.Hz,
    )

    with pytest.raises(ValueError, match="coupling-factor unit"):
        from_result(result, start_gps_ns=123, duration_ns=10)


@pytest.mark.parametrize("axis", [None, np.array([1.0, 2.0]) * u.s])
def test_from_result_rejects_absent_or_non_frequency_axis(axis: object) -> None:
    result = SimpleNamespace(
        cf=SimpleNamespace(
            value=np.array([0.1, 0.2]),
            xindex=axis,
            unit=u.m / u.V,
        ),
        valid_mask=np.array([True, True]),
        witness_name="WIT",
        target_name="TGT",
    )

    with pytest.raises((TypeError, ValueError), match="frequency axis"):
        from_result(result, start_gps_ns=123, duration_ns=10)


def test_from_result_omits_invalid_bins_without_upper_limit_method() -> None:
    table = from_result(_result(), start_gps_ns=123, duration_ns=10)

    assert list(table["frequency_hz"]) == [10.0]
    assert list(table["estimate_kind"]) == ["measurement"]
    assert not table.isna().any().any()
    validate(table)


def test_from_result_represents_dimensionless_unit_as_one() -> None:
    result = _result()
    result.cf = FrequencySeries(
        [0.1, np.nan, -1.0, np.nan],
        frequencies=result.cf.xindex,
        unit=u.dimensionless_unscaled,
    )
    result.cf_ul = None

    table = from_result(result, start_gps_ns=123, duration_ns=10)

    assert table["coupling_factor_unit"].tolist() == ["1"]
    validate(table)
