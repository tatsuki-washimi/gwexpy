# Coupling segment schema v1

Schema name: `gwexpy.coupling.segment.v1`

This document defines a domain-neutral tabular interchange format for coupling
factor estimates. The public implementation is
`gwexpy.coupling.segment.validate()` and
`gwexpy.coupling.segment.from_result()`; it does not replace the existing
`CouplingResult` CSV/TXT exporters.

## Required columns

Each row contains:

| Column | Meaning |
| --- | --- |
| `start_gps_ns` | Nonnegative signed-64-bit GPS start time in nanoseconds |
| `duration_ns` | Positive signed-64-bit duration in nanoseconds; `start_gps_ns + duration_ns` must also fit signed int64 |
| `source_channel` | Nonempty source channel name |
| `response_channel` | Nonempty response channel name |
| `frequency_hz` | Finite nonnegative frequency in Hz |
| `coupling_factor` | Finite nonnegative coupling factor |
| `coupling_factor_unit` | Nonempty Astropy unit string; the sole persisted unit authority, parsed with `Unit.to_string()` during validation without mutating input tables |

Unknown columns are rejected. Optional columns are `estimate_kind`,
`limit_method`, and `confidence_level`.

`estimate_kind` defaults to `measurement` and may also be `upper_limit`.
`limit_method` is required for an upper limit and forbidden for a measurement.
`confidence_level` is valid only for upper limits and must satisfy `0 < q < 1`.
`confidence_level` is rejected when `limit_method` is absent.

When one DataFrame contains both measurements and upper limits,
non-applicable `limit_method` and `confidence_level` cells are represented by
an empty string rather than null. This keeps interchange rows free of NaN/null
values while preserving the row-level rules above.

## Units and table representations

`coupling_factor_unit` is the sole persisted authority for every row. A pandas
DataFrame has no second unit channel. When an Astropy `coupling_factor` column
declares a unit, it must exactly match every row's parsed unit string; a
conflict is rejected rather than converted or relabeled. Thus an Astropy table
can round-trip through pandas without changing the schema authority. Result
objects supplied to `from_result()` must explicitly provide a coupling-factor
unit, including an explicit dimensionless unit where that is the intended
meaning; unitless duck objects are rejected.

## Result conversion

`from_result(result, start_gps_ns=..., duration_ns=...)` emits finite valid
measurement bins from `result.cf`. Invalid measurement bins are emitted as
finite nonnegative upper-limit bins only when a `limit_method` is supplied and
`result.cf_ul` provides such a value. All other bins are omitted. Source and
response names come from `witness_name` and `target_name`; frequency and unit
come from `cf`.

The schema is intentionally independent of any detector or experiment naming
convention and accepts pandas DataFrames as well as Astropy Tables when
Astropy is installed.

When `cf_ul` is supplied, `from_result()` requires its frequency axis to be
present and convertible to Hz. Its converted grid must agree with `cf` within
the larger of 32 binary64 ULPs and one billionth of the nearest positive bin
spacing. This accepts Hz/kHz representation roundoff while rejecting a real
bin mismatch. UL values are never relabeled onto `cf` frequencies. Its
coupling-factor unit must be equivalent to `cf`'s unit, and values are
explicitly converted to `cf`'s unit before rows are emitted. Incompatible units
are rejected.

`from_results()` is the typed adapter for zero- or multi-target mappings from
`estimate_coupling()`. Empty mappings produce an empty DataFrame with all v1
columns. Passing a mapping to `from_result()` raises a descriptive `TypeError`.

## JSON envelope

`to_json_envelope(table)` produces the strict JSON-safe envelope
`{"schema": "gwexpy.coupling.segment.v1", "columns": [...], "rows": [...]}`.
`from_json_envelope()` accepts only that exact field set and schema version.
The explicit ordered `columns` array preserves the schema for zero-row tables;
plain record-oriented JSON cannot do so.

`significance` is intentionally not a v1 field. Its witness source, statistical
normalization, formula, and applicability to upper limits lack approved physics
authority. It is omitted rather than inferred or serialized.
