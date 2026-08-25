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

For a mixed measurement/upper-limit table, non-applicable `limit_method` and
`confidence_level` cells may be either the legacy empty string or an explicit
null (`None`, pandas `NA`, or an Astropy masked value). Null is the canonical
form emitted by `from_results()` when it joins targets with heterogeneous
optional-column shapes. Floating `NaN` is not null metadata and is rejected.
This preserves absence semantics through pandas, Astropy, and the JSON
envelope without assigning limit semantics to measurement rows.

## Units and table representations

`coupling_factor_unit` is the sole persisted authority for every row. A pandas
DataFrame has no second unit channel. When an Astropy `coupling_factor` column
declares a unit, it must exactly match every row's parsed unit string; a
conflict is rejected rather than converted or relabeled. Thus an Astropy table
can round-trip through pandas without changing the schema authority.

The `_ns` and `_hz` names are the authority for the three canonical coordinate
columns. Unitless pandas columns, and unitless Astropy columns such as those
created by `Table.from_pandas()`, carry raw canonical values: nanoseconds for
`start_gps_ns` and `duration_ns`, and Hz for `frequency_hz`. An explicit
Astropy unit is accepted only when it is exactly `ns` for either time column or
exactly `Hz` for frequency. Seconds, days, kHz, and dimensionally incompatible
units are rejected without converting or mutating the input, so raw seconds or
days can never be silently interpreted as nanoseconds.

Use `to_pandas()` and `to_astropy()` for the supported pandas/Astropy
round-trip. The adapters return independent copies, preserve Astropy table and
column metadata, attach the canonical units to the Astropy result, and map
masked optional values to explicit nulls. Native `Table.to_pandas()` may turn a
mask into a floating `NaN`; that representation is ambiguous and remains
invalid input to `validate()`. Native `Table.from_pandas()` is valid only as a
unitless canonical-value import and does not preserve Astropy metadata or
units. Result objects supplied to `from_result()` must explicitly provide a
coupling-factor unit, including an explicit dimensionless unit where that is
the intended meaning; unitless duck objects are rejected.

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
the larger of exactly 32 binary64 ULPs (from the adjacent `nextafter` values,
including near zero) and one billionth of the nearest positive bin spacing.
This accepts Hz/kHz representation roundoff while rejecting a real bin
mismatch. UL values are never relabeled onto `cf` frequencies. Its
coupling-factor unit must be equivalent to `cf`'s unit, and values are
explicitly converted to `cf`'s unit before rows are emitted. Incompatible units
are rejected.

`from_results()` is the typed adapter for zero- or multi-target mappings from
`estimate_coupling()`. Its mapping keys must be strings and are processed in
lexicographic key order, so a reversed input mapping produces the same row
order. Empty mappings produce an empty DataFrame with all v1 columns. When
only some targets emit upper limits, it adds the relevant optional columns to
the measurement-only target frames before concatenation and writes explicit
nulls for their non-applicable cells. Passing a mapping to `from_result()`
raises a descriptive `TypeError`.

## JSON envelope

`to_json_envelope(table)` produces the strict JSON-safe envelope
`{"schema": "gwexpy.coupling.segment.v1", "columns": [...], "rows": [...]}`.
`from_json_envelope()` accepts only that exact field set and schema version.
The explicit ordered `columns` array preserves the schema for zero-row tables;
plain record-oriented JSON cannot do so. Every table accepted by `validate()`
can be passed to `json.dumps(to_json_envelope(table))`: signed-int64 schema
times normalize to Python `int`, while finite real frequency, coupling-factor,
and confidence values normalize to binary64 Python `float`. A finite
`Fraction` or representable NumPy extended float therefore serializes through
that binary64 normalization; non-`Real` values such as `Decimal`, and values
outside finite binary64 range, are rejected.

`significance` is intentionally not a v1 field. Its witness source, statistical
normalization, formula, and applicability to upper limits lack approved physics
authority. It is omitted rather than inferred or serialized.
