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
null (`None`, pandas `NA`, or an Astropy masked value). The empty string is
accepted only in either of those two fields on a measurement row; it is never
valid for required strings, numeric values, or upper-limit metadata.
`from_result()` and `from_results()` emit explicit nulls as the canonical
absence form: textual `limit_method` uses `None`, and nullable-binary64 pandas
`confidence_level` uses `pd.NA`. `to_pandas()`, `to_astropy()`, and the JSON
envelope normalize the permitted legacy form to those null semantics before
conversion (`pd.NA` is an Astropy mask and JSON `null`). Floating `NaN` is not
null metadata and is rejected. This preserves absence semantics through
pandas, Astropy, and the JSON envelope without assigning limit semantics to
measurement rows.

The factories always emit `estimate_kind`. With neither `limit_method` nor
`confidence_level` supplied, they retain the minimal shape of required columns
plus `estimate_kind`. Supplying `limit_method` requests that optional column on
every output row, using `None` for measurements. Supplying `confidence_level`
(which requires `limit_method`) requests both optional columns; its absent
measurement cells use `pd.NA`. The same shape rule applies to measurement-only
results, heterogeneous mappings, and empty `from_results()` mappings, so their
JSON envelopes remain deterministic.

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
column metadata (including arbitrary nested `Column.meta` mappings), column
descriptions and formats, units, and masks. The pandas frame carries the
Astropy metadata in its `DataFrame.attrs` under the reserved fully qualified
key `gwexpy.coupling.segment.v1.astropy_metadata`. Its exact schema name,
outer fields, per-column fields, and column set are checked on both adapters;
an absent carrier is allowed, but a malformed or colliding carrier is rejected
rather than partly applied. Carrier and restored metadata are deep-copied, so
subsequent mutations of the source table, frame attributes, and restored table
are independent. The adapters attach canonical units to the Astropy result and
map masked optional values, and a validation-accepted legacy measurement empty
string, to explicit nulls. This makes mixed `confidence_level` values numeric
with a mask on the Astropy result instead of coercing upper-limit values to
strings. Native `Table.to_pandas()` may turn a mask into a floating `NaN`; that
representation is ambiguous and remains invalid input to `validate()`. Native
`Table.from_pandas()` is valid only as a unitless canonical-value import and
does not preserve Astropy metadata or units. Result objects supplied to `from_result()` must explicitly provide a
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
present and convertible to Hz. Its converted grid must lie within exactly 32
IEEE-754 binary64 `nextafter` steps of `cf` in the candidate's direction, per
bin. The upward and downward limits are independent, including at powers of
two and near zero: the 33rd step is rejected in either direction. This accepts
benign Hz/kHz representation roundoff while rejecting a real bin mismatch. UL
values are never relabeled onto `cf` frequencies. Its
coupling-factor unit must be equivalent to `cf`'s unit, and values are
explicitly converted to `cf`'s unit before rows are emitted. Incompatible units
are rejected.

`from_results()` is the typed adapter for zero- or multi-target mappings from
`estimate_coupling()`. Its mapping keys must be strings and are processed in
lexicographic key order, so a reversed input mapping produces the same row
order. Empty mappings preserve the requested factory shape: the minimal schema
by default, `limit_method` when supplied, and both optional fields when a valid
confidence is supplied. When only some targets emit upper limits, requested
optional columns are present on every target frame before concatenation, with
explicit nulls for their non-applicable measurement cells. Passing a mapping to
`from_result()` raises a descriptive `TypeError`.

## Dtype contract

All public factories and adapters use one v1 dtype contract, including
zero-row tables. In pandas, `start_gps_ns` and `duration_ns` are native signed
`int64`; `frequency_hz` and `coupling_factor` are `float64`; and a present
`confidence_level` is nullable pandas `Float64`, so only its absent cells use
`pd.NA`. In Astropy, those same integer and numeric columns are `int64` and
`float64` respectively; absent optional cells are masks. Textual columns use
object-capable pandas and Astropy columns, rather than inferred or fixed-width
string arrays, so adding later valid channel, method, or unit strings cannot
truncate them. This contract is restored from the ordered JSON envelope even
when `rows` is empty; JSON need not carry separate dtype metadata.

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
the same binary64 normalization used by `validate()`. Non-`Real` values such
as `Decimal`, values outside finite binary64 range, and finite nonzero values
that underflow to binary64 zero are rejected. Accepted conversion preserves
the source sign and zero/nonzero status; it does not claim exact preservation
of every non-binary fraction. Conversion failures, including oversized
`Fraction` values, are normalized to schema `TypeError` or `ValueError` rather
than exposing `OverflowError`.

`significance` is intentionally not a v1 field. Its witness source, statistical
normalization, formula, and applicability to upper limits lack approved physics
authority. It is omitted rather than inferred or serialized.
