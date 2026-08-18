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
| `duration_ns` | Positive signed-64-bit duration in nanoseconds |
| `source_channel` | Nonempty source channel name |
| `response_channel` | Nonempty response channel name |
| `frequency_hz` | Finite nonnegative frequency in Hz |
| `coupling_factor` | Finite nonnegative coupling factor |
| `coupling_factor_unit` | Nonempty Astropy unit string, parsed with `Unit.to_string()` during validation; input tables are not mutated |

Unknown columns are rejected. Optional columns are `estimate_kind`,
`limit_method`, `confidence_level`, and `significance`.

`estimate_kind` defaults to `measurement` and may also be `upper_limit`.
`limit_method` is required for an upper limit and forbidden for a measurement.
`confidence_level` is valid only for upper limits and must satisfy `0 < q < 1`.
`significance` is finite and dimensionless.

When one DataFrame contains both measurements and upper limits,
non-applicable `limit_method` and `confidence_level` cells are represented by
an empty string rather than null. This keeps interchange rows free of NaN/null
values while preserving the row-level rules above.

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
present, convertible to Hz, and exactly equal to `cf`'s converted Hz grid; UL
values are never relabeled onto `cf` frequencies. Its coupling-factor unit must
be equivalent to `cf`'s unit, and values are explicitly converted to `cf`'s
unit before rows are emitted. Incompatible units are rejected.

The independent-ordinate median correction used by related spectral workflows
is documented against FINDCHIRP Appendix B Eq. B12 and Section VI Eq. 6.3b:
<https://arxiv.org/abs/gr-qc/0509116>. That correction is not part of this
table schema and must not be applied to correlated overlapped periodograms.
