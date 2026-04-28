# SegmentTable Contract Audit

Date: 2026-04-28
Issue: #276
Branch: `codex/wave3-276-segment-table-contracts`
Mode: docs and tests only; no runtime modules under `gwexpy/` changed.

## Scope

This slice records a minimal current-behavior baseline for `SegmentTable`
payload cells, lazy materialization, and two representative plotting paths.
The companion tests live in
`tests/table/test_segment_table_contracts.py`.

## Current Contracts Covered

- `to_pandas(meta_only=False)` is structural for unloaded lazy payload cells:
  it writes a summary such as `"<lazy: object>"`, does not invoke the loader,
  and leaves the `SegmentCell` unloaded.
- `fetch(columns=[...])` loads only the named payload column. `clear_cache()`
  returns loader-backed cells to the unloaded state.
- `select(mask=...)` currently shares the selected row's original
  `SegmentCell` object with the source table. Accessing the selected payload
  loads the same original cell. This test baselines current shared-cache
  behavior; it is not an endorsement of the long-term isolation contract.
- `copy(deep=False)` creates new `SegmentCell` wrappers while sharing loaded
  payload value references. `copy(deep=True)` isolates loaded mutable object
  payload values for plain `dict`/`list` payloads with `kind="object"`.
- `materialize(inplace=False)` returns a copied table with lazy payloads loaded
  while the original table remains unloaded.
- `segments(y="row_y", color="group")` with numeric y metadata produces x-label
  `"time"`, y-label `"row_y"`, and one rendered segment path per row.
- `segments(y="group")` with categorical y metadata currently raises
  `ValueError` from the float conversion path. The test intentionally records
  the unsupported categorical-y behavior without pinning the raw conversion
  message.
- `overlay_spectra("spectrum")` on a `frequencyseries` payload uses log/log
  axes, draws one line per row, labels x as `"Frequency [Hz]"`, and labels y
  with the payload column name.

Colorbar behavior is deliberately not asserted in this slice because the
current contract for colorbar exposure, labels, and propagation is unsettled.

## Deferred Follow-Ups

- Parent metadata and context-aware tables for preserving analysis-level
  metadata across SegmentTable operations.
- Unified segment algebra metadata priority, including reconciliation rules for
  table-level metadata, row metadata, and payload metadata after selections,
  joins, crops, and derived columns.
- Validated serializers for custom columns and payload columns, instead of
  relying on ad hoc pandas object columns or display summaries.
- Metadata-preserving export decisions separate from value-oriented pandas and
  CSV-style exports.
- Cache/cell isolation changes for `select()`, `copy(deep=False)`, and
  materialization semantics.
- Categorical y labels for `segments()` rather than requiring numeric y values.
- Richer plotting metadata and colorbar propagation, including stable colorbar
  exposure, colorbar labels, units, legends, axis metadata, and payload-derived
  labels.

## Verification Commands

```bash
rtk env MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp pytest -q tests/table/test_segment_table_contracts.py -p no:cacheprovider
rtk env MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp pytest -q tests/table/test_segment_cell.py tests/table/test_segment_table.py tests/table/test_segment_table_new.py tests/table/test_segment_table_contracts.py -p no:cacheprovider
rtk ruff check tests/table/test_segment_table_contracts.py
rtk ruff format tests/table/test_segment_table_contracts.py
rtk ruff format --check tests/table/test_segment_table_contracts.py
rtk git add -N tests/table/test_segment_table_contracts.py docs/developers/plans/2026-04-28-segment-table-contract-audit.md docs/developers/plans/audit-manifest-276-segment-table.yaml
rtk git diff --check
```
