# Low-Level Types Axis Host Contract Matrix

**Issue:** #291
**Branch:** `codex/wave4-291-axis-host-contracts`
**Date:** 2026-04-28
**Author:** Codex

## Scope

This pass records the public low-level axis host surface for `gwexpy.types`.
It does not change runtime behavior, metadata semantics, physics-facing unit
conversion, or downstream `TimeSeries`/`FrequencySeries`/`Spectrogram`
contracts.

The goal is to make the current return-class and helper-boundary behavior easy
to review before any later runtime hardening. In particular, this document
keeps the existing `Array3D` versus `Array4D` dimension-dropping asymmetry as a
documented baseline rather than treating it as a bug.

## Public Axis Host Matrix

| Host | Axis source | `isel` / `sel` slice behavior | Integer-index behavior | Permutation behavior | Notes |
| --- | --- | --- | --- | --- | --- |
| `Array` | Generic `_axis_names` with dimensionless integer coordinates. | Delegates to normal array slicing and preserves generic axis names when the resulting dimension count matches. | Dimension dropping follows the underlying GWpy/Quantity view behavior. | `swapaxes()` reorders `_axis_names`; generic transpose uses the mixin fallback. | Suitable for named dimensions, not coordinate-bearing physics axes. |
| `Array2D` | GWpy `xindex`/`yindex` exposed as `AxisDescriptor` objects. | Slice selections stay `Array2D` when two dimensions remain. | Dimension dropping follows the underlying GWpy/Quantity behavior. | `swapaxes()` and `(1, 0)` transpose reconstruct a fresh `Array2D` with axis names and indices swapped. | Dimension 0 is exposed through `xindex` and dimension 1 through `yindex`; keep this convention explicit in tests. |
| `Plane2D` | Same storage as `Array2D`, with semantic `axis1` and `axis2` aliases. | Slice selections stay `Plane2D` when two dimensions remain through inherited `Array2D` behavior. | Dimension dropping follows inherited `Array2D` / GWpy behavior. | `swapaxes()` reconstructs `Plane2D` with semantic axis names and indices swapped. | Used when a 2D slice should keep human-readable physical axis meaning. |
| `Array3D` | Explicit `_axis{0,1,2}_name` and `_axis{0,1,2}_index` arrays. | Pure slice selections return `Array3D` and preserve sliced axis coordinates. | One integer index returns `Plane2D`; lower-dimensional or unsupported indexing falls back to a plain Quantity/GWpy view. | `swapaxes()` and full transpose reorder axis names and coordinate arrays. | `plane()` is the explicit helper for 2D extraction with optional axis-order control. |
| `Array4D` | Explicit `_axis{0,1,2,3}_name` and `_axis{0,1,2,3}_index` arrays. | Pure slice selections return `Array4D` and preserve sliced axis coordinates. | Any dimension drop currently returns a plain Quantity/GWpy view, not `Array3D` or `Plane2D`. | `swapaxes()` and full transpose reorder axis names and coordinate arrays. | This asymmetry with `Array3D` is current behavior and should be changed only in a follow-up runtime contract PR. |

## Shared Axis API Contract

- `axis(key)` accepts an integer axis index or axis name and returns an
  `AxisDescriptor`.
- `axis_names` reflects the host's current axis order.
- `rename_axes(mapping, inplace=False)` returns an isolated copy by default and
  rejects duplicate names.
- `isel()` converts named or positional indexers into a tuple and delegates to
  the host's `_isel_tuple()` implementation.
- `sel()` converts coordinate values or coordinate slices to integer selections
  through `AxisDescriptor.iloc_nearest()` and `AxisDescriptor.iloc_slice()`.
  The public argument is named `method`, but current behavior is nearest-like;
  exact, pad, backfill, and tolerance policies are not implemented.
- `swapaxes()` and `transpose()` accept names or integer indices and delegate to
  host-specific permutation hooks.

## Public Versus Internal Boundary

| Object | Status | Contract |
| --- | --- | --- |
| `AxisDescriptor` | Public | Named one-dimensional coordinate descriptor with unit exposure and nearest/slice index conversion. |
| `coerce_1d_quantity` | Public helper | Converts axis-like input to a one-dimensional Quantity and rejects higher-dimensional axes. |
| `AxisApiMixin` | Public developer helper | Shared axis API for low-level hosts; concrete return classes remain host-specific. |
| `Array`, `Array2D`, `Plane2D`, `Array3D`, `Array4D` | Public | Low-level array hosts with the return-class contracts listed above. |
| `MetaData`, `MetaDataDict`, `MetaDataMatrix` | Public | Metadata containers covered by the #291 foundation audit. |
| `as_series` | Public helper | Builds identity-valued `TimeSeries` or `FrequencySeries` from one-dimensional axis-like input. |
| `gwexpy.types.mixin._protocols` | Internal | Structural typing aid for implementation and static checking; not a stable user API. |
| `gwexpy.types.mixin._collection_mixin` | Internal | Delegation factory helpers for collection implementations; downstream code should not depend on them directly. |
| `gwexpy.types.seriesmatrix_validation` | Internal | SeriesMatrix construction and validation helpers; public behavior should be tested through concrete SeriesMatrix APIs. |

## Follow-Up Candidates

1. Decide whether `Array4D` should eventually return `Array3D` / `Plane2D` for
   dimension-dropping selections. That would be a runtime behavior change.
2. Decide whether `AxisApiMixin.sel(method=...)` should reject unsupported
   methods explicitly or grow exact/tolerance semantics. Either option changes
   user-visible behavior and should be a separate PR.
3. Add downstream smoke tests only after the low-level host table above is
   accepted, so TimeSeries/FrequencySeries/Spectrogram contracts do not move
   accidentally with type-level refactors.

## Validation

This PR adds `tests/types/test_axis_host_contract_docs.py` to ensure the public
axis host table and internal-helper boundary do not silently drop required
entries.

Recommended local checks:

```bash
rtk pytest -q tests/types/test_axis_host_contract_docs.py -p no:cacheprovider
rtk ruff check tests/types/test_axis_host_contract_docs.py
rtk ruff format --check tests/types/test_axis_host_contract_docs.py
rtk python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('docs/developers/plans/audit-manifest-291-axis-hosts.yaml').read_text())"
rtk git diff --check
```

