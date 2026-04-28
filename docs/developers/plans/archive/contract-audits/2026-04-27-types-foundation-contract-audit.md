# Low-Level Types Metadata And Axis Foundation Contract Audit

Date: 2026-04-27
Issue: #291, "Audit low-level types metadata and axis foundation contracts"
Mode: contract-audit with focused regression tests; no runtime behavior changes.

## Constraints

- `AGENTS.md`, `.agent/AGENTS.md`, and the RTK instructions were reviewed. No
  issue-specific `.agent/*/SKILL.md` file was required for this docs/test-only
  audit pass.
- This pass intentionally avoids physics-sensitive changes to axis comparison,
  unit conversion, matrix math, or downstream TimeSeries/FrequencySeries/
  Spectrogram behavior.
- Adjacent audits for #244, #246, and #247 were treated as dependencies. This
  document stays at the low-level `gwexpy.types` foundation layer and does not
  redefine matrix-specific or `to_matrix()` family-specific contracts.

## Inputs Reviewed

- `gwexpy/types/axis.py`
- `gwexpy/types/axis_api.py`
- `gwexpy/types/array.py`
- `gwexpy/types/array2d.py`
- `gwexpy/types/array3d.py`
- `gwexpy/types/array4d.py`
- `gwexpy/types/plane2d.py`
- `gwexpy/types/series.py`
- `gwexpy/types/series_creator.py`
- `gwexpy/types/metadata.py`
- `gwexpy/types/mixin/_collection_mixin.py`
- `gwexpy/types/mixin/_protocols.py`
- `gwexpy/types/mixin/mixin_legacy.py`
- Existing focused tests under `tests/types/`
- Developer plans for #244, #246, and #247.

## Current Contract Summary

### AxisDescriptor And AxisApiMixin

- `AxisDescriptor` is a named, one-dimensional `Quantity` axis. It coerces
  scalar inputs to length-one axes, rejects higher-dimensional axes, exposes the
  axis unit through `unit`, and supports nearest-coordinate and coordinate-slice
  conversion.
- `regular` and `delta` are convenience checks only. They rely on the wrapped
  index's `regular` flag when present, otherwise they use `np.isclose` on
  numeric spacings. They should not be treated as precision-sensitive physics
  validation.
- `AxisApiMixin` resolves axes by integer index or name and delegates slicing,
  swap, and transpose to subclass hooks. Duplicate names after `rename_axes()`
  are rejected.
- `sel()` maps coordinates to integer indices before delegating to `isel()`.
  Its current `method` argument is effectively limited to the implemented
  nearest behavior; alternate methods are not implemented.

### Array, Array2D, Plane2D, Array3D, And Array4D

- `Array` stores generic `_axis_names` but uses default dimensionless integer
  coordinates for all axes. It is appropriate for named dimensions, not
  coordinate-bearing physics axes.
- `Array2D`/`Plane2D` use GWpy `xindex`/`yindex` storage while presenting them
  through the `AxisApiMixin` surface. Existing code treats dimension 0 as
  `xindex` and dimension 1 as `yindex`; tests should keep that convention
  explicit because it is easy to confuse with plotting terminology.
- `Array3D` and `Array4D` preserve axis names, units, and sliced coordinates
  for pure slice selections. Integer indexing intentionally drops dimensions:
  `Array3D` can return `Plane2D`, while `Array4D` lower-dimensional selections
  return plain GWpy/Quantity views instead of inventing 3D/2D wrappers.
- Transpose and swapaxes reorder both axis names and coordinate arrays. This is
  a foundational contract for later field and transform code.

### as_series

- `as_series()` creates an identity-valued `TimeSeries` or `FrequencySeries`
  from a one-dimensional axis-like input.
- For time-like axes, values use the requested value unit or the axis unit, and
  the time axis remains the input axis.
- For frequency-like axes, values use the requested frequency/angular-frequency
  unit. Angular-frequency input axes are converted to Hz for the
  `FrequencySeries.frequencies` axis, while the series values can remain angular
  frequency when no target unit is requested.
- This helper depends on registered constructors. Importing `gwexpy` before
  use remains the safest public entry point in tests and examples.

### MetaData, MetaDataDict, And MetaDataMatrix

- `MetaData` normalizes missing names/channels/units to empty name, empty
  `Channel`, and dimensionless unit. Invalid units also fall back to
  dimensionless.
- `MetaDataDict` is ordered and key-preserving. Construction from an existing
  `MetaDataDict` is shallow, but ordinary `copy.deepcopy()` isolates nested
  `MetaData` entries.
- `MetaDataMatrix(shape=..., default=...)` and `fill()` create independent
  `MetaData` entries for each cell. This prevents one cell mutation from
  silently changing another cell.
- CSV serialization for `MetaDataDict` and `MetaDataMatrix` stores unit strings
  and round-trips names/channels through `MetaData` normalization. It is a
  metadata serialization contract, not a full data-array round-trip contract.
- Discovered gap: blank `MetaDataMatrix` channel/unit cells can reload through
  pandas as `nan`, producing `Channel("nan")` and a scaled dimensionless
  `Unit("nan")` instead of the intended empty channel and dimensionless unit.
  This pass records the gap with a strict xfail rather than changing runtime
  normalization.

### Public Versus Internal Helpers And Mixins

- Public, user-facing foundation surface: `AxisDescriptor`, `coerce_1d_quantity`,
  `AxisApiMixin`, `Array`, `Array2D`, `Array3D`, `Array4D`, `Plane2D`,
  `MetaData`, `MetaDataDict`, `MetaDataMatrix`, and `as_series`.
- Internal or semi-internal helper surface: `gwexpy.types.mixin._protocols`,
  `_collection_mixin` factory helpers, and validation/creator functions used by
  SeriesMatrix internals. These should remain implementation details unless a
  later public API decision documents them.
- The shared mixins rely on host classes for units, names, channels, and
  constructor metadata. Low-level changes here can alter TimeSeries,
  FrequencySeries, Spectrogram, and collection behavior indirectly.

## Regression Coverage Added In This Pass

- `tests/types/test_types_foundation_contracts.py` checks that:
  - `rename_axes()` returns an isolated copy and rejects duplicate axis names,
  - `Array3D.isel()` preserves axis names, units, and sliced coordinates,
  - `MetaDataDict` and `MetaDataMatrix` are isolated under `copy.deepcopy()`,
  - `MetaDataMatrix` default construction and `fill()` create independent cells,
  - metadata CSV round-trips preserve keys, names, channels, and units,
  - missing metadata CSV cells are documented as a strict xfail pending a
    normalization fix,
  - angular-frequency `as_series()` keeps angular values by default while using
    a Hz frequency axis, and can emit Hz values when requested.

## Gaps And Follow-Up Candidates

- `AxisApiMixin.sel(method=...)` accepts a `method` parameter but only implements
  nearest-index behavior. Supporting exact, pad, or tolerance semantics would be
  a public behavior change and should be split into its own PR.
- `MetaDataDict(MetaDataDict)` and `MetaDataMatrix(existing_metadata_objects)`
  may preserve nested object references. A change to deep-copy on construction
  could be useful, but it is a runtime data-model change and should be reviewed
  separately.
- `MetaDataMatrix` slicing currently behaves like an ndarray object view and
  may share nested `MetaData` entries. The desired copy/view contract should be
  decided before adding stronger tests or changing behavior.
- `MetaDataMatrix.read()` should normalize pandas missing values for channel and
  unit fields before constructing `MetaData`. This is a low-level runtime
  behavior change with cross-module metadata implications, so it should be made
  in a follow-up PR with reviewer sign-off.
- `Array4D` integer indexing returns lower-dimensional plain objects, while
  `Array3D` can return `Plane2D`. This asymmetry may be intentional, but a
  future public contract should state it explicitly.
- `Series` is currently a statistical-mixin wrapper around GWpy `Series`, not
  an `AxisApiMixin` host. If a unified axis API is expected for 1D series, that
  should be designed as a separate compatibility-sensitive change.
- Shared collection-delegation mixins do not inspect or preserve metadata
  themselves; they rely entirely on each delegated element method. Downstream
  metadata guarantees must therefore be tested on concrete collection types.

## Recommended Next PRs

1. Axis API documentation PR: publish a developer table for each host class,
   including integer indexing, slicing, transpose, coordinate selection, and
   returned object type.
2. Metadata copy/view PR: decide shallow-constructor versus deep-constructor
   semantics for `MetaDataDict` and `MetaDataMatrix`, then add tests before any
   behavior change.
3. `sel()` semantics PR: either restrict/document `method="nearest"` as the
   only supported mode or implement exact/tolerance behavior with tests.
4. Public-helper boundary PR: mark internal creator, validation, protocol, and
   mixin helpers as private in developer docs so downstream code does not depend
   on accidental exports.
5. Downstream smoke PR: add low-blast-radius TimeSeries/FrequencySeries/
   Spectrogram tests that exercise axis/mixin delegation without changing their
   physics contracts.

## Audit Notes

- No production code changed in this pass.
- No physics-sensitive behavior was changed.
- Runtime changes discovered above should be treated as follow-up design work,
  not as blockers for the current docs/test-only audit PR.
