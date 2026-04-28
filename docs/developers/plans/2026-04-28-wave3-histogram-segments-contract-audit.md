# Wave 3 Histogram And Segments Contract Audit

Date: 2026-04-28
Issue: #290, "Audit histogram and segments public API contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #290 slice records deterministic public contracts for native
histogram behavior and the segments compatibility boundary. It deliberately
avoids optional IO backends, plotting, ROOT, HDF5 round trips, and runtime
behavior changes.

Covered in this slice:

- native `Histogram` unit-bearing values, edges, aliases, bin geometry, passive
  `name` and `channel` metadata, and uncertainty surfaces;
- current `fill()` non-mutating weighted-bin, flow-bin, and `sumw2` behavior;
- `rebin()` and `integral()` total-bin semantics for a simple partial-overlap
  case, including unit and metadata preservation;
- `HistogramDict.rebin()` and `HistogramList.rebin()` container type and order
  preservation;
- `gwexpy.segments` dynamic proxy exposure for representative GWpy symbols.

This slice does not change histogram algorithms, statistical semantics,
metadata propagation, IO contracts, or segment behavior. Any histogram or
segments runtime behavior change remains deferred for human review.

## Contracts Recorded

### Native Histogram Surface

- `Histogram.values` and `Histogram.edges` are `astropy.units.Quantity`
  instances preserving the value unit and edge unit supplied at construction.
- `Histogram.value` is a public alias for `Histogram.values`.
- `Histogram.bin_widths` carries the edge unit and is computed from adjacent
  edges.
- `Histogram.xindex` aliases the bin centers and carries the edge unit.
- `Histogram.name` and `Histogram.channel` are stored passively for the covered
  construction, `fill()`, and `rebin()` paths. This slice does not assert any
  channel validation or metadata synchronization behavior.

### Statistical Uncertainty Surface

- `Histogram.sumw2` has shape `(nbins,)` and unit `value_unit ** 2` when
  provided.
- `Histogram.cov` has shape `(nbins, nbins)` and unit `value_unit ** 2` when
  provided.
- If both covariance and `sumw2` are present, current `Histogram.errors` uses
  the square root of the covariance diagonal first.
- If only `sumw2` is present, current `Histogram.errors` uses the square root
  of `sumw2`.
- A `copy(deep=True)` current-gap test records that the current implementation
  reconstructs with Quantity underflow and overflow values that trip Astropy's
  "Quantity truthiness is ambiguous" guard in `Histogram.__init__`. This is a
  known gap for a follow-up runtime review, not a behavior change in this slice.

### Fill Semantics

- `fill()` returns a new `Histogram` and leaves the original histogram's
  covered values, `sumw2`, underflow, and overflow unchanged.
- Weighted fills convert Quantity weights to the histogram value unit in the
  covered case.
- In-range bin values are incremented by the sum of weights.
- In-range `sumw2` is incremented by the sum of squared weights.
- Underflow and overflow values are incremented by out-of-range weights, and
  their `sumw2` values are incremented by squared out-of-range weights.
- The covered `fill()` path preserves value unit, edge unit, `name`, and
  `channel` on the returned histogram.

### Rebin And Integral Semantics

- `rebin()` uses total-bin semantics, not density semantics. New bin totals are
  formed from overlap fractions of old bin totals.
- `rebin()` preserves the value unit, edge unit, `name`, and `channel` in the
  covered Quantity-edge case.
- `rebin()` propagates covariance as `A @ cov @ A.T` for the covered linear
  overlap matrix.
- `rebin()` propagates independent `sumw2` as `(A ** 2) @ sumw2`.
- `integral(start, end)` uses the same total-bin overlap semantics. In the
  covered partial-overlap case, only the overlapping fractions contribute and
  the result carries the histogram value unit.

### Collection Ordering

- `HistogramDict.rebin()` returns a `HistogramDict` and preserves key iteration
  order in the covered ordered mapping case.
- `HistogramList.rebin()` returns a `HistogramList` and preserves list order.
- This slice does not cover collection HDF5 layout, ROOT export, plotting,
  statistical method return-shape policy, or collection class additions.

### Segments Proxy Boundary

- `gwexpy.segments.Segment`, `gwexpy.segments.SegmentList`, and
  `gwexpy.segments.DataQualityFlag` resolve to the same objects as the
  corresponding `gwpy.segments` symbols.
- These representative names appear in `gwexpy.segments.__all__` and
  `dir(gwexpy.segments)`.
- This slice deliberately keeps the segments scope to the GWpy passthrough
  boundary. Segment metadata integration, custom segment classes, and
  SegmentTable row/cell/lazy-payload behavior are out of scope and remain
  distinct from issue #276.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Fix or redefine `Histogram.copy(deep=True)` reconstruction behavior.
- Change bin-edge inclusivity, underflow/overflow policy, or flow-bin
  uncertainty behavior.
- Change covariance or `sumw2` precedence, units, shape validation, or error
  propagation.
- Change rebin overlap precision, nonuniform/log-bin policy, out-of-range edge
  warnings, or total-bin versus density semantics.
- Change `integral()` zero-width, reversed-range, error-return, or out-of-range
  behavior.
- Change collection mapping/delegation return types or statistical method
  return policy.
- Add or change HDF5/ROOT round-trip guarantees.
- Add segment metadata integration, segment association metadata, or custom
  segment passthrough behavior.
- Audit or modify SegmentTable row/cell/lazy-payload semantics covered by
  issue #276.

## Follow-Up Slices For #290

1. Histogram copy and slicing/cropping contracts: deep copy reconstruction,
   view versus copy expectations, covariance slicing, flow-bin policy, and
   metadata preservation.
2. Rebin precision and statistical invariants: nonuniform bins, partial
   out-of-range edges, covariance/sumw2 consistency, density conversion, and
   conservation tolerances.
3. Histogram IO contracts: HDF5 and ROOT round trips for units, edges, flow
   bins, covariance, `sumw2`, names, channels, and collection manifests.
4. Collection API contracts: map/plain delegation semantics, statistical
   method return types, class preservation, mutation behavior, and error
   messages.
5. Segments passthrough contracts: full `__all__` parity, optional GWpy import
   behavior, documentation of the proxy boundary, and explicit separation from
   SegmentTable issue #276.

## Verification

Focused contract check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/histogram/test_histogram_contracts.py -p no:cacheprovider
```

Related histogram and segments regression checks:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q \
  tests/histogram/test_histogram_contracts.py \
  tests/histogram/test_histogram_core.py \
  tests/histogram/test_rebin.py \
  tests/histogram/test_histogram_collections.py \
  tests/segments/test_segments.py \
  tests/segments/test_flag.py \
  -p no:cacheprovider
```

Changed-file hygiene:

```bash
rtk ruff check tests/histogram/test_histogram_contracts.py
rtk ruff format --check tests/histogram/test_histogram_contracts.py
rtk python -c "import yaml; from pathlib import Path; yaml.safe_load(Path('docs/developers/plans/audit-manifest-290-histogram-contracts.yaml').read_text())"
rtk git diff --check
```
