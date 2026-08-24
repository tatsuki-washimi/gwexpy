# SeriesMatrix container arithmetic contract

Status: B0 frozen for v0.2.0 Phase A.  This document records the observable
contract before issue #637.  It does not adopt or implement the #637
composition redesign.

The executable canonical ledger is the typed `B0_CONTRACT` in
`tests/types/series_matrix_contract_manifest.py`.  It contains exactly 390
cells across `TimeSeriesMatrix`, `FrequencySeriesMatrix`, and
`SpectrogramMatrix`.  The typed adapter in
`tests/types/test_series_matrix_contract_manifest.py` executes every cell once
and consumes typed result-class, unit, metadata, axis, value, and mutation
expectations directly.  It checks exact values, dtypes, shapes, view aliasing,
per-cell names/channels, row/column metadata, and family axes, or an exact
exception.  The earlier 318-cell candidate was rejected by physics/data-model
review because it omitted active scalar add/sub behaviour.  The replacement
adds 72 scalar cells: Python and NumPy unitless scalars, dimensional failure
and dimensionless success scenarios, pure and reflected directions, and
atomic in-place forms for every family and both add/sub operators.  The
ndarray scenarios remain split between a dimensionless-matrix success and a
dimensional-incompatibility failure for both directions and all three
families.  `tests/types/test_series_matrix_operator_contract.py` retains
the direct behavioral checks.  A B1 implementation must update and compare
this same ledger rather than introducing a second matrix.

## B0 construction and structure

The approved structure surface is:

| Surface | B0 rule |
| --- | --- |
| `shape`, `dtype`, values | Preserve NumPy shape, dtype, and values according to the concrete matrix class. |
| Slicing, assignment, iteration | Preserve the concrete matrix family where the current implementation supports the operation; assignment is value-based and does not change the matrix identity. |
| `copy`, `astype` | Return an independent concrete matrix with copied metadata and axes. `astype` changes only the requested dtype. |
| `real`, `imag`, `conj` | Return the concrete matrix class and preserve per-cell units. SpectrogramMatrix `real`/`imag` results have deep-independent metadata/attrs and preserve both time and frequency axes. |
| `transpose`, `reshape` | Preserve the concrete class for the 3-D series families. B0 currently raises `ValueError` for these operations on `SpectrogramMatrix`; that observed exception is frozen honestly in the manifest. |
| `np.asarray(matrix)` | Return a plain `numpy.ndarray` containing the values. |
| `matrix.view(np.ndarray)` | Return a plain raw `numpy.ndarray` view containing the values. |

For a result that remains a matrix, per-cell metadata is independent of the
source and of neighboring cells.  Row/column metadata is copied for a new
logical result.  A sample axis is preserved or adjusted consistently with the
operation and with the concrete family.

The following are explicitly **NON-CONTRACT**: ndarray object identity;
typed/raw view compatibility beyond the explicit `matrix.view(np.ndarray)`
surface; the buffer protocol; `.base`, `.data`, `.flags`, `.strides`, and
`__array_interface__`.

## B0 arithmetic and units

All supported out-of-place arithmetic returns the concrete matrix family.  A
failed operation must not mutate an operand.

* Addition and subtraction convert compatible quantities to the left
  operand's cell units.  Unitless Python scalars, NumPy scalars, and raw
  ndarrays are accepted only for a dimensionless matrix.  Dimensional inputs
  fail atomically with `astropy.units.UnitConversionError`; the manifest
  records all six dimensional SpectrogramMatrix raw-ndarray add/sub cells,
  covering pure, reflected, and in-place operations, as exact atomic
  `UnitConversionError` failures.  The dimensionless ndarray cases succeed;
  the dimensionless scalar cases also succeed.  There is no
  dimensional-cell-preservation exception for these unitless inputs.
  Matrix metadata and axes remain independent and intact where the concrete
  operation provides independent metadata.
* Multiplication and true division compose units in operand order.  The
  accepted scalar families include Python and NumPy scalars, ndarrays,
  `Quantity`, bare `Unit`, and a same-class matrix where the operation is
  meaningful.  Quantity-left and Unit-left forms must retain the matrix class;
  they must not collapse to a bare `Quantity`.
* `power` accepts a dimensionless scalar exponent and raises each cell unit to
  that exponent.  A dimensional exponent or any non-scalar exponent raises
  `UnitConversionError` before values or metadata are changed, including a
  matrix-valued SpectrogramMatrix exponent.  `sqrt` is a
  documented target rule for B1, but direct `np.sqrt(matrix)` remains a B0
  rejection (see below).
* Comparisons (`<`, `<=`, `==`, `!=`, `>`, `>=`) return the concrete matrix
  class with `numpy.bool_` values and dimensionless-unscaled cell units.
  Compatible units are converted before comparison; incompatible units raise
  `UnitConversionError`.  Comparison metadata is an independent copy.
* The B0 predicate surface is deliberately conservative.  Direct
  `np.isfinite` and `np.isnan` calls are rejected along with other direct
  ufunc calls.  `np.isreal` is a `UnitConversionError` for the two 3-D series
  families and returns a boolean SpectrogramMatrix with dimensionless cells,
  preserved time/frequency axes, preserved name, and deep-independent
  metadata/attrs.
* In-place dunders compute and validate an out-of-place result first, then
  commit atomically.  Incompatible units, zero division, shape errors, and
  unsafe dtype changes leave values and metadata untouched.

Floor division, remainder, and `divmod` remain explicitly restricted or
unsupported in B0 because NumPy's raw operations do not perform the required
unit conversion.  The current tests pin these failures rather than silently
accepting a numerically wrong result.

## Direct ufunc behavior: B0 and the #637 target

### B0

`SeriesMatrix.__array_ufunc__ = None` is an intentional observable behavior.
Direct ufunc calls such as `np.sqrt(matrix)`, `np.log(matrix)`,
`np.exp(matrix)`, `np.add.reduce(matrix)`, and other direct applications raise
`TypeError`.  Phase A records this behavior; it does not make currently
failing target cases execute.

### #637 target policy (not adopted by B0)

The proposed composition redesign permits only ordinary ufunc `__call__`
operations that have a typed metadata rule.  `out=`, reduction/accumulation
methods, multiple outputs, and unsupported `where` forms raise `TypeError`.
Only atomic in-place dunders may perform destructive updates.  Target unary
rules include `sqrt` with unit propagation and `log`/`exp` only when every
cell is dimensionless; their boolean predicates return a concrete matrix with
dimensionless units.  These statements are policy for B1 review, not evidence
that #637 has passed.

## Phase A benchmark evidence (#676)

`scripts/benchmarks/series_matrix_benchmark.py` is a standard-library runner.
It does not require `pytest-benchmark` or an optional benchmark extra.  The
runner checks out the selected fixed SHA into an exact temporary detached
worktree, starts children with that tree as the only `PYTHONPATH`, and records
every loaded `gwexpy.*` source module used by each workload.  The parent
schema-validates each child payload and independently verifies every recorded
module path is a relative, existing file under the selected target checkout;
module names map exactly to `gwexpy/__init__.py`, a module `.py`, or a package
`__init__.py` as import resolution requires.  Duplicate names and paths,
traversal/absolute/missing/outside paths, and duplicate candidate runtime files
are rejected.  Child elapsed time must meet the protocol minimum and must
agree with `iterations * per_operation_seconds` within only a representation
safe few-ULP check; child-supplied paths never select the target tree.

The frozen protocol is:

* three warm-ups;
* seven independent child processes per operation;
* calibration to at least 250 ms per measured operation;
* MAD/median variability at a 5% threshold, with at most three total batch
  attempts and every attempt/stability decision recorded;
* bounded representative operations: construction, copy, slice, `np.asarray`,
  scalar multiplication, and Quantity-left multiplication.

Each JSON result records the fixed origin SHA, environment versions, operation
definitions, imported source module names and paths relative to the target
tree, every raw timing/RSS sample, median, and MAD.  Tracked evidence contains
no absolute username or worktree paths.  Candidate comparison is implemented
now with
these gates:

* each operation median delta `<= max(baseline * 20%, 10 microseconds)`;
* geometric-mean timing ratio `<= 1.10`;
* absolute child-process peak RSS is retained as a diagnostic only and is
  non-gating until a matched-process memory methodology is adopted.

The timing comparisons use exact `>` rejection: equality is accepted and the
next representable value above each timing boundary is rejected.
`adopt_candidate()`
applies the stability gate before numeric comparison, so any unstable baseline
or candidate operation is non-adoptable.

Candidate evidence also accepts a SHA-256 over an explicitly supplied,
target-relative frozen runtime-file set.  The B0 capture summarized in
`docs/plans/evidence/v0.2.0-b0/series_matrix_b0_summary.md` is fixed at
`6a13900672900551ccaf1b18fe78b9ce6f062e29`, has a recorded raw-capture digest
`5fcdab552f8c910812335e81dfb4e0f170543f69ba78f18a63784191a80bf3b5`, and
records all six operations as stable with an adoptable stability gate.  The
raw JSON was intentionally not retained, so that digest cannot independently
verify the deleted artifact.  Any future adoption review must retain a raw
artifact and make it available for designated reviewer inspection.  Its
`decision: pending` and `issue_637: not evaluated in B0` fields do not adopt
or evaluate a #637 candidate.  A separate B1 capture and comparison are
required before any numeric adoption claim.

## D21 data-model approval

This B0 contract freezes unit, metadata, axis, label, and mutation semantics.
It requires explicit human D21/data-model sign-off before merge or release.
AI review is advisory and cannot provide that sign-off.
