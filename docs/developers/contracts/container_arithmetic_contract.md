# SeriesMatrix container arithmetic contract

Status: B0 frozen for v0.2.0 Phase A.  This document records the observable
contract before issue #637.  It does not adopt or implement the #637
composition redesign.

The executable canonical ledger is the typed `B0_CONTRACT` in
`tests/types/series_matrix_contract_manifest.py`.  It contains exactly 318
cells across `TimeSeriesMatrix`, `FrequencySeriesMatrix`, and
`SpectrogramMatrix`.  The typed adapter in
`tests/types/test_series_matrix_contract_manifest.py` executes every cell once
and consumes typed result-class, unit, metadata, axis, value, and mutation
expectations directly.  It checks exact values, dtypes, shapes, view aliasing,
per-cell names/channels, row/column metadata, and family axes, or an exact
exception.  The 24-cell increase over the earlier 294-cell ledger is intentional: each
add/sub ndarray case is split into a dimensionless-matrix success scenario and
a dimensional-incompatibility scenario for both operand directions and all
three families; the same two scenarios are also covered by ndarray in-place
add/sub cells for every family.  `tests/types/test_series_matrix_operator_contract.py` retains
the direct behavioral checks.  A B1 implementation must update and compare
this same ledger rather than introducing a second matrix.

## v0.2.0 direct-ufunc limitation

- Stability: provisional.
- B0 is the v0.2.0 contract.
- B1/composition is not part of v0.2.0.
- Direct NumPy ufuncs are supported only where this canonical ledger marks them
  as supported.
- Unsupported direct ufunc operations fail explicitly and remain explicit in the
  ledger; they never silently downgrade to bare `ndarray` or `Quantity`.
- This fail-closed behavior is intentional and must be treated as observable
  contract compliance, not a regression signal.
- `SeriesMatrix.__array_ufunc__ = None` is the current implementation
  mechanism used to keep the fallback behavior explicit; it is not a long-term
  API compatibility guarantee.
- #637 is future redesign work with no version or date commitment.

### User-facing alternatives

- `np.sqrt(matrix)` is unsupported as a direct ufunc under B0.
- `matrix ** 0.5` is the validated metadata-preserving operator-path workaround
  for all three SeriesMatrix families:
  `TimeSeriesMatrix`, `FrequencySeriesMatrix`, and `SpectrogramMatrix`.
- That claim is guaranteed by a targeted operator-contract regression test,
  `test_sqrt_workaround_preserves_the_whole_b0_surface` in
  `tests/types/test_series_matrix_operator_contract.py`, which pins the concrete
  class, the values against `np.sqrt`, every cell unit, the axes, the row and
  column labels, the per-cell name and channel, the epoch, the family-specific
  attrs semantics, and the absence of source mutation.
  The canonical 318-cell manifest does not itself enumerate a `0.5` exponent
  cell; its scalar power cells are pinned at exponent 2.
- `matrix ** 0.5` is a workaround via normal operator dispatch; it is not
  direct-ufunc support and does not imply B1/composition adoption.
- No metadata-preserving B0 alternatives are currently defined for
  `np.log(matrix)`, `np.exp(matrix)`, `np.isfinite(matrix)`, or
  `np.isnan(matrix)`.
- `np.isreal(matrix)` remains family-specific: concrete boolean matrix for
  `SpectrogramMatrix`, `UnitConversionError` for `TimeSeriesMatrix` and
  `FrequencySeriesMatrix`.
- Both quantity-left and quantity-right multiplication forms are already supported:
  `(2 * u.s) * matrix` and `matrix * (2 * u.s)`.
- Improving the runtime error message so that an unsupported direct ufunc can
  itself point at the supported alternative is tracked as future design work in
  [#681](https://github.com/tatsuki-washimi/gwexpy/issues/681). v0.2.0 does not
  change runtime behavior for this, and no release version or date is assigned
  to #681. Its scope is the diagnostic only; it does not reopen the B1/
  composition redesign.

## B0 construction and structure

The approved structure surface is:

| Surface | B0 rule |
| --- | --- |
| `shape`, `dtype`, values | Preserve NumPy shape, dtype, and values according to the concrete matrix class. |
| Slicing, assignment, iteration | Preserve the concrete matrix family where the current implementation supports the operation; assignment is value-based and does not change the matrix identity. |
| `copy`, `astype` | Return an independent concrete matrix with copied metadata and axes. `astype` changes only the requested dtype. |
| `real`, `imag`, `conj` | Return the concrete matrix class and preserve per-cell units. SeriesMatrix axes are preserved; SpectrogramMatrix records the observed time-axis-only result for `real`/`imag`. |
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
  operand's cell units.  A unitless ndarray is accepted for a dimensionless
  matrix and is refused with `astropy.units.UnitConversionError` for the
  dimensional TimeSeriesMatrix/FrequencySeriesMatrix cases.  The
  dimensionless ndarray cases succeed.  The manifest records all six dimensional
  SpectrogramMatrix raw-ndarray add/sub cells, covering pure, reflected, and
  in-place operations, are exact atomic `TypeError` failures.  There is no
  dimensional-cell-preservation exception for this ndarray input.
  Matrix metadata and axes remain independent and intact where the concrete
  operation provides independent metadata.
* Multiplication and true division compose units in operand order.  The
  accepted scalar families include Python and NumPy scalars, ndarrays,
  `Quantity`, bare `Unit`, and a same-class matrix where the operation is
  meaningful.  Quantity-left and Unit-left forms must retain the matrix class;
  they must not collapse to a bare `Quantity`.
* `power` accepts a dimensionless scalar exponent and raises each cell unit to
  that exponent.  A dimensional exponent, or a non-scalar exponent applied to
  a dimensional base, raises `UnitConversionError` in B0.  `sqrt` is a
  documented target rule for B1, but direct `np.sqrt(matrix)` remains a B0
  rejection (see below).
* Comparisons (`<`, `<=`, `==`, `!=`, `>`, `>=`) return the concrete matrix
  class with `numpy.bool_` values and dimensionless-unscaled cell units.
  Compatible units are converted before comparison; incompatible units raise
  `UnitConversionError`.  Comparison metadata is an independent copy.
* The B0 predicate surface is deliberately conservative.  Direct
  `np.isfinite` and `np.isnan` calls are rejected along with other direct
  ufunc calls.  `np.isreal` is a `UnitConversionError` for the two 3-D series
  families and returns a boolean SpectrogramMatrix in the observed B0 path.
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
* RSS increase `<= max(baseline * 10%, 8 MiB)`.

These comparisons use exact `>` rejection: equality is accepted and the next
representable value above each boundary is rejected.  `adopt_candidate()`
applies the stability gate before numeric comparison, so any unstable baseline
or candidate operation is non-adoptable.

Candidate evidence also accepts a SHA-256 over an explicitly supplied,
target-relative frozen runtime-file set.  A B0 result is marked
`decision: pending` and `issue_637: not evaluated in B0`; the current frozen
slice evidence is truthfully marked unstable and
`stability_gate.adoptable: false`.  It must never claim that the #637 candidate
passed.  A clean freeze recapture is required before numeric adoption work.
