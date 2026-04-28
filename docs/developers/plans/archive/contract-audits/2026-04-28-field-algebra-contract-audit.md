# Field Algebra Contract Audit

Date: 2026-04-28
Issue: #287, "Audit field algebra contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This slice records a narrow baseline for `gwexpy.fields.ScalarField` binary
arithmetic. It does not change runtime behavior.

Reviewed inputs:

- `gwexpy/fields/base.py` domain metadata handling and unit validation.
- `gwexpy/fields/scalar.py` `ScalarField` construction and metadata
  propagation paths.
- Existing field-domain, unit, and space-transform contract tests under
  `tests/fields/`.
- Prior field-space audit notes in `docs/developers/plans/`.

## Current Observed Behavior

- Aligned same-shape `ScalarField` addition with compatible value units returns
  a `ScalarField`, converts compatible units through the inherited array
  arithmetic path, and preserves the left-hand field's `axis0_domain`,
  `space_domains`, axes, and unit.
- Same-shape fields with different `x` coordinate grids currently add without
  a field-level coordinate check. The result inherits the left-hand field's
  public axis metadata.
- Same-shape fields with different axis0 domains, for example `time` versus
  `frequency`, currently add when the data units permit arithmetic. The result
  inherits the left-hand field's axis0 domain and axis names.

The two mismatch cases are recorded with explicit `pytest.xfail()` calls only
for the current unsafe behavior where arithmetic succeeds. If runtime code later
raises a matching `ValueError`, those tests will pass normally and force the
temporary xfail branch to be removed or updated. Unexpected exception types or
non-contract error messages remain ordinary test failures.

## Why Runtime Change Is Deferred

Field algebra touches physics-sensitive metadata contracts: coordinate grid
alignment, time/frequency-domain separation, unit conversion, and compatibility
with NumPy/GWpy array arithmetic. A runtime guard should be reviewed by a human
before deciding:

- which binary operations require identical field axes;
- whether equivalent but differently represented axes may be accepted;
- how error messages should distinguish value-unit failures from metadata
  alignment failures;
- how much behavior may diverge from inherited GWpy/NumPy semantics.

This slice therefore adds only tests and audit documentation.

## Follow-Up Candidates

1. Define the public binary-operation alignment contract for `ScalarField`,
   including axis names, axis coordinates, `axis0_domain`, and `space_domains`.
2. Decide whether coordinate comparison should require exact equality or
   quantity-aware tolerances.
3. Extend the same contract review to `VectorField`, `TensorField`,
   `FieldList`, and `FieldDict` arithmetic paths.
4. Add runtime guards after physics review and replace the strict xfail tests
   with passing regression tests.

## Verification

Required focused commands for this slice:

```bash
rtk env MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp pytest -q tests/fields/test_field_algebra_contracts.py -p no:cacheprovider
rtk env MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp pytest -q tests/fields/test_field_algebra_contracts.py tests/fields/test_scalarfield_domain.py tests/fields/test_space_transform_contracts.py -p no:cacheprovider
rtk ruff check tests/fields/test_field_algebra_contracts.py
rtk ruff format --check tests/fields/test_field_algebra_contracts.py
rtk env MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp python -c "import yaml; yaml.safe_load(open('docs/developers/plans/audit-manifest-287-field-algebra.yaml'))"
rtk git diff --check
```
