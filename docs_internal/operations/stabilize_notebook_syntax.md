# Notebook Syntax Stabilization Guide

This guide documents the procedures used to stabilize Jupyter Notebook syntax and indentation, specifically for `warnings` blocks, following the major CI restoration in April 2026.

## Context
Malformed `with warnings.catch_warnings():` blocks in tutorial notebooks were causing `SyntaxError` regressions in Sphinx builds and CI checks. This was often caused by repeated automated editing or incorrect indentation of cell content.

## Maintenance Script: `fix_notebooks_warnings_v3.py`
A robust "reset-style" script is provided to normalize these blocks.

### Location
`scripts/fix_notebooks_warnings_v3.py`

### Usage
Run from the repository root:
```bash
# Dry run (default)
python3 scripts/fix_notebooks_warnings_v3.py

# Apply fixes to all notebooks under docs/web
python3 scripts/fix_notebooks_warnings_v3.py --fix
```

### What it does:
1.  Discovers `.ipynb` files in `docs/web`.
2.  Identifies cells using `warnings.catch_warnings` or related filters.
3.  **Aggressive Reset**: Removes ALL existing warnings-related lines and redundant imports.
4.  **Canonical Wrapping**: Re-wraps the entire cell content in a single, correctly indented `with` block (4 spaces).
5.  **Validation**: Ensures valid Python syntax can be parsed.

## CI Enforcement
The `.github/workflows/test.yml` includes a `Notebook syntax check` step.
- It validates all JSON structures and Python syntax in notebooks.
- It excludes `_build` artifacts to prevent false positives.
- Failure in this step will block the PR.

## Dependency Management
If a new scientific dependency is required for a tutorial:
1.  Add it to `requirements-dev.txt`.
2.  Update `test.yml` (micromamba setup), `docs.yml`, and `notebook-tests.yml`.
3.  Verify the `lint` job passes in CI.
