# scripts/

Development and maintenance scripts for the GWexpy project.
Run all scripts from the **repository root**.

## Directory structure

| Directory | Purpose |
| --- | --- |
| `notebook_gen/` | Generate and execute Jupyter notebooks |
| `validation/` | Physics validation and numerical robustness checks |
| `dev_tools/` | Development utilities and dependency management |
| `benchmarks/` | Performance profiling |

## notebook_gen/

Scripts that programmatically create or run Jupyter notebooks.

| Script | Description |
| --- | --- |
| `generate_cagmon_tutorial.py` | Generate `examples/case-studies/case_cagmon_noise_diagnostics.ipynb` |
| `generate_viz_tutorial.py` | Generate the Field Visualization tutorial notebook |
| `make_scalarfield_tutorial.py` | Generate `examples/basic-new-methods/intro_ScalarField.ipynb` |
| `make_arima_burst_notebook.py` | Generate `docs/web/en/user_guide/tutorials/case_arima_burst_search.ipynb` |
| `run_all_notebooks.py` | Execute all notebooks under `docs/` via `nbconvert` |

## validation/

Scripts that verify physical correctness and numerical stability of gwexpy computations.
These are not tests (not run by pytest) but standalone sanity checks.

| Script | Description |
| --- | --- |
| `verify_scalarfield_physics.py` | Check ScalarField physical units and metadata consistency |
| `verify_scalarfield_noise.py` | Verify noise injection and ScalarField round-trip |
| `verify_spectral_density_physics.py` | Validate PSD/ASD normalization conventions |
| `verify_timeseries_attrs.py` | Confirm TimeSeries attribute preservation after operations |
| `audit_numerical_risks.py` | Scan source code for division-by-zero and NaN risks |
| `validate_io_improvements.py` | Validate I/O reader type annotations and Path support |

## dev_tools/

Utilities used during development and CI setup.

| Script | Description |
| --- | --- |
| `install_minepy.py` | Build and install the `minepy` (MIC) C extension from source |
| `fix_scalarfield_notebook.py` | One-off patch for ScalarField notebook cell outputs |
| `a2_inventory_check_timeseries.py` | Diff public API against a CSV ledger; produces HTML/CSV diff reports |

## benchmarks/

Performance measurement scripts.

| Script | Description |
| --- | --- |
| `bruco_bench.py` | Benchmark three coherence-ranking implementations (gwexpy vs gwpy vs naive) |
| `benchmark_fields.py` | Profile ScalarField and related field operations |

```bash
# Example: run coherence benchmark
python scripts/benchmarks/bruco_bench.py --n-bins 20000 --n-channels 300

# Example: validate spectral density physics
python scripts/validation/verify_spectral_density_physics.py
```
