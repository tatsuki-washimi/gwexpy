# scripts/

Development and maintenance scripts for the GWexpy project.
Run all scripts from the **repository root**.

## Directory structure

| Directory | Purpose |
| --- | --- |
| `ci/` | Local reproduction of GitHub Actions gates |
| `notebook_gen/` | Generate and execute Jupyter notebooks |
| `validation/` | Physics validation and numerical robustness checks |
| `dev_tools/` | Development utilities and dependency management |
| `dev/ci_logs/` | CI log analysis and retrieval utilities |
| `branding/` | Asset and branding bundle generation |
| `benchmarks/` | Performance profiling |

## ci/

Scripts used to reproduce CI gates locally.

- `scripts/ci/run_gate.py`: Run the same command sets as PR Fast and nightly gate jobs.

```bash
# I/O contract gate
python scripts/ci/run_gate.py io-contract

# I/O conformance gate
python scripts/ci/run_gate.py io-conformance

# Full PR fast validation gate
python scripts/ci/run_gate.py pr-fast
```

## Root-level utility scripts

Repository-wide checks and maintenance utilities.

| Script | Description |
| --- | --- |
| `check_docs_sync.py` | Verify that EN and JA documentation are structurally synchronized |
| `check_external_links.py` | Validate external URLs in documentation |
| `check_forbidden_artifacts.py` | Block committing generated environments and docs build artifacts |
| `check_non_ascii.py` | Detect non-ASCII characters in code and docs |
| `check_release_artifacts.py` | Validate release artifacts before publishing |
| `check_release_metadata.py` | Check version consistency across `__version__`, `pyproject.toml`, and `CITATION.cff` |
| `check_repo_hygiene.py` | Guard repository hygiene for changed files, including notebook bloat |
| `check_terms.py` | Detect discouraged or inappropriate terminology in code and docs |
| `preflight_doctor.py` | Preflight Doctor — セッション開始前の環境チェックスクリプト |
| `generate_evidence_pack.py` | Generate an Evidence Pack / Audit Manifest for a PR or task |
| `generate_hero_plot.py` | Generate hero_plot.png style assets for the GWexpy gateway landing page |
| `generate_thumbnails.py` | Generate three Visual Examples thumbnail images for the GWexpy hub pages |
| `extend_fitting_notebook.py` | Extend advanced_fitting.ipynb (EN + JA) with Lorentzian / Voigt spectral-line models |
| `fix_notebooks_warnings_v3.py` | Notebook warnings fix script v3.1 (Aggressive Reset & Clean Indentation) |
| `fix_tutorial_notebooks.py` | (用途未記載) |
| `make_bruco_advanced_notebook.py` | Generate case_bruco_advanced.ipynb (EN + JA) |
| `make_bruco_ica_notebook.py` | Generate case_bruco_ica_denoising.ipynb (EN + JA) |
| `make_peak_tracking_notebook.py` | Generate advanced_peak_tracking.ipynb (EN + JA) |
| `make_schumann_notebook.py` | Generate Schumann resonance analysis tutorial notebooks (EN + JA) |
| `make_spectrogram_processing_notebook.py` | Generate advanced_spectrogram_processing.ipynb (EN + JA) |
| `make_violin_mode_notebook.py` | Generate case_violin_mode.ipynb (EN + JA) |
| `run_quickstart_test.py` | (用途未記載) |
| `strip_example_notebook_outputs.py` | Strip transient outputs from tracked notebooks before commit |
| `update_intersphinx_inventories.py` | (用途未記載) |

## notebook_gen/

Scripts that programmatically create or run Jupyter notebooks.

| Script | Description |
| --- | --- |
| `generate_cagmon_tutorial.py` | Generate `examples/case-studies/case_cagmon_noise_diagnostics.ipynb` |
| `generate_viz_tutorial.py` | Generate the Field Visualization tutorial notebook |
| `make_scalarfield_tutorial.py` | Generate `examples/basic-new-methods/intro_ScalarField.ipynb` |
| `make_arima_burst_notebook.py` | Generate `docs/web/en/user_guide/tutorials/case_arima_burst_search.ipynb` |
| `check_changed_notebooks.py` | Run CI-style checks only for notebooks changed in the current PR |
| `prepare_docs_notebook_tree.py` | Copy the repo to a temp tree and execute docs notebooks there for docs builds |
| `run_all_notebooks.py` | Execute all notebooks under `docs/` via `nbconvert` |
| `exec_notebooks.sh` | Execute a notebook glob serially and write timestamped logs under `temp_logs/notebook_exec/` |

```bash
# CI と同じ分類規則で、PR 変更 notebook だけを検証
conda run -n gwexpy python scripts/notebook_gen/check_changed_notebooks.py --base origin/main --head HEAD

# 実行せず、対象 notebook の分類だけ確認
conda run -n gwexpy python scripts/notebook_gen/check_changed_notebooks.py --list-only

# 単純な逐次実行ログを temp_logs/notebook_exec/ に残す
bash scripts/notebook_gen/exec_notebooks.sh 'docs/web/ja/user_guide/tutorials/*.ipynb'
```

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
| `check_branding_html.py` | (用途未記載) |
| `check_og_metadata.py` | (用途未記載) |

## dev_tools/

Utilities used during development and CI setup.

| Script | Description |
| --- | --- |
| `install_minepy.py` | Build and install the `minepy` (MIC) C extension from source |
| `fix_scalarfield_notebook.py` | One-off patch for ScalarField notebook cell outputs |
| `a2_inventory_check_timeseries.py` | Diff public API against a CSV ledger; produces HTML/CSV diff reports |
| `catalog_legacy_codes.py` | (用途未記載) |
| `make_calibration_tutorial.py` | Generate Counts → Strain calibration pipeline tutorial notebooks (EN + JA) |
| `make_dttxml_tutorial.py` | Generate DTTXML calibration tutorial notebooks (EN + JA) |
| `make_fields_tutorial.py` | Generate multi-dimensional field analysis tutorial notebooks (EN + JA) |
| `make_finesse_tutorial.py` | Generate Finesse 3 interoperability tutorial notebooks (EN + JA) |
| `make_glitch_tutorial.py` | Generate glitch analysis (Q-transform/Omega-scan) tutorial notebooks (EN + JA) |
| `make_modal_tutorial.py` | Generate high-precision modal analysis tutorial notebooks (EN + JA) |
| `make_physics_validation_tutorial.py` | Generate physical validity checking tutorial notebooks (EN + JA) |
| `make_provenance_tutorial.py` | Generate HDF5 provenance / reproducible metadata tutorial notebooks (EN + JA) |
| `make_pycbc_tutorial.py` | Generate PyCBC interoperability tutorial notebooks (EN + JA) |
| `patch_notebooks.py` | (用途未記載) |
| `pin_notebook_versions.py` | (用途未記載) |

## branding/

Asset and branding bundle generation for documentation and landing pages.

| Script | Description |
| --- | --- |
| `generate_docs_branding.py` | Generate the docs branding asset bundle |

## dev/ci_logs/

CI log analysis and retrieval utilities.

| Script | Description |
| --- | --- |
| `analyze_logs.py` | (用途未記載) |
| `fetch_logs.py` | (用途未記載) |

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
