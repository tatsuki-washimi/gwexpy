# v0.2.0 local integration evidence for `[Unreleased]`

This record captures the final local verification performed on 2026-08-17.
v0.2.0 remains `[Unreleased]`, unreleased, and unpublished. Local gate results
do not establish CI completion or release approval.

Version, tag, commit, push, pull request, issue, release, and publication
operations remain pending explicit USER authorization. No such operation is
claimed or performed by this lane.

## Run environments and scope

- Main environment: Python 3.12.12, NumPy 2.3.5, Astropy 7.2.0, GWpy 4.0.1.
- Conda environment: Python 3.11.14, NumPy 1.26.4, Astropy 6.1.7,
  GWpy 4.0.1.
- The minimum dependency combo was not installed/run locally: Python 3.11,
  NumPy 1.23.2, Astropy 5, and GWpy 4.0. That compatibility result remains
  pending CI.
- Some historical root reports did not capture test counts. This record uses
  only the exact counts supplied by the fresh reruns and labels older rows as
  historical.

## Local command evidence

The following rows are the fresh 2026-08-17 verification. Each row records
the observed environment, command, and result.

### Static checks and CI-shared gates

| Environment | Command | Result |
| --- | --- | --- |
| Main Python 3.12.12 | `git diff --check` | exit 0 |
| Main Python 3.12.12 | `ruff check gwexpy/ tests/ scripts/` | exit 0 |
| Main Python 3.12.12 | `ruff format --check gwexpy/ tests/ scripts/` | exit 0; 1045 files checked |
| Conda Python 3.11.14 | `/home/washimi/miniforge3/envs/gwexpy/bin/mypy gwexpy/` | exit 0; 397 sources |
| Conda Python 3.11.14 | `PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH python scripts/ci/run_gate.py pr-fast` | exit 0; 7372 passed, 137 skipped, 28 deselected, 6 xfailed, 205 warnings; internal mypy 398 sources; real493.08s |
| Conda Python 3.11.14 | `PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH python scripts/ci/run_gate.py io-contract` | exit 0; 1397 passed, 26 skipped, 1 deselected |
| Conda Python 3.11.14 | `PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH python scripts/ci/run_gate.py io-gwf` | exit 0; 97 passed, 1 skipped |
| Main Python 3.12.12 | `PYTHONNOUSERSITE=1 PYTHONPATH=$PWD:/home/washimi/.local/lib/python3.12/site-packages python scripts/ci/run_gate.py interop-mne` | exit 0; 76 passed |

### Representative full suites and documentation checks

| Environment | Command | Result |
| --- | --- | --- |
| Conda Python 3.11.14 | `PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH python scripts/ci/run_gate.py docs-notebook` | exit 0; 4 passed, 1 pre-existing MissingIDFieldWarning; real35.13s |
| Main Python 3.12.12 | `PYTHONNOUSERSITE=1 PYTHONPATH=$PWD:/home/washimi/.local/lib/python3.12/site-packages python -m pytest -q tests/` | exit 0; 9074 passed, 194 skipped, 6 xfailed, 272 warnings; pytest time626.91s/real640.86s |
| Conda Python 3.11.14 | `PYTHONNOUSERSITE=1 PYTHONPATH=$PWD /home/washimi/miniforge3/envs/gwexpy/bin/python -m pytest -q tests/` | exit 0; 8937 passed, 275 skipped, 6 xfailed, 256 warnings; pytest time562.23s/real570.41s |
| Conda Python 3.11.14 | `python scripts/check_non_ascii.py --root gwexpy` | exit 0 |
| Conda Python 3.11.14 | `PYTHONNOUSERSITE=1 PYTHONPATH=$PWD /home/washimi/miniforge3/envs/gwexpy/bin/python -m pytest --doctest-modules -q gwexpy/` | exit 0; 99 passed, 6 warnings; real16.42s |
| Main Python 3.12.12 | `PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python -m pytest --doctest-modules -q gwexpy/` | exit 0; 99 passed, 7 warnings; real16.01s |
| Conda Python 3.11.14 | `PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH sphinx-build -b html -W --keep-going docs docs/_build/en` | exit 0; 7 nbformat DuplicateCellId warnings; real119.85s |
| Conda Python 3.11.14 | `PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH sphinx-build -b html -W --keep-going -D language=ja docs docs/_build/ja` | exit 0; 7 nbformat DuplicateCellId warnings; real127.06s |

DuplicateCellId warnings are nbformat validation warnings and did not become Sphinx -W failures. The raw doctest rows and both Sphinx rows therefore support the green local Documentation gate.

### Focused evidence checks

| Environment | Command/evidence | Result |
| --- | --- | --- |
| Conda Python 3.11.14 | focused sanity checks for `median_bias` golden/ln2, coupling Hz, and `SpectrogramMatrix` dimensional ndarray behavior | exit 0 |

## Environment-delta audit

Direct collect-only totals were 9267 main vs 9209 conda. The bounded node
delta was:

- Main-only: 84 MNE/Torch interop nodes, 4 PyCBC nodes, and 1 GPS-unit case.
- Conda-only: 30 Pint-unit cases and 1 GPS-unit case.
- Xfail counts are identical between the two collections.
- Outcome-sum minus collect-only offsets remain 7 main and 9 conda. This is a bounded residual/report-count discrepancy; this record does not claim a full explanation for it.

## Harness diagnostics

The following failed or nonrepresentative runs remain disclosed and are not
used to turn representative gate rows red:

- The initial pr-fast run failed only because ambient PATH made bare mypy resolve to main mypy 1.20.2 without types-PyYAML, although `sys.executable` was pinned. Conda has mypy 1.19.1 plus types-PyYAML 6.0.12.20250915. The PATH-pinned rerun passed. No package/code change was made.
- The initial unpinned EN/JA Sphinx runs failed because ambient PATH hid conda pandoc, nbsphinx was not enabled, and `-D nbsphinx_execute` was unknown. The PATH-pinned reruns passed. No code/config change was made.
- An additional nonrepresentative main worktree-only isolation run exited 2 at collection because sphinx was unavailable. It is not the representative main gate.

## Immutable B0 and B1 evidence

The existing JSON records are referenced without modification. Verification
did not change worktree status.

| Record | Repository path | SHA-256 | Decision recorded |
| --- | --- | --- | --- |
| B0 baseline | `docs/plans/evidence/v0.2.0-b0/series_matrix_b0.json` | `ac856b9ffab86c702cb1d66a8cae7f8a826b6928eb2119a0fbf1ad73f87da01c` | Parsed JSON records `stability_gate.adoptable=false`; B0 `slice` remains the sole unstable operation. |
| B1 candidate | `docs/plans/evidence/v0.2.0-b1/series_matrix_b1.json` | `6b1fac847052d1e814f2f5501f9eed329d876a03cf67e6f637d65acc804bbd8e` | Parsed candidate-only evidence; B1 remains `adopted: false`, and no candidate runtime adoption is asserted. |

The B1 decision packet records the candidate runtime-file SHA-256
`e5acf6ce7ce87fd1d0986c5cfa094f709cfff82fbbc2934fb7df080e1cab227f` and
preserves the approved Phase A fallback. #637 composition runtime remains
deferred while approved Phase A remains in effect.
See the [B1 decision packet](../v0.2.0-b1/series_matrix_b1_decision.md) and
[B1 completion ledger](../v0.2.0-b1/completion-ledger.md).

## Focused lane evidence present in the repository

- #400 policy documents and the three-label contract tests are present under
  `docs/developers/contracts/` and `tests/docs/`.
- #402 sidecar evidence is present in `gwexpy/io/hdf5_sidecar.py`,
  `tests/io/test_hdf5_sidecar.py`, and
  `tests/io/test_hdf5_timeseries_family.py`.
- #508 provenance evidence is present in `gwexpy/provenance.py` and
  `tests/provenance/test_provenance_contract.py`; #513 nanosecond timing
  evidence is present in `gwexpy/timeseries/timeseries.py` and
  `tests/timeseries/test_t0_gps_ns.py`.
- #409/#410 evidence is present in `gwexpy/signal/spectral/_median_mean.py`
  and `tests/signal/test_spectral_median_bias.py`.
- #411/#412 evidence is present in `gwexpy/coupling/segment.py`,
  `docs/developers/coupling_segment_schema_v1.md`, and
  `tests/coupling/test_segment.py`.
- #588 evidence is present in `gwexpy/timeseries/_gwf_io.py` and
  `tests/timeseries/test_gwf_parallel_contract.py`; #590 evidence is present
  in `gwexpy/timeseries/io/ndscope_hdf5.py` and
  `tests/timeseries/test_io_ndscope_hdf5.py`.
- #612 and #676 evidence is present in
  `tests/types/series_matrix_contract_manifest.py`,
  `tests/types/test_series_matrix_contract_manifest.py`,
  `scripts/benchmarks/series_matrix_benchmark.py`, and
  `tests/types/test_series_matrix_benchmark_contract.py`.
- The #413 documentation/evidence contract is this directory, the paired
  migration pages, the `[Unreleased]` changelog text, and the roadmap outcome.

These bullets identify repository evidence; they do not assert CI completion,
release approval, or publication.

## Post-integration review remediation

The representative full suites (main 9074 passed, 194 skipped, 6 xfailed;
conda 8937 passed, 275 skipped, 6 xfailed) were run before the final narrow
on-demand-I/O/bootstrap + migration-scope remediation; they were not rerun
afterward. Sol explicitly judged a new full-suite rerun unnecessary because
the delta was narrow and directly covered.

Post-remediation root evidence:

| Environment | Evidence | Result |
| --- | --- | --- |
| Main Python 3.12.12 | relevant broader focused suite | 320 passed, 1 skipped, 6 warnings |
| Conda Python 3.11.14 | relevant broader focused suite | 320 passed, 1 skipped, 6 warnings |
| Main Python 3.12.12 | final focused closure suite | 77 passed |
| Conda Python 3.11.14 | final focused closure suite | 77 passed |

The exact direct single-TimeSeries NDScope fresh-process diagnostic succeeded.
Ruff check/format (1045 files), full mypy gwexpy/ (397 sources), compileall,
and git diff --check passed. Terra final individual rereview PASS. Sol
FINAL_INTEGRATED_REREVIEW PASS. Local integration is ready for authorized
commit/CI handoff.

CI/minimum dependencies/publication/release authorization remain pending; no
release claim is made.

## Final gates owned by the root integration run

| Gate | Status | Evidence / remaining condition |
| --- | --- | --- |
| Local static checks | passed (local) | `git diff --check`, Ruff check/format, conda mypy, and the non-ASCII check reported exit 0 on 2026-08-17. |
| Local full test suite | passed (local) | Representative main and conda full suites reported the exact counts above, both exit 0. |
| CI-shared integration gates | passed (local) | Fresh PATH-pinned conda pr-fast, io-contract, io-gwf, and main interop-mne reported exit 0 locally. |
| Documentation gates | passed (local) | Raw doctest, docs-notebook, and PATH-pinned EN/JA Sphinx rows reported exit 0; warnings are disclosed above. |
| B0/B1 and #637 adoption decision review | passed (local) | B0/B1 hashes and the deferred `adopted: false` decision were structurally verified. |
| Minimum dependency compatibility | pending CI | Python 3.11 with NumPy 1.23.2, Astropy 5, and GWpy 4.0 was not installed/run locally. |
| Full CI matrix | pending CI | Full GitHub CI must cover the required dependency and platform combinations. |
| Release version, tag, publication, and GitHub operations | pending explicit USER authorization | No version, tag, commit, push, pull request, issue, release, or publication operation is claimed or performed autonomously by this lane. |

## Historical 2026-08-16 rows (historical only)

These rows are retained only as historical context and are not fresh
2026-08-17 gate evidence: conda `io-conformance`, conda `io-optional`, and
main `interop-contract` each previously reported exit 0. They are not included
in the current shared-gate claim above.

Local gate pass is distinct from CI completion and release approval. v0.2.0
remains `[Unreleased]`.
