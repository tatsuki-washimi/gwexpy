# Full Test Environment

This project can run with a minimal dependency set, but broader test coverage
uses optional Python packages and some system-level tools. The environment
below installs the development tooling and non-GUI Python extras; tests that
need unavailable system tools or external services may still be skipped.

## Python Extras

Recommended (installs test tooling and all non-GUI optional Python extras):

```bash
pip install -e ".[dev,all]"
```

Extras responsibilities: `dev` provides test/lint/type tooling, and `all`
provides the optional Python features supported by this repository. GUI code
and GUI behavior tests are maintained in the standalone `pyaggui` project.

If you prefer to install only the pieces you need, use the explicit extras:

```bash
pip install -e ".[dev,gw,analysis,fitting,seismic,audio,io,netcdf4,zarr,plotting,control]"
```

## System-Level Dependencies

Some tests require tools that are not pure Python packages:

- GMT runtime (for `pygmt`)
- Framecpp toolchain (for `ldas-tools-framecpp` GWF support)
- Kerberos CLI tools (`kinit`, `klist`) for authentication workflows

Additional test-only Python tooling used by upstream GWpy tests:

- `freezegun` (time-freezing markers for deterministic time tests)
- `requests-mock` (provides the `requests_mock` fixture for HTTP tests)

The exact package names vary by platform. Typical options are:

- Conda-forge: `gmt`, `ldas-tools-framecpp`, `python-framel`, `krb5`
- Debian/Ubuntu: `gmt`, `krb5-user`
- RHEL/CentOS: `gmt`, `krb5-workstation`
- macOS (Homebrew): `gmt`, `krb5`

## Running the Full Suite

```bash
python -m pytest
```

## Reproducing CI Gates Locally

Gate-equivalent commands are centralized in `scripts/ci/run_gate.py` and are kept in sync with CI jobs.
The `ci_jobs=base` contract mapping currently points to `python scripts/ci/run_gate.py io-conformance`; optional logical jobs stay as separate future mappings unless they are already documented elsewhere.

```bash
python scripts/ci/run_gate.py pr-fast
python scripts/ci/run_gate.py io-contract
python scripts/ci/run_gate.py io-conformance
python scripts/ci/run_gate.py io-optional
python scripts/ci/run_gate.py io-network-backend
python scripts/ci/run_gate.py docs-notebook
python scripts/ci/run_gate.py io-zarr
python scripts/ci/run_gate.py interop-contract
```

Each gate runs the same command set as the corresponding workflow job, so use these for local reproduction before opening a PR.

GWexpy retains NDS-related library I/O for offline scientific analysis. This repository does not provide a live-NDS workflow, and its test suite does not certify connectivity to remote NDS services. Live connectivity depends on the user's environment and is not part of release qualification.

## GUI test boundary

GWexpy no longer contains the legacy GUI implementation, GUI behavior tests,
or their test runner and headless hooks. GUI development and display-dependent
testing are maintained by the standalone
[`pyaggui` project](https://github.com/tatsuki-washimi/pyaggui); follow that
project's current contributor instructions. GWexpy's `tests/test_gui_boundary.py`
checks the repository boundary and does not run or certify GUI behavior.
