# Full Test Environment

This project can run with a minimal dependency set, but the full test suite
expects optional Python packages and a few system-level tools. Use the
instructions below to provision a "full" environment that should run all tests
without skips.

## Python Extras

Recommended (installs test tooling and all optional Python extras):

```bash
pip install -e ".[dev,all]"
```

If you prefer to install only the pieces you need, use the explicit extras:

```bash
pip install -e ".[dev,gw,stats,fitting,astro,geophysics,audio,bio,interop,control,gui,plot,analysis]"
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

GUI tests are included by default. The test harness sets `QT_QPA_PLATFORM=offscreen`
to allow headless execution, but you can override it if you need a real display.
