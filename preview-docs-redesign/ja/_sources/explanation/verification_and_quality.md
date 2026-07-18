---
myst:
  html_meta:
    description: "Understand what GWexpy validates publicly for notebooks, direct I/O formats, algorithm audit notes, and repository coverage signals, and where each evidence source lives."
---

# Verification and Quality Signals

This page explains what kinds of public verification signals `gwexpy` exposes today, where those signals come from, and how to interpret their limits.

It is not a single "all features are verified" claim. Instead, it points you to the current evidence sources for notebooks, direct I/O formats, algorithm audits, and repository-level coverage reporting.

:::{important}
**Read this page as a transparency map, not as a blanket guarantee**

Different parts of the project are verified in different ways. The extended verification workflow runs cross-platform import smoke tests and targeted gates (network/backend I/O, docs notebooks, Zarr I/O) on demand, some notebooks are fully executed, some heavy notebooks are only structure-checked, and some optional-dependency tests can be skipped when the backend is unavailable.
:::

## Public Evidence Sources

| Area | Public source | What it tells you |
| --- | --- | --- |
| Notebook tutorials | [Notebook Policy](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/NOTEBOOK_POLICY.md) | Which notebook classes are treated as `Light`, `Heavy`, or `Display-only`, and how CI handles each class |
| Extended verification gates | [Extended verification workflow](https://github.com/tatsuki-washimi/gwexpy/blob/main/.github/workflows/extended-verification.yml) | Which checks the on-demand extended verification workflow covers: import smoke tests on Linux/macOS/Windows plus the `io-network-backend`, `docs-notebook`, and `io-zarr` gates driven by `scripts/ci/run_gate.py` |
| Direct I/O formats | [SUPPORTED_IO_MATRIX](https://github.com/tatsuki-washimi/gwexpy/blob/main/SUPPORTED_IO_MATRIX.md) | Which public format families are tied to which tests and which backends are optional |
| Algorithm audit trail | [Validated Algorithms](validated_algorithms.md) | Numerical tolerances, assumptions, and links to audit evidence for selected high-value algorithms |
| Repository coverage signal | [README codecov badge](https://github.com/tatsuki-washimi/gwexpy) and the linked [Codecov dashboard](https://codecov.io/gh/tatsuki-washimi/gwexpy) | Where repository-level line coverage is surfaced publicly, useful as a broad signal rather than a per-feature proof |

## Notebook Validation Policy

The public notebook policy is defined in the repository's [Notebook Policy](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/NOTEBOOK_POLICY.md).

The legacy notebook policy and the redesigned website have separate build paths.

- The legacy documentation tree follows the repository Notebook Policy: its `Light`, `Heavy`, and `Display-only` classes describe the older notebook-validation regime, including its Papermill and nbval checks.
- The redesigned website builds EN and JA HTML from an isolated temporary copy. MyST-NB executes clean notebook sources through its build cache when an execution result is needed; the cache and rendered outputs are publish artifacts, not changes to tracked `.ipynb` files.
- The redesigned-site PR, preview, and production workflows use that same isolated build path. They are distinct from the legacy-docs checks in the general Docs PR workflow.

The current public model is:

- The public [extended verification workflow](https://github.com/tatsuki-washimi/gwexpy/blob/main/.github/workflows/extended-verification.yml) runs cross-platform import smoke tests and the `io-network-backend`, `docs-notebook`, and `io-zarr` gates; it is intentionally scoped to these checks and does not run docstring doctests.
- **Light**, **Heavy**, and **Display-only** describe the legacy policy. Consult that policy before inferring the execution status of a legacy notebook.
- The redesigned website is validated by its EN and JA Sphinx builds from an isolated temporary copy. Its rendered notebook outputs are produced by the MyST-NB cache and are not committed to Git.
- The on-demand extended verification workflow remains a separate, targeted check; it does not make every documentation example a release gate.

This is why a notebook or docstring example being present in the docs is a useful signal, but not enough on its own to infer that every published sample is executed in every PR, nightly, and release path.

## Current CI Coverage and Its Limits

The current public evidence supports a narrower statement than "all sample code is universally guaranteed."

- The extended verification workflow shows that `gwexpy` runs automated import smoke tests on three platforms plus targeted I/O and docs-notebook gates when the workflow is triggered.
- The [Notebook Policy](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/NOTEBOOK_POLICY.md) shows that notebook handling is class-dependent: `Light` notebooks are executed with `papermill`, while `Heavy` notebooks are checked with `nbval --nbval-lax`.
- The redesigned-site build executes notebook sources only in its isolated temporary copy and retains the results in the MyST-NB cache.

Read those signals carefully:

- They show that public examples are not unmanaged; the redesigned-site build and the legacy notebook policy exercise different scopes.
- They do **not** mean every published code block is executed in every workflow.
- They do **not** mean Doctest or notebook coverage is a single release-blocking gate for the whole documentation set.
- They do **not** remove the need to check notebook class, optional dependencies, and workflow scope before treating an example as strongly guaranteed.

## Direct I/O Verification Visibility

The public [SUPPORTED_IO_MATRIX](https://github.com/tatsuki-washimi/gwexpy/blob/main/SUPPORTED_IO_MATRIX.md) is the main visibility layer for direct I/O verification.

Use it when you need to answer questions such as:

- "Is this format publicly documented as supported?"
- "Which test file is meant to back this format claim?"
- "Does this route depend on an optional backend?"

The matrix is especially useful together with the [File I/O Supported Formats Guide](../how-to/io_formats.md):

- the user guide explains how to choose and call a public direct-I/O path,
- the matrix shows which tests are intended to back that path,
- and the notes clarify when optional dependencies can cause skips instead of hard failures.

## Coverage Signals and Their Limits

`gwexpy` publishes a repository-level coverage signal through [Codecov](https://codecov.io/gh/tatsuki-washimi/gwexpy), and the repository [README.md](https://github.com/tatsuki-washimi/gwexpy) surfaces that badge and link publicly.

Read that signal conservatively:

- it is useful for understanding overall automated test health,
- it does **not** prove that every algorithm branch, notebook, or optional-backend path is equally exercised,
- and it should be read alongside page-specific evidence such as the notebook policy, I/O matrix, and audit notes.

## What This Page Does Not Claim

- It does **not** claim that every public notebook is fully executed in every CI run.
- It does **not** claim that every docstring example or sample code block is executed in every PR, nightly, and release workflow.
- It does **not** claim that every optional dependency is present in every test environment.
- It does **not** replace the algorithm-specific assumptions and tolerances documented on [Validated Algorithms](validated_algorithms.md).
- It does **not** turn repository-wide line coverage into a substitute for per-feature scientific validation.

## Related Pages

- [Validated Algorithms](validated_algorithms.md)
- [File I/O Supported Formats Guide](../how-to/io_formats.md)
- [Troubleshooting](../how-to/troubleshooting.md)

## Next to Read

- [Validated Algorithms](validated_algorithms.md) for algorithm-specific assumptions, tolerances, and audit links
- [File I/O Supported Formats Guide](../how-to/io_formats.md) for direct user-facing format choice and backend notes
- [Troubleshooting](../how-to/troubleshooting.md) if you need error-first guidance after checking the public verification signals
