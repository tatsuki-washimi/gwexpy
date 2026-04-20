---
myst:
  html_meta:
    description: "Understand what GWexpy validates publicly for notebooks, direct I/O formats, algorithm audit notes, and repository coverage signals, and where each evidence source lives."
---

# Verification and Quality Signals

> **Page Role:** Transparency guide

This page explains what kinds of public verification signals `gwexpy` exposes today, where those signals come from, and how to interpret their limits.

It is not a single "all features are verified" claim. Instead, it points you to the current evidence sources for notebooks, direct I/O formats, algorithm audits, and repository-level coverage reporting.

## At a Glance

| Item | Details |
| --- | --- |
| **Audience** | Users who want to judge how strongly a tutorial, format, or algorithm path is backed by public evidence |
| **Prerequisites** | None beyond basic familiarity with the user guide |
| **Use Cases** | Check how notebooks are exercised, see where I/O support is tied to tests, find algorithm audit evidence, and understand what repository coverage signals do and do not mean |
| **Search Hints** | verification, quality, coverage, notebook policy, SUPPORTED_IO_MATRIX, codecov, audit trail |

**Search hints:** verification, quality, coverage, notebook policy, SUPPORTED_IO_MATRIX, codecov, audit trail

:::{important}
**Read this page as a transparency map, not as a blanket guarantee**

Different parts of the project are verified in different ways. Some notebooks are fully executed in CI, some heavy notebooks are only structure-checked, and some optional-dependency tests can be skipped when the backend is unavailable.
:::

## Public Evidence Sources

| Area | Public source | What it tells you |
| --- | --- | --- |
| Notebook tutorials | [Notebook Policy](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/NOTEBOOK_POLICY.md) | Which notebook classes are treated as `Light`, `Heavy`, or `Display-only`, and how CI handles each class |
| Direct I/O formats | [SUPPORTED_IO_MATRIX](https://github.com/tatsuki-washimi/gwexpy/blob/main/SUPPORTED_IO_MATRIX.md) | Which public format families are tied to which tests and which backends are optional |
| Algorithm audit trail | [Validated Algorithms](validated_algorithms.md) | Numerical tolerances, assumptions, and links to audit evidence for selected high-value algorithms |
| Repository coverage signal | [README codecov badge](https://github.com/tatsuki-washimi/gwexpy) and the linked [Codecov dashboard](https://codecov.io/gh/tatsuki-washimi/gwexpy) | Where repository-level line coverage is surfaced publicly, useful as a broad signal rather than a per-feature proof |

## Notebook Validation Policy

The public notebook policy is defined in the repository's [Notebook Policy](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/NOTEBOOK_POLICY.md).

The current public model is:

- **Light** notebooks are classified for full execution and validation in CI through `papermill`.
- **Heavy** notebooks are kept in CI, but the policy describes them as `nbval --nbval-lax` checks rather than guaranteed full execution.
- **Display-only** notebooks prioritize curated outputs and are outside normal execution validation, or limited to load-style checks.
- Public tutorial notebooks under `docs/web/{en,ja}/user_guide/tutorials/` are treated as the authoritative published copies.

This is why a notebook being present in the docs is a useful signal, but not enough on its own to infer that every notebook cell is executed in every CI path.

## Direct I/O Verification Visibility

The public [SUPPORTED_IO_MATRIX](https://github.com/tatsuki-washimi/gwexpy/blob/main/SUPPORTED_IO_MATRIX.md) is the main visibility layer for direct I/O verification.

Use it when you need to answer questions such as:

- "Is this format publicly documented as supported?"
- "Which test file is meant to back this format claim?"
- "Does this route depend on an optional backend?"

The matrix is especially useful together with the [File I/O Supported Formats Guide](io_formats.md):

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
- It does **not** claim that every optional dependency is present in every test environment.
- It does **not** replace the algorithm-specific assumptions and tolerances documented on [Validated Algorithms](validated_algorithms.md).
- It does **not** turn repository-wide line coverage into a substitute for per-feature scientific validation.

## Related Pages

- [Validated Algorithms](validated_algorithms.md)
- [File I/O Supported Formats Guide](io_formats.md)
- [Troubleshooting](troubleshooting.md)

## Next to Read

- [Validated Algorithms](validated_algorithms.md) for algorithm-specific assumptions, tolerances, and audit links
- [File I/O Supported Formats Guide](io_formats.md) for direct user-facing format choice and backend notes
- [Troubleshooting](troubleshooting.md) if you need error-first guidance after checking the public verification signals
