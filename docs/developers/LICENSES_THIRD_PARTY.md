# Third-Party Code and License Policy

GWexpy takes design ideas, product semantics, and API conventions from a number of
neighbouring projects. It does **not** copy their code. This document records the licence of
each project GWexpy studies or bridges to, how that licence was verified, and what
contributors may and may not do with it.

The user-facing summary of the same relationships lives in
[`docs_redesign/explanation/ecosystem.md`](../../docs_redesign/explanation/ecosystem.md).

> Last verified: 2026-08-01.

## How Licences Are Verified

1. **Read the LICENSE file in the project's own distribution.** For a repository, fetch the
   `LICENSE` / `LICENSE.txt` blob directly. For a PyPI package, download the sdist and read the
   bundled licence file.
2. **Do not rely on secondary sources.** PyPI trove classifiers, README badges, GitHub's
   detected-licence label, and third-party summaries are hints for where to look, not evidence.
3. **No LICENSE file means all rights reserved.** Absence of a licence grant is *more*
   restrictive than GPL, not less. Nothing may be reused from such a project — not code, not
   file structure, not a transcribed algorithm implementation.
4. **If the licence cannot be confirmed, write "unverified".** Do not guess, and do not carry a
   previous document's claim forward without re-checking it.

## Per-Project Policy

| Project | Licence | How it was verified | Policy for GWexpy |
| --- | --- | --- | --- |
| [GWpy](https://github.com/gwpy/gwpy) | **GPL-3.0** | `LICENSE` at `gwpy/gwpy@main` — "GNU GENERAL PUBLIC LICENSE, Version 3" | Runtime dependency and base class provider. GWexpy subclasses its public API. No source is vendored. |
| [pemcoupling](https://github.com/pdqnguyen/pemcoupling) | **No LICENSE file → all rights reserved** | GitHub API reports `license: NONE` on the default branch (`master`); the repository root contains no `LICENSE` and no `setup.py` | **No reuse of any kind**: no code, no file structure, no transcribed implementation. Domain concepts only (coupling-product schema, measurement status flags), independently redesigned. |
| [GWDama](https://gwnoisehunt.gitlab.io/gwdama/) | **MIT** | `LICENSE` bundled in the `gwdama-0.6.0` sdist from PyPI — "MIT License, Copyright (c) 2021 gwnoisehunt" | Legally reusable with attribution, but the design decision is to stay at a thin interop / reader layer. No code copy. GWexpy reads the HDF5 *format* with h5py and does not depend on the `gwdama` package. |
| [spicypy](https://gitlab.com/pyda-group/spicypy) | **Apache-2.0** | `LICENSE.txt` at `pyda-group/spicypy@main` — "Apache License, Version 2.0, January 2004" | Reference only: API design, method naming, and documentation structure. No code copy, no adapter module. |
| [Differometor](https://github.com/artificial-scientist-lab/Differometor) | **MIT** | `LICENSE` at `artificial-scientist-lab/Differometor@main` — "MIT License, Copyright (c) 2025 Artificial Scientist Lab" | Thin conversion adapters only. The optimiser and the simulator are not reimplemented, and the package is not imported by GWexpy. |

### Note on the pemcoupling licence

An earlier internal description recorded pemcoupling as GPLv3, apparently from a `setup.py`
metadata field in a downloaded snapshot of the LIGO GitLab copy. The public GitHub mirror
carries no licence grant at all. Because the two sources disagree and the more permissive claim
cannot be confirmed from a licence file, GWexpy applies the stricter reading: **all rights
reserved**. This is a deliberately conservative position, not a legal determination.

## Rules for Contributors

- Do not vendor, copy, or transcribe source from any project in the table above, regardless of
  its licence. Where GWexpy needs equivalent behaviour, implement it from the published
  description or from first principles.
- Do not add a dependency on a project whose licence has not been verified by the procedure at
  the top of this page.
- When a plan, changelog entry, or design note states a third-party licence, verify it at the
  time of writing and record how. A licence that was correct two years ago may have changed.
- Prefer **format compatibility** over **package dependency** when bridging to another tool.
  Reading a documented on-disk format with an existing base dependency keeps GWexpy's
  dependency surface small and avoids inheriting the other project's dependency tree.

## Related Documents

- [`docs_redesign/explanation/ecosystem.md`](../../docs_redesign/explanation/ecosystem.md) — user-facing positioning and ecosystem map
- [`docs_redesign/how-to/interop.md`](../../docs_redesign/how-to/interop.md) — the conversion API catalogue
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — contribution workflow, including the vendoring prohibition
- [`ROADMAP.md`](../../ROADMAP.md) — ecosystem and interoperability backlog
