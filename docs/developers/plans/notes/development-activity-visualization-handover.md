# Development activity visualization handover

## Scope and current state

- Branch: `feature/development-activity-plot`.
- The implementation is a developer-facing CLI at
  `scripts/generate_development_activity.py`; it does not alter the GWexpy
  public API.
- Its unit tests are in `tests/test_generate_development_activity.py`.
- The first reviewed preview targets `v0.1.13`
  (`f7f836eec7e6247a01e9a1b61cc1a2121235e58d`).  It contains 1,478
  non-merge commits and its weekly CSV SHA-256 is
  `1ba9f491325d136fe5641145f4b862f8005979d56f468b3c3cb2f739982e4adb`.

## Generator contract

The generator accepts `--ref`, `--svg-output`, `--csv-output`,
`--audit-output`, and an optional `--overrides` JSON file.  It classifies each
non-merge commit at its UTC author-date, Monday-start week into exactly one
category:

1. Product development (`feat`, `refactor`, `perf`)
2. Fixes & hardening (`fix`)
3. Tests & QA (`test`)
4. Documentation & examples (`docs`)
5. Release & maintenance (`ci`, `build`, `chore`, `style`, `revert`, or the
   remaining fallback)

`[AGENT:...]` prefixes are ignored.  Conventional Commit types take priority;
legacy subject keywords and then changed paths provide deterministic fallback.
The audit CSV records the category and matching rule per commit.  Edited lines
are additions plus deletions after generated files, notebooks, gettext
catalogues, logs, caches, backups, and binary changes are excluded.

The SVG contains two weekly stacked-bar panels: commits and edited source
lines.  Bars are exactly seven days wide, so adjacent weekly bins touch and a
gap represents an inactive week.  Five colour-blind-safe colours distinguish
the categories; hatching is intentionally not used.

To preserve visibility in weeks with very uneven totals, the total bar height
uses `log10(total + 1)`, while the coloured segments retain the corresponding
raw category proportions.  Axis tick labels report raw totals and identify the
encoding as `log total; proportional stack`.  Segment boundaries therefore
show composition rather than cumulative raw values.

Every reachable stable SemVer tag is marked.  Ordinary `vX.Y.Z` labels are
vertical.  The long contextual label `v0.1.3 · GWADW 2026` is horizontal and
positioned directly above the vertical labels; the legend sits above it to
avoid a large empty gap.

## Published v0.2.2 record

The public changelog now contains a static SVG and weekly CSV generated against
the latest public stable tag, `v0.2.2`
(`2503743cf654606a5baa83c7b7e7c8b8e1e06596`). It contains 1,755 non-merge
commits and the weekly CSV SHA-256 is
`cd72102029af78b05dcb002051365092f128df341125164e4ba7c8a96d4203e3`.

The SVG and CSV are tracked under `docs_redesign/_static/`; the figure,
download link, caption, and Japanese gettext copy are in
`docs_redesign/about/changelog.md` and its catalog. Do not dynamically
aggregate Git history during a Sphinx build. The commit SHA, target tag, time
range, and weekly CSV SHA-256 are embedded in the SVG so the checked-in
visualisation remains auditable.

## Verified local checks

The implementation was checked with Python 3.13 in the `gwexpy` Conda
environment:

```console
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/gwexpy-development-activity-mpl \\
  python -m pytest -q --confcutdir=/tmp tests/test_generate_development_activity.py
ruff format --check scripts/generate_development_activity.py \\
  tests/test_generate_development_activity.py
ruff check scripts/generate_development_activity.py \\
  tests/test_generate_development_activity.py
mypy scripts/generate_development_activity.py
```

The focused test suite reported `33 passed`; Ruff and MyPy were clean.  Keep
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for this focused check because the
repository's default plugin autoload was observed to hang in this environment.
