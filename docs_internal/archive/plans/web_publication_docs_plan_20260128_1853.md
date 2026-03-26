2026-01-28T18:53:17+09:00

# Web Docs Publication Readiness Plan (GWexpy)

This document is a working plan to improve GWexpy documentation quality for public web publishing, with an explicit language-selection gateway, clearer navigation, and higher-fidelity API docs.

## Assumptions

- Branding: user-facing name is "GWexpy" (package/import name remains `gwexpy`).
- Distribution: PyPI is NOT available yet; official install guidance is via GitHub or local checkout.
- Docs are built with Sphinx from `docs/`.

## Definition of Done

- `docs/index.rst` is a clear language-selection gateway (no placeholder text; no confusing mixed-language messaging).
- Side navigation no longer presents duplicated/competing "API" vs "Detailed" reference entry points.
- "No documentation available" is eliminated for the key reported methods/properties.
- Installation pages explicitly state "PyPI not available" and lead with GitHub install.
- Branding is consistently "GWexpy" in user-facing docs pages.
- Docs build succeeds and QA checks pass (Sphinx build + Ruff + MyPy).

## Task List (with progress checks)

### P0: Landing Page (Gateway)

- [x] Update `docs/index.rst` to be an explicit language-selection gateway (replace placeholder, add brief overview in EN/JA, keep a single H1).
  - Recommended model: `openai/gpt-5.2`
- [x] Align site title/branding in `docs/conf.py` so the header/title displays "GWexpy" consistently (without changing package name `gwexpy`).
  - Recommended model: `openai/gpt-5.2`


### P0: Navigation & Structure (Remove Confusion)

- [x] Remove the duplicated "Reference" entry points from `docs/web/en/index.rst` and `docs/web/ja/index.rst` (avoid listing both `reference/api/index` and `reference/index` at the same level).
  - Recommended model: `openai/gpt-5.2`
- [x] Reframe the current detailed reference page as a clear "Class Index" page.
  - Target files: `docs/web/en/reference/index.rst`, `docs/web/ja/reference/index.rst`
  - Recommended model: `openai/gpt-5.2`
- [x] Unify sidebar naming policy on the Japanese site: use Japanese labels with class-name hints, e.g. `Jishikei (TimeSeries)` style for toctree entries (display names), while keeping class identifiers as-is.
  - Recommended model: `openai/gpt-5.2`

### P1: API Docstrings (Eliminate "No documentation available")

- [x] Add a docstring to `Plot.plot_mmm()`.
  - Target file: `gwexpy/plot/plot.py`
  - Recommended model: `openai/gpt-5.2`
- [x] Add a docstring to the statistics `median()` method.
  - Target file: `gwexpy/types/_stats.py`
  - Recommended model: `openai/gpt-5.2`
- [x] Add a docstring to `SeriesMatrixCoreMixin.xunit` explaining precedence (dx -> xindex -> x0).
  - Target file: `gwexpy/types/series_matrix_core.py`
  - Recommended model: `openai/gpt-5.2`
- [x] Improve `Spectrogram.imshow` / `Spectrogram.pcolormesh` docstrings: include key arguments briefly + link to GWpy for full details.
  - Target file: `gwexpy/spectrogram/spectrogram.py`
  - Recommended model: `openai/gpt-5.2`

### P1: Installation Guidance (No PyPI)

- [x] Update installation pages to explicitly say "PyPI not available yet" and lead with GitHub install.
  - Target files: `docs/web/en/guide/installation.rst`, `docs/web/ja/guide/installation.rst`
  - Recommended model: `openai/gpt-5.2`
- [x] Remove or adjust any PyPI-oriented cues that can mislead users.
  - Candidate: `README.md` PyPI badge and install snippet.
  - Recommended model: `openai/gpt-5.2`

### P2: Branding Consistency (GWexpy)

- [x] Replace user-facing "GWExPy" with "GWexpy" across docs landing pages and guides.
  - Target files (minimum): `docs/index.rst`, `docs/web/en/index.rst`, `docs/web/ja/index.rst`, `docs/web/en/guide/tutorials/index.rst`, `docs/web/ja/guide/tutorials/index.rst`, `docs/web/en/guide/installation.rst`, `docs/web/ja/guide/installation.rst`
  - Recommended model: `openai/gpt-5.2`

### P2: Tutorials Download Messaging

- [x] Adjust tutorial index notes so they match actual UI behavior (currently the top-right link is "Edit on GitHub" for `.ipynb`, not a direct download button).
  - Target file: `docs/web/ja/guide/tutorials/index.rst`
  - Recommended model: `openai/gpt-5.2`
- [x] (Optional) Add a consistent "Download notebook" link on notebook pages via Sphinx/nbsphinx configuration if desired.
  - Target file: `docs/conf.py`
  - Recommended model: `openai/gpt-5.2`

## Verification Checklist

- [x] Build docs locally and review key pages:
  - `docs/index.rst` gateway
  - `docs/web/ja/index.rst` and `docs/web/en/index.rst` nav
  - Reference sidebar structure (API vs Class Index)
  - API pages for the fixed docstrings
  - Recommended model: `openai/gpt-5.2`
- [x] Run Ruff.
  - Recommended model: `openai/gpt-5.2`
- [x] Run MyPy.
  - Recommended model: `openai/gpt-5.2`

## Notes

- Some "type-hint-looking" signatures (e.g., `epoch: float | int | None`) appear because Sphinx is rendering annotated attributes/protocols; fix only if it materially harms readability, otherwise treat as acceptable.
