Timestamp: 2026-01-28 14:15:42 +0900

# GWExPy Documentation Fixes & Quality Improvement Plan

This document updates the earlier Japanese plan and incorporates the current execution strategy (Coming Soon for English-only gaps) plus recommended model usage, time, and quota estimates.

## Goals

- Remove user-irrelevant warning spam from published docs (notebook outputs).
- Make English tutorial pages honest and usable: clearly marked as (Coming Soon) with working links to Japanese rendered pages.
- Keep developer-facing language mixing as a lower-priority, optional cleanup.

## Scope

### A) Silence user-irrelevant warnings in tutorials (High priority)

1) ARIMA tutorial (sklearn/pmdarima-related FutureWarning spam)
- Fix options:
  - Notebook-level: add `warnings.filterwarnings(...)` scoped to sklearn/pmdarima.
  - Library-level: adjust the pmdarima call sites to avoid triggering deprecated code paths.
- Acceptance: notebook output no longer contains repeated sklearn/pmdarima FutureWarnings.

2) TimeSeries interop / new features (TensorFlow / Protobuf warnings)
- Notebook-level: set `TF_CPP_MIN_LOG_LEVEL=3` and filter protobuf-related warnings explicitly.
- Acceptance: notebooks do not print long Protobuf/gencode warning blocks.

3) FrequencySeries advanced features (control library deprecation warning)
- Library-level: switch from deprecated `fresp` to `frdata` (or suppress narrowly, if needed).
- Acceptance: documentation build output contains no `fresp` deprecation warnings.

### B) English tutorial pages: Coming Soon + clean redirect (High priority)

Current state: multiple `docs/web/en/guide/tutorials/*.md` pages contain only a Japanese redirect note, and also mix English/Japanese in the same page.

Plan:
- For each English stub page:
  - Replace mixed-language stub with an English-only “Coming Soon” page.
  - Provide two links:
    - Rendered Japanese HTML page (preferred for readers).
    - Japanese notebook `.ipynb` (for download / local execution).
- Update `docs/web/en/guide/tutorials/index.rst` to visibly mark these items as “(Coming Soon)” in the sidebar/toctree.

Acceptance:
- English docs no longer look “empty” or misleading.
- All links from English stubs resolve (no broken relative links).
- `sphinx-build -nW` passes and `linkcheck` does not report these internal links as broken.

### C) Language mixing under Developers Guide (Optional)

- Keep as a later cleanup task unless external-facing consistency becomes a requirement.

## Implementation Steps

1) Identify targets
- List English stub tutorial pages under `docs/web/en/guide/tutorials/`.
- Map each stub to its Japanese counterpart under `docs/web/ja/guide/tutorials/`.

2) Implement Coming Soon pages (batch edit)
- Replace content of the English stub pages with a single consistent template.
- Ensure links point to the correct built path (HTML) and to the `.ipynb`.

3) Update English tutorials index
- Edit `docs/web/en/guide/tutorials/index.rst`:
  - Use `Title (Coming Soon) <page>` entries.
  - Add a short note near the top explaining the status.

4) Address warning spam in notebooks
- Add notebook-level warning filters in the relevant `.ipynb` cells or an early “setup” cell.
- Prefer narrowly-scoped filters (module/category) over global suppression.

5) Fix deprecated control API usage
- Update gwexpy code paths using deprecated `fresp` to use `frdata`.

6) Validation gates
- Build docs:
  - `sphinx-build -nW -b html docs docs/_build/html`
  - `sphinx-build -b linkcheck docs docs/_build/linkcheck`
- Run a minimal notebook execution gate for the modified tutorials (if CI has a notebook job).

## Recommended Models and Execution Strategy

Primary goal is fast, correct batch edits + build/test loops.

1) Planning + target discovery
- Model: `openai/gpt-5.2`
- Quota: Low
- Time: 5-10 min

2) Batch edits (Coming Soon pages + toctree)
- Model: `GPT-5.1-Codex-Mini` (cost-efficient) or `openai/gpt-5.2` (single-model workflow)
- Quota: Medium
- Time: 10-20 min

3) Docs build + linkcheck loop
- Model: `openai/gpt-5.2`
- Quota: Low-Medium
- Time: 10-20 min

4) Optional English copy review
- Model: `Claude Sonnet 4.5`
- Quota: Low
- Time: 3-8 min

Estimated total:
- Wall-clock: 25-55 min
- Quota: Medium

## Risks / Notes

- Relative links: ensure `en/...` to `ja/...` uses correct `../../../` depth and targets rendered `.html`.
- `linkcheck` can be slow or flaky due to external links; prefer internal-link validation first.
- Warning suppression should be scoped to avoid hiding actionable user warnings.
