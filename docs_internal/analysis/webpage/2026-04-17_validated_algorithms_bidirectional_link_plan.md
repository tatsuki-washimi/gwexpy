# Validated Algorithms / API Bi-directional Links Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen two-way navigation between `validated_algorithms` and the concrete API reference pages, and replace weak validation-evidence wording with verifiable, existing evidence links.

**Architecture:** Keep the fix inside existing pages only. Add stable section labels and evidence tables to `validated_algorithms.md`, then point API category pages and a small set of high-value reference pages back to those exact sections using explicit cross-references rather than generic page-level links. Finish by updating the audit report so sections `12-3` and `12-10` reflect the implemented state.

**Tech Stack:** MyST Markdown, Sphinx/reST, autosummary-generated API docs, `conda run -n gwexpy` for docs verification

---

## File Map

**Modify:**
- `docs/web/ja/user_guide/validated_algorithms.md`
- `docs/web/en/user_guide/validated_algorithms.md`
- `docs/web/ja/reference/api/fields.rst`
- `docs/web/en/reference/api/fields.rst`
- `docs/web/ja/reference/api/timeseries.rst`
- `docs/web/en/reference/api/timeseries.rst`
- `docs/web/ja/reference/api/spectral.rst`
- `docs/web/en/reference/api/spectral.rst`
- `docs/web/ja/reference/api/fitting.rst`
- `docs/web/en/reference/api/fitting.rst`
- `docs/web/ja/reference/api/preprocessing.rst`
- `docs/web/en/reference/api/preprocessing.rst`
- `docs/web/ja/reference/ScalarField.md`
- `docs/web/en/reference/ScalarField.md`
- `docs/web/ja/reference/TimeSeries.md`
- `docs/web/en/reference/TimeSeries.md`
- `docs/web/ja/reference/Spectral.md`
- `docs/web/en/reference/Spectral.md`
- `docs/web/ja/reference/fitting.md`
- `docs/web/en/reference/fitting.md`
- `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md`

**Read for evidence only:**
- `docs/developers/algorithms/ALGORITHM_CONTEXT.md`
- `docs/developers/algorithms/merged_validation_report.md`
- `docs/developers/algorithms/algorithm_fix_report_20260201.md`
- `docs_internal/tech_notes/scalarfield_physics_review_20260120.md`
- `docs_internal/archive/reports/report_scalarfield_physics_verification_20260122_222000.md`

## Design Constraints

- Do not add a new public page for validation evidence.
- Do not keep references to non-existent `docs_internal/verification/`.
- Do not keep unverified claims about a `verify-physics` CI gate.
- Prefer explicit labels in MyST and `:ref:` links in reST over brittle raw `#heading-slug` links.
- Keep Japanese and English structures aligned.

## Section Labels To Add

Add explicit MyST labels before the corresponding headings in both `validated_algorithms.md` files:

```md
(validated-objective-evidence)=
## Objective Evidence

(validated-k-space)=
### 1. k-space Calculation

(validated-transient-fft)=
### 2. Transient FFT

(validated-vif)=
### 3. VIF / Overlap Correction

(validated-arima-forecast)=
### 4. Forecast Timestamp (ARIMA)

(validated-mcmc-gls)=
### 5. MCMC / GLS Likelihood

(validated-adaptive-whitening)=
### 6. Adaptive Whitening
```

These labels are the backbone of the reverse-link design. API pages should point to labels, not just to the top of `validated_algorithms`.

### Task 1: Rebuild `validated_algorithms` As The Canonical Mapping Page

**Files:**
- Modify: `docs/web/ja/user_guide/validated_algorithms.md`
- Modify: `docs/web/en/user_guide/validated_algorithms.md`

- [ ] **Step 1: Add stable section labels for every validated algorithm and for the evidence section**

Use MyST labels so API pages can deep-link safely.

```md
(validated-k-space)=
### 1. k-space Calculation
```

- [ ] **Step 2: Expand the summary table to include direct API and evidence columns**

Replace the current broad table with a concrete mapping table shaped like this:

```md
| Algorithm | Primary API | Related API Page | Evidence |
| --- | --- | --- | --- |
| k-space calculation | `ScalarField.fft_space()` | `{doc}`../reference/api/fields`` | `{ref}`validated-objective-evidence`` |
```

Expected outcome:
- Each row names a concrete public API.
- The table gives the user one click to API and one click to evidence.

- [ ] **Step 3: Add a short `API links` block to each algorithm section**

At the end of each algorithm section, add a consistent block like:

```md
**Related API pages**
- {doc}`../reference/api/fields`
- {doc}`../reference/ScalarField`
```

Minimum mapping:
- `k-space` -> `api/fields.rst`, `ScalarField.md`
- `Transient FFT` -> `api/timeseries.rst`, `TimeSeries.md`
- `VIF / overlap correction` -> `api/spectral.rst`, `Spectral.md`
- `Forecast timestamp (ARIMA)` -> `api/timeseries.rst`, `TimeSeries.md`
- `MCMC / GLS likelihood` -> `api/fitting.rst`, `reference/fitting.md`
- `Adaptive whitening` -> `api/preprocessing.rst`

- [ ] **Step 4: Replace the weak Audit Trail wording with an `Objective Evidence` section**

Add a visible section near the top or immediately before `Audit Trail`:

```md
## Objective Evidence

| Evidence type | Source | What it shows |
| --- | --- | --- |
| Audit scope | `ALGORITHM_CONTEXT.md` | Which algorithms were reviewed |
| Consolidated findings | `merged_validation_report.md` | Cross-reviewed findings and severity |
| Fix history | `algorithm_fix_report_20260201.md` | What was changed after review |
```

Use existing GitHub blob links for `docs/developers/algorithms/*` because those files are real and already public in the repository.

- [ ] **Step 5: Remove dead or unverified claims**

Delete or rewrite:
- `docs_internal/verification/` references
- `verify-physics` gate wording

Replace with language grounded in existing artifacts, for example:

```md
The evidence below points to the audit scope, merged review findings, and the documented fix history for the algorithms summarized on this page.
```

- [ ] **Step 6: Verify source-level cleanup**

Run:

```bash
rg -n "docs_internal/verification|verify-physics" docs/web/ja/user_guide/validated_algorithms.md docs/web/en/user_guide/validated_algorithms.md
```

Expected:
- No matches

- [ ] **Step 7: Commit**

```bash
git add docs/web/ja/user_guide/validated_algorithms.md docs/web/en/user_guide/validated_algorithms.md
git commit -m "docs: strengthen validated algorithm evidence and api mappings"
```

### Task 2: Add Reverse Links From API Category Pages To Exact Validation Sections

**Files:**
- Modify: `docs/web/ja/reference/api/fields.rst`
- Modify: `docs/web/en/reference/api/fields.rst`
- Modify: `docs/web/ja/reference/api/timeseries.rst`
- Modify: `docs/web/en/reference/api/timeseries.rst`
- Modify: `docs/web/ja/reference/api/spectral.rst`
- Modify: `docs/web/en/reference/api/spectral.rst`
- Modify: `docs/web/ja/reference/api/fitting.rst`
- Modify: `docs/web/en/reference/api/fitting.rst`
- Modify: `docs/web/ja/reference/api/preprocessing.rst`
- Modify: `docs/web/en/reference/api/preprocessing.rst`

- [ ] **Step 1: Add a `Validation links` or `See also` block to every target API category page**

Example for reST:

```rst
.. seealso::

   :ref:`validated-k-space`
      Validation assumptions and evidence for `ScalarField.fft_space()`.
```

Required mappings:
- `fields.rst` -> `validated-k-space`
- `timeseries.rst` -> `validated-transient-fft`, `validated-arima-forecast`
- `spectral.rst` -> `validated-vif`
- `fitting.rst` -> `validated-mcmc-gls`
- `preprocessing.rst` -> `validated-adaptive-whitening`

- [ ] **Step 2: Keep the existing general `Validated Algorithms` page link, but demote it below the exact links**

The exact section links should come first. The generic page link can remain as the fallback overview.

- [ ] **Step 3: Make the compatibility stub `api/fitting.rst` useful**

Because `api/fitting.rst` currently only redirects to `../fitting`, add a small block that says which validated section applies before the redirect line.

- [ ] **Step 4: Build docs and confirm API pages now contain reverse links**

Run:

```bash
conda run -n gwexpy sphinx-build -b html docs /tmp/gwexpy-docs-validated-links
rg -n "validated_algorithms.html|Objective Evidence|K-space|Transient FFT|GLS" /tmp/gwexpy-docs-validated-links/web/en/reference/api /tmp/gwexpy-docs-validated-links/web/ja/reference/api
```

Expected:
- Sphinx build succeeds
- Built API pages contain links or text pointing back to the exact validation context

- [ ] **Step 5: Commit**

```bash
git add docs/web/ja/reference/api/fields.rst docs/web/en/reference/api/fields.rst docs/web/ja/reference/api/timeseries.rst docs/web/en/reference/api/timeseries.rst docs/web/ja/reference/api/spectral.rst docs/web/en/reference/api/spectral.rst docs/web/ja/reference/api/fitting.rst docs/web/en/reference/api/fitting.rst docs/web/ja/reference/api/preprocessing.rst docs/web/en/reference/api/preprocessing.rst
git commit -m "docs(api): add reverse links to validated algorithm sections"
```

### Task 3: Tighten Deep Links In High-Value Reference Pages

**Files:**
- Modify: `docs/web/ja/reference/ScalarField.md`
- Modify: `docs/web/en/reference/ScalarField.md`
- Modify: `docs/web/ja/reference/TimeSeries.md`
- Modify: `docs/web/en/reference/TimeSeries.md`
- Modify: `docs/web/ja/reference/Spectral.md`
- Modify: `docs/web/en/reference/Spectral.md`
- Modify: `docs/web/ja/reference/fitting.md`
- Modify: `docs/web/en/reference/fitting.md`

- [ ] **Step 1: Replace generic `Validated Algorithms` bullets with section-specific links where possible**

Examples:

```md
- {ref}`validated-k-space`
- {ref}`validated-transient-fft`
- {ref}`validated-vif`
- {ref}`validated-mcmc-gls`
```

Use only the links relevant to the page.

- [ ] **Step 2: Add a one-line explanation for why each validation link matters**

Example for `ScalarField.md`:

```md
- {ref}`validated-k-space` - Assumptions and evidence for `ScalarField.fft_space()`
```

- [ ] **Step 3: Keep related theory pages (`Physics Models`, `FFT_Conventions`) intact**

Do not remove those links. The fix is to add precision, not to narrow the reference surface too aggressively.

- [ ] **Step 4: Rebuild and inspect the specific high-value pages**

Run:

```bash
conda run -n gwexpy sphinx-build -b html docs /tmp/gwexpy-docs-validated-links
rg -n "validated_algorithms.html|K-space|Transient FFT|Variance Inflation|GLS" /tmp/gwexpy-docs-validated-links/web/en/reference/ScalarField.html /tmp/gwexpy-docs-validated-links/web/en/reference/TimeSeries.html /tmp/gwexpy-docs-validated-links/web/en/reference/Spectral.html /tmp/gwexpy-docs-validated-links/web/en/reference/fitting.html
```

Expected:
- The built pages show specific validation links, not only the generic overview page

- [ ] **Step 5: Commit**

```bash
git add docs/web/ja/reference/ScalarField.md docs/web/en/reference/ScalarField.md docs/web/ja/reference/TimeSeries.md docs/web/en/reference/TimeSeries.md docs/web/ja/reference/Spectral.md docs/web/en/reference/Spectral.md docs/web/ja/reference/fitting.md docs/web/en/reference/fitting.md
git commit -m "docs(reference): deep-link validation sections from key api pages"
```

### Task 4: Sync The Audit Report And Run Final Verification

**Files:**
- Modify: `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md`

- [ ] **Step 1: Update section `12-3` status and action text**

Change the status from `要確認` to a closed state only if the implementation actually shipped exact reverse links from API pages back to labeled algorithm sections.

- [ ] **Step 2: Update section `12-10` status and action text**

State clearly that:
- the dead `docs_internal/verification/` reference was removed
- objective evidence is now represented by existing `docs/developers/algorithms/*` artifacts
- the page no longer relies on unverifiable CI wording

- [ ] **Step 3: Re-run docs sync and build verification**

Run:

```bash
conda run -n gwexpy python scripts/check_docs_sync.py
conda run -n gwexpy sphinx-build -b html docs /tmp/gwexpy-docs-validated-links
```

Expected:
- `check_docs_sync.py` passes or reports only pre-existing accepted differences
- Sphinx build succeeds

- [ ] **Step 4: Spot-check the built HTML for the two audit targets**

Run:

```bash
rg -n "Objective Evidence|ALGORITHM_CONTEXT|merged_validation_report|algorithm_fix_report_20260201" /tmp/gwexpy-docs-validated-links/web/en/user_guide/validated_algorithms.html /tmp/gwexpy-docs-validated-links/web/ja/user_guide/validated_algorithms.html
rg -n "validated_algorithms.html" /tmp/gwexpy-docs-validated-links/web/en/reference/api /tmp/gwexpy-docs-validated-links/web/ja/reference/api
```

Expected:
- `Objective Evidence` appears in both languages
- built API pages contain backlinks to `validated_algorithms`

- [ ] **Step 5: Commit**

```bash
git add docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md
git commit -m "docs(audit): close validated algorithm backlink and evidence findings"
```

## Risks And Notes

- MyST labels must be identical between JA and EN files. If labels diverge, the reST `:ref:` links will drift.
- `api/fitting.rst` is only a compatibility stub, so the real user-facing detail remains `reference/fitting.md`. Update both.
- `docs/developers/algorithms/*` is not currently published as built site content, so evidence links should stay as GitHub blob URLs unless the docs publication model changes later.
- If `scripts/check_docs_sync.py` reports unrelated historical differences, record them explicitly rather than silently weakening this plan’s scope.

## Success Criteria

- `validated_algorithms` exposes exact, labeled sections for each validated algorithm and for objective evidence.
- The relevant API category pages link back to those exact sections, not only to the top of `validated_algorithms`.
- Key reference pages (`ScalarField`, `TimeSeries`, `Spectral`, `fitting`) also point to the matching validation sections.
- No user-facing page mentions non-existent `docs_internal/verification/`.
- No user-facing page claims an unverified `verify-physics` gate.
- The audit report can honestly mark `12-3` and `12-10` as resolved.
