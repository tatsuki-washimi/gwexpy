# Post-Preparation Validation & CI Alignment Plan

**Date**: 2026-01-25
**Target**: Release Branch CI Green & Japanese Doc Alignment

## 1. Objectives

- **Japanese Docs**: Ensure no 404s for Japanese documentation by creating stubs or redirects for missing pages.
- **CI Readiness**: Verify local equivalents of CI jobs (lint, mypy, docs) pass perfectly.
- **Installation Guide**: Update docs to explicitly mention Conda requirement for `nds2-client` and other non-PyPI deps.
- **Version Integrity**: Double-check version consistency before final tagging approval.

## 2. Detailed Roadmap

### Phase 1: Japanese Documentation Gap Fill

- [ ] **Identify Gaps**: Compare `docs/guide/tutorials/*.ipynb` vs `docs/ja/guide/tutorials/`.
- [ ] **Create Stubs**: Generate placeholder `.rst` or `.md` files for missing Japanese tutorials, redirecting to the English version or stating "Translation in progress".

### Phase 2: Installation Guide Update

- [ ] **Update Install Docs**: Edit `docs/install.md` (or equivalent) to clarify `[gw]` dependencies like `nds2-client` require Conda.

### Phase 3: Final Local Validation

- [ ] **Version Check**: Script check of `pyproject.toml` vs `gwexpy/_version.py`.
- [ ] **Full Test**: Run `pytest` (skip slow nbmake), `mypy`, `ruff`, and `sphinx-build` one last time.

## 3. Recommended Models & Skills

- **Model**: `Gemini 3 Pro` (High Context)
- **Skills**:
  - `build_docs`: For Sphinx validation.
  - `test_code`: For running final checks.

## 4. Effort Estimate

- **Total Time**: ~45 minutes.
- **Complexity**: Low.
