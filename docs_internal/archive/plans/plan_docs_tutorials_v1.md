# Documentation & Tutorial Refinement Plan

**Date**: 2026-01-25
**Target**: `v0.1.0b1` Documentation Quality

## 1. Objectives

- **Tutorial Verification**: Ensure all new Jupyter notebooks in `docs/guide/tutorials/` execute without errors and render correctly.
- **API Reference**: Guarantee that optional/lazy modules like `gwexpy.fitting` and `gwexpy.interop` are documented in the API reference.
- **Link Integrity**: Verify external links using Sphinx linkcheck and fix broken ones.
- **Feature Highlights**: Explicitly document robust serialization (Pickle support) in README.

## 2. Detailed Roadmap

### Phase 1: Tutorial Verification

- [ ] **Execute Notebooks**: Use `test_notebooks` skill (or `pytest --nbmake` if available) to run all notebooks in `docs/guide/tutorials/`.
- [ ] **Visual Check**: Verify rendering of outputs (plots/tables) if execution passes.

### Phase 2: API Reference Expansion

- [ ] **Inspect Reference**: Check `docs/reference/` for missing modules.
- [ ] **Add Missing Modules**: Add `autosummary` or `automodule` directives for `gwexpy.fitting` and `gwexpy.interop`.

### Phase 3: Link Check

- [ ] **Run Linkcheck**: Execute `sphinx-build -b linkcheck docs/ docs/_build/linkcheck`.
- [ ] **Fix Broken Links**: Resolve 404s or redirects found in the report.

### Phase 4: README & Feature Updates

- [ ] **Update README**: Add a bullet point about "Robust Serialization (Pickle round-trip support)" in the Features section of `README.md`.

## 3. Recommended Models & Skills

- **Model**: `Gemini 3 Pro` (High Context) - Good for understanding documentation structure and Sphinx quirks.
- **Skills**:
  - `build_docs`: For running Sphinx builds (linkcheck).
  - `test_notebooks`: For executing tutorial notebooks.
  - `search_web_research`: If replacement links need to be found.

## 4. Effort Estimate

- **Total Time**: ~60 minutes (Notebook execution and link checks can be time-consuming).
- **Complexity**: Low.
