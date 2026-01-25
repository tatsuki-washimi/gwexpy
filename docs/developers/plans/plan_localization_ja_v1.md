# Localization (ja) Plan

**Date**: 2026-01-25
**Target**: Initial Japanese Documentation Setup

## 1. Objectives

- **Structure Mirroring**: Create a physical `docs/ja` directory mirroring `docs/` (en).
- **Translation (Phase 1)**: Translate key landing pages (`index.rst`, `guide/index.rst`, `guide/installation.rst`) into Japanese.
- **Stub Preservation**: Keep the tutorial stubs (created previously) but ensure they link correctly within the new structure.

## 2. Detailed Roadmap

### Phase 1: Structure Setup (Done)

- Copied `*.rst`, `guide/`, `reference/` to `docs/ja/`.

### Phase 2: Core Page Translation

- [ ] **index.rst**: Translate the main landing page.
- [ ] **guide/index.rst**: Translate the guide table of contents.
- [ ] **guide/installation.rst**: Translate installation instructions.

### Phase 3: Cleanup

- [ ] **Remove Duplicate Stubs**: Ensure `docs/ja/guide/tutorials/*.md` (stubs) match the file names of `docs/guide/tutorials/*.ipynb` so `toctree` works. (Currently mixed: copies of english rst/md vs stubs).

## 3. Recommended Models & Skills

- **Model**: `Gemini 3 Pro` (High Context) - Excellent for translation.
- **Skills**: None specific (standard editing).

## 4. Effort Estimate

- **Total Time**: ~20 minutes.
- **Complexity**: Low.
