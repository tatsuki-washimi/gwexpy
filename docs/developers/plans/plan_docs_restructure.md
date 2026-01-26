---
type: plan
title: Documentation Directory Restructuring Plan
status: proposed
created: 2026-01-26
author: Antigravity
---

# Documentation Directory Restructuring Plan

This plan outlines the steps to reorganize the `docs/` directory to separate developer resources from user documentation (`web/`) and establish a symmetrical structure for English and Japanese content.

## Objectives

1. **Separate Developers Guide**: Isolate `docs/developers/` as a Git-only resource, excluded from the Sphinx web build.
2. **Symmetrical Structure**: Create a clean `web/` directory with `en/` and `ja/` subdirectories containing symmetrical `guide/` and `reference/` sections.
3. **Clean Reference**: Flatten nested language folders within `reference/` (e.g., remove `reference/en/en/`).

## Target Structure

```text
docs/
├── developers/          # Internal documentation (Git tracking only)
│   ├── plans/
│   ├── reports/
│   └── ...
├── web/                 # User-facing documentation (Sphinx build target)
│   ├── en/              # English content
│   │   ├── index.rst
│   │   ├── guide/
│   │   └── reference/
│   └── ja/              # Japanese content
│       ├── index.rst
│       ├── guide/
│       └── reference/
├── conf.py              # Sphinx configuration
└── index.rst            # Root redirect (optional, or points to web/en/index)
```

## Execution Steps

### Phase 1: Preparation & Verification

1. **Backup**: Ensure all files are committed to Git.
2. **Content Audit**: Compare `docs/developers/` and `docs/ja/developers/` to ensure no unique content is lost before deleting `docs/ja/developers/`.

### Phase 2: Restructuring

1. **Create New Roots**:
    * `mkdir -p docs/web/en docs/web/ja`
2. **Move English Content**:
    * Move `docs/guide` to `docs/web/en/guide`.
    * Move `docs/reference` to `docs/web/en/reference`.
    * Move `docs/index.rst` to `docs/web/en/index.rst`.
3. **Move Japanese Content**:
    * Move `docs/ja/guide` to `docs/web/ja/guide`.
    * Move `docs/ja/reference` to `docs/web/ja/reference`.
    * Move `docs/ja/index.rst` to `docs/web/ja/index.rst`.
4. **Consolidate Developers Guide**:
    * Merge unique content from `docs/ja/developers/` into `docs/developers/` (renaming files if necessary).
    * Delete `docs/ja/developers/`.

### Phase 3: Reference Flattening

1. **English Reference**:
    * In `docs/web/en/reference/`:
        * Move `en/*` to current directory (`.`).
        * Remove `ja/` folder (it belongs in `web/ja`).
        * Remove empty `en/` folder.
2. **Japanese Reference**:
    * In `docs/web/ja/reference/`:
        * Move `ja/*` to current directory (`.`).
        * Remove `en/` folder.
        * Remove empty `ja/` folder.

### Phase 4: Link & Config Fixes

1. **Update `index.rst`**:
    * Fix `toctree` paths in `docs/web/en/index.rst` and `docs/web/ja/index.rst` to reflect the new depth.
    * Ensure language links point to the correct peer (e.g., `../../ja/index.html`).
2. **Update `conf.py`**:
    * No major changes needed if Sphinx runs from `docs/` root, but `master_doc` might need adjustment if not using a root `index.rst`.
    * *Decision*: Keep a minimal `docs/index.rst` that redirects to `web/en/index.rst` or uses `toctree` to include `web/en/index` and `web/ja/index`.

### Phase 5: Verification

1. **Build**: Run `sphinx-build -b html docs docs/_build/html`.
2. **Check**: Verify `docs/_build/html/web/en/index.html` and `docs/_build/html/web/ja/index.html` generate correctly.
