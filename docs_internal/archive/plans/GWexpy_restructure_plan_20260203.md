# Documentation Restructuring Plan

This plan outlines the steps to restructure the GWexpy documentation as proposed in `docs/developers/plans/GWexpyドキュメント再構成提案.pdf`.

## Role Division

- **Gemini**: Structural reorganization, file moves, and build integrity checks.
- **Claude Coden**: Content rewriting, MyST conversion, and detailed guide expansion.

## Proposed Changes

### Phase 1: Structural Reorganization & Setup (Gemini)

*Objective: Establish the new directory structure and ensure build integrity without significant content rewriting.*

#### [MODIFY] `docs/web/{lang}/`

- **Rename** `guide/` to `user_guide/`.
- **Create** `examples/` directory (empty or with placeholders).
- **Cleanup** `reference/` structure (flatten redundant language folders if any).
- **Update** `toctree` references in `index.rst` files to match new paths.
- **Create** Redirect Stubs for renamed/moved major pages (e.g., old guide URLs).

Target Structure:

```
docs/web/
├── en/
│   ├── index.rst
│   ├── introduction.rst
│   ├── user_guide/             # Renamed from 'guide'
│   ├── examples/               # NEW: Empty/Placeholder
│   ├── reference/
│   ├── faq.rst
│   └── changes.rst
└── ja/
    └── (Symmetric structure)
```

### Phase 2: Content Refinement & Expansion (Claude Coden)

*Objective: Rewrite content for clarity, convert formats, and expand examples.*

- **Convert** Guide pages from RST to MyST Markdown (`.md`).
- **Write** new content for `examples/` (Gallery).
- **Review** and refine text in `user_guide/`.

## Verification Plan (Gemini)

### Automated Tests

- **Sphinx Build**: `sphinx-build -nW -b html docs docs/_build/html` (Verify structure valid).
- **Link Check**: `sphinx-build -b linkcheck docs docs/_build/linkcheck` (Verify internal links).
