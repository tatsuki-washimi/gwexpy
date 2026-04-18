# Phase 3 Prototype Isolation

This directory is the safe staging area for Phase 3 prototype work while Phase 2 remediation is still active.

## Allowed prototype paths

- `docs_internal/analysis/webpage/phase3_prototypes/`
  - design notes
  - diagram source drafts
  - asset manifests
- `docs/_static/images/phase3/`
  - generated SVG/PNG assets intended for later promotion
- `tmp/phase3/`
  - scratch renders and local experiments that should not be committed

## Do not edit during active Phase 2 work

Until the manager explicitly grants a file lock, do not edit:

- `docs/web/{en,ja}/index.rst`
- `docs/web/{en,ja}/examples/index.rst`
- `docs/web/{en,ja}/user_guide/architecture.md`
- `docs/conf.py`
- `docs/_static/custom.css`
- `docs/_templates/layout.html`

## Current split

- Phase 2 workers own live guide/reference remediation.
- Phase 3 workers own prototype assets only.
- Promotion from prototype paths into live docs happens only after an explicit merge decision.
