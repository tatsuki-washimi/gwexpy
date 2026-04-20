# GWexpy logo source package

This directory is the source-of-truth package for GWexpy branding assets.

Canonical production files:

- `logo.svg`
- `logo.png`
- `icon.svg`
- `icon.png`

Supporting source exports:

- `logo.pdf`
- `icon.pdf`

Design handoff draft:

- `GWexpy_logo_draft.pptx`

Recommendation:

- Track the four canonical render files and the two PDF exports in git.
- Keep the PPTX in this directory only if you expect ongoing slide-based editing; otherwise it is better left untracked because it is bulky and adds review noise without improving the public asset pipeline.

Public deploy copies are generated into `docs/_static/branding/` by
`scripts/branding/generate_docs_branding.py`.
