---
orphan: true
---

# Public documentation source

`docs_redesign` is the published PyData/MyST-NB site, built in English and Japanese
under `/docs/` and `/docs/ja/`.

Notebook execution cells are authoritative in `docs/web/en/user_guide/tutorials`.
English public narrative and cell identities live in this tree; Japanese public
prose uses gettext catalogs. The legacy bilingual markdown cells remain in the
canonical tree for the legacy docs. `scripts/prepare_public_docs.py` copies this
tree outside the checkout, replaces every execution cell from its canonical
notebook, and records the source mapping and Git revision. Structural changes
must update the matching lesson cell slots together; count mismatches fail.

The September 2026 reconciliation incorporated the fifteen published code-cell
differences into the canonical notebooks before enabling this derivation.
Do not repair a public execution cell without repairing its canonical source.

MyST-NB executes prepared notebooks into an untracked cache, with a 600-second
per-cell timeout and errors raised. Full tracebacks are retained on failure.
Committed notebooks stay free of outputs and execution counts. The case-study
header reports measured execution time and package version separately from
physical validation. `release_status.json` records the released package and
release against which the introductory scripts are tested; it is not inferred
from the development package's version number.

Run `python scripts/verify_public_examples.py <prepared-docs-source>` after preparation to execute the Markdown lessons and regenerate the shared Quickstart figure from its downloadable source.
