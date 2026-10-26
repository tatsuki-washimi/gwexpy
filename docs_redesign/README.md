---
orphan: true
---

# Public documentation source

`docs_redesign` is the published PyData/MyST-NB site, built in English and Japanese
under `/docs/` and `/docs/ja/`.

English narrative and executable notebook cells are maintained together in
this tree; Japanese prose uses gettext catalogs. `scripts/prepare_public_docs.py`
copies the tree outside the checkout, clears notebook outputs and execution
counts, and records the public notebook paths with the Git revision. The
notebooks here are the source used by preview, production, and pull-request
builds.

The repository still retains `docs/web/` as legacy input for older tutorial
generators and as source material referenced by historical release evidence.
It is not built or published as a second site. Production Pages generates the
supported old `/docs/web/` URL redirects from
`scripts/generate_docs_redirect_stubs.py`; that deployed path contains
redirects, while this source tree remains available until its remaining tools
and historical references no longer need it.

MyST-NB executes prepared notebooks into an untracked cache, with a 600-second
per-cell timeout and errors raised. Full tracebacks are retained on failure.
Committed notebooks stay free of outputs and execution counts. The case-study
header reports measured execution time and package version separately from
physical validation. `release_status.json` records `latest_release`, the latest published package,
and `intro_examples_release`, the release against which the introductory
scripts are tested. These are independent metadata values and are not inferred
from the development package's version number.

Run `python scripts/verify_public_examples.py <prepared-docs-source>` after preparation to execute the Markdown lessons and regenerate the shared Quickstart figure from its downloadable source.
