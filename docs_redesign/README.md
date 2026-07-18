---
orphan: true
---

# docs_redesign — website source notes

This tree is the **B1 (pydata + Diátaxis) redesign** source for the published
website. The deployment workflow builds it into the existing `/docs/` site path.

## Notebook single-source contract

The notebooks under `tutorials/`, `how-to/<topic>/`, and `how-to/case-studies/`
were **derived** from the canonical notebooks at
`docs/web/en/user_guide/tutorials/*.ipynb`.

- **Canonical source of truth:** `docs/web/en/...` (per `docs/NOTEBOOK_POLICY.md`).
  These were **not modified** by the redesign.
- **Derived here:** EN-only copies (the paired `lang-ja` markdown cells were
  stripped), with Colab bootstrap removed and links rewritten to the new layout.
  Treat the copies in this tree as **disposable prototype artifacts**.
- **Japanese recovery (P4):** Japanese prose is recovered from the canonical
  `lang-ja` cells when this tree is gettext-enabled; the EN-only copies here are
  *not* the JA recovery source. At cutover (P5) the canonical→derived transform
  should move to a build-time step rather than committed copies.

## Rendering

`nb_execution_mode = "cache"`: notebooks execute into an untracked build cache.
The rendered site therefore includes plots and other outputs, while committed
notebooks remain clean source with outputs and execution counts stripped. The
publish workflows build from an isolated temporary copy of this tree, so no
executed notebook is ever written back to the checkout.

Notebook execution is fail-closed and has a 180-second per-cell limit, which
accommodates fitting and file-I/O examples without silently publishing a page
whose output failed to generate.
