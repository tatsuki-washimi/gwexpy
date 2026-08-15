# Developer Design Index

`design/` holds dated-free, living design documents that are the canonical
architectural reference for their subject (e.g. the capability-domain roadmap
design). They differ from `../plans/`, `../contracts/`, and `../reports/`:

- `design/` (this directory): undated, maintained-in-place canonical design.
- `../design_data/`: CSV artifacts used by developer designs.
- `../gui/`: GUI analysis notes.
- `../plans/`: dated execution records and coordination plans (see
  `../plans/README.md`).
- `../contracts/`: normative I/O contracts.
- `../reports/`: generated reports.

Documents in this directory are maintainer-facing: `docs/conf.py` excludes
`developers/**` from the built Sphinx docs.
