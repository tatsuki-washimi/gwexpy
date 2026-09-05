# v0.2.3 release closure report

Date: 2026-09-06 (UTC)

This report closes the bookkeeping cycle for the already published GWexpy
v0.2.3 release. It records the authoritative release identity and the state
of the repository after publication. It does not replace or revise the
candidate-bound review and qualification reports.

## Published release identity

The immutable release source is commit
`75d3d1a89ebc8942af1f3228152fea99d2d3420e`. The annotated tag `v0.2.3` is
object `b79ad05ca51527a048dd18e5c3cf84bc9e57487a`, and the GitHub Release is
[v0.2.3](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.3)
(release ID `383250425`, published 2026-09-05T13:11:33Z). The release source,
tag, and published payload are historical and were not changed by this
cleanup.

The public distribution records agree with that identity:

| Channel | Published identity | Evidence |
| --- | --- | --- |
| PyPI | `gwexpy==0.2.3`; wheel SHA-256 `e73da65cff769615fc78e264f6f17730573472a8587031444ccef237294e1e9c`; sdist SHA-256 `a5e752c6a53b5c6cabf41de309ca3077534f068384425deff3b1bccc461377b2` | [PyPI 0.2.3](https://pypi.org/project/gwexpy/0.2.3/) |
| conda-forge | `noarch/gwexpy-0.2.3-pyhc364b38_0.conda`, build 0, SHA-256 `4723b718fb80fd9676ba085bd74ea4ca08db864df89e9539ee1da5877d67bbf6` | [conda-forge package](https://anaconda.org/conda-forge/gwexpy) and [feedstock PR #13](https://github.com/conda-forge/gwexpy-feedstock/pull/13) |
| Zenodo | record `22344992`, DOI `10.5281/zenodo.22344992`, publication date 2026-09-05 | [Zenodo record](https://zenodo.org/records/22344992) |

The publication qualification run was `33967606952`: 19/19 qualification
cells and 4/4 smoke cells passed, with 27 artifact archives, two verified
aggregates, and zero unexpected skips. The release publication manifest
records the same-payload and distribution digests:
`docs/developers/plans/manifests/audit-manifest-v0.2.3-publication.yaml`.

## Issue closure

Issues #698, #699, #700, #701, #702, #703, #704, #705, #706, #707, #709,
#710, and #711 were fixed before the immutable release source and are closed
with comments linking their tests and release evidence. #639 was closed after
the bounded GWpy compatibility inventory was completed. The initial #704
checklist and earlier issue comments remain historical; the closure comments
record the final scope without rewriting them.

The #639 audit is classified as follows:

- **A — audited and complete:** 575 logical members, 1,150 frozen cases, and
  all selected selectors in the v0.2.3 inventory; 0 unreviewed and 0
  differential-required cases remain.
- **B — parity or explicit exception:** 224 fixed cases, 44 no-finding cases,
  and 0 cases where GWpy itself fails. Parent-parity boundaries and the
  private corrected Rayleigh route remain documented exceptions. Issue #611
  has its separate approved record.
- **C — GWexpy-only:** 882 counterpart-absent entries are outside a GWpy
  parity audit and remain classified as GWexpy-only API.
- **D — unfinished:** none within the frozen inventory. Any future override
  requires a new focused audit rather than reopening this umbrella issue.

No v0.2.3 milestone existed, so no milestone was created or changed. Issue
#634 (numeric CSV time-scale contract) and #688 (full exact rational time-axis
work) remain deferred without a version assignment. Docs redesign PR #713 and
the separately approved #611 work remain outside this closure.

## Repository records synchronized

The v0.2.3 release section was added to `ROADMAP.md`. The English and Japanese
web roadmaps and the redesigned roadmap now identify v0.2.3 as the current
published maintenance baseline, link PyPI/conda-forge/Zenodo, and state that no
next-minor theme is committed. The Japanese catalogue was synchronized with
the redesigned roadmap. Existing `CHANGELOG.md`, `release_notes/v0.2.3.md`,
`CITATION.cff`, `.zenodo.json`, the publication manifest, and the generated
v0.2.3 GitHub history plot assets were already publication-correct and were
left unchanged.

The public docs deployment readback remains tied to source-only post-release
commit `23f7030f0ab08834dcd1bceeb5395f730f76b354`, Pages run `33969465299`,
and gh-pages commit `1bfae37429449efc9027ca287c85ff2fd4c89e7a`. Since that
deployment, `main` advanced to `941a377a0305336b66231d8ec8a42c14b364b244`
through the separate Docs redesign PR #713; `maint/0.2` remains at `23f703...`.
The plot and CSV assets are tied to tag `v0.2.3`; the Japanese plot contains
Japanese labels.

## Historical evidence and scope boundary

The candidate-bound review reports, qualification artifacts, publication
record, and earlier handover under the external release-evidence directory
are retained byte-for-byte. Historical candidate SHAs are not presented as
approval for a later source. Human scientific/data-model approval is kept
distinct from the three automated review lanes and from the qualification
run. The final publication source is stated explicitly above and in the
closure manifest.

This cleanup changes documentation and bookkeeping only:

> No runtime, public API, dependency, or scientific-semantic changes.

No new runtime or correctness defect was found. The deferred issues listed
above are not release failures and were not silently closed.

## Verification

The closure branch runs the release-governance and documentation checks listed
in the pull request, including the roadmap/manifest contracts, metadata
consistency, Markdown link checks, Ruff, and strict English/Japanese docs
builds. `git diff` is checked to confirm that no `gwexpy/` runtime source was
modified.

The executed results are 190 docs tests passed with 4 skips, 388 release and
governance tests passed, and 31 closure-contract tests passed. Ruff check and
format checks, release metadata, and JA/EN heading synchronization passed.
The docs_redesign strict EN/JA builds passed with notebook execution disabled;
legacy EN strict build passed. Legacy JA strict build reproduces the known 74
docutils inline-interpreted-text warnings recorded as a matching base/current
baseline in the v0.2.0 audit, so no unrelated warning cleanup was folded into
this PR. The runtime diff is empty.

The machine-readable counterpart is
`docs/developers/plans/manifests/audit-manifest-v0.2.3-release-closure.yaml`.

## Human decisions after this PR

Review and merge the single post-release cleanup PR. Future work should be
triaged independently for #634 and #688; no v0.2.4 or v0.3.0 feature theme is
committed by this closure.
