# Project Roadmap

This document describes release **themes and policy**. It is the canonical source of
*inclusion criteria* — the features and quality gates that define each release's scope.
[GitHub milestones](https://github.com/tatsuki-washimi/gwexpy/milestones) are the
canonical source of *applied membership* derived from this document's criteria; they
record which issues are being worked toward each release (one-directional — milestone
membership never rewrites this document's Definition of Done). The long-term shape of
the library is organized as a set of capability domains; see [Capability domains](#capability-domains)
below and the
[capability-domain roadmap design](docs/developers/design/capability-domain-roadmap.md)
for the full per-domain goals, the domain-by-theme matrix, and the issue triage rules.

The v0.1.x series established security, CI, release tooling, metadata integrity,
and I/O correctness through v0.1.14. The first semantically complete feature
release, v0.2.0, shipped on 2026-08-26. No next-minor theme is committed yet.

## Release policy

- **Patch releases (v0.x.y)** contain bug fixes only — no new features, no new public
  APIs, no new dependencies. Two clarifications, so that "bug fix" is not read more
  narrowly than intended:
  - Making an argument that is *already accepted but silently ignored* actually take
    effect — or raise — is a bug fix, not a new feature. The API surface does not grow;
    it stops advertising something it never did.
  - Narrowing a public API to correct a contract violation (for example, replacing a
    silent wrong result with an explicit `TypeError`) is also in scope, even though it
    is backwards-incompatible. Every such narrowing must be disclosed in `CHANGELOG.md`
    as an explicit before/after table. v0.1.13 is the worked example.
- **Maintenance releases** (v0.1.14, v0.2.1+, v0.1.15) are never pre-assigned features. They are
  issued only if regressions or newly discovered bugs require them after a release.
  "Finish feature X in v0.2.1" is explicitly not allowed: a feature is either complete
  in v0.2.0 or deferred to a later minor. Maintenance releases do not introduce new
  public API; new arguments, public types, or public functions belong to the next
  minor release. For example, a fail-closed narrowing on #632/#634 may be eligible
  for a patch, but new `timezone=`/`time_scale=` API for either issue belongs to a
  minor release.
- **Milestones are not created in advance**: the milestone for the next-but-one minor
  is created only after the preceding minor has shipped. The future themes below are
  directional, not commitments.
- **v1.0 is a stabilization declaration, not a feature release**: it adds no new
  feature domains and introduces no new public API surface. Its primary path is
  module-by-module stability labelling (see [v1.0](#v10--public-contract-stabilization)
  below); no v1.0 milestone exists or will be created in advance, consistent with the
  rule above.
- **Repository-level work (dependency-free of any release theme) does not ride on a
  release milestone.** GUI source removal and documentation-tree consolidation are
  examples: they are tracked as independent issues so they cannot block, or be
  blocked by, a feature release.
- Documentation, tests, and contract updates are part of each feature's definition of
  done — not a separate release theme.

## Capability domains

Releases are organized around eleven capability domains, four cross-cutting
foundations, and a consumer layer outside the library itself. This taxonomy is a
planning vocabulary for scoping issues and releases — it is not a promise that every
domain gains features in every release.

- **Core data**: 1. Time/Frequency series — 2. Multi-channel & matrix containers —
  3. Histogram & distribution — 4. Field & spatial data — 5. Segment & experiment datasets
- **Data access**: 6. Experimental I/O — 7. Scientific interoperability
- **Analysis**: 8. General signal & statistical analysis — 9. Commissioning &
  experimental analysis — 10. Modeling & forecasting
- **Presentation**: 11. Scientific visualization
- **Cross-cutting foundations** (not features; every domain answers to these): X1
  semantic/metadata contract, X2 persistence/provenance/reproducibility, X3 API
  stability & GWpy compatibility, X4 performance & scalability
- **Consumer layer** (outside the domain taxonomy): GWexpy Studio, pyaggui, the CLI,
  and Jupyter workflows consume the library through its public API; see
  [GWexpy Studio](#gwexpy-studio-companion-app) below.

Per-domain minimum / v1.0 / long-term goals, the domain-by-release-theme matrix, and
the issue triage rules are maintained in the
[capability-domain roadmap design](docs/developers/design/capability-domain-roadmap.md).

## v0.1.13 — Silent-corruption stabilization patch (released 2026-08-08)

> Close every known case where GWexpy returns wrong numbers, wrong units, or silently
> dropped metadata without raising.

Released from `f7f836eec7e6247a01e9a1b61cc1a2121235e58d` as
[v0.1.13](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.1.13), on PyPI as
`gwexpy==0.1.13`, archived at [10.5281/zenodo.21849416](https://doi.org/10.5281/zenodo.21849416).
Milestone [v0.1.13](https://github.com/tatsuki-washimi/gwexpy/milestone/8) closed with
no open issues. The authoritative per-change record is the `[0.1.13]` section of
`CHANGELOG.md`, including the before/after tables for every API narrowing; the
categories below are what the release set out to do.

- **Wrong numbers / units / dtype (P0)**: ROOT non-double histograms read as `float64`
  ([#593]); Quantity/Unit operands capturing containers and dropping class, unit, or
  metadata in `SeriesMatrix`, `SpectrogramMatrix`, Field collections, and `Histogram`
  ([#575], [#576], [#577], [#578], [#579]); WIN reader decoding only 8 of the 12
  sampling-rate bits ([#610]).
- **GWpy compatibility**: `TimeSeries.rms()` signature incompatibility ([#451]). Fixed
  inside the release candidate; PR [#453] was closed as superseded rather than merged.
- **Accepted-but-ignored arguments**: `start`/`end` silently ignored by some readers
  ([#611]); GWF `parallel`/`nproc` no-op ([#588]); ndscope HDF5 writer creation kwargs
  ignored ([#590]). All three now raise. Only #611 gained a working windowed read
  path; #588 and #590 ship as fail-closed contracts, with the implementations
  scheduled independently of any specific release theme.
- **Broken or nondeterministic I/O**: `TimeSeries.read(path, format="zarr")` always
  raising `IsADirectoryError` so the documented entry point never worked ([#620]);
  zarr returning a nondeterministic channel ([#614]); NetCDF4 round trips perturbing
  `t0` and `dt` ([#615]).
- **Timing and provenance precision**: `crop` perturbing `dt` by several ulp, which the
  truncating `nfft` derivation amplifies into an O(1/nfft) frequency-axis error
  ([#617]); read provenance dropped by the collection re-wrap ([#618]).
- **CI integrity**: dedicated gates that can pass with zero collected tests ([#511]).
- **Documentation vs implementation**: SegmentTable reference describing unimplemented
  APIs ([#605]); GWinc docstring pointing at a nonexistent classmethod ([#608]); and
  the one-line `VectorField.plot(stride=)` fix ([#559]).
- **Verification-only items (not release blockers)**: GWpy-only HDF5 readability
  ([#402], stays in scope for the container-semantic-contract release — manual check
  only for this release). The public documentation now correctly states that `.ffl`
  is unsupported. That correction was part of v0.1.13; [#594] remained open as a
  follow-up investigation. PR [#625] subsequently measures the path and proposes
  an implementation. Neither the investigation nor that implementation was part of
  v0.1.13.

Explicitly excluded: any new feature or API, new dependencies, large refactors, and
PR [#488] (GUI extraction — open, tracked separately from any release milestone; see
[#645](https://github.com/tatsuki-washimi/gwexpy/issues/645) for the repository-level
completion work). Monte-Carlo provenance ([#508]) and the `_t0_ns` precision follow-up
([#513]) were deferred during the release rather than shipped here.

[#593]: https://github.com/tatsuki-washimi/gwexpy/issues/593
[#575]: https://github.com/tatsuki-washimi/gwexpy/issues/575
[#576]: https://github.com/tatsuki-washimi/gwexpy/issues/576
[#577]: https://github.com/tatsuki-washimi/gwexpy/issues/577
[#578]: https://github.com/tatsuki-washimi/gwexpy/issues/578
[#579]: https://github.com/tatsuki-washimi/gwexpy/issues/579
[#610]: https://github.com/tatsuki-washimi/gwexpy/issues/610
[#451]: https://github.com/tatsuki-washimi/gwexpy/issues/451
[#453]: https://github.com/tatsuki-washimi/gwexpy/pull/453
[#611]: https://github.com/tatsuki-washimi/gwexpy/issues/611
[#588]: https://github.com/tatsuki-washimi/gwexpy/issues/588
[#590]: https://github.com/tatsuki-washimi/gwexpy/issues/590
[#511]: https://github.com/tatsuki-washimi/gwexpy/issues/511
[#614]: https://github.com/tatsuki-washimi/gwexpy/issues/614
[#615]: https://github.com/tatsuki-washimi/gwexpy/issues/615
[#617]: https://github.com/tatsuki-washimi/gwexpy/issues/617
[#618]: https://github.com/tatsuki-washimi/gwexpy/issues/618
[#620]: https://github.com/tatsuki-washimi/gwexpy/issues/620
[#508]: https://github.com/tatsuki-washimi/gwexpy/issues/508
[#513]: https://github.com/tatsuki-washimi/gwexpy/issues/513
[#605]: https://github.com/tatsuki-washimi/gwexpy/issues/605
[#608]: https://github.com/tatsuki-washimi/gwexpy/issues/608
[#559]: https://github.com/tatsuki-washimi/gwexpy/issues/559
[#402]: https://github.com/tatsuki-washimi/gwexpy/issues/402
[#594]: https://github.com/tatsuki-washimi/gwexpy/issues/594
[#488]: https://github.com/tatsuki-washimi/gwexpy/pull/488

## v0.1.14 — I/O contract and maintenance hardening (released 2026-08-15)

Released from `42eec70450b867f10b7a9331c3a0217ce589c564` as
[v0.1.14](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.1.14), on PyPI as
`gwexpy==0.1.14`, archived at [10.5281/zenodo.21941441](https://doi.org/10.5281/zenodo.21941441).
Milestone [v0.1.14](https://github.com/tatsuki-washimi/gwexpy/milestone/9) closed with
12 resolved issues. The authoritative per-change record is the `[0.1.14]` section of
`CHANGELOG.md`; the categories below are what the release set out to do.

- **Time-interpretation and conversion correctness**: readers now distinguish source-defined absolute times, naive civil times, and relative sample indices, with malformed offsets and DST fold/gap ambiguity failing closed instead of silently picking an instant ([#633], [#651]); NumPy `datetime64` scalars and vectors convert through a time-aware Astropy representation with exact-nanosecond precision, and `from_gps()` returns a consistent timezone-aware UTC value across scalar, vector, and Astropy `Time` inputs ([#646], [#650]).
- **Fail-closed cadence and topology guards**: WIN reads now require consecutive one-second global packet cadence and fail closed on internal gaps, sample-rate changes, and malformed payloads ([#647]); CSV and SDB timestamps are validated before conversion, with duplicate, backward, or overlarge gaps raising instead of being silently accepted ([#648], [#649]).
- **Experimental-I/O route and format-claim repairs**: the ATS.MTH5 reader now uses the supported `mth5` API and fails closed on missing or inconsistent metadata instead of silently degrading ([#619]); the undocumented `sqlite`/`sqlite3` SDB aliases were removed in favor of the canonical `format="sdb"` name (breaking) ([#635]).
- **CI and release-process hardening**: the I/O conformance generator smoke check now enforces a bounded timeout with full process-group cleanup ([#629], [#630]).
- **Partial mitigation, not resolution**: #632 received a partial mitigation — WIN header times are now explicitly interpreted as UTC with an explicit warning, but no public `timezone=` contract exists yet. #634 received a partial mitigation on the CSV/component-column warning path, but numeric CSV timestamps remain on the legacy GPS-second interpretation; v0.1.14 did not add the proposed `time_scale=`/`time_unit=` contract. Neither issue is closed by this release.

Note: The CHANGELOG `[0.1.14]` entry records "#634 for v0.2.0" as a working assumption from earlier planning. The current assignment of #632 and #634 (deferred from this release), together with #636, is to the "I/O time and dispatch semantics" theme (Directional, not v0.2.0 Committed).

[#633]: https://github.com/tatsuki-washimi/gwexpy/issues/633
[#651]: https://github.com/tatsuki-washimi/gwexpy/issues/651
[#646]: https://github.com/tatsuki-washimi/gwexpy/issues/646
[#650]: https://github.com/tatsuki-washimi/gwexpy/issues/650
[#647]: https://github.com/tatsuki-washimi/gwexpy/issues/647
[#648]: https://github.com/tatsuki-washimi/gwexpy/issues/648
[#649]: https://github.com/tatsuki-washimi/gwexpy/issues/649
[#619]: https://github.com/tatsuki-washimi/gwexpy/issues/619
[#635]: https://github.com/tatsuki-washimi/gwexpy/issues/635
[#629]: https://github.com/tatsuki-washimi/gwexpy/issues/629
[#630]: https://github.com/tatsuki-washimi/gwexpy/issues/630
[#632]: https://github.com/tatsuki-washimi/gwexpy/issues/632
[#634]: https://github.com/tatsuki-washimi/gwexpy/issues/634
[#636]: https://github.com/tatsuki-washimi/gwexpy/issues/636

## Release Theme Status Vocabulary

Releases and future themes are classified as follows. Status labels appear as `Status: <term>` directly below the theme heading and apply to the entire theme block.

| Status | Definition | When |
|---|---|---|
| **Committed** | Theme is represented by an active release milestone and work is underway. | Zero or one selected theme may be `Committed`; there is none between minor-release selections. |
| **Directional** | Candidate theme for a future minor release; no version, date, or scope is committed. Scope may be re-assigned or dropped. | All themes in "Future themes" section. |
| **Backlog** | Capability recognized but not yet part of the acceptance scope of any Committed or Directional theme. | "Ecosystem & Interoperability" section. |

**Status vocabulary does not apply to:** released sections, Release policy, v1.0 criteria, or Engineering hygiene sections. The absence of a status label in those sections is not ambiguity — they are simply not classified by this scheme.

## v0.2.0 — Container Semantic Contract (released 2026-08-26)

> Every supported operation on a GWexpy container preserves class, unit, axes,
> labels, and metadata, or raises explicitly — never a silent downgrade.

Released from `5c91cf2d1087616c9815d0cbcc082c5f21bb36e9` as annotated tag
[`v0.2.0`](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.0), on
[PyPI](https://pypi.org/project/gwexpy/0.2.0/) and
[conda-forge](https://anaconda.org/conda-forge/gwexpy), and archived at
[10.5281/zenodo.22106588](https://doi.org/10.5281/zenodo.22106588). Milestone
[v0.2.0](https://github.com/tatsuki-washimi/gwexpy/milestone/3) closed with no
open issues. The canonical publication and post-release record is the
[v0.2.0 closeout ledger](https://github.com/tatsuki-washimi/gwexpy/pull/687#issuecomment-5429503803).

Release outcomes:

- **SeriesMatrix B0 / Phase A shipped:** the 480-cell container contract freezes
  metadata-preserving supported operations and explicit failures for unsupported
  operations. Direct NumPy ufunc compatibility remains a documented limitation.
- **SeriesMatrix B1 deferred:** the composition runtime was not adopted. It remains
  future design work on [#637](https://github.com/tatsuki-washimi/gwexpy/issues/637),
  with no assigned version or date.
- **Completed bounded work:** exact GPS-nanosecond state, GWpy-readable HDF5 sidecars
  and provenance, GWF `parallel=` with `nproc=` as a compatibility alias, coupling v1,
  and `median-mean` / `median_bias` shipped under the v0.2.0 contract.
- **Residual work:** #513, #588, and #590 remain open outside the closed milestone.

Field I/O, the eager SegmentTable workflow, Histogram arithmetic, coordinate
transforms and reprojection, layered visualization, lazy or aggregating segment
workflows, mesh-aware field models, and Fisher analysis were not v0.2.0 scope.
GUI removal ([#645](https://github.com/tatsuki-washimi/gwexpy/issues/645), PR
[#488]) and documentation-tree consolidation
([#606](https://github.com/tatsuki-washimi/gwexpy/issues/606)) remain independent
repository-level work.

## v0.2.3 — GWpy behavioral compatibility stabilization (released 2026-09-05)

Released from `75d3d1a89ebc8942af1f3228152fea99d2d3420e` as annotated tag
[`v0.2.3`](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.3), on
[PyPI](https://pypi.org/project/gwexpy/0.2.3/) and
[conda-forge](https://anaconda.org/conda-forge/gwexpy), and archived at
[10.5281/zenodo.22344992](https://doi.org/10.5281/zenodo.22344992). The
publication record is the
[v0.2.3 closure manifest](docs/developers/plans/manifests/audit-manifest-v0.2.3-release-closure.yaml)
and its [closure report](docs/developers/reports/report_v0.2.3_release_closure_20260906.md).
No v0.2.3 milestone was created; the release Issues were tracked directly and
closed after the published evidence was read back.

Release outcomes:

- **GWpy compatibility surface:** the audited GWpy-derived methods and their
  differential tests cover 575 logical members, with 224 fixed dispositions,
  44 no-finding dispositions, zero unreviewed entries, and 882 GWexpy-only
  entries explicitly outside the parent-override audit.
- **Qualification and publication:** the tag-triggered run passed 19/19
  qualification cells and 4/4 smoke cells. The qualified wheel and sdist were
  published to PyPI with the recorded SHA-256 digests; the same release source
  was archived by Zenodo and packaged on conda-forge.
- **Known boundaries:** the separately approved #611
  `non_intersecting_window_safety` exception remains limited to a completely
  disjoint HDF5 read window. Retained parent-parity risks and the private
  corrected Rayleigh route remain documented in the
  [v0.2.3 changelog](CHANGELOG.md) and
  [GWpy compatibility policy](docs_redesign/explanation/gwpy_compatibility_policy.md).

The v0.2.3 compatibility audit and release-control scope are closed. Explicit
numeric CSV time semantics (#634), the exact rational GPS time-axis contract
(#688), and other future themes remain unscheduled; the roadmap does not assign
them to a next version. No next-minor theme is committed.

## Future themes (not scheduled)

No milestones exist for these yet, and the themes may be re-scoped. Each theme below
carries a one-line release statement and a few headline user stories to make the
theme testable — this is a drafting convention, **not** a commitment to scope, order,
or a version number. Consistent with the release policy, an issue belongs to one of
these themes only if a headline user story's named acceptance artifact (a test or an
example notebook) would not pass without it. If a theme's timeline slips, drop the
items that no headline user story's acceptance artifact needs, rather than the theme
itself.

### I/O time and dispatch semantics (domain: io)

**Status: Directional**

Consistent time interpretation (time zones, numeric time scales and epochs) and
uniform reader behaviour across supported experiment data formats.

- **Track A — Time interpretation contract** (#632, #634, #636). Headline user
  story: a user reading data from any supported format gets a value with an
  explicit, documented time reference, never a silently-assumed one. Affected
  readers, and writers where applicable, interpret time through an explicit
  timezone / scale / unit / epoch contract; the implicit legacy GPS-seconds
  interpretation is deprecated. Named acceptance artifact: a cross-format
  time-interpretation conformance matrix covering #632, #634, and #636,
  including explicit-zone/scale cases and required fail-closed cases.
  Non-goals: no unrelated format expansion, and no changes to existing
  on-disk time encodings.
- **Track B — Dispatch / reader semantics** (#444 → #616). Headline user story: a
  user reading across multiple backends gets identical collection-fallback and
  gap/pad behaviour regardless of which reader served the request. First decides
  the collection fallback registry contract for `FrequencySeriesDict`/`List`/
  `Matrix` `.read()`/`.write()` — whether to keep the Astropy registry or
  converge on the GWpy default registry (#444) — then makes gap/pad behaviour
  consistent across supported `TimeSeriesDict.read` backends (#616). Named
  acceptance artifacts: (a) a `FrequencySeriesDict`/`List`/`Matrix` collection
  dispatch/reachability matrix covering the chosen registry, and (b) a
  cross-backend `TimeSeriesDict.read` gap/pad test matrix. Non-goals: this
  track does not implement a new registry mechanism, add new file formats, or
  change unrelated backend behaviour.

Both tracks complete independently; theme completion = both tracks green.

### Experiment data workflow

**Status: Directional**

Read, transform, and persist spatial Field data and
per-segment experiment records through GWpy-style APIs, without escaping to pandas
for metadata-bearing state. Covers Field I/O (`ScalarField`/`VectorField`/
`TensorField`/`FieldDict` read/write, canonical full-fidelity HDF5, GSI DEM and
GeoTIFF readers with a geospatial baseline) and the eager SegmentTable workflow
(explicit cell status model, `errors=` policy, column expressions, row/column
operations, HDF5 persistence with schema version and provenance). On-disk schema
versioning and the unknown-field policy are decided once, as a single project-wide
rule, before either format ships its first file. Design:
[terrain/ScalarField I/O plan](docs/developers/plans/active/2026-07-31-terrain-scalarfield-io-design.md),
[SegmentTable workflow plan](docs/developers/plans/active/2026-08-01-segmenttable-workflow-design.md).

### Advanced segment workflows

**Status: Directional**

Reducers, `groupby`/`aggregate`, and a lazy
`SegmentFrame` for aggregating across many experiment segments. A lazy or
out-of-core execution path is added only once a demonstrated usage requirement
exists — no segment-count target is set in advance. Design groundwork already in
the [SegmentTable workflow plan](docs/developers/plans/active/2026-08-01-segmenttable-workflow-design.md).

### Spatial geometry and layered visualization

**Status: Directional**

Grid geometry, detector frames, and
component-correct rotations (#556 theme — rotating coordinates without rotating
vector/tensor components is treated as a defect, not an approximation), plus
terrain/basemap/marker layer composition (#558 series). Changes to `gwexpy/fields/`
require physics-reviewer sign-off per project convention; this theme does not start
until that review capacity is available. Design:
[layered visualization plan](docs/developers/plans/active/2026-08-01-layered-visualization-design.md).

### Mesh-aware fields and solver interoperability

**Status: Directional**

Bringing simulation output
(OpenFOAM / FLOW-3D / SPECFEM3D / SimPEG) into the same metadata-aware workflow as
measured Field data. Before building a bespoke mesh topology model, this theme
evaluates delegating mesh representation to an existing library (`meshio`,
`PyVista`) — a bespoke `MeshField` (#522) is only justified if that delegation
cannot preserve unit/axis/metadata.

### Fisher forecasting and advanced analysis

**Status: Directional**

A labeled matrix layer, spectral
models, numerical derivatives, `FisherMatrix`, bias, and the overlap reduction
function (#570–#574). The overlap reduction function specifically is gated on
physics review and does not start until a reviewer is available.

### API compatibility and stabilization (foundation: X3)

**Status: Directional**

Continuation of the v0.2.0 "API stability labelling" workstream (#400) that
establishes and documents the public API surface. Covers auditing GWpy method
overrides against the documented compatibility principle (#639), and #640, which
splits into a behavioural-contract issue for the `ignore_nan` default mismatch
between `TimeSeriesMatrix` and `FrequencySeriesMatrix`, and a documentation-only
issue for the `TimeSeriesDict.append` docstring gap. Named acceptance artifact:
published compatibility matrix and per-module behaviour contract document.
Non-goals: this theme does not redesign any container's arithmetic behaviour —
see [v0.2.0](#v020--container-semantic-contract) for that contract.

### Later 0.x — Ecosystem and application readiness (deliberately unnumbered)

**Status: Directional**

One or more minors between the themes above and v1.0 that graduate matured
[Ecosystem backlog](#ecosystem--interoperability-backlog) items into a release
(each gets a milestone only when started), build out the headless application
contract that GWexpy Studio / pyaggui / the CLI need (source inspection, format
capability introspection, serializable operation parameters, provenance), and do
measurement-driven performance work building on the
[#581](https://github.com/tatsuki-washimi/gwexpy/issues/581) benchmark harness.
  Their number and order are decided release-by-release under the standard milestone
  rule, not fixed here.

## v1.0 — Public contract stabilization

v1.0 is defined by acceptance criteria, not by a milestone. No v1.0 milestone exists;
the historical empty `v1.0.0` milestone was closed in the 2026-08-01 reorganization,
and a milestone will be created only after the final pre-1.0 minor has shipped, per
the release policy above.

Its primary path is **module-by-module stability labelling**
([#400](https://github.com/tatsuki-washimi/gwexpy/issues/400)): each module is
labelled experimental, provisional, or stable, with a documented deprecation window,
and v1.0 is reached when every core module has graduated to stable — not by a single
release that stabilizes everything at once. This is deliberately more granular than a
one-shot "public contract" declaration, so that stabilization can proceed
incrementally as each domain's contract work actually finishes.

Criteria:
- Every capability domain (see [Capability domains](#capability-domains)) meets at
  least its per-domain minimum goal, as tracked in the
  [capability-domain roadmap design](docs/developers/design/capability-domain-roadmap.md).
- Cross-cutting foundations X1 (semantic contract) through X4 (performance) apply
  across all domains, not only the ones that happened to ship them first.
- The public API surface is frozen under the #400 stability labels, with a
  documented deprecation window (in releases and in time).
- No new feature domains are introduced as part of reaching v1.0 — it is a
  stabilization milestone, not a feature release.

## Roadmap maintenance

This document is a living guide, not a frozen specification. It is updated as follows to prevent stale decisions from becoming technical debt:

- **Releases**: a new released section is added (e.g., `## v0.3.0 — ...`) only after the release ships and will include the final SHA, tag, DOI, and per-change summary.
- **Directional → Committed escalation**: a Directional theme is promoted to Committed (assigned a v0.x.0 milestone and moved to a dedicated section) immediately when its milestone is created. This rule enforces that no Committed theme enters a release without a dedicated section and named Definition of Done.
- **Committed uniqueness**: only one theme is Committed at any time (it is the currently open release). No pre-assignment of future milestones happens.
- **Theme retirement**: a Directional theme is removed or archived once its named acceptance artifact (test, example, or document) is green on main and the decision to ship it or defer it further has been made.
- **Label semantics**: `domain:*` labels on issues are a search convenience, not an authority. Domain taxonomy and per-issue triage are governed by the [capability-domain roadmap design](docs/developers/design/capability-domain-roadmap.md); this document governs release inclusion criteria (which themes and issues belong to which release). The two authorities are disjoint — this document does not redefine domain taxonomy, and the design document does not redefine release scope.

## Ecosystem & Interoperability (Backlog)

**Status: Backlog**

Unscheduled work on connecting GWexpy to neighbouring projects. Nothing here is assigned to a
release; items move to a milestone only when started. In the domain taxonomy, everything here
belongs to domain 7 (scientific interoperability); graduation into a release happens through
the unnumbered later-0.x themes above. The user-facing positioning statement is
[`docs_redesign/explanation/ecosystem.md`](docs_redesign/explanation/ecosystem.md), and the
licence policy that constrains all of it is
[`docs/developers/LICENSES_THIRD_PARTY.md`](docs/developers/LICENSES_THIRD_PARTY.md).

Ordered by expected value per unit of effort. This ordering is an engineering judgement, not a
measurement of demand, and is deliberately kept out of the public documentation.

1. **GWDama HDF5 reader/writer** (`format="hdf.gwdama"`). Highest value: it needs no new
   dependency because GWDama's HDF5 layout is readable with the existing `h5py` base
   dependency, and it reuses the `hdf.ndscope` reader pattern wholesale. Blocked on three
   design decisions: how to preserve unknown attributes, how to resolve the `.h5` extension
   collision with `hdf5` and `hdf.ndscope`, and whether to support arbitrary group nesting.
   The unknown-attribute question is the same class of problem as the on-disk schema
   versioning decision in the Experiment data workflow theme above; resolve them together.
2. **Differometor converters** (`#423`–`#427`). Design is settled but the existing issue bodies
   assume result objects that Differometor does not have; `#423` must be closed with the real
   API shape before `#424`–`#426` are implemented.
3. **Virgo data-path completion** (`#591` umbrella): a separately reviewed FFL I/O
   contract, followed by `.ffl` support once that contract lands; FFL expander
   hardening (`#638`: exception-contract normalization and per-line bound); the
   dataDisplay ROOT product converters (`#598`–`#600`), which follow the ROOT
   class structural inventory (`#595`).
4. **LPSD / Daniell's method / huddle test.** Concepts worth evaluating for GWexpy's own
   spectral estimation, taking spicypy's API as a design reference only. No code reuse, and no
   design work has been done yet.
5. **Trigger-table exchange with pyomicron and search pipelines.** Reading and writing the
   products, never the orchestration.
6. **I/O performance** (`#580` umbrella): the shared benchmark harness (`#581`) comes
   first; individual optimizations are prioritized from measurements, not assigned to
   releases in advance.

Explicitly not planned: a spicypy adapter module (GWpy objects are already the shared
language), and any absorption of gwdetchar / gwsumm / gwvet / hveto workflows.

## Unassigned ideas

Recorded, but deliberately without a version:

- Distributed / out-of-core execution (Dask or similar) — separate from the lazy
  `SegmentFrame` theme; needs a demonstrated usage requirement first. (domain X4)
- GUI redesign beyond pyaggui — a support-policy question, not just a feature.
  (consumer layer)
- Meta-analysis and population-level statistics. (domains 8/10)
- 3D surface/volume data models and rendering. (domains 4/11)

## GWexpy Studio (companion app)

A first-party desktop workbench for beginners and quick-look analysis (working name
*GWexpy Studio*) is planned as a **separate repository**, not part of the gwexpy
package: open files by drag & drop, inspect their structure as a tree, process data
interactively, and export every action as plain Python (or Marimo) code that uses only
public gwexpy APIs. Studio is part of the consumer layer, outside the domain taxonomy;
what gwexpy promises it is the headless contract below, delivered through the
unnumbered later-0.x application-readiness theme. Before v0.2.0 only design prototypes
are in scope. What gwexpy itself promises is the headless contract Studio needs:
lightweight source inspection, format capability introspection, serializable operation
parameters, and provenance.

## Engineering hygiene (release-independent)

Carried over from the previous roadmap as continuous engineering work rather than
release themes: dependency locking for CI reproducibility (version ranges stay in
`pyproject.toml`), mypy strictness expansion (fail-on-error for new and core modules
first), test fixture standardization (synthetic, specification-derived, and
external-tool-generated fixtures kept distinct), release metadata automation, and
zero-warning documentation builds with a deterministic notebook execution policy.
Cross-cutting foundation X4 (performance) is release-scoped, measurement-driven work;
the items here are continuous hygiene and never gate a release.

---

> [!NOTE]
> This roadmap reflects current project priorities and is subject to change based on
> experimental needs and community feedback.
