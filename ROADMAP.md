# Project Roadmap

This document describes release **themes and policy**. For the upcoming releases, the
[GitHub milestones](https://github.com/tatsuki-washimi/gwexpy/milestones) are
**authoritative for exact issue-level scope**; this document explains what each release
means and what it deliberately excludes. The long-term shape of the library is
organized as a set of capability domains; see [Capability domains](#capability-domains)
below and the
[capability-domain roadmap design](docs/developers/plans/active/2026-08-09-capability-domain-roadmap-design.md)
for the full per-domain goals, the domain-by-theme matrix, and the issue triage rules.

The v0.1.x series established security, CI, release tooling, and metadata integrity.
The project is now closing that phase with one final stabilization patch (v0.1.13) and
moving to the first semantically complete feature release (v0.2.0).

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
- **Maintenance releases** (v0.1.14, v0.2.1+) are never pre-assigned features. They are
  issued only if regressions or newly discovered bugs require them after a release.
  "Finish feature X in v0.2.1" is explicitly not allowed: a feature is either complete
  in v0.2.0 or deferred to a later minor.
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
[capability-domain roadmap design](docs/developers/plans/active/2026-08-09-capability-domain-roadmap-design.md).

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
  is unsupported; #594's resolution is that documentation correction, not an
  implementation commitment.

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

## v0.2.0 — Container Semantic Contract

> Every supported operation on a GWexpy container preserves class, unit, axes,
> labels, and metadata, or raises explicitly — never a silent downgrade.

Milestone: [v0.2.0](https://github.com/tatsuki-washimi/gwexpy/milestone/3). This
release is deliberately narrower than earlier drafts of this document: it freezes the
arithmetic contract that every other container feature will depend on, before any of
those features are built on top of it. Field I/O and the eager SegmentTable workflow
move to the next theme (below) so that they are built against an already-frozen
contract rather than against a data model that is changing underneath them at the
same time. Workstreams (individual issues live in the milestone):

- **Container arithmetic contract**
  ([#612](https://github.com/tatsuki-washimi/gwexpy/issues/612) umbrella): a
  declarative, human-reviewed regression matrix — class x operand x operator x side
  (left/right) x in-place — that is green against the *current* implementation first,
  with every currently-unsupported combination registered as an explicit `raises`
  entry (not a silent gap). The suite asserts its own collected-case count so it
  cannot regress to zero collected tests silently (the failure mode fixed by #511 in
  v0.1.13).
- **SeriesMatrix composition redesign**
  ([#637](https://github.com/tatsuki-washimi/gwexpy/issues/637)): moving the
  `ndarray`-subclass data model to composition so that `np.sqrt(matrix)` and
  `(2 * u.s) * matrix` both return the correct class, values, units, and metadata.
  This has a fixed decision date and a documented fallback: if the composition
  prototype is not green against the full test suite by that date, v0.2.0 ships with
  the current `__array_ufunc__ = None` limitation documented, and the redesign moves
  to the next theme rather than blocking this release indefinitely.
- **Pre-refactor performance baseline**: representative container operations are
  benchmarked *before* the #637 redesign lands, and the redesign is required to stay
  within a documented regression budget — bringing forward the shared benchmark
  harness ([#581](https://github.com/tatsuki-washimi/gwexpy/issues/581)) rather than
  discovering regressions after the fact.
- **GWpy-native HDF5 readability** ([#402]): golden tests that write with GWexpy and
  read back with a GWpy-only process (no `import gwexpy`), for the containers already
  covered by HDF5 I/O.
- **API stability labelling** ([#400](https://github.com/tatsuki-washimi/gwexpy/issues/400)):
  define the stable/provisional/experimental labels used from this release onward,
  since the contract above needs somewhere to record its own status.
- **Carried-over reproducibility work**: Monte-Carlo provenance across copy, slice,
  and serialization ([#508]); the `_t0_ns` precision follow-up ([#513]).

Definition of done:
1. The class x operand x operator x side x in-place contract matrix (#612) is green,
   with unsupported combinations as explicit `raises` entries and an asserted
   collected-case count.
2. `np.sqrt(matrix)` and `(2 * u.s) * matrix` both succeed with correct class, values,
   units, and metadata (#637), or the release ships with a documented
   `__array_ufunc__ = None` limitation per the fallback above.
3. HDF5 written by GWexpy for GWpy-derived containers is readable by a GWpy-only
   process (#402).
4. The #637 redesign, if it lands, stays within the documented performance
   regression budget from the pre-refactor baseline.

Non-goals for v0.2.0: Field I/O and the eager SegmentTable workflow (the
"Experiment data workflow" theme below), Histogram arithmetic (bin-compatibility and
uncertainty-propagation rules are undesigned; only the current fail-closed behaviour
is registered in the #612 contract matrix as explicit `raises` entries), coordinate
transforms and reprojection, layered visualization, lazy or aggregating segment
workflows, mesh-aware field models, and Fisher analysis.

Bounded additions — included only if completed within the release window, otherwise
deferred with no replacement release and no milestone assignment: median-mean PSD
averaging ([#409](https://github.com/tatsuki-washimi/gwexpy/issues/409),
[#410](https://github.com/tatsuki-washimi/gwexpy/issues/410)), the coupling segment
schema ([#411](https://github.com/tatsuki-washimi/gwexpy/issues/411),
[#412](https://github.com/tatsuki-washimi/gwexpy/issues/412)), and the I/O
fail-closed-to-implemented follow-ups for GWF `parallel` ([#588]) and the ndscope
HDF5 writer ([#590]).

GUI removal ([#645](https://github.com/tatsuki-washimi/gwexpy/issues/645), PR [#488])
and documentation-tree consolidation
([#606](https://github.com/tatsuki-washimi/gwexpy/issues/606)) proceed as independent
repository-level work, not as part of this milestone (see Release policy above).

## Future themes (not scheduled)

No milestones exist for these yet, and the themes may be re-scoped. Each theme below
carries a one-line release statement and a few headline user stories to make the
theme testable — this is a drafting convention, **not** a commitment to scope, order,
or a version number. Consistent with the release policy, an issue belongs to one of
these themes only if a headline user story's named acceptance artifact (a test or an
example notebook) would not pass without it. If a theme's timeline slips, drop the
items that no headline user story's acceptance artifact needs, rather than the theme
itself.

- **Experiment data workflow**: read, transform, and persist spatial Field data and
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
- **Advanced segment workflows**: reducers, `groupby`/`aggregate`, and a lazy
  `SegmentFrame` for aggregating across many experiment segments. A lazy or
  out-of-core execution path is added only once a demonstrated usage requirement
  exists — no segment-count target is set in advance. Design groundwork already in
  the [SegmentTable workflow plan](docs/developers/plans/active/2026-08-01-segmenttable-workflow-design.md).
- **Spatial geometry and layered visualization**: grid geometry, detector frames, and
  component-correct rotations (#556 theme — rotating coordinates without rotating
  vector/tensor components is treated as a defect, not an approximation), plus
  terrain/basemap/marker layer composition (#558 series). Changes to `gwexpy/fields/`
  require physics-reviewer sign-off per project convention; this theme does not start
  until that review capacity is available. Design:
  [layered visualization plan](docs/developers/plans/active/2026-08-01-layered-visualization-design.md).
- **Mesh-aware fields and solver interoperability**: bringing simulation output
  (OpenFOAM / FLOW-3D / SPECFEM3D / SimPEG) into the same metadata-aware workflow as
  measured Field data. Before building a bespoke mesh topology model, this theme
  evaluates delegating mesh representation to an existing library (`meshio`,
  `PyVista`) — a bespoke `MeshField` (#522) is only justified if that delegation
  cannot preserve unit/axis/metadata.
- **Fisher forecasting and advanced analysis**: a labeled matrix layer, spectral
  models, numerical derivatives, `FisherMatrix`, bias, and the overlap reduction
  function (#570–#574). The overlap reduction function specifically is gated on
  physics review and does not start until a reviewer is available.
- **Later 0.x — Ecosystem and application readiness** (deliberately unnumbered):
  one or more minors between the themes above and v1.0 that graduate matured
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
  [capability-domain roadmap design](docs/developers/plans/active/2026-08-09-capability-domain-roadmap-design.md).
- Cross-cutting foundations X1 (semantic contract) through X4 (performance) apply
  across all domains, not only the ones that happened to ship them first.
- The public API surface is frozen under the #400 stability labels, with a
  documented deprecation window (in releases and in time).
- No new feature domains are introduced as part of reaching v1.0 — it is a
  stabilization milestone, not a feature release.

## Ecosystem & Interoperability (Backlog)

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
3. **Virgo data-path completion** (`#591` umbrella): `.ffl` support is deferred until it has a
   separately reviewed I/O contract; the dataDisplay ROOT product converters (`#598`–`#600`)
   follow the structural inventory (`#595`).
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
