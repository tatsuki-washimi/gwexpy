# Project Roadmap

This document describes release **themes and policy**. For the upcoming releases, the
[GitHub milestones](https://github.com/tatsuki-washimi/gwexpy/milestones) are
**authoritative for exact issue-level scope**; this document explains what each release
means and what it deliberately excludes.

The v0.1.x series established security, CI, release tooling, and metadata integrity.
The project is now closing that phase with one final stabilization patch (v0.1.13) and
moving to the first semantically complete feature release (v0.2.0).

## Release policy

- **Patch releases (v0.x.y)** contain bug fixes only — no new features, no new public
  APIs, no new dependencies.
- **Maintenance releases** (v0.1.14, v0.2.1+) are never pre-assigned features. They are
  issued only if regressions or newly discovered bugs require them after a release.
  "Finish feature X in v0.2.1" is explicitly not allowed: a feature is either complete
  in v0.2.0 or deferred to a later minor.
- **Milestones are not created in advance**: the milestone for the next-but-one minor
  is created only after the preceding minor has shipped. The future themes below are
  directional, not commitments.
- Documentation, tests, and contract updates are part of each feature's definition of
  done — not a separate release theme.

## v0.1.13 — Silent-corruption stabilization patch (next)

> Close every known case where GWexpy returns wrong numbers, wrong units, or silently
> dropped metadata without raising.

Milestone: [v0.1.13](https://github.com/tatsuki-washimi/gwexpy/milestone/8). Scope by
category (this is the only section of this document that enumerates issues
exhaustively; the milestone remains authoritative):

- **Wrong numbers / units / dtype (P0)**: ROOT non-double histograms read as `float64`
  ([#593]); Quantity/Unit operands capturing containers and dropping class, unit, or
  metadata in `SeriesMatrix`, `SpectrogramMatrix`, Field collections, and `Histogram`
  ([#575], [#576], [#577], [#578], [#579]); WIN reader decoding only 8 of the 12
  sampling-rate bits ([#610]).
- **GWpy compatibility**: `TimeSeries.rms()` signature incompatibility ([#451], fixed
  via PR [#453]).
- **Accepted-but-ignored arguments**: `start`/`end` silently ignored by some readers
  ([#611]); GWF `parallel`/`nproc` no-op ([#588]); ndscope HDF5 writer creation kwargs
  ignored ([#590]).
- **CI integrity**: dedicated gates that can pass with zero collected tests ([#511]).
- **Documentation vs implementation**: SegmentTable reference describing unimplemented
  APIs ([#605]); GWinc docstring pointing at a nonexistent classmethod ([#608]); and
  the one-line `VectorField.plot(stride=)` fix ([#559]).
- **Verification-only items (not release blockers)**: GWpy-only HDF5 readability
  ([#402], stays in v0.2.0 — manual check only for this release). The public
  documentation now correctly states that `.ffl` is unsupported; #594's resolution
  is that documentation correction, not an implementation commitment.

Explicitly excluded: any new feature or API, new dependencies, large refactors, and
PR [#488] (GUI extraction — merged right after this release).

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
[#508]: https://github.com/tatsuki-washimi/gwexpy/issues/508
[#513]: https://github.com/tatsuki-washimi/gwexpy/issues/513
[#605]: https://github.com/tatsuki-washimi/gwexpy/issues/605
[#608]: https://github.com/tatsuki-washimi/gwexpy/issues/608
[#559]: https://github.com/tatsuki-washimi/gwexpy/issues/559
[#402]: https://github.com/tatsuki-washimi/gwexpy/issues/402
[#594]: https://github.com/tatsuki-washimi/gwexpy/issues/594
[#488]: https://github.com/tatsuki-washimi/gwexpy/pull/488

## v0.2.0 — Field I/O and Semantic Contracts

> Read, analyse, and persist experiment and simulation data with units, axes, and
> metadata intact — under an explicit, tested behavioural contract.

Milestone: [v0.2.0](https://github.com/tatsuki-washimi/gwexpy/milestone/3). Five
workstreams (individual issues live in the milestone and the umbrella issues):

1. **Container semantic contract**
   ([#612](https://github.com/tatsuki-washimi/gwexpy/issues/612) umbrella): promote
   the v0.1.13 arithmetic fixes to a formal public contract — supported operands and
   operators, preserved metadata, and an explicit-failure policy instead of silent
   downgrade to bare `ndarray`/`Quantity` — enforced by a parametrized
   class x operand x operator regression matrix.
2. **Field I/O**: GWpy-style `.read()`/`.write()` for `ScalarField`, `VectorField`,
   `TensorField`, and `FieldDict`, including a canonical full-fidelity HDF5
   persistence format, GSI DEM and GeoTIFF readers with a geospatial baseline
   (angle-domain axes, geo metadata, explicit `fill_missing`), and inclusion of Field
   formats in the public I/O contract. Design:
   [terrain/ScalarField I/O plan](docs/developers/plans/active/2026-07-31-terrain-scalarfield-io-design.md).
3. **SegmentTable eager workflow**
   ([#592](https://github.com/tatsuki-washimi/gwexpy/issues/592) umbrella): explicit
   cell status model, `errors=` policy with missing reporting, column expressions,
   row/column operations, and HDF5 persistence with schema version and provenance.
   Design:
   [SegmentTable workflow plan](docs/developers/plans/active/2026-08-01-segmenttable-workflow-design.md).
4. **Documentation consolidation**
   ([#606](https://github.com/tatsuki-washimi/gwexpy/issues/606)): `docs_redesign`
   becomes the single documentation tree; the legacy `docs/web` is redirected or
   retired so contract tests and translations have one source of truth.
5. **GUI separation completion** (PR
   [#488](https://github.com/tatsuki-washimi/gwexpy/pull/488)): gwexpy ships without
   the GUI package, which now lives in the separate pyaggui project.
6. **Deferred reproducibility and Virgo I/O**: preserve Monte-Carlo provenance across
   copy, slice, and serialization ([#508]); complete the `_t0_ns` precision follow-up
   ([#513]); and implement `.ffl` support only with a separately reviewed I/O contract.

Non-goals for v0.2.0: coordinate transforms and reprojection (the
[#556](https://github.com/tatsuki-washimi/gwexpy/issues/556) theme), basemap and
layered visualization (the
[#558](https://github.com/tatsuki-washimi/gwexpy/issues/558) series), lazy or
aggregating segment workflows (v0.3.0 theme), mesh-aware field models
([#522](https://github.com/tatsuki-washimi/gwexpy/issues/522)), and Fisher analysis
([#570](https://github.com/tatsuki-washimi/gwexpy/issues/570)).

## Future themes (not scheduled)

No milestones exist for these yet, and the themes may be re-scoped:

- **v0.3.0 — Advanced segment workflows**: reducers, `groupby`/`aggregate`,
  lazy `SegmentFrame`, `reshape`/`explode`. Design groundwork already in the
  [SegmentTable workflow plan](docs/developers/plans/active/2026-08-01-segmenttable-workflow-design.md).
- **v0.4.0 — Spatial geometry and layered visualization**: grid geometry, detector
  frames, and component-correct rotations (#556 theme), plus terrain/basemap/marker
  layer composition (#558 series). Design:
  [layered visualization plan](docs/developers/plans/active/2026-08-01-layered-visualization-design.md).
- **v0.5.0 — Mesh-aware fields and solver interoperability**: a mesh-topology-aware
  field model (#522) with explicit interpolation, then OpenFOAM / FLOW-3D / SPECFEM3D
  / SimPEG readers on top of it.
- **v0.6.0 — Fisher forecasting**: labeled matrix layer, spectral models, numerical
  derivatives, `FisherMatrix`, bias, and the overlap reduction function
  (#570–#574; ORF gated on physics review).

## Ecosystem & Interoperability (Backlog)

Unscheduled work on connecting GWexpy to neighbouring projects. Nothing here is assigned to a
release; items move to a milestone only when started. The user-facing positioning statement is
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
2. **Differometor converters** (`#423`–`#427`). Design is settled but the existing issue bodies
   assume result objects that Differometor does not have; `#423` must be closed with the real
   API shape before `#424`–`#426` are implemented.
3. **Virgo data-path completion** (`#591` umbrella): `.ffl` support is deferred to
   v0.2.0 after its separately reviewed I/O contract; the dataDisplay ROOT product
   converters (`#598`–`#600`) follow the structural inventory (`#595`).
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
  `SegmentFrame` theme; needs a demonstrated usage requirement first.
- GUI redesign beyond pyaggui — a support-policy question, not just a feature.
- Meta-analysis and population-level statistics.
- 3D surface/volume data models and rendering.

## GWexpy Studio (companion app)

A first-party desktop workbench for beginners and quick-look analysis (working name
*GWexpy Studio*) is planned as a **separate repository**, not part of the gwexpy
package: open files by drag & drop, inspect their structure as a tree, process data
interactively, and export every action as plain Python (or Marimo) code that uses only
public gwexpy APIs. Before v0.2.0 only design prototypes are in scope; application
development starts against `gwexpy>=0.2.0`. What gwexpy itself promises is the
headless contract Studio needs: lightweight source inspection, format capability
introspection, serializable operation parameters, and provenance.

## Engineering hygiene (release-independent)

Carried over from the previous roadmap as continuous engineering work rather than
release themes: dependency locking for CI reproducibility (version ranges stay in
`pyproject.toml`), mypy strictness expansion (fail-on-error for new and core modules
first), test fixture standardization (synthetic, specification-derived, and
external-tool-generated fixtures kept distinct), release metadata automation, and
zero-warning documentation builds with a deterministic notebook execution policy.

---

> [!NOTE]
> This roadmap reflects current project priorities and is subject to change based on
> experimental needs and community feedback.
