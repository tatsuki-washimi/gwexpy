# Roadmap

This page provides the public roadmap entry point for GWexpy documentation and feature planning.
It is intended as a lightweight overview rather than a release contract.

*Last updated: 2026-08-01.*

## How to Read This Page

- **Near-term**: areas we expect to improve in upcoming iterations
- **Mid-term**: work we want to expose more broadly after the current docs and API cleanup
- **Long-term**: exploratory directions that are useful but not yet scheduled

The roadmap is public, but priorities can change based on research needs, maintenance cost, and upstream dependencies.

## Near-Term Focus

- Reliability hardening: eliminating the known cases where data could be returned with
  silently wrong values, units, or dropped metadata instead of an explicit error
- Documentation quality improvements across navigation, accessibility, and search
- Clearer migration guidance for GWpy users
- Better notebook and tutorial reliability in CI
- More explicit references between tutorials, guides, and API pages

## Mid-Term Focus

- Direct file reading and writing for field containers (regular-grid scalar, vector,
  and tensor fields), including terrain/elevation and GeoTIFF sources, with units,
  axes, and metadata preserved end to end
- A complete segment-based analysis workflow: load segments, select, process with
  explicit error and missing-data handling, persist, and resume
- An explicit behavioural contract for container arithmetic: predictable units,
  metadata propagation, and explicit failures instead of silent type downgrades
- Broader CLI coverage beyond the current prototype-stage interface
- More analysis workflows for noise characterization and time-frequency studies
- Expanded interoperability guides for external scientific Python libraries, including reading
  HDF5 products written by data-preparation tools and importing outputs from interferometer
  design and simulation packages
- More public validation notes for numerical and physics-facing algorithms

For how these bridges relate to the surrounding ecosystem, and which projects are deliberately
left out of scope, see [Where GWexpy Sits in the GW Python Ecosystem](ecosystem).

## Long-Term Directions

- Spatial geometry for field data: coordinate frames, detector-aligned rotations, and
  layered map visualization combining terrain, physical fields, and site markers
- Mesh-aware field models and interoperability with numerical solver outputs
- Forecasting and inference utilities built on the matrix containers, such as Fisher
  analysis
- Richer visual discovery paths across examples and reference content
- More automation around documentation verification and sample-code validation
- Potential improvements to theme, search quality, and interactive visual components

## Public Tracking Sources

- [Feedback form for lightweight bug reports and feature requests](https://forms.gle/c8jJaf9UCs5tb5cC8)
- [GitHub Issues](https://github.com/tatsuki-washimi/gwexpy/issues)
- [Security policy](https://github.com/tatsuki-washimi/gwexpy/blob/main/SECURITY.md) for vulnerability reports; do not include vulnerability details in the form or public issues.
- [GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases)
- [Changelog](../about/changelog.md)
- [Developer roadmap (`ROADMAP.md`)](https://github.com/tatsuki-washimi/gwexpy/blob/main/ROADMAP.md) for release themes and policy details

## Scope Note

This roadmap is a planning aid, not a guarantee of delivery order or release timing.
GWexpy is research-oriented software and some priorities may shift as the package evolves.
