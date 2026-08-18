# v0.2.0 API classification for the current `[Unreleased]` record

This is the smallest useful classification of the public surfaces covered by
the current implementation evidence.
It does not classify every legacy API.
An API label is not a release promise, and a deferred release outcome is not an
API tier.

| Label | Public surfaces in this lane | Rationale |
| --- | --- | --- |
| stable | `SeriesMatrix arithmetic contract`; `API stability policy semantics` | These are explicit, reviewed contracts. A breaking change requires the policy's compatibility and release-note treatment. |
| provisional | `t0_ns`; `t0_gps_ns`; `HDF5 sidecar restoration`; `provenance mapping and operation schema`; `median_bias`; `GWF parallel`; `nproc compatibility alias`; `NDScope dataset_options` | These surfaces have focused implementation evidence and documented failure behavior, but remain bounded to this release's evidence and may change through the provisional migration process. |
| experimental | `coupling segment v1 schema` | The schema is validated and versioned, but its scientific generality and downstream interchange stability are not yet established. It is opt-in evidence, not a broad coupling standard. |

## Release outcomes

`deferred` is a release outcome, not a stability label.
It does not classify an API, create a fourth compatibility promise, or serve as
a substitute for an API stability label.
For this lane, #637 is partial with release outcome deferred and
`adopted: false`; the #403 broad `nproc` migration is partial with release
outcome deferred/out-of-scope.
