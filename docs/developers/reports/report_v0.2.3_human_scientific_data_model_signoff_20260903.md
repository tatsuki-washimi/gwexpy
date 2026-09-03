---
schema: gwexpy-v023-human-scientific-data-model-signoff-v1
status: approved
date: "2026-09-03"
approver_role: release owner
approved_candidate_sha: c7b79db7fee2e646069679a0efe3d65c7ed4e562
approval_scope:
  release: v0.2.3
  kind: human scientific/data-model sign-off
  covered_records:
    - docs/developers/plans/manifests/audit-manifest-v0.2.3-phase2.yaml
    - docs/developers/plans/manifests/audit-manifest-v0.2.3-phase3.yaml
    - docs/developers/plans/manifests/audit-manifest-v0.2.3-timeseries-signal.yaml
    - docs/developers/plans/manifests/audit-manifest-v0.2.3-timeseries-terminal.yaml
    - docs/developers/plans/manifests/audit-manifest-v0.2.3-constructor-terminal.yaml
    - docs/developers/plans/manifests/audit-manifest-v0.2.3-stats-compat.yaml
    - docs/developers/plans/manifests/audit-manifest-v0.2.3-type-collision-compat.yaml
    - docs/developers/plans/manifests/audit-manifest-v0.2.3-scalarfield-diff-comparison.yaml
    - docs/developers/plans/manifests/audit-manifest-506-rayleigh-null-model.yaml
  excludes:
    - release GO decision
    - same-candidate scientific/data-model review
    - same-candidate release-security review
    - candidate-wide QA
    - 19-cell qualification
accepted_parent_parity_risks:
  - mixed_unit_csd_v2_per_hz_label
  - public_rayleigh_parent_segments_private_corrected_route_finite_mc_limits
  - signal_dimensionless_raw_quantity_float32_underflow
  - stale_array2d_plane2d_min_max_indices
  - stale_numeric_swapaxes_transpose_metadata
unconditionally_approved_contracts:
  - ifft_exact_time_lifecycle
  - constructor_prefix_keyword_only_extensions
  - coherent_dimension_reductions
  - quantity_out_validation_precedence_atomic_conversion_dimensionless_success
  - bifrequencymap_axes
  - scalarfield_diff_comparison
non_intersecting_window_safety:
  issue: "#611"
  status: previously-approved-not-reapproved
release_decision:
  status: HOLD
  remaining_gates:
    - same-candidate scientific/data-model review
    - same-candidate release-security review
    - candidate-wide QA
    - 19-cell qualification
invalidation_rule: >-
  Any later runtime/data-model semantic change invalidates this sign-off and
  requires reapproval.
---

# v0.2.3 human scientific/data-model sign-off

## Decision and candidate boundary

The v0.2.3 scientific and data-model contracts listed below are approved.
This approval is bound only to commit
`c7b79db7fee2e646069679a0efe3d65c7ed4e562` on branch
`fix/v023-gwpy-behavioral-compat`.

Documentation-only commits may follow this record without rebinding the
approved candidate. The canonical #639 inventory semantics and its candidate
identity remain unchanged. Any later runtime/data-model semantic change
invalidates this sign-off and requires reapproval against the new candidate.

## Unconditionally approved contracts

The following contracts are approved without a residual scientific or
data-model condition:

- `FrequencySeries.ifft()` metadata and the private exact-time lifecycle,
  including propagation and invalidation rules.
- The inherited constructor prefix and keyword-only GWexpy physical-axis
  extensions.
- Coherent dimension reductions, including result dimensions and surviving
  axis authority.
- Quantity `out=` validation precedence, atomic unit conversion, and successful
  dimensionless results.
- BifrequencyMap axis orientation and frequency-axis authority.
- ScalarField finite-difference and comparison contracts, including units,
  physical-grid validation, metadata authority, and nonfinite behavior.

## Accepted parent-parity risks

Exactly five parent-parity risks are accepted for this candidate:

1. The mixed-unit CSD `V²/Hz` label is retained because the public default
   follows the parent result and unit.
2. The public Rayleigh route retains parent segment selection while the
   corrected route remains private; the known finite-Monte-Carlo limitations
   remain disclosed and are not represented as corrected by this approval.
3. Signal compatibility retains dimensionless signal outputs, raw-magnitude
   frequency `Quantity` handling, and float32 RMS underflow on the public
   parent-compatible routes.
4. Array2D/Plane2D `min`/`max` retain stale indices where GWpy does so.
5. Numeric `swapaxes`/`transpose` retain stale metadata where GWpy does so.

These are approvals to preserve observable parent behavior, not authority to
introduce additional default divergence. Corrected GWexpy-only routes remain
explicit or private as already recorded.

## Existing #611 decision

The `non_intersecting_window_safety` decision for #611 was previously approved
under its own evidence and is not reapproved by this aggregate decision. Its
scope remains only the completely disjoint HDF5 read-window subcase, and its
separate approval record remains authoritative.

## Release gate

This human sign-off does not make the release GO. The global release decision
remains **HOLD** until all of the following complete against the required
candidate:

- same-candidate scientific/data-model review;
- same-candidate release-security review;
- candidate-wide QA; and
- 19-cell qualification.

No push, tag, publication, or release action is authorized by this record.
