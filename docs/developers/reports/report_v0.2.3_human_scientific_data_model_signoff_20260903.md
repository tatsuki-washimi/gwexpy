---
schema: gwexpy-v023-human-scientific-data-model-signoff-v1
status: approved
historical_approval:
  status: approved
  date: "2026-09-03"
  approver_role: release owner
  candidate_sha: c7b79db7fee2e646069679a0efe3d65c7ed4e562
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
current_candidate:
  sha: 0a3d09a117827113b02e4a2ce73bccd3b1ba95d2
  status: approved
  date: "2026-09-03"
  approver_role: release owner
  approval_scope:
    accepted_parent_parity_risks:
      - mixed_unit_csd_v2_per_hz_label
      - public_rayleigh_parent_segments_private_corrected_route_finite_mc_limits
      - signal_dimensionless_raw_quantity_float32_underflow
      - stale_array2d_plane2d_min_max_indices
      - stale_numeric_swapaxes_transpose_metadata
    other_contracts: excluded
inventory_evidence:
  historical_approved_candidate:
    sha: c7b79db7fee2e646069679a0efe3d65c7ed4e562
    logical_members: 575
    evidence_selectors: 59
    executed_cases_per_oracle: 384
  current_candidate:
    sha: 0a3d09a117827113b02e4a2ce73bccd3b1ba95d2
    logical_members: 575
    evidence_selectors: 62
    executed_cases_per_oracle: 396
non_intersecting_window_safety:
  issue: "#611"
  status: approved-separately-unchanged
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

## Historical decision and current candidate approval boundary

The v0.2.3 scientific and data-model contracts listed below received human
approval on 2026-09-03. This historical approval is bound only to commit
`c7b79db7fee2e646069679a0efe3d65c7ed4e562` on branch
`fix/v023-gwpy-behavioral-compat`.

Candidate `0a3d09a117827113b02e4a2ce73bccd3b1ba95d2` restores native HDF5
auto-identification and is therefore a later runtime/data-model semantic
change. On 2026-09-03, the release owner reapproved the human
scientific/data-model sign-off for that exact runtime candidate. The current
approval is bound only to
`0a3d09a117827113b02e4a2ce73bccd3b1ba95d2` and exactly the five parent-parity
risks below. It does not approve any other contract. The intervening
documentation-only correction at
`5d27c4ae8a1341e79aadf4751449784b2f575455` and this recording change do not
alter or rebind the runtime candidate.

Setting `gwexpy/_version.py` to `0.2.3` and synchronizing its citation metadata
is release identity only. It is not a runtime/data-model semantic change, does
not rebind the current approval from
`0a3d09a117827113b02e4a2ce73bccd3b1ba95d2`, and does not broaden the approval
beyond exactly the five accepted parent-parity risks. The global decision
remains **HOLD**, with all four later gates still pending.

The historical c7 inventory evidence covered 575 logical members with 59
selectors and 384 executed cases per oracle. The current 0a3d inventory still
has 575 logical members, but its HDF5 evidence expands the executable closure
to 62 selectors and 396 executed cases per oracle. The generated inventory's
1,150 case rows remain a separate logical version-row count.

## Historically approved contracts

The following contracts were approved without a residual scientific or
data-model condition for the historical c7 candidate. They remain historical
evidence and are not represented as approved by the current-candidate
reapproval:

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

Exactly five parent-parity risks were accepted for the historical c7 candidate.
The current-candidate approval is strictly limited to exactly the same five
risks:

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

These historical and current-candidate approvals preserve observable parent
behavior; they are not authority to introduce additional default divergence.
Corrected GWexpy-only routes remain explicit or private as already recorded.

## Existing #611 decision

The `non_intersecting_window_safety` decision for #611 was previously approved
under its own evidence, remains approved separately and unchanged, and is not
reapproved by this aggregate record. It is not invalidated by the aggregate
reset and does not need reapproval. Its scope remains only the completely
disjoint HDF5 read-window subcase, and its separate approval record remains
authoritative.

## Release gate

This human sign-off does not make the release GO. The global release decision
remains **HOLD** until all of the following complete against the current
candidate:

- same-candidate scientific/data-model review;
- same-candidate release-security review;
- candidate-wide QA; and
- 19-cell qualification.

No push, tag, publication, or release action is authorized by this record.
