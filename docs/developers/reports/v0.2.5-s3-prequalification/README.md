# v0.2.5 S3 pre-qualification of 38 historical audit scenarios

Status: **pre-qualification only; release gate OPEN**. Source: R2
`159a338081e1fca2c77030cabe160a458217563b` (the local convergence
commit has the same tree). This record does not replace exact-S3/R3 artifact
qualification.

The [historical audit matrix](../2026-09-27-public-io-cross-format-audit/runtime-characterization-matrix.json)
contains 74 scenarios. The [post-fix matrix](../2026-09-27-public-io-cross-format-post-fix-matrix.json)
verified 36 defects and left 38 scenarios `NOT_REQUALIFIED`. The R2 wheel
(`96e1f72e68404c718ad6f881119ede983782d469db6d146141659eac21bd469d`)
and sdist (`515bb91a960e6dfc6f85537bcf4775fa6cbf7e1c60c14141608593160f2d2f7d`)
were installed separately. Both completed the original 17 audit probe
commands. The relevant raw JSONL outputs for both artifacts are stored under
`raw/`; `prequalification-38.json` binds each result to the raw SHA-256 and row
indices. Run `python verify_prequalification.py` here to regenerate it.

The 26 previously non-blocked scenarios pass case-specific assertions on
both installed artifacts. `NC-ROUTE-001` needed a supplemental public-read
probe because the original command recorded only a complete matrix read and
six public writes. `probe_netcdf_routes.py --fixture-root <generated-netcdf-dir>`
reads the six written fixtures through explicit and auto routes; wheel and
sdist return identical exact int64 values. Its output is stored in
`raw/nc-routes-{wheel,sdist}.jsonl`.

## Unresolved historical cases

These twelve cases remain **BLOCKED**. A successful probe process does not
turn them into passing scenarios. The row indices and raw hashes in
`prequalification-38.json` identify the current observations for each case.
The verifier asserts each observed blocked condition on wheel and sdist;
that characterization does not settle its fixture or contract authority.

| Finding | Current observation | Required disposition before S3 freeze |
| --- | --- | --- |
| `NC-MATRIX-008` | The attempted per-cell length mismatch cannot be represented on the shared NetCDF sample dimension. | Confirm that this fixture is inapplicable to the v2 schema, or provide a valid independent fixture. |
| `NC-MATRIX-009` | Extra per-cell time attributes do not override the file-global v2 axis. | Confirm the file-global timing authority, or provide an independently authoritative cell axis. |
| `ZARR-DTYPE-003` | Native int16, int32, and float32 values remain exact but are exposed as float64. | Decide whether dtype-only widening is accepted existing compatibility behavior; a new dtype requirement would require source work and review. |
| `TDMS-TIME-007` | When channel `wf_start_time` is absent, the reader uses a custom root `DateTime` as `t0`. | Decide whether the custom root property is authoritative for channel acquisition time. |
| `TDMS-TIME-008` | With no absolute time property, the reader returns relative `t0=0` and `dt=0.25`. | Decide whether relative zero is accepted or missing absolute time must fail. |
| `TDMS-UNIT-001` | Source `unit_string=V` is not imported; public unit is dimensionless. | Decide whether unit import belongs to the current contract or a later feature. |
| `GBD-COUNT-001` | A header declaring two rows with three payload rows returns the first two without warning. | Set malformed surplus-payload policy; preserve current behavior until approved otherwise. |
| `GBD-LEGACY-001` | No non-GL500 authoritative malformed-header fixture exists. | Limit this candidate's claim to GL500 firmware 1.00–1.21, or supply model-specific authority. |
| `ATS-TRUNC-001` | A five-sample header with three payload samples returns three scaled samples and warns. | Approve or reject partial-data salvage for malformed ATS input. |
| `HDF5-DISCOVERY-TS-DICT-GROUP-001` | The manifest-free group-layout control fails as an empty plain GWpy `TimeSeriesDict`; no valid control exists. | Decide whether that layout is outside the plain HDF5 compatibility contract, or create a valid public control before judging corrupt-entry behavior. |
| `HDF5-HIST-DATASET-001` | The histogram writer creates entry groups even when dataset layout is requested; `Histogram.read` requires a group containing `values` and `edges`. | Confirm dataset-per-entry is inapplicable, or define and test an independent supported layout. |
| `OBSPY-DUP-BASE-001` | The base environment has no ObsPy, so the duplicate-ID reader cannot run. | Treat this as an expected dependency absence and qualify duplicate behavior in the optional cell, or define a different base-cell oracle. |

No new runtime defect was established by the 26 passing checks. The twelve
blocked dispositions still need explicit review. No release readiness checkbox
is closed by this pre-qualification.
The [disposition proposal](disposition-proposal.md) records a reviewable
way to account for all twelve without labeling them as passing runtime tests.
