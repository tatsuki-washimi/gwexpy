# Independent pre-freeze check of twelve proposed dispositions

This read-only check examined the local candidate at
`8be700962a973bf3de1c7486b3c813365c81aec0` before S3 freeze. It is
not one of the required independent reviews of an exact S3 SHA, and it does
not approve a release-owner disposition. The reviewer checked the proposed
finding against the historical audit, current R2 observation, implementation,
and documented contract.

| Finding | Pre-review | Required resolution before S3 review |
| --- | --- | --- |
| `NC-MATRIX-008` | Accept | Keep the v2 shared-dimension boundary explicit. |
| `NC-MATRIX-009` | Accept | Keep the v2 file-global time-axis boundary explicit. |
| `ZARR-DTYPE-003` | Accept with qualification | Verify exact values and widened dtype on both exact R3 artifacts; disclose dtype widening. |
| `TDMS-TIME-007` | Insufficient authority | Do not imply a custom root `DateTime` is guaranteed acquisition time; disclose its assumption and `epoch=` override. |
| `TDMS-TIME-008` | Contract correction needed | Document the relative-zero route separately from normal absolute timestamps. |
| `TDMS-UNIT-001` | Contract correction needed | Disclose ignored `unit_string`, the `unit=` override, and the misleading legacy provenance marker. |
| `GBD-COUNT-001` | Accept with qualification | Approve silent surplus-row handling only as a malformed-file policy. |
| `GBD-LEGACY-001` | Accept | Restrict the claim to tested GL500 firmware 1.00–1.21. |
| `ATS-TRUNC-001` | Accept with qualification | Approve warned partial-data salvage only as a malformed-file policy. |
| `HDF5-DISCOVERY-TS-DICT-GROUP-001` | Accept | Restrict the finding to the manifest-free group-only control. |
| `HDF5-HIST-DATASET-001` | Contract correction needed | Disclose that `layout="dataset"` writes physical groups while its manifest says `dataset-per-entry`; do not qualify a nonexistent physical dataset schema. |
| `OBSPY-DUP-BASE-001` | Accept | Confirm dependency absence in base and duplicate-ID cases in the optional cell. |

The [revised proposal](disposition-proposal.md), public I/O contract, user
guide, and release note make the four questioned boundaries explicit without
changing the R2 runtime tree. Exact-S3 scientific/data-model review must still
decide each of the twelve findings individually. Exact-R3 artifact evidence
and release-owner approval remain separate gates.

A read-only follow-up on `373b6c0046bedc3326a092a0676a6adbf7530ead`
accepted all four revised dispositions **as descriptions of retained behavior
and known limits only**. It did not establish device-level timing authority,
unit import, or a physical histogram dataset-per-entry schema. The twelve
formal S3 dispositions remain pending.
