# Proposed disposition of twelve historical BLOCKED findings

Status: **accounting policy selected; individual dispositions pending**. This
accompanies the [pre-qualification record](README.md). The release owner chose
to measure the 62 executable scenarios and review these twelve individually
as applicability or retained-behavior dispositions. The S3
scientific/data-model review and release owner must still decide whether each
proposed boundary is acceptable. A probe's exit code is not acceptance
evidence.

| Proposed disposition | Finding IDs | Basis | S3 acceptance evidence |
| --- | --- | --- | --- |
| Outside the valid v2 NetCDF schema | `NC-MATRIX-008`, `NC-MATRIX-009` | The shared sample dimension fixes every cell length; the v2 time axis is file-global. The attempted mismatches do not create independent valid cells. | Inspect the generated file with xarray/netCDF4 and retain the two attempted-case probe rows. Review the schema boundary explicitly. |
| Existing dtype widening, no value loss | `ZARR-DTYPE-003` | int16, int32, and float32 native arrays return float64 with exact values, as in the historical observation. There is no agreed dtype-only requirement for these native stores. | Assert exact values and characterize dtype for each public route on exact R3 wheel/sdist. Review this as retained behavior, not a new optimization exception. |
| Existing TDMS fallback behavior | `TDMS-TIME-007`, `TDMS-TIME-008`, `TDMS-UNIT-001` | Missing channel start time falls back to custom root `DateTime` or relative zero; `unit_string` is not imported. The documented contract does not give these fields stronger authority. | Assert the current `t0`, `dt`, values, and unit in present dependency cells; explicitly decide whether these remain compatibility behavior. |
| Existing malformed-file behavior | `GBD-COUNT-001`, `ATS-TRUNC-001` | GBD uses declared Counts and ignores surplus complete rows; ATS uses available rows with a warning when its payload is short. No format-authoritative requirement for a different response has been established. | Assert result and warning/error behavior on exact R3 artifacts. Scientific review must accept retention; otherwise make the source correction before S3 freeze. |
| Outside the qualified device generation | `GBD-LEGACY-001` | The available Graphtec authority covers GL500 firmware 1.00–1.21 only. | Keep the release claim limited to that generation; do not generalize to other GBD devices. |
| Unsupported plain HDF5 group control | `HDF5-DISCOVERY-TS-DICT-GROUP-001` | Removing the GWexpy manifest routes the group-only file through plain GWpy dictionary discovery. It contains no eligible root datasets and the uncorrupted control already fails. | Confirm the plain HDF5 compatibility boundary. A new group-discovery promise needs a valid public control and separately reviewed source change. |
| Unsupported histogram dataset-per-entry layout | `HDF5-HIST-DATASET-001` | The writer physically creates groups with `values`/`edges`; the histogram reader also requires a group. The requested dataset layout does not produce a dataset-per-entry histogram. | Confirm that no supported dataset-per-entry histogram schema is claimed. Retain the physical-layout inspection. |
| Expected missing optional dependency | `OBSPY-DUP-BASE-001` | ObsPy is absent in the base cell, so the duplicate-ID reader cannot run there. The present cell exercises the four duplicate-ID variants. | Assert absence in base and qualify the variants in the optional cell. |

When all individual dispositions are approved, the release gate can say
**74 scenarios accounted for**:
36 previously fixed defect scenarios, 26 newly asserted behavior scenarios,
and twelve reviewed dispositions with their applicability boundaries. The
twelve must never be reported as passing runtime assertions. No current
disposition is approved merely by selection of this accounting policy.
