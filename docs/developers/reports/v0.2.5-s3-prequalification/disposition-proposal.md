# Proposed disposition of twelve historical BLOCKED findings

Status: **accounting policy selected; individual dispositions pending**. This
accompanies the [pre-qualification record](README.md). The release owner chose
to measure the 62 executable scenarios and review these twelve individually
as applicability or retained-behavior dispositions. The S3
scientific/data-model review and release owner must still decide whether each
proposed boundary is acceptable. A probe's exit code is not acceptance
evidence.

| Finding | Proposed individual disposition | Basis and required S3/R3 evidence |
| --- | --- | --- |
| `NC-MATRIX-008` | Inapplicable fixture | A v2 NetCDF matrix uses one shared sample dimension; the attempted per-cell length mismatch cannot be represented as a valid independent cell. Review the schema and retained attempted-case row. |
| `NC-MATRIX-009` | Inapplicable independent-axis fixture | The v2 time axis is file-global. Extra per-cell time attributes do not create an authoritative independent axis. Review the schema and both attempted-case rows. |
| `ZARR-DTYPE-003` | Retain characterized dtype widening | Native int16, int32, and float32 arrays return float64 with exact values. Assert values and dtype on exact R3 wheel/sdist and decide whether dtype-only preservation is required by the public contract. |
| `TDMS-TIME-007` | Retain root `DateTime` fallback | Without channel `wf_start_time`, the current reader uses custom root `DateTime`. Assert `t0`, `dt`, and values on exact R3 artifacts and review whether that property is authoritative. |
| `TDMS-TIME-008` | Retain relative-zero fallback | Without an absolute source time, the current reader returns `t0=0` and `dt=0.25`. Assert the public result on exact R3 artifacts and review whether missing absolute time should instead fail. |
| `TDMS-UNIT-001` | Retain dimensionless result | The source `unit_string=V` is not imported. Assert source property and public unit on exact R3 artifacts and review whether unit import belongs to the current contract. |
| `GBD-COUNT-001` | Retain declared-count truncation | With Counts=2 and three complete payload rows, the reader returns the first two without warning. Assert result and warning behavior on exact R3 artifacts; review malformed surplus-payload policy. |
| `GBD-LEGACY-001` | Outside qualified device generation | Available Graphtec authority covers only GL500 firmware 1.00–1.21. Keep the release claim limited to that generation; a non-GL500 case needs model-specific authority. |
| `ATS-TRUNC-001` | Retain partial-data salvage | With a five-sample header and three payload samples, the reader returns three scaled samples and warns. Assert result and warning on exact R3 artifacts; review whether this malformed-input policy remains acceptable. |
| `HDF5-DISCOVERY-TS-DICT-GROUP-001` | Inapplicable plain HDF5 control | Removing the GWexpy manifest leaves a group-only file with no eligible root datasets; its uncorrupted public control already fails under plain GWpy dictionary discovery. Review that compatibility boundary and physical layout. |
| `HDF5-HIST-DATASET-001` | Unsupported requested layout | The writer physically creates entry groups containing `values` and `edges`, even when asked for dataset layout; the reader requires groups. Review the physical-layout evidence and absence of a dataset-per-entry promise. |
| `OBSPY-DUP-BASE-001` | Expected missing dependency | ObsPy is absent in the base cell, so that cell cannot execute duplicate-ID reading. Assert absence in base and the four duplicate-ID variants in the present optional cell. |

When all individual dispositions are approved, the release gate can say
**74 scenarios accounted for**:
36 previously fixed defect scenarios, 26 newly asserted behavior scenarios,
and twelve reviewed dispositions with their applicability boundaries. The
twelve must never be reported as passing runtime assertions. No current
disposition is approved merely by selection of this accounting policy.
