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
| `TDMS-TIME-007` | Retain root `DateTime` fallback with limited authority | Without channel `wf_start_time`, the reader uses custom root `DateTime`. No device-level evidence establishes that it is channel acquisition time. Assert `t0`, `dt`, and values on exact R3 artifacts; disclose the UTC assumption and direct users to `epoch=` when authoritative timing is required. Scientific review must explicitly accept this limited claim. |
| `TDMS-TIME-008` | Retain relative-zero fallback as a documented route | Without any source timestamp, the reader returns `t0=0` and `dt=0.25`. The prior top-level `absolute` contract was too broad for this route. Assert the public result on exact R3 artifacts and review the new route-level contract entry; relative zero must never be described as an absolute source timestamp. |
| `TDMS-UNIT-001` | Retain dimensionless result with known provenance limitation | The source `unit_string=V` is ignored while provenance says `unit_source=tdms`. Assert source property and public unit on exact R3 artifacts. Public docs now require `unit=` for known units and state that the legacy provenance marker does not certify import. Scientific review must accept this disclosed limitation or require a source correction and new candidate cycle. |
| `GBD-COUNT-001` | Retain declared-count truncation | With Counts=2 and three complete payload rows, the reader returns the first two without warning. Assert result and warning behavior on exact R3 artifacts; review malformed surplus-payload policy. |
| `GBD-LEGACY-001` | Outside qualified device generation | Available Graphtec authority covers only GL500 firmware 1.00–1.21. Keep the release claim limited to that generation; a non-GL500 case needs model-specific authority. |
| `ATS-TRUNC-001` | Retain partial-data salvage | With a five-sample header and three payload samples, the reader returns three scaled samples and warns. Assert result and warning on exact R3 artifacts; review whether this malformed-input policy remains acceptable. |
| `HDF5-DISCOVERY-TS-DICT-GROUP-001` | Inapplicable plain HDF5 control | Removing the GWexpy manifest leaves a group-only file with no eligible root datasets; its uncorrupted public control already fails under plain GWpy dictionary discovery. Review that compatibility boundary and physical layout. |
| `HDF5-HIST-DATASET-001` | Inapplicable physical dataset fixture; retain known layout mismatch | The public `layout="dataset"` argument and manifest label `dataset-per-entry` conflict with the writer's physical entry groups containing `values` and `edges`; the reader requires groups. A true dataset-per-entry fixture is not produced by the current writer. Review this discrepancy and its disclosure in the contract and release note; do not claim the requested layout is silently unsupported or that the physical schema is qualified. |
| `OBSPY-DUP-BASE-001` | Expected missing dependency | ObsPy is absent in the base cell, so that cell cannot execute duplicate-ID reading. Assert absence in base and the four duplicate-ID variants in the present optional cell. |

When all individual dispositions are approved, the release gate can say
**74 scenarios accounted for**:
36 previously fixed defect scenarios, 26 newly asserted behavior scenarios,
and twelve reviewed dispositions with their applicability boundaries. The
twelve must never be reported as passing runtime assertions. No current
disposition is approved merely by selection of this accounting policy.
