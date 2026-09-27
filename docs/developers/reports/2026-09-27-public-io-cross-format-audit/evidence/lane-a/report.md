# Lane A: NetCDF and Zarr public I/O characterization

Source SHA: `ae10234a1e37853508c54901bf7c9e80878f25aa`. No package source, permanent test, contract, or docs were edited. No commit.

Commands, all run with cwd `/home/washimi/.config/superpowers/worktrees/gwexpy-public-io-audit/A-netcdf-zarr`:

```bash
rtk conda run --prefix /home/washimi/.config/superpowers/worktrees/gwexpy-public-io-audit/envs/present python .audit_tmp/netcdf_probe.py > .audit_tmp/netcdf_present.jsonl 2>&1
rtk conda run --prefix /home/washimi/.config/superpowers/worktrees/gwexpy-public-io-audit/envs/present python .audit_tmp/zarr_probe.py > .audit_tmp/zarr_present.jsonl 2>&1
rtk conda run --prefix /home/washimi/.config/superpowers/worktrees/gwexpy-public-io-audit/envs/base python .audit_tmp/missing_probe.py > .audit_tmp/missing_base.jsonl 2>&1
```

Dependencies: present has NumPy 1.26.4, GWpy 4.0.2, netCDF4 1.7.4, xarray 2026.2.0, Zarr 3.1.5; base has NumPy 1.26.4, GWpy 4.0.2 and lacks netCDF4/xarray/Zarr. Both import this worktree's `gwexpy/__init__.py`. All three commands exit 0; individual public route exceptions are recorded as JSON rows.

Authority: `public_io_contract.json` lists `TimeSeries`, `TimeSeriesDict`, and `TimeSeriesMatrix` read/write and `public_auto_identify: true` for both `nc` and `zarr`. Native xarray/Zarr fixture values and metadata supply the independent oracle. GWpy 4.0.2 `TimeSeries.read(.nc)` returns `ValueError` requesting an HDF5 `path=`, so it supplies no finite correct NetCDF oracle. NetCDF v2 timing is file-global. Native NetCDF variables sharing `sample` cannot have differing lengths, and arbitrary per-variable `t0`/`dt` attrs are not an authoritative v2 time axis.

| ID | Status | Impact | Public entrypoint and independent fixture | Expected from authority | Actual |
| --- | --- | --- | --- | --- | --- |
| NC-MATRIX-001 | CONFIRMED_CORRECTNESS_DEFECT | silent_value_corruption | `TimeSeriesMatrix.read(path, format='nc')`; xarray v2 2x2 metadata with only c00/c01/c10 (values 11/12/21) | Reject incomplete cell topology | Returns `TimeSeriesMatrix` shape `(2,2,3)` with cell r1,c1 never assigned from any source variable; do not rely on its incidental `np.empty` bytes |
| NC-MATRIX-002 | CONFIRMED_CORRECTNESS_DEFECT | silent_entry_loss | Same entrypoint; complete 2x2 plus fifth variable `duplicate` at (r0,c0), value 99 versus original 11 | Reject two variables for one cell | Returns shape `(2,2,3)` with r0,c0 = 99, original entry dropped |
| NC-MATRIX-003 | CONFIRMED_CORRECTNESS_DEFECT | silent_metadata_corruption | Same entrypoint; r1,c0 index 1 and r1,c1 index 2; symmetric c1 index 1/2 fixture | Reject conflicting index for same key | Both variants return shape `(2,2,3)` and normalize to labels r0/r1, c0/c1 |
| NC-MATRIX-004 | CONFIRMED_CORRECTNESS_DEFECT | silent_metadata_corruption | Same entrypoint; r0 and r1 both index 0; symmetric c0 and c1 both index 0 | Reject different keys sharing one index | Both variants return shape `(2,2,3)` with distinct labels |
| NC-MATRIX-005 | CONFIRMED_CORRECTNESS_DEFECT | silent_metadata_corruption | Same entrypoint; r1 index -1; symmetric c1 index -1 | Reject negative index | Both return; row variant orders rows `[r1,r0]`, column variant columns `[c1,c0]` |
| NC-MATRIX-006 | CONFIRMED_CORRECTNESS_DEFECT | silent_metadata_corruption | Same entrypoint; r1 or c1 index 3 while only two keys; separately index 1000 | Reject sparse/out-of-range index | All four variants return shape `(2,2,3)` with dense output positions, dropping the source index meaning |
| NC-MATRIX-007 | CONFIRMED_CORRECTNESS_DEFECT | silent_metadata_corruption | Same entrypoint; xarray variables c00/c01/c10 have `units='V'`, c11 has `units='m'` | Reject incompatible per-cell units or retain the distinct unit | Returns matrix with all four cell units `V`, including c11 value 22 mislabeled as volts |
| NC-MATRIX-008 | BLOCKED: NO_VALID_FIXTURE | explicit_error | Same entrypoint; attempted per-cell length mismatch | Valid NetCDF variables on one `sample` dimension have its fixed length | No valid distinct-length variable fixture can reach this matrix reader through the shared dimension |
| NC-MATRIX-009 | BLOCKED: NO_VALID_FIXTURE | silent_axis_corruption | Same entrypoint; attempted per-cell t0/dt mismatch | v2 `t0`/`dt` are file-global | Extra `t0` or `dt` attrs on c11 were ignored; they do not establish a distinct authoritative per-cell axis |
| NC-AXIS-001 | CONFIRMED_CORRECTNESS_DEFECT | silent_axis_corruption | `TimeSeries.read(.nc, format='nc')` and auto; independent xarray legacy numeric coordinate `[0,1,2,4,5]`, samples `[10,11,12,14,15]` | Preserve source sample times or reject irregular cadence | Returns times `[0,1,2,3,4]`, same samples; generic warning only says legacy timing precision limited; Dict and Matrix public routes also accept |
| NC-AXIS-002 | CONFIRMED_CORRECTNESS_DEFECT | silent_axis_corruption | Same routes; independent xarray datetime64 coordinate `2020-01-01T00:00:00,01,02,04,05`, same samples | Preserve 2-second gap or reject irregular cadence; GPS conversion of first timestamp is 1261872018 | Returns GPS times `[1261872018,1261872019,1261872020,1261872021,1261872022]`, compressing gap; Dict and Matrix also accept |
| NC-MATRIX-UNIT-001 | CONFIRMED_CORRECTNESS_DEFECT | silent_metadata_corruption | `TimeSeriesMatrix(..., unit='V').write(.nc, format='nc')` and auto; source cell unit V; independent xarray opens output | Store `units='V'` as single/Dict writers do | Both matrix routes store `units=''`; int64 values and t0/dt otherwise preserved |
| NC-MATRIX-UNIT-002 | CONFIRMED_CORRECTNESS_DEFECT | silent_metadata_corruption | `TimeSeriesMatrix.read(legacy_irregular.nc, format='nc')` and auto; xarray source variable `units='V'` | Retain the sole channel's V unit | Matrix cell unit is empty; public single/Dict routes retain V |
| NC-ROUTE-001 | REFUTED_STATIC_SUSPICION | api_route_mismatch | Public single/Dict/Matrix explicit and auto `.nc` reads/writes in present env | Contract routes should work | All six write routes preserve int64 values via xarray; legacy read routes return declared classes; NetCDF v2 complete matrix returns correct 2x2 values/labels |
| ZARR-DTYPE-001 | CONFIRMED_CORRECTNESS_DEFECT | silent_value_corruption | Public `TimeSeries`/Dict/Matrix `.read(..., format='zarr')` on native Zarr int64 and public `.write(..., format='zarr')` inspected by native Zarr | Native int64 `[9007199254740993,-9007199254740995,17]` exact | Read and write both yield float64 `[9007199254740992,-9007199254740996,17]`, no precision warning; t0=1234567890.25, dt=.0625, unit V retained |
| ZARR-DTYPE-002 | CONFIRMED_CORRECTNESS_DEFECT | silent_value_corruption | Same public routes; native Zarr complex64/128 stores with nonzero imaginary values | Preserve real, imaginary and phase or reject unsupported dtype | Read/write yield float64 real parts only: e.g. complex128 `(1.1+2.2j)` becomes `1.1`, phase 1.1071487 becomes 0; `ComplexWarning` is emitted |
| ZARR-DTYPE-003 | BLOCKED: AMBIGUOUS_CONTRACT | silent_dtype_change (new class: data values remain exact but dtype changes) | Same routes; native int16/int32/float32 stores and public writes inspected natively | Zarr itself preserves native dtype; public contract does not explicitly state dtype fidelity for representable values | All three become float64, values remain exact. This is a measured dtype change; classification as defect needs a dtype preservation contract decision |
| ZARR-DTYPE-004 | REFUTED_STATIC_SUSPICION | silent_value_corruption | Same routes; native float64 `[1.1,-2.2,3.3]` | Exact values and dtype | Public read and write retain float64 and exact values; axes/unit retained |
| ZARR-AUTO-001 | CONFIRMED_CORRECTNESS_DEFECT | api_route_mismatch | `TimeSeries.read(native_int64.zarr)` in present env; independent native store exists; public auto identify true | Identify `.zarr` and read one channel | `IsADirectoryError`; explicit `format='zarr'` succeeds; Dict/Matrix auto reads succeed |
| OPTIONAL-001 | CONFIRMED_INTENTIONAL_BEHAVIOR | explicit_error | Base env public explicit single/Dict/Matrix NetCDF and Zarr read/write; actual valid fixtures | Contract says missing optional dependency raises ImportError | All 12 explicit operations raise ImportError with extra installation hint; single/Dict NetCDF auto and Dict Zarr auto do likewise |
| OPTIONAL-002 | CONFIRMED_CORRECTNESS_DEFECT | api_route_mismatch | Base env `TimeSeriesMatrix.read/write(.nc)` and `.read/write(.zarr)` auto | `unavailable_behavior` says ImportError | All four return generic `ValueError: Could not identify format...`, hiding missing dependency; explicit routes raise ImportError |
| OPTIONAL-003 | CONFIRMED_CORRECTNESS_DEFECT | api_route_mismatch | Base env `TimeSeries.read(native_float64.zarr)` auto | Missing Zarr should raise ImportError | Returns `IsADirectoryError`; same auto route fails in present env |

Raw JSONL contains each source cell's row/column keys and indices, exact values, dtype, shape, axis, unit, labels, warnings, return type or exception. Fixture paths are under `.audit_tmp/netcdf/` and `.audit_tmp/zarr/`. The temporary scripts are `.audit_tmp/netcdf_probe.py`, `.audit_tmp/zarr_probe.py`, and `.audit_tmp/missing_probe.py`.

Suggested follow-up issue titles:

1. Reject incomplete and contradictory NetCDF matrix topology before allocating data.
2. Preserve or reject irregular legacy NetCDF time coordinates.
3. Preserve TimeSeriesMatrix units in NetCDF writes and validate mixed-unit cells on reads.
4. Preserve Zarr numeric dtype and complex values, or reject unsupported values before read/write.
5. Restore public auto Zarr TimeSeries reads and preserve optional dependency errors on Matrix auto routes.
