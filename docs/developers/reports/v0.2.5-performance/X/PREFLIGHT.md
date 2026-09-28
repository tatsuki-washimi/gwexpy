# X / #586 copy audit preflight

Status: `UNBASELINED`. This is a source audit, not performance evidence. Freeze
B-X against the integrated tree after the remaining F2 SDB disposition; do not
edit runtime code before that freeze.

| Path | Current copy | Decision for B-X |
|---|---|---|
| ATS native reader | `data_raw.astype(float64) * lsb_mV / 1000.0` creates an output-sized cast and arithmetic temporaries before the final float64 result. | Measure a large int32 and int64 file. Test a single preallocated float64 result with ordered in-place multiply and divide if B1 bit patterns, warnings, metadata, and dtype remain identical. |
| NetCDF native matrix reader | Its `lossless()` check casts and round-trips each cell even when the source and common result dtype are identical. | Measure homogeneous float64 and int64 matrices. A same-numeric-dtype shortcut is eligible only if B1 result and warning/error behavior remain identical, including nonfinite cells. Keep cross-dtype validation. |
| Seismic reader | `trace.data.astype(float)` unconditionally copies float64 traces; integer traces may also be cast earlier for NaN gap padding. | Characterize B1 ownership and all public ObsPy routes before considering `copy=False`. Do not optimize by aliasing mutable backend buffers without a public-behavior check. |
| GBD reader | `bytes`-backed integer samples become writable float64 through `astype(float64)`. | Retain the conversion until an independent output-ownership and dtype proof identifies an avoidable copy. |
| Zarr/NetCDF collection stacking | `stack` and matrix assembly create the required combined result; format dtype guards protect exact values. | Retain in this patch unless B-X isolates a redundant intermediate and preserves exact dtype. |
| WAV and interop modules | Selection conversion is already handled in C2; source inspection found no clearly redundant whole-payload copy here. | No X claim without a positive B-X counter. |

B-X will freeze fixture bytes/hash, a harness digest independent of the
candidate package, B0/B1 and immediate pre-X installed-wheel fingerprints and
resource samples, and a structural count of avoidable output-sized arrays.
Run structural instrumentation separately from timing and memory. The X claim
requires fewer unnecessary copied bytes, exact result/dtype/metadata/warning
parity, and no confirmed small-input regression. If the primary counter is zero
or the measured candidate does not clear its declared gate, leave that subroute
unchanged and record `HOLD`.
