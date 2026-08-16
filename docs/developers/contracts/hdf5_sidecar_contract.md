# GWexpy HDF5 sidecar contract

GWexpy’s canonical `format="hdf5"` handlers preserve native GWpy payloads
unchanged.  GWexpy-only state is stored as one UTF-8 JSON root-file attribute:
`_gwexpy_sidecar_json_v1`.

The JSON document is exactly:

```json
{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{
  "relative/object/path":{"metadata":{},"provenance":{}}
}}
```

Object keys are normalized POSIX relative HDF5 paths.  Absolute paths,
empty/dot/dotdot/NUL components, duplicate members, traversal, unknown schema
or version, and extra entry keys are invalid.  Metadata and provenance are
strict JSON-safe mappings using the canonical GWexpy provenance helpers;
NumPy arrays, non-finite values, cycles, and unknown type tags are rejected.

For a `TimeSeries`, metadata reserves `_gwexpy_t0_gps_state` for the exact
state `{"_gwex_t0_gps_ns": <int>, "precision": "exact"|"quantized"}`.
User metadata may not use this key.  The reserved member is removed from the
public `metadata` mapping after reading, while `_gwex_t0_gps_ns` and its
precision are restored.

The sidecar is rooted at the HDF5 file even when writing through a containing
`h5py.Group`.  Append preserves unrelated object entries and replaces only the
normalized path being written.  Missing sidecars and entries are empty.
GWpy-only readers ignore the extra root attribute and therefore continue to
read the native payload, axes, units, names, channels, and segment structure.

Before a native write touches its real target, the captured native HDF5 writer
is run once against a disposable HDF5 file (including the containing-group
prefix when the caller supplied an `h5py.Group`).  This bounded preflight
validates native HDF5 options and attribute writes after sidecar metadata,
provenance, path, and document validation.  It does not invoke the sidecar
wrapper, close or mutate caller-owned HDF5 objects, persist sidecar data, or
execute arbitrary application callbacks.  Native errors therefore leave both
the payload and sidecar unchanged.

When multiple HDF5 files are merged into a `SegmentList`, zero or one marked
input preserves its state, and multiple marked inputs must have deeply equal
metadata and provenance.  Equal state is returned as an independent deep
copy; any conflict raises `ValueError` before a merged result is returned.
Missing sidecars or entries are marked reads with empty state, so they follow
the same equality and conflict rules.  Unmarked, non-HDF5 reads retain GWpy's
native merge behavior.

Only these existing native `hdf5` handlers are wrapped: `TimeSeries`,
`FrequencySeries`, `Spectrogram`, `StateVector`, `SegmentList`, and
`DataQualityFlag`.  Nested segment writes performed by a flag are suppressed;
the flag entry is authoritative.  `hdf.ndscope` and its aliases are not
wrapped.
