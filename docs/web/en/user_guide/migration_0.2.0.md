# Migration notes for the v0.2.0 implementation lane

This page documents the current `[Unreleased]` implementation evidence.
v0.2.0 is not published or released, and the final root verification gates are
still pending.

## API labels

The API stability policy has exactly three labels: `stable`, `provisional`, and
`experimental`.
`deferred` is a release outcome, not a fourth API tier.
The classification is conservative and does not implicitly label all legacy
APIs.

## Exact nanosecond origins

Use keyword-only `t0_ns` when the source provides an exact GPS-nanosecond
origin.
The accepted range is exactly `0 <= t0_ns <= 2**63 - 1`.
The read-only `t0_gps_ns` property exposes the exact value when it is retained.

Before, a floating-point epoch could be accepted but only represented at its
available precision:

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

samples = [0.0, 1.0]
series = TimeSeries(samples, sample_rate=4, epoch=1_400_000_000.000000001)
```

After, pass the integer explicitly:

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

samples = [0.0, 1.0]
series = TimeSeries(samples, sample_rate=4, t0_ns=1_400_000_000_000_000_001)
assert series.t0_gps_ns == 1_400_000_000_000_000_001
```

`t0_ns` is keyword-only and must be an exact integer in that range.
Boolean values, negative values, values greater than 2**63 - 1, and
disagreement with `t0` or `epoch` fail with `TypeError` or `ValueError` before
construction.
Floating-point seconds retain the existing API but are documented as
`quantized`, not exact.

## Automatic HDF5 metadata and provenance

The canonical HDF5 writer keeps the GWpy-native payload unchanged and writes
one automatic file-root `_gwexpy_sidecar_json_v1` attribute for GWexpy metadata,
exact timing state, and provenance.
The sidecar is restored only by GWexpy; a GWpy-only reader can still read the
native payload.
The automatic transition is attached to public GWexpy read/write entry points.
That entry point also installs the six native HDF5 sidecar handlers, including
`StateVector`, `SegmentList`, and `DataQualityFlag`; a GWpy-only caller that
never enters GWexpy does not trigger GWexpy bootstrap.
The root attribute is writer-managed; update metadata through the object and
write API.

Before, a native round trip exposed only the payload to a GWpy-only reader:

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

source = TimeSeries([0.0, 1.0])
source.write("data.h5", format="hdf5", path="data")
restored = TimeSeries.read("data.h5", format="hdf5", path="data")
```

After, the same executable entry point restores the exact timing, metadata,
and provenance state automatically:

```python
# executable-roundtrip
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gwexpy import register_all
from gwexpy.timeseries import TimeSeries

register_all()
metadata = {"channel": "K1:TEST-STRAIN", "labels": ["synthetic", "tutorial"]}
provenance = {
    "schema": "gwexpy.provenance",
    "algorithm": "migration-roundtrip",
    "parameters": {"sample_rate_hz": 4.0},
}
origin_ns = 1_400_000_000_000_000_001

with TemporaryDirectory() as temporary_directory:
    path = Path(temporary_directory) / "data.h5"
    source = TimeSeries(np.arange(4, dtype=float), sample_rate=4, t0_ns=origin_ns)
    source.metadata = metadata
    source.provenance = provenance
    source.write(path, format="hdf5", path="data")
    restored = TimeSeries.read(path, format="hdf5", path="data")

    assert restored.t0_gps_ns == origin_ns
    assert restored.metadata == metadata
    assert restored.provenance == provenance
```

Malformed JSON, an unknown sidecar schema or version, and non-JSON provenance
fail with `ValueError` or `TypeError`; they are not silently ignored.

## GWF parallel reads and the `nproc` alias

`parallel` is the preferred option for GWF reads.
`nproc` remains a compatibility alias and is not deprecated or removed here.
The two names cannot be supplied together.

Before, existing callers could use the compatibility spelling:

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

files = ["example.gwf"]
series = TimeSeries.read(files, format="gwf", nproc=4)
```

After, new code should use the preferred spelling:

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

files = ["example.gwf"]
series = TimeSeries.read(files, format="gwf", parallel=4)
```

This is an error and does not choose one option silently:

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

files = ["example.gwf"]
TimeSeries.read(files, format="gwf", parallel=4, nproc=4)  # TypeError
```

These are static signature examples; they do not claim that the example file
was read externally.
The implemented path uses spawn, deterministic merge order, cancellation on
failure, and real lalframe evidence.

## NDScope `dataset_options`

NDScope writer options have one public surface: `dataset_options`.
Do not pass a legacy collection of top-level filter or chunk keywords.

Before, a top-level option was an unsupported or ambiguous surface:

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

series = TimeSeries([0.0, 1.0])
series.write("data.h5", format="hdf.ndscope", compression="gzip")  # TypeError
```

After, put approved HDF5 creation options in the mapping:

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

series = TimeSeries([0.0, 1.0])
series.write(
    "data.h5",
    format="hdf.ndscope",
    dataset_options={"compression": "gzip", "shuffle": True},
)
```

These are static signature examples; they do not claim external NDScope file
creation.
Unknown keys, invalid filters, incompatible chunks, and unavailable codecs are
rejected during preflight before file creation.

## Provenance, median bias, and coupling segments

Provenance is a strict JSON mapping.
Record the RNG method, bit generator and seed when applicable, software
versions, and all algorithm parameters.
Runtime propagation is implemented only for provenance-bearing `Spectrogram`
analysis outputs: supported copy, slice, ufunc, and binary operations preserve
validated independent provenance snapshots and operation trees.
Supported HDF5 sidecar round-trips preserve JSON-safe provenance on supported
objects.
`GauChResult.metadata` remains an alias of its provenance mapping.

```python
# static-signature-example
from gwexpy.spectrogram import Spectrogram

result = Spectrogram([[0.0]], dt=1, f0=0, df=1)
result.provenance = {
    "schema": "gwexpy.provenance",
    "algorithm": "example",
    "parameters": {"window": "hann"},
    "rng": {"method": "numpy", "bit_generator": "PCG64", "seed": 7},
}
```

Arrays, non-finite numbers, and unsupported objects fail instead of becoming
ambiguous JSON.
`median_bias(N)` uses the reviewed independent chi-square-2 or
exponential-sample formula and documented limits:

```python
# static-signature-example
from gwexpy.signal.spectral import median_bias

alpha_3 = median_bias(3)
assert alpha_3 == 5 / 6
```

The correction is not a claim about overlapping or correlated samples.

Coupling segment v1 uses the required long-form columns
`start_gps_ns`, `duration_ns`, `source_channel`, `response_channel`,
`frequency_hz`, `coupling_factor`, and `coupling_factor_unit`.
Upper-limit fields are allowed only for an `upper_limit` row, and frequencies
are normalized to Hz.
The schema is experimental and intentionally does not claim broad scientific
generality.

```python
# static-signature-example
import pandas as pd

from gwexpy.coupling import validate

segments = pd.DataFrame(
    {
        "start_gps_ns": [1_400_000_000_000_000_000],
        "duration_ns": [1_000_000_000],
        "source_channel": ["K1:PEM-MIC"],
        "response_channel": ["K1:STRAIN"],
        "frequency_hz": [10.0],
        "coupling_factor": [0.25],
        "coupling_factor_unit": ["1"],
    }
)
validate(segments)
```

## SeriesMatrix direct-ufunc limitation and #637 fallback

**Stability:** provisional.

The #637 composition prototype and evidence were completed in isolation, but
the candidate runtime was not copied into integration. v0.2.0 keeps the B0
fallback and does not adopt `SeriesMatrix` composition/B1 behavior.

`np.sqrt(matrix)` is not supported as a direct NumPy ufunc under the v0.2.0 B0
contract.

Use:

```python
result = matrix ** 0.5
```

This alternative is contract-tested for `TimeSeriesMatrix`, `FrequencySeriesMatrix`,
and `SpectrogramMatrix`, and preserves B0 semantics (concrete class, numerical
value, per-cell unit, axes, labels, and metadata).

No metadata-preserving B0 alternatives are currently defined for direct
`np.log(matrix)`, `np.exp(matrix)`, `np.isfinite(matrix)`, or `np.isnan(matrix)`.

`np.isreal(matrix)` remains family-specific: it is supported directly for
`SpectrogramMatrix`, and is `UnitConversionError` for `TimeSeriesMatrix` and
`FrequencySeriesMatrix` under B0.

Both forms below are already B0-supported:

```python
(2 * u.s) * matrix
matrix * (2 * u.s)
```

`np.asarray(matrix)` remains the intentional metadata-escape boundary and returns a
raw `numpy.ndarray` under B0. It is not a metadata-preserving workaround.

Unsupported direct calls must fail explicitly and must never silently degrade
to bare `ndarray` or `Quantity`; this is the required v0.2.0 contract behavior,
not a correctness regression.

The future #637 redesign remains open with no assigned release version or date,
and this page does not promise B1 adoption in v0.2.0.

## Release status

This page is implementation documentation and evidence preparation for
`[Unreleased]`.
It does not claim a published version, tag, commit, pull request, issue state,
or completed final integration gate.
