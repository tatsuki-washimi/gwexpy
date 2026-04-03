# GWexpy Physics Standards

> This file defines physics-specific coding rules for the GWexpy project.
> It extends the global `common/coding-style.md` with gravitational-wave domain requirements.

## Unit Handling (CRITICAL)

ALWAYS use `astropy.units` for physical quantities:

```python
# WRONG
def bandpass(data, flow, fhigh):  # raw floats — no unit info
    ...

# CORRECT
from astropy.units import Quantity
import astropy.units as u

def bandpass(data: Quantity, flow: u.Quantity, fhigh: u.Quantity) -> Quantity:
    ...
```

- Never strip units with `.value` unless storing in a unit-aware container
- Explicit unit conversions required: `x.to(u.Hz)`, not `float(x)`
- Document output units in docstrings: `Returns: Quantity in Hz`

## Domain Separation (CRITICAL)

Time domain and frequency domain objects must never be mixed:

| Type | Domain | Key attributes |
|------|--------|---------------|
| `TimeSeries` | time | `t0`, `dt`, `sample_rate` |
| `FrequencySeries` | frequency | `f0`, `df` |
| `Spectrogram` | time-frequency | both axes |

- FFT normalization convention must be documented in every function docstring
- `sample_rate` ≠ `df` — do not confuse them

## Metadata Preservation (REQUIRED)

When implementing any operation on `ScalarField`, `TimeSeries`, `FrequencySeries`, `Spectrogram`, or `VectorField`:

- Preserve axis metadata: `t0`, `dt`, `f0`, `df`, channel name, unit
- Return new objects (non-destructive) — see `common/coding-style.md` immutability rule
- Test metadata preservation explicitly: `assert result.t0 == input.t0`

## Numerical Stability (REQUIRED)

```python
# Division — always guard
eps = np.finfo(float).eps
result = numerator / np.where(denominator == 0, eps, denominator)

# Matrix operations — check finite first
assert np.all(np.isfinite(matrix)), "Matrix contains NaN/Inf"

# FFT — document normalization
# Uses one-sided normalization: S(f) = 2 * |X(f)|^2 / (N * fs)
```

## GWpy Compatibility

- New public APIs must not break `gwpy` semantics
- Add `test_gwpy_compat_*.py` tests when public API diverges
- Migration notes required in docstrings if behavior differs

## Physics Review Triggers

Changes requiring `physics-reviewer` agent review:
- Any file in `gwexpy/fields/`
- Any FFT/PSD/ASD computation
- New filter design in `gwexpy/signal/`
- Changes to `gwexpy/spectrogram/`
