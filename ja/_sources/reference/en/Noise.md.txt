# Noise

Utilities for detector and environmental noise models, including ASD helpers
and time-domain synthesis.

## `from_asd`

```python
from_asd(asd, duration, sample_rate, t0=0.0, rng=None) -> TimeSeries
```

Generate a colored noise `TimeSeries` from an ASD (`FrequencySeries`).

Notes:
- Returns a `TimeSeries` (not a NumPy array).
- `name` and `channel` are propagated from the input ASD.
- Output unit is `asd.unit * sqrt(Hz)`.
- Use `t0` to set the start time.
