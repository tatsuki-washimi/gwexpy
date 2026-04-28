# GUI Payload Contract Audit

Date: 2026-04-28
Issue: #274
Scope: docs/test-only baseline for current GUI payload shape and metadata absence.

## Summary

This slice records the current GUI result payload contracts before any
metadata-rich behavior is introduced. It does not change runtime GUI,
plotting, spectral, or physics code.

The covered paths are:

- `gwexpy.gui.engine.Engine.compute()`
- `gwexpy.gui.streaming.SpectralAccumulator.get_results()`

Both paths currently emit value-oriented payloads for GUI rendering. They do
not carry value units, display units, source names, channel identifiers, or a
metadata object in the result payload.

## Current Contracts

### Time Series

`Engine.compute(..., graph_type="Time Series", ...)` currently returns one
payload per active trace as:

```python
(times, values)
```

`SpectralAccumulator.get_results()` returns the same tuple shape for streaming
time-series display history.

The tuple shape means there are no metadata keys in the payload. The tests
assert the payload is not a dict, which explicitly records the current absence
of `unit`, `name`, `channel`, and `metadata` fields.

### Spectrogram

`Engine.compute(..., graph_type="Spectrogram", ...)` currently returns one
payload per active trace as a dict with exactly these keys:

```python
{"type", "times", "freqs", "value"}
```

`SpectralAccumulator.get_results()` returns the same dict shape from
spectrogram history.

The new tests assert the exact current key set and explicitly assert that the
dict does not expose `unit`, `name`, `channel`, or `metadata`.

## Test Strategy

The tests use local lightweight fakes and direct accumulator state setup. This
keeps the coverage headless and independent of Qt, a live display, NDS,
network, and expensive spectral computation. The tests are contract baselines,
not gwpy spectral-math tests.

## Deferred Behavior

Metadata-rich GUI payloads remain follow-up work. A future behavior-changing PR
can add structured fields such as x/y/value units, display units, names,
channels, and metadata after the broader plot/helper contracts settle.

That future PR should update these tests intentionally rather than treating
their failures as incidental breakage.

## Physics Review

Physics review is not required for this slice. No runtime algorithms, unit
conversions, spectral math, plotting semantics, or GUI rendering behavior were
changed.
