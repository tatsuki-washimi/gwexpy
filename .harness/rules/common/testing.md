# GWexpy Testing Rules

> This file extends the global `common/testing.md` with GWexpy-specific testing requirements.

## Conda Environment

All tests must run in the gwexpy conda environment:

```bash
conda run -n gwexpy pytest tests/
```

## Test Organization

```
tests/
├── test_timeseries.py       # Unit tests per module
├── test_frequencyseries.py
├── test_fields.py           # Physics-critical — physics-reviewer required
├── test_signal.py
├── gui/                     # GUI tests (marker: gui)
└── integration/             # Integration tests
```

## Pytest Markers

Always mark tests appropriately:

```python
import pytest

@pytest.mark.gui
def test_plot_window(): ...       # Requires Qt display

@pytest.mark.nds
def test_nds_fetch(): ...         # Requires NDS2 server

@pytest.mark.slow
def test_long_psd(): ...          # > 10 seconds

@pytest.mark.integration
def test_full_pipeline(): ...     # Multi-component
```

## Physics Test Requirements

For any test touching `gwexpy/fields/`, `gwexpy/signal/`, or `gwexpy/spectrogram/`:

```python
def test_fft_normalization():
    """Test that PSD normalization satisfies Parseval's theorem."""
    ts = TimeSeries(np.random.randn(1024), sample_rate=1024 * u.Hz)
    psd = ts.psd(fftlength=1.0)
    # Parseval: sum(x^2) * dt == 2 * sum(PSD) * df (one-sided)
    time_power = np.sum(ts.value ** 2) * ts.dt.value
    freq_power = 2 * np.sum(psd.value) * psd.df.value
    np.testing.assert_allclose(time_power, freq_power, rtol=0.01)
```

- Include numerical tolerance in assertions (`rtol`, `atol`)
- Test both normal and edge cases (zero-length, single-sample, NaN input)
- Test metadata preservation: `assert result.unit == expected_unit`

## Coverage Target

Minimum 80% coverage per modified module. Check with:

```bash
conda run -n gwexpy pytest --cov=gwexpy --cov-report=term-missing tests/
```
