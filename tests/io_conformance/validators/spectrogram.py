from __future__ import annotations

import numpy as np


def assert_spectrogram_close(
    actual,
    expected_values: np.ndarray | None = None,
    *,
    dt: float | None = None,
    df: float | None = None,
    t0: float | None = None,
    f0: float | None = None,
    name: str | None = None,
    unit: str | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Validate a round-tripped Spectrogram against expected metadata."""
    if expected_values is not None:
        expected = np.asarray(expected_values, dtype=float)
        actual_values = np.asarray(actual.value, dtype=float)
        assert actual_values.shape == expected.shape
        np.testing.assert_allclose(actual_values, expected, rtol=rtol, atol=atol)

    if dt is not None:
        np.testing.assert_allclose(float(actual.dt.value), dt)

    if df is not None:
        np.testing.assert_allclose(float(actual.df.value), df)

    if t0 is not None:
        np.testing.assert_allclose(float(actual.t0.value), t0)

    if f0 is not None:
        np.testing.assert_allclose(float(actual.f0.value), f0)

    if name is not None:
        assert getattr(actual, "name", None) == name

    if unit is not None:
        assert str(getattr(actual, "unit", "")) == unit
