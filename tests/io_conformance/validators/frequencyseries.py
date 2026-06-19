from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def assert_frequencyseries_close(
    actual,
    expected_values: Sequence[float] | np.ndarray | None = None,
    *,
    df: float | None = None,
    f0: float | None = None,
    name: str | None = None,
    unit: str | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Validate a round-tripped FrequencySeries against expected metadata."""
    if expected_values is not None:
        expected = _as_array(expected_values)
        actual_values = _as_array(actual.value)
        assert actual_values.shape == expected.shape
        np.testing.assert_allclose(actual_values, expected, rtol=rtol, atol=atol)

    if df is not None:
        np.testing.assert_allclose(float(actual.df.value), df)

    if f0 is not None:
        np.testing.assert_allclose(float(actual.f0.value), f0)

    if name is not None:
        assert getattr(actual, "name", None) == name

    if unit is not None:
        assert str(getattr(actual, "unit", "")) == unit
