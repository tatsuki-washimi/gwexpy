from __future__ import annotations

import numpy as np


def assert_histogram_close(
    actual,
    expected_values: np.ndarray | None = None,
    *,
    expected_edges: np.ndarray | None = None,
    name: str | None = None,
    unit: str | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Validate a round-tripped Histogram against expected bin contents/edges."""
    if expected_values is not None:
        np.testing.assert_allclose(
            np.asarray(actual.value, dtype=float),
            np.asarray(expected_values, dtype=float),
            rtol=rtol,
            atol=atol,
        )

    if expected_edges is not None:
        np.testing.assert_allclose(
            np.asarray(actual.edges, dtype=float),
            np.asarray(expected_edges, dtype=float),
            rtol=rtol,
            atol=atol,
        )

    if name is not None:
        assert getattr(actual, "name", None) == name

    if unit is not None:
        assert str(getattr(actual, "unit", "")) == unit
