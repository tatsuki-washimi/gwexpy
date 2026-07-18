from __future__ import annotations

import numpy as np


def assert_eventtable_close(
    actual,
    expected,
    *,
    columns: tuple[str, ...] | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Validate a round-tripped EventTable against an expected table."""
    assert len(actual) == len(expected)

    cols = columns if columns is not None else tuple(expected.colnames)
    assert tuple(actual.colnames) == tuple(expected.colnames)

    for col in cols:
        np.testing.assert_allclose(
            np.asarray(actual[col], dtype=float),
            np.asarray(expected[col], dtype=float),
            rtol=rtol,
            atol=atol,
        )
