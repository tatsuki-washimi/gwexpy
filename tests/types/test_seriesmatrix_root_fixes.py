#!/usr/bin/env python
"""Regression tests for the Wave 1 shared-root I/O fixes.

These cover the *roots* of two historically recurring bug classes that were
previously patched only per-call-site:

* **B1 / #443** -- ``SeriesMatrix.append(gap="pad")`` with no explicit ``pad``
  must fill the gap with ``NaN`` ("no data"), not a valid-looking ``0.0``.
  The unsafe ``pad=0.0`` default used to live in
  ``series_matrix_analysis.py`` while only the NetCDF4/Zarr call sites were
  patched to pass ``pad=np.nan``.

* **B2 / #442** -- operations that build a *new* matrix object must not alias
  the source's mutable ``attrs`` dict.  The original fix only added
  ``deepcopy`` to ``astype``/``real``/``imag``; the same defect lived in
  ``matmul``/``inv``/``diagonal``/``angle``, the structural ops routed through
  ``_get_meta_for_constructor`` (``crop``/``append``/``diff``/``pad``/
  ``interpolate``), and ``__array_ufunc__``.
"""

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeriesMatrix


def _square_matrix(attrs=None, x0=0.0):
    """A small invertible 2x2 (x6 samples) TimeSeriesMatrix with attrs.

    Uses a plain (unitless) integer xindex so structural ops such as
    ``append``/``shift`` stay arithmetic-friendly.
    """
    data = np.tile(np.eye(2)[:, :, None], (1, 1, 6)).astype(float)
    # make it genuinely invertible / non-degenerate per sample
    data += np.arange(6, dtype=float)[None, None, :] * 0.1
    return TimeSeriesMatrix(
        data,
        xindex=np.arange(x0, x0 + 6, dtype=float),
        attrs=attrs if attrs is not None else {"pipeline": "wave1", "nested": {"v": 1}},
    )


# --------------------------------------------------------------------------- #
# B1 -- gap padding defaults to NaN, not 0.0
# --------------------------------------------------------------------------- #


def test_append_gap_pad_defaults_to_nan_not_zero():
    """Merging matrix segments across a gap must yield NaN, not 0.0."""
    dt = 1.0  # seconds per sample
    x1 = np.arange(0.0, 3.0, dt)  # t = 0,1,2
    x2 = np.arange(6.0, 9.0, dt)  # t = 6,7,8  -> 3-sample gap at t=3,4,5
    sm1 = TimeSeriesMatrix(np.ones((1, 1, 3)), xindex=x1)
    sm2 = TimeSeriesMatrix(np.ones((1, 1, 3)) * 2.0, xindex=x2)

    merged = sm1.append(sm2, inplace=False, gap="pad")

    assert merged.shape == (1, 1, 9)
    gap_region = merged.value[0, 0, 3:6]
    assert np.all(np.isnan(gap_region)), (
        f"gap should be NaN (missing data), got {gap_region!r}"
    )
    # real samples are untouched
    assert np.array_equal(merged.value[0, 0, :3], np.ones(3))
    assert np.array_equal(merged.value[0, 0, 6:], np.ones(3) * 2.0)


def test_append_explicit_pad_still_honoured():
    """An explicit pad value is still respected (no silent override)."""
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([5.0, 6.0, 7.0])
    sm1 = TimeSeriesMatrix(np.ones((1, 1, 3)), xindex=x1)
    sm2 = TimeSeriesMatrix(np.ones((1, 1, 3)), xindex=x2)

    merged = sm1.append(sm2, inplace=False, gap="pad", pad=-1.0)

    assert np.array_equal(merged.value[0, 0, 3:5], np.array([-1.0, -1.0]))


def test_append_nan_pad_rejected_for_integer_dtype():
    """NaN cannot represent missing data in an integer matrix -> clear error."""
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([5.0, 6.0, 7.0])
    sm1 = TimeSeriesMatrix(np.ones((1, 1, 3), dtype=np.int64), xindex=x1)
    sm2 = TimeSeriesMatrix(np.ones((1, 1, 3), dtype=np.int64), xindex=x2)

    with pytest.raises(ValueError, match="NaN.*dtype|dtype.*NaN"):
        sm1.append(sm2, inplace=False, gap="pad")


# --------------------------------------------------------------------------- #
# B2 -- derived objects must not alias the source's attrs dict
# --------------------------------------------------------------------------- #


def _derive(op, matrix):
    """Apply a single-argument matrix operation, returning a new object."""
    if op == "copy":
        return matrix.copy()
    if op == "crop":
        return matrix.crop(matrix.xspan[0], matrix.xspan[1])
    if op == "diff":
        return matrix.diff()
    if op == "pad":
        return matrix.pad(2, constant_values=0)
    if op == "astype":
        return matrix.astype(np.float64)
    if op == "real":
        return matrix.real
    if op == "imag":
        return matrix.imag
    if op == "matmul":
        return matrix @ matrix
    if op == "inv":
        return matrix.inv()
    if op == "diagonal":
        return matrix.diagonal(output="matrix")
    if op == "angle":
        return matrix.angle()
    if op == "ufunc_add":
        return matrix + 1.0
    if op == "append":
        # a contiguous second segment starting right after the source
        other = _square_matrix(attrs={"pipeline": "other"}, x0=6.0)
        return matrix.append(other, inplace=False)
    raise AssertionError(op)


DERIVING_OPS = [
    "copy",
    "crop",
    "diff",
    "pad",
    "astype",
    "real",
    "imag",
    "matmul",
    "inv",
    "diagonal",
    "angle",
    "ufunc_add",
    "append",
]


@pytest.mark.parametrize("op", DERIVING_OPS)
def test_derived_object_does_not_alias_source_attrs(op):
    """Mutating a derived matrix's attrs must not touch the source's attrs."""
    matrix = _square_matrix()
    derived = _derive(op, matrix)

    # The derived object owns an independent top-level attrs dict ...
    derived.attrs["new_key"] = "derived"
    assert "new_key" not in matrix.attrs, (
        f"{op}: derived.attrs aliases source.attrs (top level)"
    )

    # ... and an independent *nested* payload (deep copy, not shallow).
    if "nested" in derived.attrs:
        derived.attrs["nested"]["v"] = 999
        assert matrix.attrs["nested"]["v"] == 1, (
            f"{op}: derived.attrs shares nested payload with source"
        )


def test_view_still_shares_attrs_by_reference():
    """Sanity: a plain .view() intentionally keeps the by-reference contract."""
    matrix = _square_matrix()
    viewed = matrix.view(TimeSeriesMatrix)
    assert viewed.attrs is matrix.attrs
