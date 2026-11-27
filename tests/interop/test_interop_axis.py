"""Axis checks used by interop converters before emitting x0/dx (U-7)."""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.interop._axis import affine_representable_axis, exactly_affine_axis

_JITTER_887 = np.arange(100) * 1e-6 + np.where(np.arange(100) % 2, 1e-9, 0.0)


@pytest.mark.parametrize(
    "values",
    [
        np.array([]),
        np.array([np.nan, 1.0, 2.0]),
        np.array([3.0, 2.0, 1.0]),
        np.array([0.0, 1.0, 1.0, 2.0]),
        np.arange(6.0).reshape(2, 3),
    ],
    ids=["empty", "nan", "decreasing", "duplicate", "2d"],
)
def test_invalid_axes(values):
    assert not affine_representable_axis(values) or values.size == 0
    assert not exactly_affine_axis(values) or values.size == 0


@pytest.mark.parametrize(
    "values",
    [
        np.array([5.0]),
        np.array([5.0, 7.0]),
        np.arange(4096) / 4096 + 1264316116,
        np.linspace(1, 10, 1000),
        10 + np.arange(300) * (1 / 3),
        np.arange(4) * 1e-21,
    ],
    ids=["one", "two", "gps", "linspace", "define-1/3", "1e-21"],
)
def test_uniform_axes_are_affine_representable(values):
    assert affine_representable_axis(values)


@pytest.mark.parametrize(
    "values",
    [
        _JITTER_887,
        np.geomspace(1, 10, 100),
        np.array([0.0, 1e-21, 3e-21, 6e-21]),
    ],
    ids=["jitter887", "geomspace", "1e-21-irregular"],
)
def test_nonuniform_axes_are_rejected_at_any_scale(values):
    assert not affine_representable_axis(values)
    assert not exactly_affine_axis(values)


def test_exact_check_distinguishes_roundoff_from_exact_reconstruction():
    # Uniform up to round-off, but x0/dx from the first interval do not
    # reproduce every sample: affine-only exports accept it, explicit-coordinate
    # imports keep the samples.
    values = np.linspace(1, 10, 1000)
    assert affine_representable_axis(values)
    assert not exactly_affine_axis(values)
    assert exactly_affine_axis(np.arange(16) * 0.25 + 3.0)
