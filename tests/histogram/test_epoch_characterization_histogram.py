"""Characterization: Histogram has no epoch (#725, v0.3.0 Step 6)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.histogram import Histogram


def test_histogram_rejects_epoch_keyword_current():
    # characterizes current behavior, not a desired contract (#725):
    # Histogram is outside the #725 scope and has no epoch argument.
    with pytest.raises(TypeError, match="epoch"):
        Histogram(np.ones(3), np.arange(4.0), epoch=0)


def test_histogram_has_no_epoch_attribute_current():
    # characterizes current behavior, not a desired contract (#725)
    h = Histogram(np.ones(3), np.arange(4.0), unit="m")
    assert not hasattr(h, "epoch")
    assert h.unit == u.m
    assert h.values.shape == (3,)
    np.testing.assert_array_equal(np.asarray(h.edges.value), np.arange(4.0))
