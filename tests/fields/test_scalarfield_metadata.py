"""Metadata integrity tests for ScalarField."""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwexpy.fields import ScalarField


def test_custom_metadata_preservation():
    """Verify that custom metadata survives operations."""
    data = np.ones((4, 2, 2, 2))
    # ScalarField inherits from Array -> GwpyArray, which supports name, channel, etc.
    field = ScalarField(data, name="MyField", channel="L1:TEST-CHANNEL")

    # Slicing
    sliced = field[1:3, :, :, :]
    assert sliced.name == "MyField"
    assert getattr(sliced.channel, "name", sliced.channel) == "L1:TEST-CHANNEL"

    # FFT
    freq = field.fft_time()
    assert freq.name in (None, "MyField")
    assert getattr(freq.channel, "name", freq.channel) in (None, "L1:TEST-CHANNEL")


def test_metadata_cloning_vs_referencing():
    """Verify that metadata is correctly cloned/copied when needed to avoid mutation."""
    data = np.zeros((4, 2, 2, 2))
    domains = {"x": "real", "y": "real", "z": "real"}
    field = ScalarField(data, space_domain=domains)

    # Slicing should result in a dict copy of space_domains
    sliced = field[:, 0:1, :, :]
    assert sliced.space_domains is not field.space_domains
    assert sliced.space_domains == field.space_domains

    # Mutation of parent domains should not affect child (if implemented safely)
    # field._space_domains["y"] = "k"  # This is a private attribute access
    # but in ScalarField implementation, space_domains property returns a copy.


def test_axis_descriptors_survival():
    """Verify axis descriptors survive slicing and transforms."""
    field = ScalarField(
        np.ones((4, 2, 2, 2)),
        axis1=np.arange(2) * 1.0 * u.m,
        axis_names=["t", "x", "y", "z"],
    )

    # Axis name
    assert field.axes[1].name == "x"

    # After slice
    sliced = field[:, 1, :, :]
    assert sliced.axes[1].name == "x"
    assert sliced.axes[1].size == 1

    # After FFT
    freq = field.fft_time()
    assert freq.axes[0].name == "f"
    assert freq.axes[1].name == "x"
