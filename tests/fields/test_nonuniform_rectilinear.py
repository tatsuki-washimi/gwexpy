"""Tests for #903 nonuniform rectilinear spatial-axis semantics."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField, TensorField, VectorField
from gwexpy.fields import signal as field_signal


@pytest.mark.parametrize(
    "x_coords",
    [
        np.array([0.0, 0.2, 0.7, 1.8]) * u.m,  # ascending nonuniform
        np.array([5.0, 3.0, 2.2, 0.1]) * u.m,  # descending nonuniform
        np.array([1.5]) * u.m,  # singleton
    ],
)
def test_nonuniform_coordinates_preserved_and_isolated(x_coords):
    t = np.arange(8, dtype=float) * 0.1 * u.s
    y = np.array([0.0, 0.5, 2.0]) * u.m
    z = np.array([0.0]) * u.m
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((len(t), len(x_coords), len(y), len(z)))

    sf = ScalarField(arr, unit=u.V, axis0=t, axis1=x_coords, axis2=y, axis3=z)
    vf = VectorField({"x": sf})
    tf = TensorField({(0, 0): sf})

    # copy() isolates coordinate arrays from mutation
    sf_copy = sf.copy()
    sf_copy._axis1_index[0] = 999.0 * u.m
    assert sf._axis1_index.to_value(u.m)[0] != 999.0

    # isel / indexing and VectorField/TensorField reductions preserve exact nonuniform coordinates
    if len(x_coords) >= 2:
        sub = sf.isel(x=slice(1, 3))
        np.testing.assert_array_equal(
            sub._axis1_index.to_value(u.m), x_coords.to_value(u.m)[1:3]
        )
        np.testing.assert_array_equal(
            vf["x"].isel(x=slice(1, 3))._axis1_index.to_value(u.m),
            x_coords.to_value(u.m)[1:3],
        )
        np.testing.assert_array_equal(
            tf[(0, 0)].isel(x=slice(1, 3))._axis1_index.to_value(u.m),
            x_coords.to_value(u.m)[1:3],
        )
    np.testing.assert_array_equal(
        vf.norm()._axis1_index.to_value(u.m), x_coords.to_value(u.m)
    )
    np.testing.assert_array_equal(
        tf.trace()._axis1_index.to_value(u.m), x_coords.to_value(u.m)
    )

    # Alignment arithmetic checks exact coordinates
    added = sf + sf
    np.testing.assert_array_equal(
        added._axis1_index.to_value(u.m), x_coords.to_value(u.m)
    )
    np.testing.assert_allclose(added.value, 2.0 * sf.value)

    if len(x_coords) >= 2:
        x_other = x_coords.copy()
        x_other[1] = x_other[1] + 0.05 * u.m
        sf_other = ScalarField(arr, unit=u.V, axis0=t, axis1=x_other, axis2=y, axis3=z)
        with pytest.raises(ValueError):
            _ = sf + sf_other


def test_nonuniform_fft_and_uniform_grid_guards():
    t_nu = np.array([0.0, 0.1, 0.4, 1.0, 1.2, 2.0]) * u.s
    x_nu = np.array([0.0, 0.25, 1.0, 3.0]) * u.m
    y_nu = np.array([4.0, 2.5, 0.5]) * u.m
    z = np.array([0.0]) * u.m
    sf = ScalarField(
        np.ones((len(t_nu), len(x_nu), len(y_nu), 1)),
        unit=u.V,
        axis0=t_nu,
        axis1=x_nu,
        axis2=y_nu,
        axis3=z,
    )

    with pytest.raises(ValueError, match="uniformly spaced|regularly spaced"):
        sf.fft_space("x")
    with pytest.raises(ValueError, match="uniformly spaced|regularly spaced"):
        sf.fft_space("y")
    with pytest.raises(ValueError, match="regularly spaced"):
        field_signal.spectral_density(sf, axis="x")
    with pytest.raises(ValueError, match="regularly spaced"):
        sf.resample(10.0 * u.Hz)
    with pytest.raises(ValueError, match="regularly spaced"):
        sf.filter([1.0, -1.0], [1.0])

    # Time uniform + space nonuniform: coherence_map with Quantity stride must fail cleanly
    t_reg = np.arange(32, dtype=float) * 0.05 * u.s
    sf_reg_t = ScalarField(
        np.ones((len(t_reg), len(x_nu), len(y_nu), 1)),
        unit=u.V,
        axis0=t_reg,
        axis1=x_nu,
        axis2=y_nu,
        axis3=z,
    )
    with pytest.raises(ValueError, match="regularly spaced"):
        field_signal.coherence_map(
            sf_reg_t,
            ref_point=(0.0 * u.m, 4.0 * u.m, 0.0 * u.m),
            plane="xy",
            freq=2.0 * u.Hz,
            stride=0.5 * u.m,
        )
    with pytest.raises(ValueError, match="regularly spaced"):
        field_signal.time_delay_map(
            sf_reg_t,
            ref_point=(0.0 * u.m, 4.0 * u.m, 0.0 * u.m),
            plane="xy",
            stride=0.5 * u.m,
        )

    # imshow requires uniform coordinates; pcolormesh / default handles nonuniform
    with pytest.raises(ValueError, match="regularly spaced"):
        sf_reg_t.plot_map2d(plane="xy", at={"t": 0.0 * u.s}, method="imshow")
    plot_pcol = sf_reg_t.plot_map2d(
        plane="xy", at={"t": 0.0 * u.s}, method="pcolormesh"
    )
    assert plot_pcol is not None

    # coherence_map axis_names ordering for xz / yz planes
    t_sig = np.arange(64, dtype=float) * 0.01 * u.s
    s_reg = ScalarField(
        np.random.default_rng(3).standard_normal((64, 3, 3, 4)),
        unit=u.V,
        axis0=t_sig,
        axis1=np.arange(3, dtype=float) * u.m,
        axis2=np.arange(3, dtype=float) * u.m,
        axis3=np.arange(4, dtype=float) * u.m,
    )
    cmap_xz = field_signal.coherence_map(
        s_reg,
        ref_point=(0.0 * u.m, 0.0 * u.m, 0.0 * u.m),
        plane="xz",
        at={"y": 0.0 * u.m},
        freq=5.0 * u.Hz,
        nfft=16,
    )
    assert cmap_xz.axis_names[1] == "x" and cmap_xz.axis_names[3] == "z"
