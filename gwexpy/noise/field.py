"""gwexpy.noise.field - Generate 4D field noise and signals.

This module provides functions for generating common 4D field waveforms
including noise (Gaussian) and coherent signals (plane waves).

All functions return gwexpy.fields.ScalarField objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from ..fields.scalar import ScalarField


def _generate_axes(
    shape: tuple[int, int, int, int],
    sample_rate: u.Quantity,
    space_step: u.Quantity,
    t0: u.Quantity = 0 * u.s,
    origin: tuple[u.Quantity, u.Quantity, u.Quantity] = (0 * u.m, 0 * u.m, 0 * u.m),
) -> dict[str, Any]:
    """Generate default Time/Space axes."""
    nt, nx, ny, nz = shape

    # Axis 0: Time
    dt = (1 / sample_rate).to(u.s)
    times = (np.arange(nt) * dt) + t0

    # Axes 1-3: Space
    dx = space_step
    x = (np.arange(nx) * dx) + origin[0]
    y = (np.arange(ny) * dx) + origin[1]
    z = (np.arange(nz) * dx) + origin[2]

    return {
        "axis0": times,
        "axis1": x,
        "axis2": y,
        "axis3": z,
    }


def gaussian(
    shape: tuple[int, int, int, int] = (100, 10, 10, 10),
    sample_rate: u.Quantity = 1.0 * u.Hz,
    space_step: u.Quantity = 1.0 * u.m,
    mean: float = 0.0,
    std: float = 1.0,
    t0: u.Quantity = 0.0 * u.s,
    origin: tuple[u.Quantity, u.Quantity, u.Quantity] = (0 * u.m, 0 * u.m, 0 * u.m),
    seed: int | None = None,
    unit: u.Unit | None = None,
    axes: dict[str, Any] | None = None,
) -> ScalarField:
    """Generate Gaussian (normal) white noise field.

    Parameters
    ----------
    shape : tuple of int
        (nt, nx, ny, nz)
    sample_rate : Quantity
        Sample rate for time axis (e.g. 1024 Hz).
    space_step : Quantity
        Grid spacing for spatial axes (e.g. 1 m).
    mean : float
        Mean of distribution.
    std : float
        Standard deviation.
    t0 : Quantity
        Start time.
    origin : tuple of Quantity
        (x0, y0, z0) for spatial origin.
    seed : int, optional
        Random seed.
    unit : Unit, optional
        Output unit.
    axes : dict, optional
        Explicit axes dict (axis0, axis1, axis2, axis3).
        If provided, overrides shape/rate/step generation logic.

    Returns
    -------
    ScalarField
    """
    from ..fields.scalar import ScalarField

    rng = np.random.default_rng(seed)

    if axes is None:
        axes = _generate_axes(shape, sample_rate, space_step, t0, origin)

    # Infer shape from axes if explicit axes are given
    if axes:
        shape = (
            len(axes["axis0"]),
            len(axes["axis1"]),
            len(axes["axis2"]),
            len(axes["axis3"]),
        )

    data = rng.normal(loc=mean, scale=std, size=shape)

    return ScalarField(data, unit=unit, **axes)


def plane_wave(
    frequency: u.Quantity,
    k_vector: tuple[u.Quantity, u.Quantity, u.Quantity],
    amplitude: float = 1.0,
    phase: float = 0.0,
    shape: tuple[int, int, int, int] = (100, 10, 10, 10),
    sample_rate: u.Quantity = 1.0 * u.Hz,
    space_step: u.Quantity = 1.0 * u.m,
    t0: u.Quantity = 0.0 * u.s,
    origin: tuple[u.Quantity, u.Quantity, u.Quantity] = (0 * u.m, 0 * u.m, 0 * u.m),
    unit: u.Unit | None = None,
    axes: dict[str, Any] | None = None,
) -> ScalarField:
    """Generate a coherent plane wave A * cos(omega*t - k*r + phi).

    Parameters
    ----------
    frequency : Quantity
        Temporal frequency (Hz).
    k_vector : tuple of Quantity
        Wave vector (kx, ky, kz) in 1/m (or rad/m depending on convention).
        Here we assume standard physics convention: k = 2pi/lambda.
        Phase argument will be (2pi*f*t - k*r + phi).
        Ensure k is in rad/m or 1/m appropriate for dot product.
        Typically k has unit 1/m, so phase = 2pi*f*t - (k*r) * (2pi?)
        GWpy/Astropy: If k has 1/m, k*r is dimensionless.
        We will implement: cos(2*pi*f*t - (kx*x + ky*y + kz*z) * 2*pi + phi)
        OR if k is angular wavenumber (rad/m), then don't multiply by 2pi.
        CONVENTION: k_vector input is in standard wavenumbers (1/length), NOT angular.
        So phase = 2*pi * (f*t - k*r).
    ...
    """
    from ..fields.scalar import ScalarField

    if axes is None:
        axes = _generate_axes(shape, sample_rate, space_step, t0, origin)

    times = axes["axis0"]
    x = axes["axis1"]
    y = axes["axis2"]
    z = axes["axis3"]

    # Meshgrid for vectorized calculation
    # Using 'ij' indexing implies (t, x, y, z) order if input is (t, x, y, z)
    # But meshgrid supports only list.

    # We can use broadcasting.
    # t: (nt, 1, 1, 1)
    # x: (1, nx, 1, 1)
    # y: (1, 1, ny, 1)
    # z: (1, 1, 1, nz)

    T = times.value[:, np.newaxis, np.newaxis, np.newaxis]
    X = x.value[np.newaxis, :, np.newaxis, np.newaxis]
    Y = y.value[np.newaxis, np.newaxis, :, np.newaxis]
    Z = z.value[np.newaxis, np.newaxis, np.newaxis, :]

    f_val = frequency.to_value(u.Hz)

    # k should be same unit base as x,y,z
    k_unit = 1 / x.unit
    kx = k_vector[0].to_value(k_unit)
    ky = k_vector[1].to_value(k_unit)
    kz = k_vector[2].to_value(k_unit)

    # Phase = 2pi * (f*t - (kx*x + ky*y + kz*z)) + phi
    # Dot product k*r
    dot_kr = kx * X + ky * Y + kz * Z

    phi_total = 2 * np.pi * (f_val * T - dot_kr) + phase

    data = amplitude * np.cos(phi_total)

    return ScalarField(data, unit=unit, **axes)
