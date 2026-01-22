"""Demo data generation utilities for ScalarField testing and tutorials.

This module provides deterministic ScalarField generators with known waveforms
for testing, documentation, and reproducible examples.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from astropy import units as u

from .scalar import ScalarField

__all__ = [
    "make_demo_scalar_field",
    "make_propagating_gaussian",
    "make_sinusoidal_wave",
    "make_standing_wave",
]


def make_demo_scalar_field(
    pattern: Literal["gaussian", "sine", "standing", "noise"] = "gaussian",
    nt: int = 100,
    nx: int = 16,
    ny: int = 16,
    nz: int = 16,
    dt: u.Quantity = 0.01 * u.s,
    dx: u.Quantity = 0.1 * u.m,
    dy: u.Quantity | None = None,
    dz: u.Quantity | None = None,
    unit: u.Unit = u.m / u.s,
    seed: int = 42,
    **kwargs,
) -> ScalarField:
    """Generate a deterministic demo ScalarField with known waveforms.

    This function provides reproducible ScalarField data for testing, tutorials,
    and examples. All generated fields have proper axis metadata and units.

    Parameters
    ----------
    pattern : {'gaussian', 'sine', 'standing', 'noise'}
        Type of demo pattern:
        - 'gaussian': Propagating Gaussian pulse (default)
        - 'sine': Plane sinusoidal wave
        - 'standing': Standing wave pattern
        - 'noise': White noise (for PSD validation)
    nt : int
        Number of time points. Default is 100.
    nx, ny, nz : int
        Number of spatial points per axis. Default is 16.
    dt : Quantity
        Time step. Default is 0.01 s.
    dx : Quantity
        Spatial step for x-axis. Default is 0.1 m.
    dy, dz : Quantity, optional
        Spatial steps for y and z. If None, uses dx.
    unit : Unit
        Physical unit of the data. Default is m/s.
    seed : int
        Random seed for reproducibility. Default is 42.
    **kwargs
        Additional keyword arguments passed to the specific pattern generator:
        - For 'gaussian': velocity, sigma_t, sigma_x
        - For 'sine': frequency, k_direction
        - For 'standing': mode_x, mode_y, mode_z, omega

    Returns
    -------
    ScalarField
        Deterministic demo field with proper metadata.

    Examples
    --------
    >>> from gwexpy.fields.demo import make_demo_scalar_field
    >>> field = make_demo_scalar_field('gaussian', nt=50, nx=32)
    >>> field.shape
    (50, 32, 16, 16)
    >>> field.axis0_domain
    'time'

    >>> # White noise for PSD testing
    >>> noise_field = make_demo_scalar_field('noise', seed=123)
    """
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx

    # Build axis coordinates
    times = np.arange(nt) * dt
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz

    # Generate pattern
    rng = np.random.default_rng(seed)

    if pattern == "gaussian":
        data = _generate_propagating_gaussian(
            times, x, y, z, rng, **kwargs
        )
    elif pattern == "sine":
        data = _generate_sinusoidal_wave(
            times, x, y, z, rng, **kwargs
        )
    elif pattern == "standing":
        data = _generate_standing_wave(
            times, x, y, z, rng, **kwargs
        )
    elif pattern == "noise":
        data = rng.standard_normal((nt, nx, ny, nz))
    else:
        raise ValueError(
            f"Unknown pattern '{pattern}'. Must be one of "
            "'gaussian', 'sine', 'standing', 'noise'."
        )

    return ScalarField(
        data,
        unit=unit,
        axis0=times,
        axis1=x,
        axis2=y,
        axis3=z,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
        space_domain="real",
    )


def _generate_propagating_gaussian(
    times: u.Quantity,
    x: u.Quantity,
    y: u.Quantity,
    z: u.Quantity,
    rng: np.random.Generator,
    velocity: u.Quantity | None = None,
    sigma_t: u.Quantity | None = None,
    sigma_x: u.Quantity | None = None,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """Generate a propagating Gaussian pulse.

    The pulse travels in the specified direction with given velocity.
    """
    nt, nx, ny, nz = len(times), len(x), len(y), len(z)

    # Defaults based on axis extent
    t_extent = (times[-1] - times[0]).to_value(u.s)
    x_extent = (x[-1] - x[0]).to_value(u.m) if len(x) > 1 else 1.0

    if velocity is None:
        if t_extent > 0:
            velocity = (x_extent / t_extent) * u.m / u.s
        else:
            velocity = 1.0 * u.m / u.s
    if sigma_t is None:
        sigma_t = t_extent / 4.0 * u.s if t_extent > 0 else 0.1 * u.s
    if sigma_x is None:
        sigma_x = x_extent / 4.0 * u.m if x_extent > 0 else 0.1 * u.m

    v = velocity.to_value(u.m / u.s)
    st = sigma_t.to_value(u.s)
    sx = sigma_x.to_value(u.m)

    # Normalize direction
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)

    # Create coordinate grids
    tt = times.to_value(u.s)
    xx = x.to_value(u.m)
    yy = y.to_value(u.m)
    zz = z.to_value(u.m)

    data = np.zeros((nt, nx, ny, nz))

    for it, t_val in enumerate(tt):
        # Position of pulse center at time t
        # Center starts at origin and moves along direction
        center = v * t_val * d

        for ix, x_val in enumerate(xx):
            for iy, y_val in enumerate(yy):
                for iz, z_val in enumerate(zz):
                    # Distance from pulse center
                    r = np.array([x_val, y_val, z_val]) - center
                    r_mag = np.linalg.norm(r)

                    # Gaussian amplitude
                    data[it, ix, iy, iz] = np.exp(
                        -0.5 * (r_mag / sx) ** 2
                        - 0.5 * ((t_val - tt[len(tt) // 2]) / st) ** 2
                    )

    return data


def _generate_sinusoidal_wave(
    times: u.Quantity,
    x: u.Quantity,
    y: u.Quantity,
    z: u.Quantity,
    rng: np.random.Generator,
    frequency: u.Quantity | None = None,
    k_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    wavelength: u.Quantity | None = None,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a plane sinusoidal wave.

    The wave propagates in the k_direction with specified frequency and wavelength.
    """
    nt, nx, ny, nz = len(times), len(x), len(y), len(z)

    # Defaults
    t_extent = (times[-1] - times[0]).to_value(u.s)
    x_extent = (x[-1] - x[0]).to_value(u.m) if len(x) > 1 else 1.0

    if frequency is None:
        # ~4 cycles over time extent
        frequency = 4.0 / t_extent * u.Hz if t_extent > 0 else 10.0 * u.Hz
    if wavelength is None:
        # ~2 wavelengths over x extent
        wavelength = x_extent / 2.0 * u.m if x_extent > 0 else 1.0 * u.m

    omega = (2 * np.pi * frequency).to_value(1 / u.s)
    k_mag = (2 * np.pi / wavelength).to_value(1 / u.m)

    # Normalize k direction
    kd = np.array(k_direction, dtype=float)
    kd /= np.linalg.norm(kd)
    k_vec = k_mag * kd

    # Create coordinate grids
    tt = times.to_value(u.s)
    xx = x.to_value(u.m)
    yy = y.to_value(u.m)
    zz = z.to_value(u.m)

    # Meshgrid for spatial coordinates
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')
    kr = k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z

    data = np.zeros((nt, nx, ny, nz))
    for it, t_val in enumerate(tt):
        data[it] = np.sin(omega * t_val - kr + phase)

    return data


def _generate_standing_wave(
    times: u.Quantity,
    x: u.Quantity,
    y: u.Quantity,
    z: u.Quantity,
    rng: np.random.Generator,
    mode_x: int = 1,
    mode_y: int = 1,
    mode_z: int = 0,
    omega: u.Quantity | None = None,
) -> np.ndarray:
    """Generate a standing wave pattern.

    Product of spatial modes with temporal oscillation.
    """
    nt, nx, ny, nz = len(times), len(x), len(y), len(z)

    # Default frequency
    t_extent = (times[-1] - times[0]).to_value(u.s)
    if omega is None:
        omega = 2 * np.pi * 3.0 / t_extent / u.s if t_extent > 0 else 6 * np.pi / u.s

    omega_val = omega.to_value(1 / u.s)

    # Coordinate values (normalized to [0, 1] range for modes)
    tt = times.to_value(u.s)
    x_norm = (x.to_value(u.m) - x[0].to_value(u.m))
    if len(x) > 1:
        x_norm = x_norm / (x[-1].to_value(u.m) - x[0].to_value(u.m))
    y_norm = (y.to_value(u.m) - y[0].to_value(u.m))
    if len(y) > 1:
        y_norm = y_norm / (y[-1].to_value(u.m) - y[0].to_value(u.m))
    z_norm = (z.to_value(u.m) - z[0].to_value(u.m))
    if len(z) > 1:
        z_norm = z_norm / (z[-1].to_value(u.m) - z[0].to_value(u.m))

    # Spatial mode shapes (sinusoidal)
    X_mode = np.sin(np.pi * mode_x * x_norm) if mode_x > 0 else np.ones(nx)
    Y_mode = np.sin(np.pi * mode_y * y_norm) if mode_y > 0 else np.ones(ny)
    Z_mode = np.sin(np.pi * mode_z * z_norm) if mode_z > 0 else np.ones(nz)

    # Outer product for spatial pattern
    spatial = np.einsum('i,j,k->ijk', X_mode, Y_mode, Z_mode)

    # Time evolution
    data = np.zeros((nt, nx, ny, nz))
    for it, t_val in enumerate(tt):
        data[it] = np.cos(omega_val * t_val) * spatial

    return data


# Convenience aliases
def make_propagating_gaussian(**kw) -> ScalarField:
    """Generate a propagating Gaussian demo field. See make_demo_scalar_field."""
    return make_demo_scalar_field("gaussian", **kw)


def make_sinusoidal_wave(**kw) -> ScalarField:
    """Generate a sinusoidal wave demo field. See make_demo_scalar_field."""
    return make_demo_scalar_field("sine", **kw)


def make_standing_wave(**kw) -> ScalarField:
    """Generate a standing wave demo field. See make_demo_scalar_field."""
    return make_demo_scalar_field("standing", **kw)
