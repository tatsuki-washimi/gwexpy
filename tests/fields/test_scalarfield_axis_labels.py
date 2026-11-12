"""Exact axis labels of ``ScalarField`` plots (#569)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField

DIMLESS = u.dimensionless_unscaled


def _field(*, unit, t_unit, space_unit):
    rng = np.random.default_rng(569)
    return ScalarField(
        rng.normal(size=(6, 4, 4, 4)),
        unit=unit,
        axis0=np.arange(6) * 0.1 * t_unit,
        axis1=np.arange(4) * 1.0 * space_unit,
        axis2=np.arange(4) * 1.0 * space_unit,
        axis3=np.arange(4) * 1.0 * space_unit,
        axis_names=["t", "x", "y", "z"],
    )


def _at(field, **names):
    index = {"t": field._axis0_index, "x": field._axis1_index, "z": field._axis3_index}
    index["y"] = field._axis2_index
    return {name: index[name][pos] for name, pos in names.items()}


@pytest.fixture
def unitful():
    return _field(unit=u.V, t_unit=u.s, space_unit=u.m)


@pytest.fixture
def unitless():
    return _field(unit=DIMLESS, t_unit=DIMLESS, space_unit=DIMLESS)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _colorbar_label(fig):
    return fig.axes[-1].get_ylabel()


def test_map2d_labels_with_units(unitful):
    fig, ax = unitful.plot_map2d("xy", at=_at(unitful, t=1, z=2))

    # The second plane axis is drawn horizontally.
    assert ax.get_xlabel() == "y [m]"
    assert ax.get_ylabel() == "x [m]"
    assert _colorbar_label(fig) == "real [V]"


def test_map2d_labels_without_units(unitless):
    fig, ax = unitless.plot_map2d("xy", at=_at(unitless, t=1, z=2))

    assert ax.get_xlabel() == "y"
    assert ax.get_ylabel() == "x"
    assert _colorbar_label(fig) == "real"


def test_timeseries_points_labels_with_units(unitful):
    point = tuple(unitful._axis1_index[[1, 1, 1]])
    _, ax = unitful.plot_timeseries_points([point])

    assert ax.get_xlabel() == "t [s]"
    assert ax.get_ylabel() == "[V]"


def test_timeseries_points_labels_without_units(unitless):
    point = tuple(unitless._axis1_index[[1, 1, 1]])
    _, ax = unitless.plot_timeseries_points([point])

    assert ax.get_xlabel() == "t"
    assert ax.get_ylabel() == ""


def test_profile_labels_with_units(unitful):
    _, ax = unitful.plot_profile("x", at=_at(unitful, t=1, y=2, z=2))

    assert ax.get_xlabel() == "x [m]"
    assert ax.get_ylabel() == "real [V]"


def test_profile_labels_without_units(unitless):
    _, ax = unitless.plot_profile("x", at=_at(unitless, t=1, y=2, z=2))

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "real"


def test_time_space_map_labels_with_units(unitful):
    fig, ax = unitful.plot_time_space_map("x", at=_at(unitful, y=2, z=2))

    assert ax.get_xlabel() == "x [m]"
    assert ax.get_ylabel() == "t [s]"
    assert _colorbar_label(fig) == "real [V]"


def test_time_space_map_labels_without_units(unitless):
    fig, ax = unitless.plot_time_space_map("x", at=_at(unitless, y=2, z=2))

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "t"
    assert _colorbar_label(fig) == "real"


def test_mixed_units_only_drop_the_empty_bracket():
    field = _field(unit=DIMLESS, t_unit=u.s, space_unit=u.m)

    fig, ax = field.plot_time_space_map("x", at=_at(field, y=2, z=2))

    assert ax.get_xlabel() == "x [m]"
    assert ax.get_ylabel() == "t [s]"
    assert _colorbar_label(fig) == "real"


def test_plotted_values_do_not_depend_on_labels(unitful, unitless):
    """Label formatting must leave the drawn data untouched."""
    _, ax = unitful.plot_profile("x", at=_at(unitful, t=1, y=2, z=2))
    line = ax.get_lines()[0]

    np.testing.assert_array_equal(line.get_xdata(), unitful._axis1_index.value)
    np.testing.assert_array_equal(line.get_ydata(), unitful.value[1, :, 2, 2])
