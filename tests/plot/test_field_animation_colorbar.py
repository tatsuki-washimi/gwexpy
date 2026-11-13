"""``FieldBase.animate`` keeps exactly one colorbar across frames (#565).

Each frame clears the plot axes and redraws the slice with its colorbar.
GWpy draws that colorbar as an inset child of the plot axes, so clearing the
axes removes it and nothing accumulates.  These tests pin that behavior and
the per-frame content, so a change in either library that broke it would be
caught.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from matplotlib.collections import QuadMesh

from gwexpy.fields import ScalarField

N_T = 5


@pytest.fixture
def field():
    rng = np.random.default_rng(565)
    data = rng.normal(size=(N_T, 4, 3, 2))
    return ScalarField(data, axis0=np.arange(N_T) * u.s, unit=u.V)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _draw_frames(ani, frames):
    """Advance the animation through ``frames`` and return the figure."""
    fig = ani._fig
    for frame in frames:
        ani._draw_frame(frame)
    return fig


def _mesh(fig):
    (mesh,) = [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)]
    return mesh


def _all_axes(fig):
    """Every axes in the figure, including inset children."""
    return [ax for top in fig.axes for ax in (top, *top.child_axes)]


def test_axes_and_colorbar_counts_are_stable_across_frames(field):
    ani = field.animate(x="x", y="y", axis="t", z=0)

    counts = []
    for frame in field._axis0_index:
        fig = _draw_frames(ani, [frame])
        counts.append((len(fig.axes), len(_all_axes(fig)), len(fig.get_children())))

    # One plot axes plus one colorbar axes, however many frames are drawn.
    assert counts == [counts[0]] * N_T
    assert counts[0][1] == 2


def test_colorbar_belongs_to_the_current_frame(field):
    ani = field.animate(x="x", y="y", axis="t", z=0)

    for frame in field._axis0_index:
        fig = _draw_frames(ani, [frame])
        colorbar = fig.last_field_colorbar

        assert colorbar.mappable is _mesh(fig)
        assert colorbar.ax in _all_axes(fig)


def test_plot_axes_position_does_not_shrink(field):
    ani = field.animate(x="x", y="y", axis="t", z=0)
    frames = list(field._axis0_index)

    fig = _draw_frames(ani, frames[:1])
    fig.canvas.draw()
    first = fig.axes[0].get_position().bounds
    fig = _draw_frames(ani, frames[1:])
    fig.canvas.draw()
    last = fig.axes[0].get_position().bounds

    np.testing.assert_allclose(last, first, rtol=1e-9, atol=1e-12)


def test_each_frame_shows_its_own_slice_and_title(field):
    ani = field.animate(x="x", y="y", axis="t", z=0)

    for index, frame in enumerate(field._axis0_index):
        fig = _draw_frames(ani, [frame])

        assert fig.axes[0].get_title() == f"t = {frame:.3g}"
        drawn = np.asarray(_mesh(fig).get_array()).ravel()
        expected = field.value[index, :, :, 0]
        np.testing.assert_array_equal(np.sort(drawn), np.sort(expected.ravel()))


def test_color_scale_is_fixed_to_the_whole_field(field):
    ani = field.animate(x="x", y="y", axis="t", z=0)

    for frame in field._axis0_index:
        fig = _draw_frames(ani, [frame])
        vmin, vmax = _mesh(fig).get_clim()
        assert vmin == np.nanmin(field.value)
        assert vmax == np.nanmax(field.value)


def test_labels_survive_frames(field):
    ani = field.animate(x="x", y="y", axis="t", z=0)
    frames = list(field._axis0_index)

    def labels(fig):
        return (
            fig.axes[0].get_xlabel(),
            fig.axes[0].get_ylabel(),
            fig.last_field_colorbar.ax.get_ylabel(),
        )

    first = labels(_draw_frames(ani, frames[:1]))
    last = labels(_draw_frames(ani, frames[1:]))

    assert last == first
    assert "V" in first[2]


def test_explicit_color_limits_are_kept(field):
    ani = field.animate(x="x", y="y", axis="t", z=0, vmin=-1.0, vmax=2.0)

    fig = _draw_frames(ani, list(field._axis0_index))

    assert _mesh(fig).get_clim() == (-1.0, 2.0)
    assert len(_all_axes(fig)) == 2
