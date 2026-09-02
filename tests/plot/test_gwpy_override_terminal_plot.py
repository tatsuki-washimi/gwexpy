"""Direct GWpy-oracle evidence for the terminal plot override inventory.

The comparisons in this module intentionally cover call binding and the data
given to Matplotlib artists.  Pixel output, styling, layout, colorbars, and
GWexpy's explicit presentation defaults are outside the compatibility policy.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from gwpy.plot import Plot as GwpyPlot
from gwpy.spectrogram import Spectrogram as GwpySpectrogram
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.types import Array2D as GwpyArray2D
from matplotlib.colors import LogNorm

from gwexpy.plot import FieldPlot, Plot, SkyMap
from gwexpy.spectrogram import Spectrogram
from gwexpy.types import Array2D, Plane2D


def _exception_class(call: Callable[[], Any]) -> type[BaseException] | None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - exception class is the oracle
        return type(exc)
    return None


def _assert_parameter_layout_matches(actual: Any, expected: Any) -> None:
    """Compare the public binding surface without return annotations."""
    assert (
        inspect.signature(actual).parameters == inspect.signature(expected).parameters
    )


def _plot_pair(actual_type: type[Plot]) -> tuple[Plot, GwpyPlot]:
    actual_data = GwpyTimeSeries(
        np.arange(4.0), t0=1_000_000_000, dt=0.25, unit="V", name="signal"
    )
    expected_data = actual_data.copy()
    return actual_type(actual_data), GwpyPlot(expected_data)


def _assert_plot_set_contract(actual_type: type[Plot]) -> None:
    _assert_parameter_layout_matches(actual_type.set, GwpyPlot.set)

    actual, expected = _plot_pair(actual_type)
    try:
        assert actual.set() == expected.set() == []

        kwargs = {
            "label": "audit-figure",
            "visible": False,
            "dpi": 87.5,
            "size_inches": (4.5, 3.0),
        }
        actual_result = actual.set(**kwargs)
        expected_result = expected.set(**kwargs)
        assert actual_result == expected_result == [None] * len(kwargs)
        assert actual.get_label() == expected.get_label() == "audit-figure"
        assert actual.get_visible() is expected.get_visible() is False
        assert actual.dpi == expected.dpi == 87.5
        np.testing.assert_array_equal(
            actual.get_size_inches(), expected.get_size_inches()
        )

        # Figure mutation must not alter the plotted scientific source values.
        np.testing.assert_array_equal(
            actual.axes[0].lines[0].get_xdata(),
            expected.axes[0].lines[0].get_xdata(),
        )
        np.testing.assert_array_equal(
            actual.axes[0].lines[0].get_ydata(),
            expected.axes[0].lines[0].get_ydata(),
        )

        invalid_calls = [
            lambda plot: plot.set("positional-is-invalid"),
            lambda plot: plot.set(not_a_figure_property=True),
            lambda plot: plot.set(label="first", **{"label": "duplicate"}),
        ]
        for call in invalid_calls:
            actual_error = _exception_class(lambda: call(actual))
            expected_error = _exception_class(lambda: call(expected))
            assert actual_error is expected_error
            assert expected_error is not None
    finally:
        plt.close(actual)
        plt.close(expected)


def test_plot_set_matches_gwpy() -> None:
    _assert_plot_set_contract(Plot)


def test_fieldplot_set_matches_gwpy() -> None:
    _assert_plot_set_contract(FieldPlot)


def test_skymap_set_matches_gwpy() -> None:
    # Data-bearing construction intentionally selects the inherited Plot route.
    _assert_plot_set_contract(SkyMap)


def _array2d_pair(actual_type: type[Array2D]) -> tuple[Array2D, GwpyArray2D]:
    values = np.array(
        [
            [1.0, np.nan, 3.0, 4.0],
            [5.0, 6.0, -1.0, 8.0],
            [9.0, 10.0, 11.0, np.inf],
        ]
    )
    kwargs = {
        "unit": u.V,
        "xindex": [10, 20, 40] * u.Hz,
        "yindex": [1, 2, 4, 8] * u.Hz,
        "name": "audit-map",
    }
    return actual_type(values.copy(), **kwargs), GwpyArray2D(values.copy(), **kwargs)


def _spectrogram_pair() -> tuple[Spectrogram, GwpySpectrogram]:
    values = np.array(
        [
            [1.0, np.nan, 3.0, 4.0],
            [5.0, 6.0, -1.0, 8.0],
            [9.0, 10.0, 11.0, np.inf],
        ]
    )
    kwargs = {
        "unit": u.V,
        "t0": 1_000_000_000,
        "dt": 0.25,
        "f0": 10,
        "df": 2,
        "name": "audit-spectrogram",
    }
    return Spectrogram(values.copy(), **kwargs), GwpySpectrogram(
        values.copy(), **kwargs
    )


def _artist(plot: GwpyPlot, method: str) -> Any:
    axes = plot.axes[0]
    artists = axes.images if method == "imshow" else axes.collections
    assert len(artists) == 1
    return artists[0]


def _assert_artist_data_matches(
    actual_plot: GwpyPlot, expected_plot: GwpyPlot, method: str
) -> None:
    actual_artist = _artist(actual_plot, method)
    expected_artist = _artist(expected_plot, method)
    actual_source = np.ma.asarray(actual_artist.get_array())
    expected_source = np.ma.asarray(expected_artist.get_array())

    assert actual_source.shape == expected_source.shape
    np.testing.assert_array_equal(
        np.ma.getdata(actual_source), np.ma.getdata(expected_source)
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual_source), np.ma.getmaskarray(expected_source)
    )
    assert np.ma.getmaskarray(expected_source).any()

    # Compare coordinates supplied to the artist, not presentation-only view
    # choices such as scale, final limits, pixel interpolation, or colorbars.
    if method == "imshow":
        np.testing.assert_array_equal(
            actual_artist.get_extent(), expected_artist.get_extent()
        )
    else:
        np.testing.assert_array_equal(
            actual_artist.get_coordinates(), expected_artist.get_coordinates()
        )
    np.testing.assert_array_equal(
        actual_plot.axes[0].dataLim.bounds,
        expected_plot.axes[0].dataLim.bounds,
    )


def _assert_direct_artist_contract(
    actual: Array2D | Spectrogram,
    expected: GwpyArray2D | GwpySpectrogram,
    method: str,
) -> None:
    _assert_parameter_layout_matches(
        getattr(type(actual), method), getattr(type(expected), method)
    )

    # **kwargs-only binding, duplicate wrapper routing, and backend validation.
    invalid_calls = [
        lambda obj: getattr(obj, method)("positional-is-invalid"),
        lambda obj: getattr(obj, method)(method="other-method"),
        lambda obj: getattr(obj, method)(not_an_artist_property=True),
    ]
    for call in invalid_calls:
        actual_error = _exception_class(lambda: call(actual))
        expected_error = _exception_class(lambda: call(expected))
        assert actual_error is expected_error
        assert expected_error is not None
        # A backend property error may leave a partially constructed figure.
        plt.close("all")

    kwargs_factories = (
        dict,
        lambda: {
            "cmap": "viridis",
            "alpha": 0.6,
            "norm": LogNorm(vmin=1, vmax=11),
        },
    )
    for make_kwargs in kwargs_factories:
        # Norms are mutable Matplotlib objects, so keep both oracle paths
        # independent instead of sharing one instance between the calls.
        actual_plot = getattr(actual, method)(**make_kwargs())
        expected_plot = getattr(expected, method)(**make_kwargs())
        try:
            _assert_artist_data_matches(actual_plot, expected_plot, method)
        finally:
            plt.close(actual_plot)
            plt.close(expected_plot)

    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.xindex.value, expected.xindex.value)
    np.testing.assert_array_equal(actual.yindex.value, expected.yindex.value)


def test_array2d_imshow_matches_gwpy() -> None:
    _assert_direct_artist_contract(*_array2d_pair(Array2D), "imshow")


def test_array2d_pcolormesh_matches_gwpy() -> None:
    _assert_direct_artist_contract(*_array2d_pair(Array2D), "pcolormesh")


def test_plane2d_imshow_matches_gwpy() -> None:
    _assert_direct_artist_contract(*_array2d_pair(Plane2D), "imshow")


def test_plane2d_pcolormesh_matches_gwpy() -> None:
    _assert_direct_artist_contract(*_array2d_pair(Plane2D), "pcolormesh")


def test_spectrogram_imshow_matches_gwpy() -> None:
    _assert_direct_artist_contract(*_spectrogram_pair(), "imshow")


def test_spectrogram_pcolormesh_matches_gwpy() -> None:
    _assert_direct_artist_contract(*_spectrogram_pair(), "pcolormesh")
