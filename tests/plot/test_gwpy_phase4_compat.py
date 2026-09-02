"""GWpy call-surface contracts for the v0.2.3 Phase 4 plot fixes."""

from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.plot import Plot as GwpyPlot
from gwpy.spectrogram import Spectrogram as GwpySpectrogram
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

PLOT_TYPES = {
    "timeseries": (TimeSeries, GwpyTimeSeries),
    "frequencyseries": (FrequencySeries, GwpyFrequencySeries),
    "spectrogram": (Spectrogram, GwpySpectrogram),
    "timeseriesdict": (TimeSeriesDict, GwpyTimeSeriesDict),
}

POSITIONAL_PREFIXES = {
    "timeseries": [
        (),
        ("plot",),
        ("plot", (10, 4)),
        ("plot", (10, 4), "linear"),
    ],
    "frequencyseries": [(), ("plot",), ("plot", "linear")],
    "spectrogram": [
        (),
        ("imshow",),
        ("imshow", (10, 6)),
        ("imshow", (10, 6), "linear"),
    ],
    "timeseriesdict": [
        (),
        ("name",),
        ("name", "plot"),
        ("name", "plot", (10, 4)),
        ("name", "plot", (10, 4), "linear"),
    ],
}

DUPLICATE_CALLS = {
    "timeseries": [
        (("plot",), {"method": "plot"}),
        (("plot", (10, 4)), {"figsize": (10, 4)}),
        (("plot", (10, 4), "linear"), {"xscale": "linear"}),
    ],
    "frequencyseries": [
        (("plot",), {"method": "plot"}),
        (("plot", "linear"), {"xscale": "linear"}),
    ],
    "spectrogram": [
        (("imshow",), {"method": "imshow"}),
        (("imshow", (10, 6)), {"figsize": (10, 6)}),
        (("imshow", (10, 6), "linear"), {"xscale": "linear"}),
    ],
    "timeseriesdict": [
        (("name",), {"label": "name"}),
        (("name", "plot"), {"method": "plot"}),
        (("name", "plot", (10, 4)), {"figsize": (10, 4)}),
        (("name", "plot", (10, 4), "linear"), {"xscale": "linear"}),
    ],
}

EXCESS_CALLS = {
    "timeseries": ("plot", (10, 4), "linear", "extra"),
    "frequencyseries": ("plot", "linear", "extra"),
    "spectrogram": ("imshow", (10, 6), "linear", "extra"),
    "timeseriesdict": ("name", "plot", (10, 4), "linear", "extra"),
}

STEP_CALLS = {
    "timeseries": ("step",),
    "frequencyseries": ("step",),
    "spectrogram": ("step",),
    "timeseriesdict": ("name", "step"),
}

ARTIST_KWARGS = {
    "timeseries": {"color": "red", "linewidth": 1.5, "alpha": 0.5},
    "frequencyseries": {"color": "red", "linewidth": 1.5, "alpha": 0.5},
    "spectrogram": {"method": "imshow", "cmap": "viridis", "alpha": 0.5},
    "timeseriesdict": {
        "label": "name",
        "color": "red",
        "linewidth": 1.5,
        "alpha": 0.5,
    },
}


def _make_plot_object(name: str, *, gwpy: bool):
    actual_type, parent_type = PLOT_TYPES[name]
    cls = parent_type if gwpy else actual_type
    if name == "timeseries":
        return cls(np.arange(8.0), t0=0, dt=1, name="series")
    if name == "frequencyseries":
        return cls(np.arange(1.0, 9.0), f0=1, df=1, name="spectrum")
    if name == "spectrogram":
        return cls(
            np.arange(12.0).reshape(3, 4),
            t0=0,
            dt=1,
            f0=1,
            df=1,
            name="spectrogram",
        )
    series_type = GwpyTimeSeries if gwpy else TimeSeries
    series = series_type(np.arange(8.0), t0=0, dt=1, name="series")
    return cls({"A": series, "B": series.copy()})


def _plot_outcome(obj, args=(), kwargs=None):
    try:
        plot = obj.plot(*args, **(kwargs or {}))
    except Exception as exc:  # outcome comparison deliberately includes failures
        return type(exc), None
    return None, plot


def _close(plot) -> None:
    if plot is not None:
        plt.close(plot)


@pytest.mark.parametrize("name", PLOT_TYPES)
def test_plot_signature_layout_matches_gwpy(name):
    actual_type, parent_type = PLOT_TYPES[name]

    assert inspect.signature(actual_type.plot) == inspect.signature(parent_type.plot)


@pytest.mark.parametrize("name", PLOT_TYPES)
def test_every_supported_positional_prefix_matches_gwpy(name):
    for args in POSITIONAL_PREFIXES[name]:
        actual_error, actual_plot = _plot_outcome(
            _make_plot_object(name, gwpy=False), args
        )
        parent_error, parent_plot = _plot_outcome(
            _make_plot_object(name, gwpy=True), args
        )
        try:
            assert actual_error is parent_error is None
            assert isinstance(actual_plot, GwpyPlot)
        finally:
            _close(actual_plot)
            _close(parent_plot)


@pytest.mark.parametrize("name", PLOT_TYPES)
def test_every_positional_keyword_duplicate_matches_gwpy(name):
    for args, kwargs in DUPLICATE_CALLS[name]:
        actual_error, actual_plot = _plot_outcome(
            _make_plot_object(name, gwpy=False), args, kwargs
        )
        parent_error, parent_plot = _plot_outcome(
            _make_plot_object(name, gwpy=True), args, kwargs
        )
        try:
            assert actual_error is parent_error is TypeError
        finally:
            _close(actual_plot)
            _close(parent_plot)


@pytest.mark.parametrize("name", PLOT_TYPES)
def test_excess_positional_argument_matches_gwpy(name):
    actual_error, actual_plot = _plot_outcome(
        _make_plot_object(name, gwpy=False), EXCESS_CALLS[name]
    )
    parent_error, parent_plot = _plot_outcome(
        _make_plot_object(name, gwpy=True), EXCESS_CALLS[name]
    )
    try:
        assert actual_error is parent_error is TypeError
    finally:
        _close(actual_plot)
        _close(parent_plot)


@pytest.mark.parametrize("name", PLOT_TYPES)
def test_step_success_or_failure_matches_each_gwpy_class(name):
    actual_error, actual_plot = _plot_outcome(
        _make_plot_object(name, gwpy=False), STEP_CALLS[name]
    )
    parent_error, parent_plot = _plot_outcome(
        _make_plot_object(name, gwpy=True), STEP_CALLS[name]
    )
    try:
        assert actual_error is parent_error
    finally:
        _close(actual_plot)
        _close(parent_plot)


@pytest.mark.parametrize("name", PLOT_TYPES)
def test_artist_kwargs_are_forwarded_without_mutating_input(name):
    actual = _make_plot_object(name, gwpy=False)
    if name == "timeseriesdict":
        values_before = {
            key: (series.value.copy(), series.xindex.value.copy())
            for key, series in actual.items()
        }
    else:
        values_before = (actual.value.copy(), actual.xindex.value.copy())

    error, plot = _plot_outcome(actual, kwargs=ARTIST_KWARGS[name])
    try:
        assert error is None
        assert isinstance(plot, GwpyPlot)
    finally:
        _close(plot)

    if name == "timeseriesdict":
        for key, series in actual.items():
            np.testing.assert_array_equal(series.value, values_before[key][0])
            np.testing.assert_array_equal(series.xindex.value, values_before[key][1])
    else:
        np.testing.assert_array_equal(actual.value, values_before[0])
        np.testing.assert_array_equal(actual.xindex.value, values_before[1])


@pytest.mark.parametrize("separate", [False, True], ids=["overlay", "separate"])
def test_timeseriesdict_values_and_labels_match_gwpy(separate):
    actual_error, actual_plot = _plot_outcome(
        _make_plot_object("timeseriesdict", gwpy=False),
        kwargs={"label": "key", "separate": separate},
    )
    parent_error, parent_plot = _plot_outcome(
        _make_plot_object("timeseriesdict", gwpy=True),
        kwargs={"label": "key", "separate": separate},
    )
    try:
        assert actual_error is parent_error is None
        actual_labels = [
            line.get_label() for axis in actual_plot.axes for line in axis.lines
        ]
        parent_labels = [
            line.get_label() for axis in parent_plot.axes for line in axis.lines
        ]
        assert actual_labels == parent_labels == ["A", "B"]
    finally:
        _close(actual_plot)
        _close(parent_plot)
