"""Contract baselines for extended plot helpers."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from gwpy.spectrogram import Spectrogram
from gwpy.timeseries import TimeSeries
from matplotlib.collections import QuadMesh

from gwexpy.fields import ScalarField
from gwexpy.plot.field import FieldPlot
from gwexpy.plot.gauch_dashboard import plot_gauch_dashboard
from gwexpy.plot.pairplot import PairPlot
from gwexpy.plot.plot import plot_summary


def _deterministic_scalar_field() -> ScalarField:
    return ScalarField(
        np.arange(48, dtype=float).reshape(2, 3, 4, 2),
        unit=u.V,
        name="field",
        axis0=np.arange(2) * u.s,
        axis1=np.arange(3) * u.m,
        axis2=np.arange(4) * u.cm,
        axis3=np.arange(2) * u.mm,
    )


def _deterministic_spectrogram() -> Spectrogram:
    return Spectrogram(
        np.array(
            [
                [0.5, 0.25, 0.125, 0.0625],
                [0.4, 0.2, 0.1, 0.05],
                [0.3, 0.15, 0.075, 0.0375],
            ]
        ),
        times=np.arange(3) * u.s,
        frequencies=np.array([1.0, 2.0, 4.0, 8.0]) * u.Hz,
    )


def test_field_plot_add_scalar_current_axis_label_contract():
    field = _deterministic_scalar_field()
    plot = FieldPlot()

    try:
        mesh = plot.add_scalar(
            field,
            x="x",
            y="y",
            slice_kwargs={"t": 0, "z": 0},
        )

        assert isinstance(mesh, QuadMesh)
        ax = plot.gca()
        assert ax.get_xlabel() == "x [m]"
        assert ax.get_ylabel() == "y [cm]"
    finally:
        plt.close(plot.figure)


def test_plot_gauch_dashboard_current_structural_labels():
    ts = TimeSeries(np.arange(6, dtype=float), t0=0, dt=1, unit=u.m)
    sg = _deterministic_spectrogram()
    gauch_result = SimpleNamespace(pvalue_map=sg, statistic_map=sg)

    fig = plot_gauch_dashboard(ts, gauch_result)

    try:
        axes_by_title = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}
        assert {
            "GauCh p-value Map",
            "GauCh KS Statistic Map",
            "Time Series",
        }.issubset(axes_by_title)

        assert axes_by_title["GauCh p-value Map"].get_ylabel() == "Frequency [Hz]"
        assert axes_by_title["GauCh KS Statistic Map"].get_ylabel() == "Frequency [Hz]"

        time_panel = axes_by_title["Time Series"]
        assert time_panel.get_xlabel() == "Time [s]"
        assert time_panel.get_ylabel() == "Amplitude [m]"

    finally:
        plt.close(fig)


def test_plot_gauch_dashboard_omits_unit_brackets_for_unitless_series():
    ts = TimeSeries(np.arange(6, dtype=float), t0=0, dt=1)
    sg = _deterministic_spectrogram()
    gauch_result = SimpleNamespace(pvalue_map=sg, statistic_map=sg)

    fig = plot_gauch_dashboard(ts, gauch_result)

    try:
        axes_by_title = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}
        time_panel = axes_by_title["Time Series"]
        assert time_panel.get_ylabel() == "Amplitude"
    finally:
        plt.close(fig)


def test_pair_plot_current_corner_label_contract():
    ts1 = TimeSeries(np.arange(8, dtype=float), t0=0, dt=1)
    ts2 = TimeSeries(np.arange(8, dtype=float) + 10.0, t0=0, dt=1)

    pair_plot = PairPlot({"H1": ts1, "L1": ts2}, corner=True, bins=4)

    try:
        assert pair_plot.axes.shape == (2, 2)
        assert not pair_plot.axes[0, 1].get_visible()
        assert pair_plot.axes[1, 0].get_xlabel() == "H1"
        assert pair_plot.axes[1, 0].get_ylabel() == "L1"
        assert pair_plot.axes[1, 1].get_xlabel() == "L1"
    finally:
        plt.close(pair_plot.figure)


def test_pair_plot_rejects_mismatched_series_lengths():
    ts1 = TimeSeries(np.arange(8, dtype=float), t0=0, dt=1)
    ts2 = TimeSeries(np.arange(6, dtype=float), t0=0, dt=1)

    with pytest.raises(ValueError, match="same length"):
        PairPlot({"H1": ts1, "L1": ts2}, corner=True, bins=4)


def test_pair_plot_rejects_mismatched_time_spans():
    ts1 = TimeSeries(np.arange(8, dtype=float), t0=0, dt=1)
    ts2 = TimeSeries(np.arange(8, dtype=float), t0=1, dt=1)

    with pytest.raises(ValueError, match="same time span"):
        PairPlot({"H1": ts1, "L1": ts2}, corner=True, bins=4)


def test_pair_plot_rejects_mismatched_sample_rates():
    ts1 = TimeSeries(np.arange(8, dtype=float), t0=0, dt=1)
    ts2 = TimeSeries(np.arange(8, dtype=float), t0=0, dt=0.5)

    with pytest.raises(ValueError, match="same sample rate"):
        PairPlot({"H1": ts1, "L1": ts2}, corner=True, bins=4)


def test_plot_summary_accepts_unitless_spectrogram_without_empty_colorbar_label():
    sg = Spectrogram(
        np.ones((3, 4)),
        t0=0,
        dt=1,
        f0=1,
        df=1,
        name="unitless",
    )

    fig, _ = plot_summary([sg])

    try:
        colorbar_labels = [
            ax.get_ylabel()
            for ax in fig.axes
            if not ax.get_title() and ax.get_ylabel() != "Frequency [Hz]"
        ]
        assert "" in colorbar_labels
        assert "[]" not in colorbar_labels
    finally:
        plt.close(fig)


def test_geomap_optional_backend_error_contract(monkeypatch: pytest.MonkeyPatch):
    import gwexpy.plot.geomap as geomap

    monkeypatch.setattr(geomap, "HAS_PYGMT", False)
    monkeypatch.setattr(
        geomap,
        "_PYGMT_IMPORT_ERROR",
        RuntimeError("pygmt unavailable"),
    )

    with pytest.raises(ImportError) as excinfo:
        geomap.GeoMap()

    message = str(excinfo.value)
    assert "pygmt is required for GeoMap" in message
    assert "pip install pygmt" in message
