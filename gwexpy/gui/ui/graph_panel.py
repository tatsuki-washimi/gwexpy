from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PyQt5 import QtWidgets


def _small_spin_int(
    min_val: int = -1000000000, max_val: int = 1000000000, width: int | None = None
) -> QtWidgets.QSpinBox:
    w = QtWidgets.QSpinBox()
    w.setRange(min_val, max_val)
    if width:
        w.setFixedWidth(width)
    return w


def _small_spin_dbl(
    decimals: int = 1,
    width: int | None = None,
    min_val: float = -1e12,
    max_val: float = 1e12,
    step: float = 0.1,
) -> QtWidgets.QDoubleSpinBox:
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(min_val, max_val)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    if width:
        w.setFixedWidth(width)
    return w


class GraphPanel(QtWidgets.QFrame):
    def __init__(
        self,
        plot_idx: int,
        target_plot: pg.PlotWidget,
        traces_items: list[dict[str, Any]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plot_idx = plot_idx
        self.target_plot = target_plot
        self.traces_items = (
            traces_items  # List of {'curve': ..., 'bar': ..., 'img': ...}
        )

        self.trace_controls: list[dict[str, Any]] = []
        self.graph_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.display_y_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()

        self._init_ui()

    def _init_ui(self) -> None:
        from gwexpy.gui.ui._tab_builders import (
            _assemble_stack,
            _build_axis_tab,
            _build_config_tab,
            _build_cursor_tab,
            _build_legend_tab,
            _build_param_tab,
            _build_range_tab,
            _build_style_tab,
            _build_tab_bar_setup,
            _build_traces_tab,
            _build_units_tab,
            _connect_axis_signals,
        )

        _build_tab_bar_setup(self)

        # Build each tab
        traces_w = _build_traces_tab(self)
        range_w = _build_range_tab(self)
        units_w = _build_units_tab(self)
        cursor_w = _build_cursor_tab(self)
        style_w = _build_style_tab(self)
        _build_axis_tab(self, "x")
        _build_axis_tab(self, "y")
        _connect_axis_signals(self)
        legend_w = _build_legend_tab(self)
        param_w = _build_param_tab(self)
        config_w = _build_config_tab(self)

        # Assemble all tabs into the stacked widget and wire cross-tab logic
        _assemble_stack(
            self,
            {
                "traces": traces_w,
                "range": range_w,
                "units": units_w,
                "cursor": cursor_w,
                "config": config_w,
                "style": style_w,
                "legend": legend_w,
                "param": param_w,
            },
        )

    def reset(self) -> None:
        """Reset plot settings, axes, and clear data."""
        self.graph_combo.blockSignals(True)
        self.graph_combo.setCurrentIndex(0)  # Time Series
        self.graph_combo.blockSignals(False)

        # Reset trace selections
        for ctrl in self.trace_controls:
            ctrl["active"].setChecked(False)
            if "chan_a" in ctrl:
                ctrl["chan_a"].clear()
            if "chan_b" in ctrl:
                ctrl["chan_b"].clear()

        # Reset axis scale radio buttons to linear
        self.rb_y_lin.setChecked(True)
        self.rb_x_lin.setChecked(True)
        self.rb_y_auto.setChecked(True)
        self.rb_x_auto.setChecked(True)

        # Reset axis to linear and auto
        self.target_plot.setLogMode(x=False, y=False)
        self.target_plot.enableAutoRange()
        self.target_plot.setLabel("bottom", "Time")
        self.target_plot.setLabel("left", "Signal")
        self.target_plot.setTitle(None)

        # Clear curves
        for item in self.traces_items:
            item["curve"].setData([], [])
            item["bar"].setOpts(height=[])
            item["img"].clear()

        # Sync UI labels
        self.update_range_logic()

    def to_graph_info(self) -> dict[str, Any]:
        return {
            "graph_combo": self.graph_combo,
            "traces": self.trace_controls,
            "range_updater": self.update_range_logic,
            "units": {"display_y": self.display_y_combo},
            "panel": self,
            "plot": self.target_plot,
        }
