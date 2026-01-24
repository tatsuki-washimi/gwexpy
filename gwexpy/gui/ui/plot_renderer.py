"""
gwexpy.gui.ui.plot_renderer - Plot rendering abstraction layer.

This module provides the PlotRenderer class which handles all plot rendering
logic, abstracting away the details of Spectrogram vs Series drawing,
unit conversions (dB, Phase, Magnitude), and coordinate transformations (Log-Y).

Inspired by DTT's TLGPlot architecture where rendering is separated from
measurement logic.
"""

import logging

import numpy as np
from PyQt5 import QtCore

logger = logging.getLogger(__name__)


class PlotRenderer:
    """
    Handles rendering of analysis results to pyqtgraph plot items.

    This class abstracts the rendering logic from MainWindow, providing:
    - Type-based dispatching (Spectrogram vs 1D Series)
    - Unit conversions (dB, Phase, Magnitude)
    - Log-scale coordinate transformations for Spectrogram
    - Time axis relative offset handling

    Parameters
    ----------
    main_window : MainWindow
        Reference to the main window for accessing UI state.
    """

    def __init__(self, main_window):
        """
        Initialize the PlotRenderer.

        Parameters
        ----------
        main_window : MainWindow
            Reference to the main window instance.
        """
        self.mw = main_window

    def render_panel(self, plot_idx, info_root, results, graph_type,
                     start_time_gps=None, is_time_axis=False):
        """
        Render all traces for a single graph panel.

        Parameters
        ----------
        plot_idx : int
            Index of the graph panel (0 or 1).
        info_root : dict
            The graph info dictionary containing plot items and configuration.
        results : list
            List of analysis results for each trace.
        graph_type : str
            The type of graph (e.g., 'ASD', 'Spectrogram', 'Time Series').
        start_time_gps : float, optional
            GPS time to use as t=0 reference.
        is_time_axis : bool
            Whether the X axis represents time.
        """
        traces_items = [self.mw.traces1, self.mw.traces2][plot_idx]

        for t_idx, result in enumerate(results):
            try:
                tr = traces_items[t_idx]
                curve, bar, img = tr["curve"], tr["bar"], tr["img"]

                if result is None:
                    self._clear_trace(curve, bar, img)
                    continue

                if self._is_spectrogram_result(result):
                    self._render_spectrogram(
                        img, curve, bar, result, info_root,
                        start_time_gps, graph_type
                    )
                else:
                    self._render_series(
                        curve, bar, img, result, info_root,
                        start_time_gps, is_time_axis, graph_type
                    )
            except Exception as e:
                logger.warning(f"Error updating Graph {plot_idx + 1} Trace {t_idx}: {e}")

    def _clear_trace(self, curve, bar, img):
        """Clear all trace items."""
        curve.setData([], [])
        if bar.isVisible():
            bar.setOpts(height=[])
        img.clear()

    def _is_spectrogram_result(self, result):
        """Check if the result is a spectrogram-type result."""
        return isinstance(result, dict) and result.get("type") == "spectrogram"

    def _get_display_unit(self, info_root):
        """Get the current display unit from UI controls."""
        try:
            return info_root.get("units", {}).get("display_y").currentText()
        except (AttributeError, KeyError):
            return "None"

    def _apply_unit_conversion(self, data, display_unit, graph_type, is_spectrogram=False):
        """
        Apply unit conversion to data based on display settings.

        Parameters
        ----------
        data : ndarray
            The data values to convert.
        display_unit : str
            One of 'dB', 'Phase', 'Magnitude', 'None'.
        graph_type : str
            The graph type for determining dB scaling.
        is_spectrogram : bool
            Whether this is spectrogram data.

        Returns
        -------
        ndarray
            The converted data.
        """
        if display_unit == "dB":
            if is_spectrogram:
                # Spectrogram always uses 10*log10 (power-like)
                return 10 * np.log10(np.abs(data) + 1e-20)
            else:
                # For amplitude (ASD), use 20*log10; for power, use 10*log10
                factor = 10 if ("Power" in graph_type or "Squared" in graph_type) else 20
                return factor * np.log10(np.abs(data) + 1e-20)

        elif display_unit == "Phase":
            if np.iscomplexobj(data):
                return np.angle(data, deg=True)
            return np.zeros_like(data)

        elif display_unit == "Magnitude":
            return np.abs(data)

        # "None" - return as-is
        return data

    def _render_spectrogram(self, img, curve, bar, result, info_root,
                            start_time_gps, graph_type):
        """
        Render a spectrogram result to an ImageItem.

        Parameters
        ----------
        img : pg.ImageItem
            The image item to render to.
        curve : pg.PlotDataItem
            The curve item (will be hidden).
        bar : pg.BarGraphItem
            The bar item (will be hidden).
        result : dict
            Spectrogram result with 'value', 'times', 'freqs' keys.
        info_root : dict
            Graph info dictionary.
        start_time_gps : float or None
            GPS time reference for t=0.
        graph_type : str
            The graph type string.
        """
        curve.setData([], [])  # Hide curve
        if bar.isVisible():
            bar.setOpts(height=[])

        data = result["value"]
        times = result["times"]
        freqs = result["freqs"]

        # Shift to relative time
        if start_time_gps is not None:
            times = times - start_time_gps

        # Unit conversion
        display_unit = self._get_display_unit(info_root)
        data = self._apply_unit_conversion(data, display_unit, graph_type, is_spectrogram=True)

        # Set image data
        img.setImage(data, autoLevels=False)
        img.setLevels([np.min(data), np.max(data)])

        if len(freqs) <= 1:
            img.setVisible(True)
            return

        # Calculate geometry
        df = freqs[1] - freqs[0]
        height = df * len(freqs)

        # Estimate dt
        dt = 1.0
        if len(times) > 1:
            dt = times[1] - times[0]

        width = dt * len(times)

        # Handle Log-Y
        f0 = freqs[0]
        y_pos = f0
        h_val = height

        is_log_y = self._is_log_y(info_root)

        if is_log_y:
            # Offset 0Hz to a small positive value
            min_f = (df * 0.5) if df > 0 else 1e-6
            if f0 < min_f:
                f0 = min_f

            # Calculate f_end from original linear height
            f_end = freqs[0] + height
            if f_end <= f0:
                f_end = f0 + min_f

            y_pos = np.log10(f0)
            h_val = np.log10(f_end) - y_pos

        # Center alignment: times[0] is the center of the first bin
        n_bins = len(times)
        dt_bin = width / n_bins if n_bins > 0 else 0
        x_start = times[0] - (dt_bin / 2.0)

        img.setRect(QtCore.QRectF(x_start, y_pos, width, h_val))
        img.setVisible(True)

    def _render_series(self, curve, bar, img, result, info_root,
                       start_time_gps, is_time_axis, graph_type):
        """
        Render a 1D series result to a curve or bar graph.

        Parameters
        ----------
        curve : pg.PlotDataItem
            The curve item to render to.
        bar : pg.BarGraphItem
            The bar item (for histogram-style display).
        img : pg.ImageItem
            The image item (will be hidden).
        result : tuple
            (x_vals, y_vals) tuple.
        info_root : dict
            Graph info dictionary.
        start_time_gps : float or None
            GPS time reference for t=0.
        is_time_axis : bool
            Whether X axis is time-based.
        graph_type : str
            The graph type string.
        """
        img.setVisible(False)

        x_vals, y_vals = result

        # Shift time axis to relative
        if is_time_axis and start_time_gps is not None:
            x_vals = x_vals - start_time_gps

        # Unit conversion
        display_unit = self._get_display_unit(info_root)
        y_vals = self._apply_unit_conversion(y_vals, display_unit, graph_type, is_spectrogram=False)

        # Update curve
        curve.setData(x_vals, y_vals)

        # Update bar if visible
        if bar.isVisible():
            bar_width = (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 1
            bar.setOpts(x=x_vals, height=y_vals, width=bar_width)

    def _is_log_y(self, info_root):
        """Check if Y-axis is in log scale."""
        if "panel" not in info_root:
            return False
        try:
            return info_root["panel"].rb_y_log.isChecked()
        except (AttributeError, KeyError):
            return False

    def update_axis_labels(self, info_root, start_time_gps, start_time_utc):
        """
        Update axis labels with time information.

        Parameters
        ----------
        info_root : dict
            Graph info dictionary.
        start_time_gps : float
            GPS time value.
        start_time_utc : str
            UTC time string.
        """
        label_text = f"Time [s] (Start: {start_time_utc} / GPS: {start_time_gps})"
        info_root["plot"].setLabel("bottom", label_text)

    def stabilize_streaming_range(self, info_root, is_streaming, is_time_axis, nds_window):
        """
        Stabilize plot range during streaming mode.

        Parameters
        ----------
        info_root : dict
            Graph info dictionary.
        is_streaming : bool
            Whether currently streaming.
        is_time_axis : bool
            Whether X axis is time-based.
        nds_window : float
            NDS window duration in seconds.
        """
        if "range_updater" not in info_root:
            return

        if is_streaming and is_time_axis:
            # Only update Y-axis, fix X-axis during streaming
            plot = info_root.get("plot")
            if plot:
                plot.enableAutoRange(axis="x", enable=False)
                plot.setXRange(0, nds_window, padding=0.02)

            # Still update Y-axis if auto mode
            panel = info_root.get("panel")
            if panel and hasattr(panel, "rb_y_auto") and panel.rb_y_auto.isChecked():
                plot.enableAutoRange(axis="y")
        else:
            info_root["range_updater"]()
