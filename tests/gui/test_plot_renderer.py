from __future__ import annotations

import numpy as np
import pytest


def _require_gui_deps():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")


class _ComboBox:
    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:
        return self._text


class _CheckBox:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _Panel:
    def __init__(self, *, y_log: bool = False, y_auto: bool = False) -> None:
        self.rb_y_log = _CheckBox(y_log)
        self.rb_y_auto = _CheckBox(y_auto)


class _Curve:
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def setData(self, x, y) -> None:
        self.x = np.asarray(x)
        self.y = np.asarray(y)


class _Bar:
    def __init__(self, visible: bool = False) -> None:
        self._visible = visible
        self.opts = {}

    def isVisible(self) -> bool:
        return self._visible

    def setOpts(self, **kwargs) -> None:
        self.opts.update(kwargs)


class _Image:
    def __init__(self) -> None:
        self.image = None
        self.levels = None
        self.rect = None
        self.visible = None
        self.cleared = False

    def setImage(self, data, autoLevels=False) -> None:
        self.image = np.asarray(data)
        self.auto_levels = autoLevels

    def setLevels(self, levels) -> None:
        self.levels = list(levels)

    def setRect(self, rect) -> None:
        self.rect = rect

    def setVisible(self, visible: bool) -> None:
        self.visible = visible

    def clear(self) -> None:
        self.cleared = True


class _Plot:
    def __init__(self) -> None:
        self.labels = []
        self.auto_range_calls = []
        self.x_ranges = []

    def setLabel(self, axis: str, text: str) -> None:
        self.labels.append((axis, text))

    def enableAutoRange(self, axis=None, enable=True) -> None:
        self.auto_range_calls.append((axis, enable))

    def setXRange(self, start: float, end: float, padding: float = 0.0) -> None:
        self.x_ranges.append((start, end, padding))


class _MainWindow:
    traces1: list[dict[str, object]] = []
    traces2: list[dict[str, object]] = []


def test_plot_renderer_renders_series_with_relative_time_and_bar_width():
    _require_gui_deps()
    from gwexpy.gui.ui.plot_renderer import PlotRenderer

    renderer = PlotRenderer(_MainWindow())
    curve = _Curve()
    bar = _Bar(visible=True)
    img = _Image()
    info_root = {"units": {"display_y": _ComboBox("dB")}}

    x = np.array([10.0, 11.0, 12.0])
    y = np.array([1.0, 10.0, 100.0])

    renderer._render_series(
        curve,
        bar,
        img,
        (x, y),
        info_root,
        start_time_gps=10.0,
        is_time_axis=True,
        graph_type="ASD",
    )

    assert img.visible is False
    assert np.allclose(curve.x, [0.0, 1.0, 2.0])
    assert np.allclose(curve.y, [0.0, 20.0, 40.0])
    assert np.allclose(bar.opts["x"], [0.0, 1.0, 2.0])
    assert np.allclose(bar.opts["height"], [0.0, 20.0, 40.0])
    assert bar.opts["width"] == 1.0


def test_plot_renderer_renders_spectrogram_with_log_y_rect():
    _require_gui_deps()
    from gwexpy.gui.ui.plot_renderer import PlotRenderer

    renderer = PlotRenderer(_MainWindow())
    curve = _Curve()
    bar = _Bar(visible=True)
    img = _Image()
    info_root = {
        "units": {"display_y": _ComboBox("dB")},
        "panel": _Panel(y_log=True),
    }
    result = {
        "type": "spectrogram",
        "value": np.array([[1.0, 10.0], [100.0, 1000.0]]),
        "times": np.array([100.0, 102.0]),
        "freqs": np.array([1.0, 3.0]),
    }

    renderer._render_spectrogram(
        img,
        curve,
        bar,
        result,
        info_root,
        start_time_gps=100.0,
        graph_type="Spectrogram",
    )

    assert np.allclose(curve.x, [])
    assert bar.opts["height"] == []
    assert img.visible is True
    assert np.allclose(img.image, [[0.0, 10.0], [20.0, 30.0]])
    assert img.levels == [0.0, 30.0]
    assert img.rect is not None
    assert img.rect.x() == -1.0
    assert pytest.approx(img.rect.width()) == 4.0
    assert pytest.approx(img.rect.y()) == 0.0
    assert pytest.approx(img.rect.height()) == np.log10(5.0)


def test_plot_renderer_updates_axis_labels_and_streaming_range():
    _require_gui_deps()
    from gwexpy.gui.ui.plot_renderer import PlotRenderer

    renderer = PlotRenderer(_MainWindow())
    plot = _Plot()
    updated = {"called": False}

    def _range_updater() -> None:
        updated["called"] = True

    info_root = {
        "plot": plot,
        "panel": _Panel(y_auto=True),
        "range_updater": _range_updater,
    }

    renderer.update_axis_labels(info_root, 1234.5, "2026-03-09 12:00:00 UTC")
    renderer.stabilize_streaming_range(
        info_root,
        is_streaming=True,
        is_time_axis=True,
        nds_window=30.0,
    )

    assert plot.labels[-1][0] == "bottom"
    assert "GPS: 1234.5" in plot.labels[-1][1]
    assert ("x", False) in plot.auto_range_calls
    assert ("y", True) in plot.auto_range_calls
    assert plot.x_ranges[-1] == (0, 30.0, 0.02)

    renderer.stabilize_streaming_range(
        info_root,
        is_streaming=False,
        is_time_axis=True,
        nds_window=30.0,
    )
    assert updated["called"] is True
