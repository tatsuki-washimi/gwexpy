import pytest


def test_axisitem_paint_ignores_deleted_qt_runtime(monkeypatch):
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("gwexpy.gui")
    axisitem_module = pytest.importorskip("pyqtgraph.graphicsItems.AxisItem")

    axis = axisitem_module.AxisItem("left")

    def raise_deleted_error(*args, **kwargs):
        raise RuntimeError("wrapped C/C++ object of type ViewBox has been deleted")

    monkeypatch.setattr(axis, "generateDrawSpecs", raise_deleted_error)

    axis.paint(object(), None, None)


def test_axisitem_paint_keeps_unrelated_runtime_errors(monkeypatch):
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("gwexpy.gui")
    axisitem_module = pytest.importorskip("pyqtgraph.graphicsItems.AxisItem")

    axis = axisitem_module.AxisItem("left")

    def raise_other_error(*args, **kwargs):
        raise RuntimeError("unexpected paint failure")

    monkeypatch.setattr(axis, "generateDrawSpecs", raise_other_error)

    with pytest.raises(RuntimeError, match="unexpected paint failure"):
        axis.paint(object(), None, None)
