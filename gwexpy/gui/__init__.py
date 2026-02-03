from __future__ import annotations

from .pyaggui import main


def _patch_pyqtgraph_axisitem_deleted_guard() -> None:
    """Avoid rare Qt paint crashes when AxisItem is deleted mid-update.

    In headless/CI runs, Qt can still dispatch paint events while widgets are
    being torn down. Some pyqtgraph versions call QWidget methods on deleted
    wrappers, raising RuntimeError inside the Qt event loop and occasionally
    leading to a segfault. We defensively no-op the update when sip reports the
    underlying C++ object is gone.
    """

    try:
        import sip  # type: ignore[import-not-found]
        from pyqtgraph.graphicsItems.AxisItem import AxisItem
    except Exception:
        return

    orig_update_height = getattr(AxisItem, "_updateHeight", None)
    if orig_update_height is None:
        return

    def _updateHeight(self, *args, **kwargs):  # noqa: N802
        try:
            if sip.isdeleted(self):
                return
        except Exception:
            # Best-effort guard; fall back to original behavior.
            pass
        return orig_update_height(self, *args, **kwargs)

    AxisItem._updateHeight = _updateHeight  # type: ignore[method-assign]


_patch_pyqtgraph_axisitem_deleted_guard()

__all__ = ["main"]
