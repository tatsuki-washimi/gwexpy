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

    if getattr(AxisItem, "_gwexpy_deleted_guard", False):
        return

    def _is_deleted_error(exc: RuntimeError) -> bool:
        return "has been deleted" in str(exc)

    def _is_deleted(obj) -> bool:
        try:
            return sip.isdeleted(obj)
        except Exception:
            return False

    orig_update_height = getattr(AxisItem, "_updateHeight", None)
    orig_paint = getattr(AxisItem, "paint", None)

    if orig_update_height is not None:

        def _updateHeight(self, *args, **kwargs):  # noqa: N802
            if _is_deleted(self):
                return
            try:
                return orig_update_height(self, *args, **kwargs)
            except RuntimeError as exc:
                if _is_deleted_error(exc):
                    return
                raise

        AxisItem._updateHeight = _updateHeight  # type: ignore[method-assign]

    if orig_paint is not None:

        def paint(self, *args, **kwargs):
            if _is_deleted(self):
                return
            try:
                return orig_paint(self, *args, **kwargs)
            except RuntimeError as exc:
                if _is_deleted_error(exc):
                    return
                raise

        AxisItem.paint = paint  # type: ignore[method-assign]

    AxisItem._gwexpy_deleted_guard = True


_patch_pyqtgraph_axisitem_deleted_guard()

__all__ = ["main"]
