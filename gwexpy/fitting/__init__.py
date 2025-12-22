from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gwpy.types import Series

if TYPE_CHECKING:
    from .core import FitResult, fit_series

__all__ = ["fit_series", "FitResult"]


def _lazy_series_fit(self: Series, *args: Any, **kwargs: Any) -> Any:
    from gwexpy.fitting import fit_series

    return fit_series(self, *args, **kwargs)


if not hasattr(Series, "fit"):
    Series.fit = _lazy_series_fit


def __getattr__(name: str) -> Any:
    if name in ("fit_series", "FitResult"):
        try:
            from .core import FitResult, fit_series
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "gwexpy.fitting requires optional dependencies (e.g. iminuit) and a working numba setup."
            ) from exc
        return fit_series if name == "fit_series" else FitResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
