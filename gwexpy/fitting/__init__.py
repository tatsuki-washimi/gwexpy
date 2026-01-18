from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from gwpy.types import Series

if TYPE_CHECKING:
    from .core import Fitter, FitResult, fit_series
    from .gls import GLS, GeneralizedLeastSquares
    from .highlevel import fit_bootstrap_spectrum

__all__ = [
    "fit_series",
    "FitResult",
    "Fitter",
    "GeneralizedLeastSquares",
    "GLS",
    "fit_bootstrap_spectrum",
    "enable_series_fit",
    "enable_fitting_monkeypatch",
]


def _lazy_series_fit(self: Series, *args: Any, **kwargs: Any) -> Any:
    from gwexpy.fitting import fit_series

    return fit_series(self, *args, **kwargs)


def enable_series_fit() -> None:
    """
    Opt-in monkeypatch for gwpy.types.Series.fit.

    Note: standard gwexpy classes (TimeSeries, FrequencySeries) already have the .fit() method
    via inheritance. This function is generally not needed unless you are using base gwpy objects
    directly.
    """
    if not hasattr(Series, "fit"):
        Series.fit = _lazy_series_fit


if os.environ.get("GWEXPY_ENABLE_SERIES_FIT") == "1":
    enable_series_fit()


# Backward compatibility alias (used in README.md)
enable_fitting_monkeypatch = enable_series_fit


def __getattr__(name: str) -> Any:
    if name in ("fit_series", "FitResult", "Fitter"):
        try:
            from .core import Fitter, FitResult, fit_series
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "gwexpy.fitting requires optional dependencies (e.g. iminuit) and a working numba setup."
            ) from exc
        if name == "fit_series":
            return fit_series
        if name == "FitResult":
            return FitResult
        return Fitter
    if name in ("GeneralizedLeastSquares", "GLS"):
        try:
            from .gls import GLS, GeneralizedLeastSquares
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "gwexpy.fitting requires optional dependencies (e.g. iminuit)."
            ) from exc
        return GeneralizedLeastSquares if name == "GeneralizedLeastSquares" else GLS
    if name == "fit_bootstrap_spectrum":
        try:
            from .highlevel import fit_bootstrap_spectrum
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "gwexpy.fitting.highlevel requires optional dependencies."
            ) from exc
        return fit_bootstrap_spectrum
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
