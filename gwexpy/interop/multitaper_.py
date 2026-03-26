"""
gwexpy.interop.multitaper_
--------------------------

Interoperability with multitaper spectral estimation packages.

Two packages are supported:

* **multitaper** (Prieto) — object-oriented API via ``MTSpec`` / ``MTSine``.
  Install with ``pip install multitaper``.

* **mtspec** (Krischer) — function-based API returning ``(spectrum, freq)``.
  Install with ``pip install mtspec``.

Both are converted to ``FrequencySeries`` (or ``FrequencySeriesDict`` when
confidence intervals are requested).

References
----------
Prieto, G. A. (2022). *The multitaper spectrum analysis package in Python*.
Seismological Research Letters.

Krischer, L. et al. — https://github.com/krischer/mtspec
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from gwexpy.interop._registry import ConverterRegistry

__all__ = ["from_mtspec", "from_mtspec_array"]


def from_mtspec(
    cls: type,
    mt: Any,
    *,
    quantity: Literal["psd", "asd"] = "psd",
    include_ci: bool = True,
) -> Any:
    """Convert a Prieto ``MTSpec`` / ``MTSine`` object to a GWexpy type.

    Parameters
    ----------
    cls : type
        Target type.  Use ``FrequencySeries`` for a plain spectrum, or
        ``FrequencySeriesDict`` to request a dict that also contains
        confidence-interval series.
    mt : multitaper.mtspec.MTSpec or multitaper.mtsine.MTSine
        Computed multitaper object.  Expected attributes:

        - ``freq``    : ndarray (nf,) — frequency axis [Hz]
        - ``spec``    : ndarray (nf,) — adaptive-weighted PSD
        - ``spec_ci`` : ndarray (nf, 2) — 95 % jackknife CI ``[lower, upper]``
          (optional; not present on ``MTSine``)
        - ``se``      : ndarray (nf,) — degrees of freedom per bin (optional)

    quantity : {"psd", "asd"}, default "psd"
        Whether to return power or amplitude spectral density.
        ``"asd"`` applies ``np.sqrt`` to the PSD.
    include_ci : bool, default True
        If *True* **and** the object carries ``spec_ci``, return a
        ``FrequencySeriesDict`` with keys ``"psd"``/``"asd"``,
        ``"ci_lower"``, and ``"ci_upper"``.  Ignored when the CI attribute
        is absent.

    Returns
    -------
    FrequencySeries
        When CI is unavailable or ``include_ci=False``.
    FrequencySeriesDict
        When CI is available and ``include_ci=True``.

    Raises
    ------
    ValueError
        If ``quantity`` is not ``"psd"`` or ``"asd"``, or if the frequency
        axis is empty.
    """
    quantity_lower = quantity.lower()
    if quantity_lower not in ("psd", "asd"):
        raise ValueError(f"quantity must be 'psd' or 'asd', got '{quantity}'")

    freq = np.asarray(mt.freq, dtype=np.float64).ravel()
    spec = np.asarray(mt.spec, dtype=np.float64).ravel()

    if len(freq) == 0:
        raise ValueError("MTSpec object has an empty frequency axis")

    # Validate that freq is (approximately) equally spaced
    _check_equal_spacing(freq)

    values = np.sqrt(spec) if quantity_lower == "asd" else spec
    spec_ci = getattr(mt, "spec_ci", None)
    has_ci = spec_ci is not None and include_ci

    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
    FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")
    want_dict = FrequencySeriesDict is not None and (
        cls is FrequencySeriesDict or _is_subclass_safe(cls, FrequencySeriesDict)
    )

    label = quantity_lower
    fs_main = FrequencySeries(
        values,
        frequencies=freq,
        name=f"multitaper {label}",
    )

    # Store multitaper metadata as instance attributes (post-construction)
    for attr in ("nw", "kspec", "kuse"):
        val = getattr(mt, attr, None)
        if val is not None and np.ndim(val) == 0:
            try:
                object.__setattr__(fs_main, f"_mt_{attr}", float(val))
            except Exception:
                pass

    if not has_ci or not want_dict:
        return fs_main

    # Build CI series
    spec_ci_arr = np.asarray(spec_ci, dtype=np.float64)  # (nf, 2)
    ci_lo_vals = spec_ci_arr[:, 0]
    ci_hi_vals = spec_ci_arr[:, 1]

    if quantity_lower == "asd":
        ci_lo_vals = np.sqrt(np.maximum(ci_lo_vals, 0.0))
        ci_hi_vals = np.sqrt(np.maximum(ci_hi_vals, 0.0))

    fs_lo = FrequencySeries(
        ci_lo_vals,
        frequencies=freq,
        name=f"multitaper {label} ci_lower",
    )
    fs_hi = FrequencySeries(
        ci_hi_vals,
        frequencies=freq,
        name=f"multitaper {label} ci_upper",
    )

    result = FrequencySeriesDict()
    result[label] = fs_main
    result["ci_lower"] = fs_lo
    result["ci_upper"] = fs_hi
    return result


def from_mtspec_array(
    cls: type,
    spectrum: np.ndarray,
    freq: np.ndarray,
    *,
    quantity: Literal["psd", "asd"] = "psd",
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    unit: Any | None = None,
) -> Any:
    """Convert Krischer ``mtspec`` function output to a GWexpy type.

    Parameters
    ----------
    cls : type
        Target type.  Use ``FrequencySeries`` for a plain spectrum, or
        ``FrequencySeriesDict`` to receive CI series alongside the main one.
    spectrum : array-like, shape (nf,)
        PSD (or ASD, depending on ``quantity``) array returned by
        ``mtspec.mtspec()``.
    freq : array-like, shape (nf,)
        Frequency axis [Hz] returned by ``mtspec.mtspec()``.
    quantity : {"psd", "asd"}, default "psd"
        Whether *spectrum* is a PSD or ASD.  Set to ``"asd"`` to interpret
        the input as amplitude spectral density.
    ci_lower : array-like, shape (nf,), optional
        Lower confidence-interval bound (same units as *spectrum*).
    ci_upper : array-like, shape (nf,), optional
        Upper confidence-interval bound (same units as *spectrum*).
    unit : str or astropy.units.Unit, optional
        Physical unit of the spectral density values.

    Returns
    -------
    FrequencySeries
        When *ci_lower* and *ci_upper* are both *None*.
    FrequencySeriesDict
        When either CI array is provided.  Keys: ``"psd"``/``"asd"``,
        ``"ci_lower"``, ``"ci_upper"``.

    Raises
    ------
    ValueError
        If ``quantity`` is not ``"psd"`` or ``"asd"``, if *freq* is not
        equally spaced, or if shapes are inconsistent.
    """
    quantity_lower = quantity.lower()
    if quantity_lower not in ("psd", "asd"):
        raise ValueError(f"quantity must be 'psd' or 'asd', got '{quantity}'")

    freq_arr = np.asarray(freq, dtype=np.float64).ravel()
    spec_arr = np.asarray(spectrum, dtype=np.float64).ravel()

    if len(freq_arr) == 0:
        raise ValueError("freq array is empty")
    if len(freq_arr) != len(spec_arr):
        raise ValueError(
            f"freq length ({len(freq_arr)}) != spectrum length ({len(spec_arr)})"
        )

    _check_equal_spacing(freq_arr)

    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
    FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")
    want_dict = FrequencySeriesDict is not None and (
        cls is FrequencySeriesDict or _is_subclass_safe(cls, FrequencySeriesDict)
    )
    label = quantity_lower

    fs_main = FrequencySeries(
        spec_arr,
        frequencies=freq_arr,
        name=f"mtspec {label}",
        unit=unit,
    )

    has_ci = ci_lower is not None or ci_upper is not None
    if not has_ci or not want_dict:
        return fs_main

    ci_lo_arr = np.asarray(ci_lower, dtype=np.float64).ravel() if ci_lower is not None else spec_arr
    ci_hi_arr = np.asarray(ci_upper, dtype=np.float64).ravel() if ci_upper is not None else spec_arr

    result = FrequencySeriesDict()
    result[label] = fs_main
    result["ci_lower"] = FrequencySeries(
        ci_lo_arr, frequencies=freq_arr, name=f"mtspec {label} ci_lower", unit=unit
    )
    result["ci_upper"] = FrequencySeries(
        ci_hi_arr, frequencies=freq_arr, name=f"mtspec {label} ci_upper", unit=unit
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_subclass_safe(cls: type, parent: type) -> bool:
    """Return True if *cls* is a subclass of *parent*, without raising."""
    try:
        return issubclass(cls, parent)
    except TypeError:
        return False


def _check_equal_spacing(freq: np.ndarray, rtol: float = 1e-4) -> None:
    """Raise ValueError if *freq* is not (approximately) equally spaced."""
    if len(freq) < 2:
        return
    diffs = np.diff(freq)
    mean_diff = float(np.mean(diffs))
    if mean_diff <= 0:
        raise ValueError("freq must be strictly increasing")
    if np.any(np.abs(diffs - mean_diff) > rtol * mean_diff):
        raise ValueError(
            "FrequencySeries requires equally spaced frequencies, "
            "but the provided freq axis is not uniform "
            f"(max deviation {float(np.max(np.abs(diffs - mean_diff))):.3g} Hz)"
        )
