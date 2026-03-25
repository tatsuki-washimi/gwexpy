"""
gwexpy.interop.gwinc_
---------------------

Interoperability with pygwinc (gravitational wave interferometer noise budget).

Provides conversion from a gwinc ``Budget`` trace hierarchy to GWexpy
``FrequencySeries`` (total noise) or ``FrequencySeriesDict`` (all sub-traces).

Notes
-----
The simpler ``gwexpy.noise.gwinc_.from_pygwinc`` helper (strain/DARM only) is
preserved unchanged. This module adds the richer trace-expansion API.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from gwexpy.interop._registry import ConverterRegistry

from ._optional import require_optional

__all__ = ["from_gwinc_budget"]

_DEFAULT_FMIN = 10.0
_DEFAULT_FMAX = 4000.0
_DEFAULT_DF = 1.0


def _collect_traces(trace: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively collect all named sub-traces from a gwinc BudgetTrace.

    Parameters
    ----------
    trace : gwinc.BudgetTrace
        Root or sub-trace object.
    prefix : str
        Key prefix for nested traces.

    Returns
    -------
    dict[str, BudgetTrace]
        Flat mapping of trace name → trace object.
    """
    result: dict[str, Any] = {}
    # gwinc traces expose sub-traces via dict-like access
    try:
        sub_names = list(trace.keys())
    except AttributeError:
        sub_names = []

    for name in sub_names:
        full_name = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
        sub = trace[name]
        result[full_name] = sub
        result.update(_collect_traces(sub, prefix=full_name))

    return result


def from_gwinc_budget(
    cls: type,
    budget_or_model: Any,
    *,
    frequencies: np.ndarray | None = None,
    quantity: Literal["asd", "psd"] = "asd",
    trace_name: str | None = None,
    fmin: float = _DEFAULT_FMIN,
    fmax: float = _DEFAULT_FMAX,
    df: float = _DEFAULT_DF,
) -> Any:
    """Create FrequencySeries or FrequencySeriesDict from a gwinc Budget.

    Parameters
    ----------
    cls : type
        ``FrequencySeries`` or ``FrequencySeriesDict`` class to instantiate.
    budget_or_model : gwinc.Budget or str
        Pre-loaded ``Budget`` object, or a model name string (e.g., ``"aLIGO"``,
        ``"Aplus"``) which is passed to ``gwinc.load_budget``.
    frequencies : array-like, optional
        Frequency array in Hz. If *None*, generated from ``fmin``/``fmax``/``df``.
    quantity : {"asd", "psd"}, default "asd"
        Whether to return amplitude or power spectral density.
    trace_name : str, optional
        Name of a specific sub-trace to extract (e.g., ``"Quantum"``).
        If *None*:

        - ``FrequencySeries`` cls → total noise only.
        - ``FrequencySeriesDict`` cls → total + all sub-traces.
    fmin : float, default 10.0
        Minimum frequency [Hz] when *frequencies* is not provided.
    fmax : float, default 4000.0
        Maximum frequency [Hz] when *frequencies* is not provided.
    df : float, default 1.0
        Frequency step [Hz] when *frequencies* is not provided.

    Returns
    -------
    FrequencySeries
        When ``trace_name`` is given, or ``cls`` is ``FrequencySeries``.
    FrequencySeriesDict
        When ``trace_name`` is *None* and ``cls`` is ``FrequencySeriesDict``.
        Keys: ``"Total"``, plus sub-trace names.

    Raises
    ------
    ValueError
        If ``trace_name`` does not exist in the budget trace hierarchy, or if
        ``quantity`` is not ``"asd"`` or ``"psd"``.
    ImportError
        If pygwinc is not installed.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
    >>> asd = FrequencySeries.from_gwinc_budget("aLIGO")
    >>> noise_budget = FrequencySeriesDict.from_gwinc_budget("aLIGO")
    >>> quantum = FrequencySeries.from_gwinc_budget("aLIGO", trace_name="Quantum")
    """
    gwinc = require_optional("gwinc")

    quantity_lower = quantity.lower()
    if quantity_lower not in ("asd", "psd"):
        raise ValueError(f"quantity must be 'asd' or 'psd', got '{quantity}'")

    # Build frequency array
    if frequencies is None:
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        frequencies = np.arange(fmin, fmax + df, df).astype(float)
    else:
        frequencies = np.asarray(frequencies, dtype=float)

    # Load budget if a string was given
    if isinstance(budget_or_model, str):
        model_name = "Aplus" if budget_or_model == "A+" else budget_or_model
        budget = gwinc.load_budget(model_name)
    else:
        budget = budget_or_model
        model_name = getattr(budget, "name", str(budget_or_model))

    # Run the simulation
    trace = budget.run(freq=frequencies)

    # Helper: psd → output value
    def _to_value(tr: Any) -> np.ndarray:
        psd = np.asarray(tr.psd)
        return np.sqrt(psd) if quantity_lower == "asd" else psd

    unit_str = "1 / Hz(1/2)" if quantity_lower == "asd" else "1 / Hz"

    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")

    def _make_fs(values: np.ndarray, label: str) -> Any:
        return FrequencySeries(
            values,
            frequencies=frequencies,
            unit=unit_str,
            name=f"{model_name} {label}",
        )

    # Single trace extraction
    if trace_name is not None:
        if trace_name == "Total":
            return _make_fs(_to_value(trace), "Total")

        subtrace_map = _collect_traces(trace)
        if trace_name not in subtrace_map:
            available = ["Total"] + sorted(subtrace_map.keys())
            raise ValueError(
                f"Trace '{trace_name}' not found. "
                f"Available traces: {available}"
            )
        return _make_fs(_to_value(subtrace_map[trace_name]), trace_name)

    # Total-only (FrequencySeries cls without trace_name)
    FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")
    _is_dict_cls = issubclass(cls, FrequencySeriesDict)

    if not _is_dict_cls:
        return _make_fs(_to_value(trace), "Total")

    # Full dict: Total + all sub-traces
    result = FrequencySeriesDict()
    result["Total"] = _make_fs(_to_value(trace), "Total")
    subtrace_map = _collect_traces(trace)
    for name, subtrace in subtrace_map.items():
        try:
            values = _to_value(subtrace)
        except Exception:
            continue  # Skip traces without a psd attribute
        result[name] = _make_fs(values, name)

    return result
