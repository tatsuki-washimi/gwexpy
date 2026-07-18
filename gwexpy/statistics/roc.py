"""gwexpy.statistics.roc - Receiver Operating Characteristic (ROC) evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..timeseries import TimeSeries


def calculate_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_points: int = 100,
    pos_label: Any = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate ROC curve (FPR, TPR) and AUC.

    Parameters
    ----------
    y_true : np.ndarray
        Binary class labels. The positive class is ``pos_label``; any other
        label is treated as negative. This follows the sklearn convention, so
        ``{-1, +1}``, ``{0, 1}`` and ``{1, 2}`` encodings are all handled
        consistently.
    y_score : np.ndarray
        Probability or statistic where a HIGH value means glitch (positive).
        If using p-values, pass ``1 - p_value``.
    n_points : int, optional
        Number of thresholds sampled across the score range. Must be ``>= 2``.
    pos_label : Any, optional
        Label value treated as the positive class (default ``1``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(fpr, tpr, auc)``.

    Raises
    ------
    ValueError
        For any degenerate input where the ROC curve is undefined: mismatched
        shapes, ``n_points < 2``, non-finite ``y_score``, all-equal ``y_score``
        (no threshold variation), or an empty positive/negative class. The
        function never returns a meaningless numeric default (e.g. AUC=0.5/0.0)
        that would be indistinguishable from a real result.

    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape != y_score.shape:
        raise ValueError(
            "ROC undefined: y_true and y_score must have the same shape "
            f"(got {y_true.shape} vs {y_score.shape})"
        )
    if y_true.size == 0:
        raise ValueError(
            "ROC undefined: y_true/y_score must contain at least one sample"
        )
    if n_points < 2:
        raise ValueError(
            f"ROC undefined: n_points must be >= 2 (got {n_points})"
        )
    if not np.all(np.isfinite(y_score)):
        raise ValueError(
            "ROC undefined: y_score contains non-finite values (NaN/Inf)"
        )

    s_min = float(np.min(y_score))
    s_max = float(np.max(y_score))
    if s_max == s_min:
        raise ValueError(
            "ROC undefined: all y_score values are identical "
            f"({s_min}); no threshold variation"
        )

    is_pos = y_true == pos_label
    n_pos = int(np.sum(is_pos))
    n_neg = int(np.sum(~is_pos))

    if n_pos == 0:
        raise ValueError(
            f"ROC undefined: positive class is empty (n_pos=0, pos_label={pos_label!r})"
        )
    if n_neg == 0:
        raise ValueError(
            "ROC undefined: negative class is empty (n_neg=0); "
            f"all labels equal pos_label={pos_label!r}"
        )

    thresholds = np.linspace(s_min, s_max, n_points)
    tpr_list: list[float] = []
    fpr_list: list[float] = []

    for thresh in thresholds:
        y_pred = y_score >= thresh
        tp = np.sum(y_pred & is_pos)
        fp = np.sum(y_pred & ~is_pos)
        tpr_list.append(float(tp / n_pos))
        fpr_list.append(float(fp / n_neg))

    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)

    # Sort by FPR (primary) then TPR (secondary) so np.trapz integrates a
    # monotone curve; a plain argsort(fpr) leaves ties unordered in TPR and
    # biases the AUC.
    idx = np.lexsort((tpr, fpr))
    fpr = fpr[idx]
    tpr = tpr[idx]

    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc = trapz(tpr, fpr)
    return fpr, tpr, float(auc)


def evaluate_detection_performance(
    method_func: Callable[[TimeSeries], Any],
    glitch_generator: Callable[..., TimeSeries],
    n_trials: int = 50,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate detection performance by comparing clean and glitchy data."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1 (got {n_trials})")

    y_true = []
    y_score = []

    for _ in range(n_trials):
        # Clean case
        ts_clean = glitch_generator(A1=0, **kwargs)  # Assuming 0 amplitude is clean
        score_clean = method_func(ts_clean)
        y_true.append(0)
        y_score.append(_reduce_score(score_clean))

        # Glitchy case
        ts_glitch = glitch_generator(**kwargs)
        score_glitch = method_func(ts_glitch)
        y_true.append(1)
        y_score.append(_reduce_score(score_glitch))

    return calculate_roc(np.array(y_true), np.array(y_score))


def _reduce_score(score: Any) -> float:
    """Reduce a per-trial detection score to a single positive-sense value.

    A score map (anything exposing ``.value``, e.g. a Spectrogram) is treated
    as p-values and reduced via ``max(1 - value)`` over its finite entries.
    An empty or all-NaN map is a degenerate input (e.g. a fully-cropped map),
    so it raises ``ValueError`` instead of crashing inside ``np.nanmax`` or
    silently producing a NaN score.
    """
    if not hasattr(score, "value"):
        return score

    arr = np.asarray(1.0 - score.value, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(
            "evaluate_detection_performance: score map is empty or all-NaN; "
            "cannot reduce to a detection statistic"
        )
    return float(np.max(finite))
