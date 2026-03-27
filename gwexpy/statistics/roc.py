"""gwexpy.statistics.roc - Receiver Operating Characteristic (ROC) evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ..timeseries import TimeSeries


def calculate_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve (FPR, TPR) and AUC.
    y_score: probability or statistic where HIGH value means glitch.
    If using p-values, pass 1 - p-value.
    """
    thresholds = np.linspace(np.min(y_score), np.max(y_score), n_points)
    tpr = []
    fpr = []
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
        
    for thresh in thresholds:
        y_pred = y_score >= thresh
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    
    # Sort by FPR for AUC calculation
    idx = np.argsort(fpr)
    fpr = fpr[idx]
    tpr = tpr[idx]
    
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, float(auc)


def evaluate_detection_performance(
    method_func: Callable[[TimeSeries], Any],
    glitch_generator: Callable[..., TimeSeries],
    n_trials: int = 50,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate detection performance (ROC) by comparing clean vs glitchy data.
    """
    y_true = []
    y_score = []
    
    for _ in range(n_trials):
        # Clean case
        ts_clean = glitch_generator(A1=0, **kwargs) # Assuming 0 amplitude is clean
        score_clean = method_func(ts_clean)
        # Handle if score is a map (take max/min depending on sense)
        if hasattr(score_clean, "value"):
            val = np.nanmax(1.0 - score_clean.value) # if score is p-value
        else:
            val = score_clean
        y_true.append(0)
        y_score.append(val)
        
        # Glitchy case
        ts_glitch = glitch_generator(**kwargs)
        score_glitch = method_func(ts_glitch)
        if hasattr(score_glitch, "value"):
            val = np.nanmax(1.0 - score_glitch.value)
        else:
            val = score_glitch
        y_true.append(1)
        y_score.append(val)
        
    return calculate_roc(np.array(y_true), np.array(y_score))
