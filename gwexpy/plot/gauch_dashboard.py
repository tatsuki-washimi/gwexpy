"""gwexpy.plot.gauch_dashboard - GauCh/Rayleigh dashboard plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

if TYPE_CHECKING:
    from ..timeseries import TimeSeries
    from ..statistics.gauch import GauChResult
    from ..spectrogram import Spectrogram


def plot_gauch_dashboard(
    ts: TimeSeries,
    gauch_res: GauChResult,
    rayleigh_spec: Spectrogram | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """
    Plot a composite dashboard for GauCh/Rayleigh analysis.

    Parameters
    ----------
    ts : TimeSeries
        The original (or whitened) time series.
    gauch_res : GauChResult
        Result from ts.gauch().
    rayleigh_spec : Spectrogram, optional
        Rayleigh statistic spectrogram.
    **kwargs
        Additional plot settings.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figsize = kwargs.pop("figsize", (12, 14))
    fig = plt.figure(figsize=figsize)
    
    # 3 rows, 1 column
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
    
    # Panel 1: GauCh p-value map
    ax1 = fig.add_subplot(gs[0])
    p_map = gauch_res.pvalue_map
    if p_map is not None:
        # Use -log10(p) for better contrast
        log_p = -np.log10(np.clip(p_map.value, 1e-10, 1.0))
        pc1 = ax1.pcolormesh(p_map.times.value, p_map.frequencies.value, log_p.T, 
                            shading="auto", cmap="RdYlBu_r")
        fig.colorbar(pc1, ax=ax1, label="-log10(p-value)")
        ax1.set_title("GauCh p-value Map")
        ax1.set_ylabel("Frequency [Hz]")
        ax1.set_yscale("log")
    
    # Panel 2: Rayleigh statistic (if provided) or GauCh statistic
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if rayleigh_spec is not None:
        pc2 = ax2.pcolormesh(rayleigh_spec.times.value, rayleigh_spec.frequencies.value, 
                            rayleigh_spec.value.T, shading="auto", cmap="viridis")
        fig.colorbar(pc2, ax=ax2, label="Rayleigh Statistic")
        ax2.set_title("Rayleigh Spectrogram")
    else:
        s_map = gauch_res.statistic_map
        if s_map is not None:
            pc2 = ax2.pcolormesh(s_map.times.value, s_map.frequencies.value, s_map.value.T, 
                                shading="auto", cmap="viridis")
            fig.colorbar(pc2, ax=ax2, label="KS Statistic (Dn)")
            ax2.set_title("GauCh KS Statistic Map")
            
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_yscale("log")
    
    # Panel 3: Time Series
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(ts.times.value, ts.value, color="black", linewidth=0.5)
    ax3.set_title("Time Series")
    ax3.set_ylabel(f"Amplitude [{ts.unit}]")
    ax3.set_xlabel("Time [s]")
    
    # Auto-gps scale
    try:
        ax1.set_xscale("auto-gps")
    except Exception:
        pass
        
    return fig
