"""gwexpy._warnings - Package-level warning filters.

Suppresses noisy warnings from external libraries (gwpy, LAL, matplotlib)
that are irrelevant to end-users. Filters are registered once at import time.

Only warnings that carry no actionable information for the user are suppressed
here. Warnings that indicate potential issues with user data or parameters are
left in place (with appropriate stacklevel so they point to user code).
"""
from __future__ import annotations

import warnings

# ---------------------------------------------------------------------------
# LAL / gwpy initialisation noise
# ---------------------------------------------------------------------------
# Emitted by lal when its stdio redirection is activated. No user action needed.
warnings.filterwarnings(
    "ignore",
    message="Wswiglal-redir-stdio",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# matplotlib backend warnings
# ---------------------------------------------------------------------------
# Fired when show() is called in a non-interactive backend (e.g. Agg in CI).
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# gwpy / matplotlib axis warnings
# ---------------------------------------------------------------------------
# Fired when a log-scaled axis receives a non-positive limit – handled
# internally by matplotlib and has no effect on the output.
warnings.filterwarnings(
    "ignore",
    message="Attempt to set non-positive",
    category=UserWarning,
)

# Fired when a legend is requested but no labelled artists exist. Harmless.
warnings.filterwarnings(
    "ignore",
    message="No artists with labels found",
    category=UserWarning,
)
