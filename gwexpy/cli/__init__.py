"""
gwexpy CLI module.

NOTE: The CLI is currently a placeholder. Main functionality is provided
through the Python API. Subcommands will be implemented in future versions.
"""

from __future__ import annotations

import sys

from gwpy.cli import (
    PRODUCTS,
    CliProduct,
    CoherencegramProduct,
    CoherenceProduct,
    QtransformProduct,
    SpectrogramProduct,
    SpectrumProduct,
    TimeSeriesProduct,
    TransferFunctionProduct,
    annotations,
    cliproduct,
    coherence,
    coherencegram,
    qtransform,
    spectrogram,
    spectrum,
    timeseries,
    transferfunction,
)

from .._version import __version__

__all__ = ["main", "__version__"]


def main(args=None):
    """Entry point for the gwexpy command."""
    if args is None:
        args = sys.argv[1:]

    # Basic version handling
    if "--version" in args or "-v" in args:
        print(f"gwexpy {__version__}")
        return

    # For now, just show a message or fallback to gwpy if applicable
    # In the future, this will dispatch to various subcommands
    if not args or args[0].startswith("-"):
        print("gwexpy: Experimental gravitational wave analysis tool.")
        print(f"Version: {__version__}")
        print("\nAvailable subcommands: (planned)")
        print("  - spectrogram: Compute and plot spectrograms")
        print("  - spectrum: Compute and plot ASD/PSD")
        print("\nUse --version to see version.")
        return

    print(f"gwexpy: unknown command or option '{args[0]}'")
    sys.exit(1)
