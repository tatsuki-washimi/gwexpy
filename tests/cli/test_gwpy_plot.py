import sys

if hasattr(sys, "ps1"):
    delattr(sys, "ps1")
from gwpy.cli import gwpy_plot as _gwpy_plot
from gwpy.cli.tests.test_gwpy_plot import *  # noqa: F403

_gwpy_plot.INTERACTIVE = False
