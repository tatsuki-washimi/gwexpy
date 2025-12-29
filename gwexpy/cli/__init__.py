from gwpy.cli import *


def main():
    """Entry point for the ``gwexpy`` console script.

    The project re-uses the upstream :mod:`gwpy.cli` implementation, so this
    wrapper simply forwards invocation to ensure the advertised entry point is
    available.
    """

    from gwpy.cli import main as gwpy_main

    return gwpy_main()
