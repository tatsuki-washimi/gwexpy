"""Package version contract tests."""

from __future__ import annotations

from importlib.metadata import version

import gwexpy


def test_package_version_matches_installed_metadata() -> None:
    assert gwexpy.__version__ == version("gwexpy")
