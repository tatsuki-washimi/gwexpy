"""Package version contract tests."""

from __future__ import annotations

import gwexpy


def test_package_version_matches_latest_local_release_lineage() -> None:
    assert gwexpy.__version__ == "0.1.5"
