from __future__ import annotations

import pytest


def test_series_import_smoke():
    # Cover-Agent will extend this file; keep a minimal smoke test as a base.
    from gwexpy.types.series import Series  # noqa: F401
