"""Tests for gwexpy/interop/_optional.py — require_optional."""
from __future__ import annotations

import pytest

from gwexpy.interop._optional import require_optional


class TestRequireOptional:
    def test_existing_package(self):
        # numpy is always available
        np = require_optional("scipy")
        assert np is not None

    def test_unknown_package_not_in_map(self):
        # Lines 75-77 — fallback: use name as pkg_name
        # Also tests lines 155-165 — ImportError with unknown extra
        with pytest.raises(ImportError, match="not installed"):
            require_optional("_gwexpy_nonexistent_pkg_xyz_")

    def test_known_package_with_extra_map(self):
        # Lines 155-162 — known package with extra in _EXTRA_MAP (meep not installed)
        with pytest.raises(ImportError, match="pip install"):
            require_optional("meep")

    def test_import_error_with_extra_hint(self):
        # Line 157-158 — extra found in map → install_cmd includes gwexpy[...]
        # openems not installed and is in map without specific extra → plain pip install
        with pytest.raises(ImportError):
            require_optional("openems")

    def test_import_error_without_extra_hint(self):
        # Lines 159-160 — no extra in map → plain pip install
        with pytest.raises(ImportError, match="pip install"):
            require_optional("_totally_unknown_package_abc123_")

    def test_existing_pandas(self):
        # Should import pandas without error
        pd = require_optional("pandas")
        assert hasattr(pd, "DataFrame")
