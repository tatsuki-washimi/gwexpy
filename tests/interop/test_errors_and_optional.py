"""Tests for gwexpy/interop/errors.py and gwexpy/interop/_optional.py."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from gwexpy.interop._optional import require_optional
from gwexpy.interop.errors import IoNotImplementedError, raise_unimplemented_io

# ---------------------------------------------------------------------------
# errors.py
# ---------------------------------------------------------------------------


class TestIoNotImplementedError:
    def test_is_not_implemented_error_subclass(self):
        assert issubclass(IoNotImplementedError, NotImplementedError)

    def test_can_be_raised(self):
        with pytest.raises(IoNotImplementedError):
            raise IoNotImplementedError("test")


class TestRaiseUnimplementedIo:
    def test_raises_io_not_implemented_error(self):
        with pytest.raises(IoNotImplementedError):
            raise_unimplemented_io("WIN")

    def test_message_contains_format_name(self):
        with pytest.raises(IoNotImplementedError, match="WIN"):
            raise_unimplemented_io("WIN")

    def test_with_hint(self):
        with pytest.raises(IoNotImplementedError, match="use obspy"):
            raise_unimplemented_io("WIN", hint="use obspy")

    def test_with_refs(self):
        with pytest.raises(IoNotImplementedError, match="http://example.com"):
            raise_unimplemented_io("SDB", refs="http://example.com")

    def test_with_plugin_help(self):
        with pytest.raises(IoNotImplementedError, match="docs/plugins"):
            raise_unimplemented_io("SDB", plugin_help="docs/plugins")

    def test_without_plugin_help_includes_contribution_message(self):
        with pytest.raises(IoNotImplementedError, match="welcome"):
            raise_unimplemented_io("SDB")

    def test_all_options(self):
        with pytest.raises(IoNotImplementedError) as exc_info:
            raise_unimplemented_io(
                "FOO", hint="try bar", refs="ref_url", plugin_help="plugin_url"
            )
        msg = str(exc_info.value)
        assert "FOO" in msg
        assert "try bar" in msg
        assert "ref_url" in msg
        assert "plugin_url" in msg


# ---------------------------------------------------------------------------
# _optional.py
# ---------------------------------------------------------------------------


class TestRequireOptional:
    def test_returns_module_for_known_package(self):
        mod = require_optional("scipy")
        import scipy
        assert mod is scipy

    def test_returns_module_for_h5py(self):
        mod = require_optional("h5py")
        import h5py
        assert mod is h5py

    def test_missing_package_raises_import_error(self):
        with pytest.raises(ImportError, match="not installed"):
            require_optional("_nonexistent_package_xyz_")

    def test_missing_package_error_includes_install_command(self):
        with pytest.raises(ImportError, match="pip install"):
            require_optional("_nonexistent_package_xyz_")

    def test_known_extra_maps_to_correct_install_cmd(self):
        """Known packages suggest the correct extras group."""
        with patch.dict(sys.modules, {"iminuit": None}):
            # Remove from cache if present so import fails
            sys.modules.pop("iminuit", None)
            try:
                import iminuit  # noqa: F401
                # Already installed — skip
                pytest.skip("iminuit is installed")
            except ImportError:
                pass
        with pytest.raises(ImportError, match="fitting"):
            require_optional("iminuit")

    def test_unknown_key_falls_back_to_name(self):
        """Packages not in _OPTIONAL_DEPENDENCIES map use the name directly."""
        with pytest.raises(ImportError):
            require_optional("_unknown_package_abc_123_")

    def test_declared_extra_suggests_gwexpy_extra_and_all(self):
        """Packages in extras included by all suggest both install paths."""
        with patch.dict(sys.modules, {"xarray": None}):
            with pytest.raises(ImportError) as excinfo:
                require_optional("xarray")

        msg = str(excinfo.value)
        assert "pip install 'gwexpy[netcdf4]'" in msg
        assert 'pip install "gwexpy[all]"' in msg

    def test_gui_dependencies_use_bare_install_hint(self):
        """GUI dependencies are not exposed as a first-release PyPI extra."""
        with patch.dict(sys.modules, {"PyQt5": None}):
            with pytest.raises(ImportError) as excinfo:
                require_optional("PyQt5")

        msg = str(excinfo.value)
        assert "pip install PyQt5" in msg
        assert "gwexpy[gui]" not in msg
        assert "gwexpy[all]" not in msg

    @pytest.mark.parametrize("name", ["librosa", "pyroomacoustics"])
    def test_undeclared_audio_neighbors_use_bare_install(self, name):
        """Packages without declared extras should not suggest gwexpy[audio]."""
        with patch.dict(sys.modules, {name: None}):
            with pytest.raises(ImportError) as excinfo:
                require_optional(name)

        msg = str(excinfo.value)
        assert f"pip install {name}" in msg
        assert "gwexpy[audio]" not in msg
        assert "gwexpy[all]" not in msg

    @pytest.mark.parametrize("name", ["finesse", "pycbc"])
    def test_contract_bare_install_interop_packages_use_bare_install(self, name):
        """Interop packages with extras: [] should not suggest a GWexpy extra."""
        with patch.dict(sys.modules, {name: None}):
            with pytest.raises(ImportError) as excinfo:
                require_optional(name)

        msg = str(excinfo.value)
        assert f"pip install {name}" in msg
        assert "gwexpy[gw]" not in msg
        assert "gwexpy[all]" not in msg

    def test_dask_array_uses_distribution_name_in_install_hint(self):
        """dask.array is an import namespace; users install the dask package."""
        with patch.dict(sys.modules, {"dask": None, "dask.array": None}):
            with pytest.raises(ImportError) as excinfo:
                require_optional("dask.array")

        msg = str(excinfo.value)
        assert "pip install dask" in msg
        assert "pip install dask.array" not in msg

    def test_returns_already_imported_module(self):
        """Calling twice returns the same module object."""
        mod1 = require_optional("scipy")
        mod2 = require_optional("scipy")
        assert mod1 is mod2
