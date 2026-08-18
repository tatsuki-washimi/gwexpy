"""Import and export contracts for GWpy 4 compatibility shims."""

from __future__ import annotations

import importlib

import pytest

AFFECTED_SHIMS = (
    "gwexpy.table.filter",
    "gwexpy.table.table",
    "gwexpy.timeseries.core",
    "gwexpy.timeseries.io.gwf.framecpp",
    "gwexpy.timeseries.io.gwf.framel",
    "gwexpy.utils.lal",
    "gwexpy.utils.misc",
    "gwexpy.utils.shell",
    "gwexpy.utils.sphinx",
    "gwexpy.utils.sphinx.ex2rst",
    "gwexpy.utils.sphinx.zenodo",
)

UPSTREAM_EXPORT_SHIMS = (
    ("gwexpy.table.filter", "gwpy.table.filter"),
    ("gwexpy.table.table", "gwpy.table.table"),
    ("gwexpy.timeseries.core", "gwpy.timeseries.core"),
    ("gwexpy.timeseries.io.gwf.framel", "gwpy.timeseries.io.gwf.framel"),
    ("gwexpy.utils.lal", "gwpy.utils.lal"),
    ("gwexpy.utils.misc", "gwpy.utils.misc"),
)

STALE_EXPORTS = {
    "gwexpy.table.filter": {"OrderedDict"},
    "gwexpy.table.table": {
        "attrgetter",
        "gps_types",
        "inherit_io_registrations",
        "io_read_multi",
        "registry",
        "vstack",
    },
    "gwexpy.timeseries.core": {
        "ChannelList",
        "OrderedDict",
        "ceil",
        "gps_types",
        "io_registry",
    },
    "gwexpy.timeseries.io.gwf.framel": {"FRAME_LIBRARY"},
    "gwexpy.utils.lal": {"LAL_UNIT_INDEX"},
    "gwexpy.utils.misc": {"OrderedDict", "nullcontext"},
}


@pytest.mark.parametrize("module_name", AFFECTED_SHIMS)
def test_affected_shims_import(module_name: str) -> None:
    """Every legacy compatibility module remains safe to import."""
    module = importlib.import_module(module_name)
    assert isinstance(module.__all__, list)
    assert getattr(module, "__test__", False) is False


@pytest.mark.parametrize("shim_name, upstream_name", UPSTREAM_EXPORT_SHIMS)
def test_shim_exports_match_available_upstream_symbols(
    shim_name: str,
    upstream_name: str,
) -> None:
    """A shim exports only symbols that still exist in GWpy 4."""
    shim = importlib.import_module(shim_name)
    try:
        upstream = importlib.import_module(upstream_name)
    except ImportError:
        if shim_name != "gwexpy.timeseries.io.gwf.framel":
            raise
        assert shim.__all__ == []
        return

    assert set(shim.__all__).isdisjoint(STALE_EXPORTS.get(shim_name, set()))
    for name in shim.__all__:
        assert hasattr(upstream, name), f"{shim_name}.{name} is stale"
        assert getattr(shim, name) is getattr(upstream, name)


@pytest.mark.parametrize(
    "shim_name, upstream_name, symbol",
    (
        (
            "gwexpy.timeseries.io.gwf.framecpp",
            "gwpy.timeseries.io.gwf.framecpp",
            "read",
        ),
        (
            "gwexpy.timeseries.io.gwf.framel",
            "gwpy.timeseries.io.gwf.framel",
            "read",
        ),
    ),
)
def test_optional_gwf_shims_keep_backend_optional(
    shim_name: str,
    upstream_name: str,
    symbol: str,
) -> None:
    """Missing GWF backends fail when used, not while importing the shim."""
    shim = importlib.import_module(shim_name)
    try:
        upstream = importlib.import_module(upstream_name)
    except ImportError:
        assert shim.__all__ == []
        with pytest.raises(ImportError, match="unavailable"):
            getattr(shim, symbol)
    else:
        for name in shim.__all__:
            assert getattr(shim, name) is getattr(upstream, name)


@pytest.mark.parametrize(
    "shim_name, symbol",
    (
        ("gwexpy.utils.shell", "which"),
        ("gwexpy.utils.sphinx.ex2rst", "ex2rst"),
        ("gwexpy.utils.sphinx.zenodo", "main"),
    ),
)
def test_removed_upstream_module_shims_fail_only_when_used(
    shim_name: str,
    symbol: str,
) -> None:
    """Removed GWpy modules import safely and report a useful failure on use."""
    shim = importlib.import_module(shim_name)
    assert shim.__all__ == []
    with pytest.raises(ImportError, match="unavailable"):
        getattr(shim, symbol)
