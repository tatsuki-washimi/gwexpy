"""GWpy 4 compatibility contracts for deliberately curated proxies."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXPECTED_EXPORTS = {
    "gwexpy.table.filter": (
        "DELIM_REGEX", "OPERATORS", "OPERATORS_INV", "QUOTE_REGEX",
        "filter_table", "generate_tokens", "is_filter_tuple",
        "parse_column_filter", "parse_column_filters", "parse_operator",
    ),
    "gwexpy.table.table": (
        "DEFAULT_GWOSC_URL", "TIME_LIKE_COLUMN_NAMES", "EventTable", "Table",
        "filter_table", "parse_operator",
    ),
    "gwexpy.timeseries.core": (
        "GWOSC_DEFAULT_HOST", "Channel", "ChannelList", "LIGOTimeGPS",
        "SegmentList", "Series", "Time", "TimeSeriesBase",
        "TimeSeriesBaseDict", "TimeSeriesBaseList", "to_gps", "units",
    ),
    "gwexpy.utils.lal": (
        "LAL_DETECTORS", "LAL_NUMPY_FROM_TYPE_STR", "LAL_TYPE_FROM_NUMPY",
        "LAL_TYPE_FROM_STR", "LAL_TYPE_REGEX", "LAL_TYPE_STR",
        "LAL_TYPE_STR_FROM_NUMPY", "find_typed_function", "from_lal_type",
        "from_lal_unit", "gwpy_units", "to_gps", "to_lal_ligotimegps",
        "to_lal_type_str", "to_lal_unit",
    ),
    "gwexpy.utils.misc": (
        "if_not_none", "property_alias", "round_to_power", "unique",
    ),
}

REMOVED_MODULES = (
    "gwexpy.utils.shell",
    "gwexpy.utils.sphinx",
    "gwexpy.utils.sphinx.ex2rst",
    "gwexpy.utils.sphinx.zenodo",
)

REMOVED_NAMES = {
    "gwexpy.table.filter": ("OrderedDict", "StringIO", "numpy", "operator", "re", "token"),
    "gwexpy.table.table": ("attrgetter", "ceil", "gps_types", "inherit_io_registrations", "io_read_multi", "registry", "vstack", "wraps"),
    "gwexpy.timeseries.core": ("OrderedDict", "as_series_dict_class", "ceil", "gps_types", "io_registry", "property_alias"),
    "gwexpy.utils.lal": ("LAL_UNIT_INDEX",),
    "gwexpy.utils.misc": ("OrderedDict", "nullcontext"),
}

FRAME_EXPORTS = (
    "FRAME_LIBRARY", "Segment", "TimeSeries", "file_list", "file_path",
    "framel", "read", "warnings", "write",
)
GWF_EXPORTS = (
    "BACKENDS", "backend", "channel_exists", "core", "data_segments",
    "get_backend", "get_backend_function", "get_channel_names",
    "get_channel_type", "identify_gwf", "import_backend",
    "iter_channel_names", "num_channels",
)


def _import_name(module_name: str, name: str) -> None:
    exec(f"from {module_name} import {name}", {})


@pytest.mark.parametrize("module_name, expected", EXPECTED_EXPORTS.items())
def test_curated_proxy_exports_are_exact(module_name: str, expected: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)
    assert tuple(module.__all__) == expected


@pytest.mark.parametrize(
    ("module_name", "name"),
    [(module, name) for module, names in REMOVED_NAMES.items() for name in names],
)
def test_removed_proxy_leaks_raise_import_error(module_name: str, name: str) -> None:
    with pytest.raises(ImportError):
        _import_name(module_name, name)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_deleted_proxy_source_paths_are_absent(module_name: str) -> None:
    relative = Path(*module_name.split("."))
    assert not relative.with_suffix(".py").exists()
    assert not (relative / "__init__.py").exists()


def test_timeseries_core_uses_maintained_owners() -> None:
    from astropy import units
    from gwosc.api import DEFAULT_URL
    from gwpy.detector.channel import Channel, ChannelList
    from gwpy.segments import SegmentList
    from gwpy.time import LIGOTimeGPS, Time, to_gps
    from gwpy.timeseries.core import TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList
    from gwpy.types import Series
    from gwexpy.timeseries import core

    assert core.Channel is Channel
    assert core.ChannelList is ChannelList
    assert core.LIGOTimeGPS is LIGOTimeGPS
    assert core.SegmentList is SegmentList
    assert core.Series is Series
    assert core.Time is Time
    assert core.TimeSeriesBase is TimeSeriesBase
    assert core.TimeSeriesBaseDict is TimeSeriesBaseDict
    assert core.TimeSeriesBaseList is TimeSeriesBaseList
    assert core.to_gps is to_gps
    assert core.units is units
    assert core.GWOSC_DEFAULT_HOST == DEFAULT_URL


def test_table_filter_uses_maintained_owner() -> None:
    from gwpy.table import filter as owner
    from gwexpy.table import filter as proxy

    for name in EXPECTED_EXPORTS["gwexpy.table.filter"]:
        assert getattr(proxy, name) is getattr(owner, name)


def test_table_table_uses_maintained_owners() -> None:
    from astropy.table import Table
    from gwosc.api import DEFAULT_URL
    from gwpy.table.filter import filter_table, parse_operator
    from gwpy.table.table import EventTable, TIME_LIKE_COLUMN_NAMES
    from gwexpy.table import table as proxy

    assert proxy.DEFAULT_GWOSC_URL == DEFAULT_URL
    assert proxy.TIME_LIKE_COLUMN_NAMES is TIME_LIKE_COLUMN_NAMES
    assert proxy.EventTable is EventTable
    assert proxy.Table is Table
    assert proxy.filter_table is filter_table
    assert proxy.parse_operator is parse_operator


def test_lal_proxy_uses_maintained_owners() -> None:
    from gwpy.detector import units as gwpy_units
    from gwpy.time import to_gps
    from gwpy.utils import lal as owner
    from gwexpy.utils import lal as proxy

    for name in EXPECTED_EXPORTS["gwexpy.utils.lal"]:
        expected = gwpy_units if name == "gwpy_units" else to_gps if name == "to_gps" else getattr(owner, name)
        assert getattr(proxy, name) is expected


def test_misc_proxy_uses_maintained_owner() -> None:
    from gwpy.utils import misc as owner
    from gwexpy.utils import misc as proxy

    for name in EXPECTED_EXPORTS["gwexpy.utils.misc"]:
        assert getattr(proxy, name) is getattr(owner, name)


def test_generic_gwf_boundary_is_frozen() -> None:
    gwf = importlib.import_module("gwexpy.io.gwf")

    assert tuple(gwf.__all__) == GWF_EXPORTS


def test_generic_gwf_boundary_imports_the_gwexpy_wrapper() -> None:
    gwf = importlib.import_module("gwexpy.io.gwf")

    assert gwf.__name__ == "gwexpy.io.gwf"
    assert tuple(gwf.__all__) == GWF_EXPORTS


def test_framel_proxy_does_not_load_backend_for_static_surface(monkeypatch) -> None:
    sys.modules.pop("gwexpy.timeseries.io.gwf.framel", None)
    original_import = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name == "gwpy.timeseries.io.gwf.framel":
            raise AssertionError("FrameL backend loaded eagerly")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    proxy = importlib.import_module("gwexpy.timeseries.io.gwf.framel")
    assert tuple(proxy.__all__) == FRAME_EXPORTS
    assert dir(proxy) == sorted(FRAME_EXPORTS)


def test_framel_proxy_preserves_missing_backend_error(monkeypatch) -> None:
    proxy = importlib.import_module("gwexpy.timeseries.io.gwf.framel")
    monkeypatch.setattr(proxy, "_module", None)

    def missing_backend(name: str):
        raise ModuleNotFoundError("No module named 'framel'", name="framel")

    monkeypatch.setattr(proxy, "import_module", missing_backend)
    with pytest.raises(ModuleNotFoundError, match="No module named 'framel'"):
        proxy.read
    assert dir(proxy) == sorted(FRAME_EXPORTS)


def test_framel_proxy_forwards_backend_objects_when_available() -> None:
    pytest.importorskip("framel")
    proxy = importlib.reload(importlib.import_module("gwexpy.timeseries.io.gwf.framel"))
    owner = importlib.import_module("gwpy.timeseries.io.gwf.framel")
    assert proxy.read is owner.read
    assert proxy.TimeSeries is owner.TimeSeries
    assert dir(proxy) == sorted(FRAME_EXPORTS)


def test_required_framel_subprocess_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    child_env = os.environ | {
        "GWEXPY_REQUIRE_GWF_FRAMEL": "1",
        "PYTHONPATH": str(root),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/timeseries/test_io_gwf_timeseriesdict.py::test_read_gwf_timeseries_with_single_channel_by_format_gwf",
        ],
        cwd=root,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    try:
        from gwpy.io.gwf.core import get_channel_names

        backend_available = bool(get_channel_names(root / "tests/fixtures/data/test.gwf", backend="framel"))
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        backend_available = False

    if backend_available:
        assert result.returncode == 0, output
    else:
        assert result.returncode == 1, output
        assert "GWEXPY_REQUIRE_GWF_FRAMEL=1 requires the framel GWF backend" in output
