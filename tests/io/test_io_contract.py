"""Contract tests for public direct I/O formats.

These tests compare declared contract entries against the live GWpy registry.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from astropy.io.registry.base import IORegistryError
from gwpy.io.registry import default_registry as io_registry

import gwexpy  # noqa: F401  # bootstrap registry
from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)
from gwexpy.gui.loaders.loaders import load_products
from gwexpy.histogram import Histogram, HistogramDict, HistogramList
from gwexpy.segments import DataQualityDict, DataQualityFlag, SegmentList
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.table import EventTable
from gwexpy.timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)
from gwexpy.timeseries._gwf_io import _normalize_gwf_format

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_io_contract.json"
FIXTURE_DATA = ROOT / "tests/fixtures/data/test.gwf"

CLASS_MAP = {
    "TimeSeries": TimeSeries,
    "TimeSeriesDict": TimeSeriesDict,
    "TimeSeriesList": TimeSeriesList,
    "TimeSeriesMatrix": TimeSeriesMatrix,
    "FrequencySeries": FrequencySeries,
    "FrequencySeriesDict": FrequencySeriesDict,
    "FrequencySeriesList": FrequencySeriesList,
    "Spectrogram": Spectrogram,
    "SpectrogramDict": SpectrogramDict,
    "SpectrogramList": SpectrogramList,
    "Histogram": Histogram,
    "HistogramDict": HistogramDict,
    "HistogramList": HistogramList,
    "SegmentList": SegmentList,
    "DataQualityFlag": DataQualityFlag,
    "DataQualityDict": DataQualityDict,
    "EventTable": EventTable,
}

VALID_EXTRAS = {
    "audio",
    "io",
    "netcdf4",
    "seismic",
    "zarr",
}

VALID_UNAVAILABLE_BEHAVIORS = {
    "available_in_base_install",
    "conditional_registration",
    "not_public",
    "raises_import_error",
    "raises_import_error_for_optional_metadata",
}


def _load_contracts():
    import json

    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _reader_callable(fmt: str, cls):
    try:
        return io_registry.get_reader(fmt, cls)
    except IORegistryError:
        return None


def _writer_callable(fmt: str, cls):
    try:
        return io_registry.get_writer(fmt, cls)
    except IORegistryError:
        return None


def _resolved_alias_for_contract(canonical: str, alias: str) -> str:
    if canonical == "gwf":
        return _normalize_gwf_format(alias) or alias
    return alias


def _has_gwf_backend() -> bool:
    try:
        from gwpy.io.gwf.core import get_channel_names

        return bool(get_channel_names(FIXTURE_DATA))
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        return False


def _api_classes(entry: dict, scope: str, operation: str) -> list[str]:
    return entry.get(scope, {}).get(operation, [])


def test_public_contract_schema_is_well_formed():
    data = _load_contracts()
    assert data["schema_version"] >= 2

    for entry in data["formats"]:
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["canonical"], str) and entry["canonical"]
        assert isinstance(entry.get("aliases", []), list)
        assert isinstance(entry["public_auto_identify"], bool)
        assert isinstance(entry["registry_auto_identify"], bool)
        assert isinstance(entry["trusted_only"], bool)
        assert isinstance(entry.get("metadata_requirements", []), list)
        assert isinstance(entry.get("notes", []), list)
        assert isinstance(entry["optional_dependencies"], list)
        assert isinstance(entry["extras"], list)
        assert set(entry["unavailable_behavior"]) == {"read", "write"}
        assert set(entry["required_args"]) == {"read", "write"}
        assert set(entry["direct_api"]) == {"read", "write"}

        optional_dependencies = entry["optional_dependencies"]
        extras = entry["extras"]
        assert all(
            isinstance(dependency, str) and dependency
            for dependency in optional_dependencies
        )
        assert all(extra in VALID_EXTRAS for extra in extras)
        assert len(optional_dependencies) == len(set(optional_dependencies))
        assert len(extras) == len(set(extras))

        public_read = set(_api_classes(entry, "public_api", "read"))
        public_write = set(_api_classes(entry, "public_api", "write"))
        direct_read = set(_api_classes(entry, "direct_api", "read"))
        direct_write = set(_api_classes(entry, "direct_api", "write"))
        registry_read = set(_api_classes(entry, "registry_api", "read"))
        registry_write = set(_api_classes(entry, "registry_api", "write"))

        assert public_read <= set(CLASS_MAP)
        assert public_write <= set(CLASS_MAP)
        assert direct_read <= set(CLASS_MAP)
        assert direct_write <= set(CLASS_MAP)
        assert registry_read <= set(CLASS_MAP)
        assert registry_write <= set(CLASS_MAP)
        assert public_read <= (registry_read | direct_read)
        assert public_write <= (registry_write | direct_write)

        for scope in ("required_args", "direct_api"):
            for operation in ("read", "write"):
                values = entry[scope][operation]
                assert isinstance(values, list)
                assert all(isinstance(value, str) and value for value in values)

        for operation in ("read", "write"):
            required_args = entry["required_args"][operation]
            unavailable_behavior = entry["unavailable_behavior"][operation]
            assert isinstance(required_args, list)
            assert all(isinstance(arg, str) and arg for arg in required_args)
            assert unavailable_behavior in VALID_UNAVAILABLE_BEHAVIORS
            if not entry["public_api"][operation]:
                assert unavailable_behavior == "not_public"


def test_registry_contract_is_registered_in_registry():
    data = _load_contracts()

    for entry in data["formats"]:
        canonical = entry["canonical"]
        aliases = entry.get("aliases", [])
        read_classes = _api_classes(entry, "registry_api", "read")
        write_classes = _api_classes(entry, "registry_api", "write")
        if canonical == "gwf" and not _has_gwf_backend():
            continue

        for class_name in read_classes:
            cls = CLASS_MAP[class_name]
            canonical_reader = _reader_callable(canonical, cls)
            if canonical_reader is None:
                assert entry["unavailable_behavior"]["read"] == "conditional_registration"
                assert entry["optional_dependencies"]
                continue
            for alias in aliases:
                alias_reader = _reader_callable(alias, cls)
                resolved = _resolved_alias_for_contract(canonical, alias)
                if alias_reader is None:
                    if resolved == canonical:
                        continue
                    raise AssertionError(
                        f"{canonical} read alias '{alias}' is not registered for {class_name}"
                    )
                if resolved != canonical:
                    resolved_reader = _reader_callable(resolved, cls)
                    if resolved_reader is not None:
                        canonical_reader = resolved_reader
                assert (
                    alias_reader.__name__ == canonical_reader.__name__
                ) or alias_reader is canonical_reader

        for class_name in write_classes:
            cls = CLASS_MAP[class_name]
            canonical_writer = _writer_callable(canonical, cls)
            if canonical_writer is None:
                assert entry["unavailable_behavior"]["write"] == "conditional_registration"
                assert entry["optional_dependencies"]
                continue
            for alias in aliases:
                alias_writer = _writer_callable(alias, cls)
                resolved = _resolved_alias_for_contract(canonical, alias)
                if alias_writer is None:
                    if resolved == canonical:
                        continue
                    raise AssertionError(
                        f"{canonical} write alias '{alias}' is not registered for {class_name}"
                    )
                if resolved != canonical:
                    resolved_writer = _writer_callable(resolved, cls)
                    if resolved_writer is not None:
                        canonical_writer = resolved_writer
                assert (
                    alias_writer.__name__ == canonical_writer.__name__
                ) or alias_writer is canonical_writer


def test_current_public_boundary_decisions_are_recorded():
    data = _load_contracts()
    entries = {entry["canonical"]: entry for entry in data["formats"]}

    assert _api_classes(entries["gwf"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert _api_classes(entries["hdf.ndscope"], "public_api", "read") == [
        "TimeSeriesDict"
    ]
    assert _api_classes(entries["hdf.ndscope"], "public_api", "write") == [
        "TimeSeriesDict"
    ]
    assert _api_classes(entries["csv"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert _api_classes(entries["csv"], "public_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert _api_classes(entries["csv"], "direct_api", "read") == [
        "TimeSeriesDict",
        "TimeSeriesList",
        "FrequencySeriesDict",
        "FrequencySeriesList",
    ]
    assert _api_classes(entries["csv"], "direct_api", "write") == [
        "TimeSeriesDict",
        "TimeSeriesList",
        "FrequencySeriesDict",
        "FrequencySeriesList",
    ]
    assert _api_classes(entries["csv"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
        "FrequencySeries",
        "Spectrogram",
        "EventTable",
    ]
    assert _api_classes(entries["csv"], "registry_api", "write") == [
        "TimeSeries",
        "FrequencySeries",
        "Spectrogram",
        "EventTable",
    ]
    assert entries["csv"]["public_auto_identify"] is False
    assert entries["csv"]["registry_auto_identify"] is False
    assert entries["csv"]["metadata_requirements"]
    assert _api_classes(entries["txt"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert _api_classes(entries["txt"], "public_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert _api_classes(entries["txt"], "direct_api", "read") == [
        "TimeSeriesDict",
        "TimeSeriesList",
        "FrequencySeriesDict",
        "FrequencySeriesList",
    ]
    assert _api_classes(entries["txt"], "direct_api", "write") == [
        "TimeSeriesDict",
        "TimeSeriesList",
        "FrequencySeriesDict",
        "FrequencySeriesList",
    ]
    assert _api_classes(entries["txt"], "registry_api", "read") == [
        "TimeSeries",
        "FrequencySeries",
    ]
    assert _api_classes(entries["txt"], "registry_api", "write") == [
        "TimeSeries",
        "FrequencySeries",
    ]
    assert entries["txt"]["public_auto_identify"] is False
    assert entries["txt"]["registry_auto_identify"] is False
    assert entries["txt"]["metadata_requirements"]
    assert _api_classes(entries["pickle"], "public_api", "read") == []
    assert _api_classes(entries["pickle"], "public_api", "write") == []
    assert _api_classes(entries["pickle"], "direct_api", "read") == []
    assert _api_classes(entries["pickle"], "direct_api", "write") == []
    assert _api_classes(entries["pickle"], "registry_api", "read") == []
    assert _api_classes(entries["pickle"], "registry_api", "write") == []
    assert entries["pickle"]["trusted_only"] is True
    assert _api_classes(entries["sdb"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert _api_classes(entries["sdb"], "public_api", "write") == []
    assert _api_classes(entries["sdb"], "direct_api", "read") == []
    assert _api_classes(entries["sdb"], "direct_api", "write") == []
    assert _api_classes(entries["sdb"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["sdb"], "registry_api", "write") == []
    assert entries["sdb"]["aliases"] == ["sqlite", "sqlite3"]
    assert entries["sdb"]["public_auto_identify"] is True
    assert entries["sdb"]["registry_auto_identify"] is True
    assert entries["sdb"]["metadata_requirements"]
    assert _api_classes(entries["wav"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert _api_classes(entries["wav"], "public_api", "write") == ["TimeSeries"]
    assert _api_classes(entries["wav"], "direct_api", "read") == []
    assert _api_classes(entries["wav"], "direct_api", "write") == []
    assert _api_classes(entries["wav"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["wav"], "registry_api", "write") == ["TimeSeries"]
    assert entries["wav"]["public_auto_identify"] is True
    assert entries["wav"]["registry_auto_identify"] is True
    assert entries["wav"]["metadata_requirements"]
    for fmt in ("flac", "ogg", "mp3", "m4a"):
        assert _api_classes(entries[fmt], "public_api", "read") == [
            "TimeSeries",
            "TimeSeriesDict",
        ]
        assert _api_classes(entries[fmt], "public_api", "write") == [
            "TimeSeries",
            "TimeSeriesDict",
        ]
        assert _api_classes(entries[fmt], "registry_api", "read") == [
            "TimeSeries",
            "TimeSeriesDict",
            "TimeSeriesMatrix",
        ]
        assert _api_classes(entries[fmt], "registry_api", "write") == [
            "TimeSeries",
            "TimeSeriesDict",
            "TimeSeriesMatrix",
        ]
        assert entries[fmt]["public_auto_identify"] is True
        assert entries[fmt]["registry_auto_identify"] is True
        assert entries[fmt]["metadata_requirements"]
    assert _api_classes(entries["gbd"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["gbd"], "public_api", "write") == []
    assert _api_classes(entries["gbd"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["gbd"], "registry_api", "write") == []
    assert entries["gbd"]["required_args"]["read"] == ["timezone"]
    assert entries["gbd"]["public_auto_identify"] is True
    assert entries["gbd"]["registry_auto_identify"] is True
    assert entries["gbd"]["metadata_requirements"]
    assert _api_classes(entries["tdms"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["tdms"], "public_api", "write") == []
    assert _api_classes(entries["tdms"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["tdms"], "registry_api", "write") == []
    assert entries["tdms"]["required_args"]["read"] == []
    assert entries["tdms"]["public_auto_identify"] is True
    assert entries["tdms"]["registry_auto_identify"] is True
    assert entries["tdms"]["metadata_requirements"]
    assert _api_classes(entries["mseed"], "public_api", "read") == ["TimeSeriesDict"]
    assert _api_classes(entries["mseed"], "public_api", "write") == ["TimeSeriesDict"]
    assert _api_classes(entries["mseed"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["mseed"], "registry_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert entries["mseed"]["aliases"] == ["miniseed"]
    assert entries["mseed"]["metadata_requirements"]
    assert _api_classes(entries["sac"], "public_api", "read") == ["TimeSeriesDict"]
    assert _api_classes(entries["sac"], "public_api", "write") == ["TimeSeriesDict"]
    assert _api_classes(entries["sac"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["sac"], "registry_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert entries["sac"]["metadata_requirements"]
    assert _api_classes(entries["gse2"], "public_api", "read") == ["TimeSeriesDict"]
    assert _api_classes(entries["gse2"], "public_api", "write") == ["TimeSeriesDict"]
    assert _api_classes(entries["gse2"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["gse2"], "registry_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert entries["gse2"]["metadata_requirements"]
    assert _api_classes(entries["knet"], "public_api", "read") == ["TimeSeriesDict"]
    assert _api_classes(entries["knet"], "public_api", "write") == []
    assert _api_classes(entries["knet"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["knet"], "registry_api", "write") == []
    assert entries["knet"]["metadata_requirements"]
    assert _api_classes(entries["win"], "public_api", "read") == ["TimeSeriesDict"]
    assert _api_classes(entries["win"], "public_api", "write") == []
    assert _api_classes(entries["win"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["win"], "registry_api", "write") == []
    assert entries["win"]["aliases"] == ["win32"]
    assert entries["win"]["optional_dependencies"] == ["obspy"]
    assert entries["win"]["extras"] == ["seismic"]
    assert entries["win"]["unavailable_behavior"]["read"] == "conditional_registration"
    assert entries["win"]["metadata_requirements"]
    assert _api_classes(entries["ats"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
    ]
    assert _api_classes(entries["ats"], "public_api", "write") == []
    assert _api_classes(entries["ats"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["ats"], "registry_api", "write") == []
    assert entries["ats"]["metadata_requirements"]
    assert _api_classes(entries["ats.mth5"], "public_api", "read") == ["TimeSeries"]
    assert _api_classes(entries["ats.mth5"], "public_api", "write") == []
    assert _api_classes(entries["ats.mth5"], "registry_api", "read") == ["TimeSeries"]
    assert _api_classes(entries["ats.mth5"], "registry_api", "write") == []
    assert entries["ats.mth5"]["public_auto_identify"] is False
    assert entries["ats.mth5"]["registry_auto_identify"] is False
    assert entries["ats.mth5"]["optional_dependencies"] == ["mth5"]
    assert entries["ats.mth5"]["extras"] == ["seismic"]
    assert (
        entries["ats.mth5"]["unavailable_behavior"]["read"] == "raises_import_error"
    )
    assert entries["ats.mth5"]["metadata_requirements"]
    assert _api_classes(entries["root"], "public_api", "read") == ["EventTable"]
    assert _api_classes(entries["root"], "public_api", "write") == ["EventTable"]
    assert _api_classes(entries["root"], "direct_api", "read") == []
    assert _api_classes(entries["root"], "direct_api", "write") == [
        "TimeSeriesDict",
        "TimeSeriesList",
        "FrequencySeriesDict",
        "FrequencySeriesList",
        "SpectrogramDict",
        "SpectrogramList",
        "HistogramDict",
        "HistogramList",
    ]
    assert _api_classes(entries["root"], "registry_api", "read") == ["EventTable"]
    assert _api_classes(entries["root"], "registry_api", "write") == ["EventTable"]
    assert entries["root"]["public_auto_identify"] is False
    assert entries["root"]["registry_auto_identify"] is False
    assert _api_classes(entries["hdf5"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesList",
        "TimeSeriesMatrix",
        "FrequencySeries",
        "FrequencySeriesDict",
        "FrequencySeriesList",
        "Spectrogram",
        "SpectrogramDict",
        "SpectrogramList",
        "Histogram",
        "HistogramDict",
        "HistogramList",
        "EventTable",
    ]
    assert _api_classes(entries["hdf5"], "public_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesList",
        "TimeSeriesMatrix",
        "FrequencySeries",
        "FrequencySeriesDict",
        "FrequencySeriesList",
        "Spectrogram",
        "SpectrogramDict",
        "SpectrogramList",
        "Histogram",
        "HistogramDict",
        "HistogramList",
        "EventTable",
    ]
    assert _api_classes(entries["hdf5"], "registry_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "FrequencySeries",
        "Spectrogram",
        "SegmentList",
        "DataQualityFlag",
        "DataQualityDict",
        "EventTable",
    ]
    assert _api_classes(entries["hdf5"], "registry_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
        "FrequencySeries",
        "Spectrogram",
        "SegmentList",
        "DataQualityFlag",
        "DataQualityDict",
        "EventTable",
    ]
    assert _api_classes(entries["hdf5"], "direct_api", "read") == [
        "TimeSeriesDict",
        "TimeSeriesList",
        "TimeSeriesMatrix",
        "FrequencySeriesDict",
        "FrequencySeriesList",
        "SpectrogramDict",
        "SpectrogramList",
        "Histogram",
        "HistogramDict",
        "HistogramList",
    ]
    assert _api_classes(entries["hdf5"], "direct_api", "write") == [
        "TimeSeriesDict",
        "TimeSeriesList",
        "TimeSeriesMatrix",
        "FrequencySeriesDict",
        "FrequencySeriesList",
        "SpectrogramDict",
        "SpectrogramList",
        "Histogram",
        "HistogramDict",
        "HistogramList",
    ]
    assert entries["hdf5"]["public_auto_identify"] is False
    assert entries["hdf5"]["registry_auto_identify"] is False
    assert entries["hdf5"]["metadata_requirements"]
    assert _api_classes(entries["xml.diaggui"], "public_api", "read") == [
        "TimeSeriesDict"
    ]
    assert entries["xml.diaggui"]["public_auto_identify"] is False
    assert entries["xml.diaggui"]["registry_auto_identify"] is True
    assert entries["xml.diaggui"]["required_args"]["read"] == ["products"]
    assert _api_classes(entries["nc"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["nc"], "public_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert entries["nc"]["aliases"] == ["netcdf4"]
    assert _api_classes(entries["zarr"], "public_api", "read") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert _api_classes(entries["zarr"], "public_api", "write") == [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesMatrix",
    ]
    assert entries["zarr"]["public_auto_identify"] is True
    assert entries["zarr"]["registry_auto_identify"] is True
    assert entries["zarr"]["metadata_requirements"]


def test_gwf_load_products_contract():
    if not FIXTURE_DATA.exists():
        pytest.skip("test.gwf fixture not found")
    if not _has_gwf_backend():
        pytest.skip("gwf backend not available")

    products = load_products(str(FIXTURE_DATA))

    assert "TS" in products
    products_keys = set(products["TS"].keys())
    tsd = TimeSeriesDict.read(str(FIXTURE_DATA))
    assert products_keys == set(tsd.keys())
