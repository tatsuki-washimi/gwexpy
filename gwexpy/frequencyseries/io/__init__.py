"""Registrations for `FrequencySeries` readers."""

from __future__ import annotations

from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.io.registry import default_registry as io_registry
from gwpy.io.registry import identify_factory

from ..frequencyseries import FrequencySeries
from . import (
    dttxml,  # noqa: F401
    stubs,  # noqa: F401
)

_native_csv_reader = io_registry.get_reader("csv", GwpyFrequencySeries)
_native_csv_writer = io_registry.get_writer("csv", GwpyFrequencySeries)
io_registry.register_reader("csv", FrequencySeries, _native_csv_reader, force=True)
io_registry.register_writer("csv", FrequencySeries, _native_csv_writer, force=True)
io_registry.register_identifier(
    "csv", FrequencySeries, identify_factory("csv"), force=True
)

__all__: list[str] = []
