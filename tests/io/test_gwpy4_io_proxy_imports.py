from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "gwexpy.io.hdf5",
        "gwexpy.types.io.ascii",
        "gwexpy.types.io.hdf5",
        "gwexpy.types.io.ligolw",
        "gwexpy.timeseries.io.ascii",
        "gwexpy.timeseries.io.core",
        "gwexpy.timeseries.io.gwf",
        "gwexpy.timeseries.io.hdf5",
        "gwexpy.timeseries.io.nds2",
    ],
)
def test_public_io_proxy_module_imports_under_gwpy4(module_name: str) -> None:
    module = importlib.import_module(module_name)

    for public_name in module.__all__:
        assert hasattr(module, public_name), f"{module_name}.{public_name} is missing"
