from __future__ import annotations

import importlib
import pkgutil

import gwexpy.table.io as table_io


def test_table_io_proxy_modules_import_under_gwpy4() -> None:
    for module_info in pkgutil.iter_modules(table_io.__path__, table_io.__name__ + "."):
        module = importlib.import_module(module_info.name)

        for public_name in module.__all__:
            assert hasattr(module, public_name), (
                f"{module_info.name}.{public_name} is missing"
            )


def test_table_io_package_public_submodules_import_under_gwpy4() -> None:
    for public_name in table_io.__all__:
        module_name = f"{table_io.__name__}.{public_name}"
        module = importlib.import_module(module_name)

        for exported_name in module.__all__:
            assert hasattr(module, exported_name), (
                f"{module_name}.{exported_name} is missing"
            )
