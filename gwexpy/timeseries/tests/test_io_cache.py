import pytest


def test_cache_reexports_gwpy():
    gwexpy_cache = pytest.importorskip("gwexpy.timeseries.io.cache")
    gwpy_cache = pytest.importorskip("gwpy.timeseries.io.cache")

    exported = getattr(gwpy_cache, "__all__", [])
    if not exported:
        pytest.skip("gwpy.timeseries.io.cache.__all__ is empty")
    for name in exported:
        assert getattr(gwexpy_cache, name) is getattr(gwpy_cache, name)
