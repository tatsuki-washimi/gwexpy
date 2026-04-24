import pytest


def test_cache_reexports_gwpy():
    gwexpy_cache = pytest.importorskip(
        "gwexpy.timeseries.io.cache", exc_type=ImportError
    )
    gwpy_cache = pytest.importorskip("gwpy.timeseries.io.cache", exc_type=ImportError)

    exported = getattr(gwpy_cache, "__all__", [])
    if not exported:
        gwpy_attrs = [a for a in dir(gwpy_cache) if not a.startswith("_")]
        if not gwpy_attrs:
            pytest.skip("gwpy.timeseries.io.cache has no public exports")
        exported = gwpy_attrs
    for name in exported:
        assert getattr(gwexpy_cache, name) is getattr(gwpy_cache, name)
