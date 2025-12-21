import pytest


def test_losc_reexports_gwpy():
    gwexpy_mod = pytest.importorskip("gwexpy.timeseries.io.losc")
    gwpy_mod = pytest.importorskip("gwpy.timeseries.io.losc")

    exported = getattr(gwpy_mod, "__all__", [])
    if not exported:
        pytest.skip("gwpy.timeseries.io.losc.__all__ is empty")
    for name in exported:
        assert getattr(gwexpy_mod, name) is getattr(gwpy_mod, name)
