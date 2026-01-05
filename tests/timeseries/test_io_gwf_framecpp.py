import pytest


def test_gwf_framecpp_reexports_gwpy():
    gwexpy_mod = pytest.importorskip("gwexpy.timeseries.io.gwf.framecpp")
    gwpy_mod = pytest.importorskip("gwpy.timeseries.io.gwf.framecpp")

    exported = getattr(gwpy_mod, "__all__", [])
    assert exported, "gwpy.timeseries.io.gwf.framecpp.__all__ is empty"
    for name in exported:
        assert getattr(gwexpy_mod, name) is getattr(gwpy_mod, name)
