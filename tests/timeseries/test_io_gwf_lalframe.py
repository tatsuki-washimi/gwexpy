import pytest


def test_gwf_lalframe_reexports_gwpy():
    gwexpy_mod = pytest.importorskip("gwexpy.timeseries.io.gwf.lalframe")
    gwpy_mod = pytest.importorskip("gwpy.timeseries.io.gwf.lalframe")

    exported = getattr(gwpy_mod, "__all__", [])
    if not exported:
        pytest.skip("gwpy.timeseries.io.gwf.lalframe.__all__ is empty")
    for name in exported:
        assert getattr(gwexpy_mod, name) is getattr(gwpy_mod, name)
