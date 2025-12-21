import pytest


def test_statevector_reexports_gwpy():
    gwexpy_mod = pytest.importorskip("gwexpy.timeseries.statevector")
    gwpy_mod = pytest.importorskip("gwpy.timeseries.statevector")

    exported = getattr(gwpy_mod, "__all__", [])
    assert exported, "gwpy.timeseries.statevector.__all__ is empty"
    for name in exported:
        assert getattr(gwexpy_mod, name) is getattr(gwpy_mod, name)
