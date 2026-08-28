"""Public post-release claims; deliberately executed only by the harness."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GWEXPY_POST_RELEASE_QUALIFICATION") != "1",
    reason="post-release qualification is opt-in",
)


def test_timeseries_t0_ns_is_exact_through_constructor_copy_slice_and_pickle() -> None:
    """Exact epoch is a passing public lifecycle contract."""
    from gwexpy.timeseries import TimeSeries

    t0_ns = 1234567890123456789
    series = TimeSeries(np.arange(8), t0_ns=t0_ns, sample_rate=1)
    expected = (t0_ns, t0_ns, t0_ns + 1_000_000_000, t0_ns)
    values = (series, series.copy(), series[1:], pickle.loads(pickle.dumps(series)))
    assert tuple(value.t0_gps_ns for value in values) == expected


def test_timeseries_hdf5_roundtrip_retains_exact_t0_gps_ns(tmp_path: Path) -> None:
    """Published HDF5 must restore the integer epoch, not a float approximation."""
    from gwexpy.timeseries import TimeSeries

    t0_ns = 1234567890123456789
    series = TimeSeries(np.arange(8), t0_ns=t0_ns, sample_rate=1)
    filename = tmp_path / "epoch.hdf5"
    series.write(filename, format="hdf5", path="series")
    recovered = TimeSeries.read(filename, format="hdf5", path="series")
    assert recovered.t0_gps_ns == t0_ns


def _clean_python(code: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-P", "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_plain_import_is_lazy() -> None:
    _clean_python('''
import importlib
import sys
import gwexpy
assert importlib.import_module("gwexpy._bootstrap")._bootstrapped is False
assert "gwexpy.timeseries.io" not in sys.modules
''')


def test_constructors_only_bootstrap_does_not_register_io() -> None:
    _clean_python('''
import sys
import gwexpy
gwexpy.register_all(include_io=False)
from gwexpy.interop._registry import ConverterRegistry
assert ConverterRegistry.has_constructor("TimeSeries")
assert "gwexpy.timeseries.io" not in sys.modules
''')


def test_full_bootstrap_promotes_io_and_is_idempotent() -> None:
    _clean_python('''
import sys
import gwexpy
gwexpy.register_all()
assert "gwexpy.timeseries.io" in sys.modules
assert "gwexpy.frequencyseries.io" in sys.modules
before = tuple(sorted(sys.modules))
gwexpy.register_all()
assert before == tuple(sorted(sys.modules))
''')
