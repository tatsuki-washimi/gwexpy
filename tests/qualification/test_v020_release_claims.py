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


def test_timeseries_t0_ns_is_exact_through_copy_slice_pickle_and_hdf5(tmp_path: Path) -> None:
    """A published distribution must retain nanosecond epoch precision."""
    from gwexpy.timeseries import TimeSeries

    t0_ns = 1234567890123456789
    series = TimeSeries(np.arange(8), t0_ns=t0_ns, sample_rate=1)
    filename = tmp_path / "epoch.hdf5"
    series.write(filename, format="hdf5", path="series")
    recovered = TimeSeries.read(filename, format="hdf5", path="series")
    for value in (series, series.copy(), series[1:], pickle.loads(pickle.dumps(series)), recovered):
        assert value.t0.value * 1_000_000_000 == t0_ns


def test_bootstrap_is_lazy_promotable_and_idempotent() -> None:
    """Top-level import must not pre-register I/O; explicit bootstrap promotes it."""
    code = '''
import importlib
import sys
import gwexpy
assert "gwexpy.timeseries.io" not in sys.modules
bootstrap = importlib.import_module("gwexpy._bootstrap")
assert bootstrap._bootstrapped is False
gwexpy.register_all(include_io=False)
assert bootstrap._bootstrapped is False
assert "gwexpy.timeseries.io" not in sys.modules
gwexpy.register_all()
assert bootstrap._bootstrapped is True
assert "gwexpy.timeseries.io" in sys.modules
before = tuple(sorted(sys.modules))
gwexpy.register_all()
assert before == tuple(sorted(sys.modules))
'''
    completed = subprocess.run(
        [sys.executable, "-P", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
