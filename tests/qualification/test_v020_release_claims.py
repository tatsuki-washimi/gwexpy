"""Public post-release claims; deliberately executed only by the harness."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
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
    assert (
        recovered.t0_gps_ns,
        getattr(recovered, "_gwex_t0_gps_ns", None),
    ) == (t0_ns, t0_ns)


def test_timeseries_mne_roundtrip_retains_exact_t0_gps_ns() -> None:
    """The latest provisioned MNE must preserve the public integer epoch."""
    import mne

    from gwexpy.timeseries import TimeSeries

    assert mne.__version__
    t0_ns = 1_234_567_890_123_456_789
    original = TimeSeries(
        np.arange(8, dtype=float),
        t0_ns=t0_ns,
        sample_rate=4,
        unit="V",
        name="X1:QUALIFICATION",
        channel="X1:QUALIFICATION",
    )

    raw = original.to_mne()
    recovered = TimeSeries.from_mne(raw, "X1:QUALIFICATION", unit=original.unit)

    assert recovered.t0_gps_ns == t0_ns
    assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns
    np.testing.assert_array_equal(recovered.value, original.value)
    assert recovered.dt == original.dt
    assert recovered.unit == original.unit


def _clean_python(code: str) -> None:
    environment = {
        key: os.environ[key]
        for key in (
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "LANG",
            "LC_ALL",
        )
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="gwexpy-clean-python-") as temporary:
        runtime = Path(temporary)
        environment.update(
            {
                "HOME": str(runtime),
                "USERPROFILE": str(runtime),
                "TEMP": str(runtime),
                "TMP": str(runtime),
                "TMPDIR": str(runtime),
                "XDG_CACHE_HOME": str(runtime / "cache"),
                "XDG_CONFIG_HOME": str(runtime / "config"),
            }
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code],
            check=False,
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_plain_import_is_lazy() -> None:
    _clean_python("""
import sys
import gwexpy
from gwexpy import _bootstrap
from gwexpy.interop._registry import ConverterRegistry
observed = (
    _bootstrap._bootstrapped,
    len(ConverterRegistry._constructors),
    "gwexpy.timeseries.io" in sys.modules,
    "gwexpy.frequencyseries.io" in sys.modules,
)
assert observed == (False, 0, False, False), (
    "plain import eagerly changed (bootstrapped, constructors, timeseries_io, "
    f"frequencyseries_io): {observed}"
)
""")


def test_constructors_only_bootstrap_does_not_register_io() -> None:
    _clean_python("""
import sys
import gwexpy
gwexpy.register_all(include_io=False)
from gwexpy.interop._registry import ConverterRegistry
assert ConverterRegistry.has_constructor("TimeSeries")
assert ConverterRegistry.has_constructor("Spectrogram")
assert "gwexpy.timeseries.io" not in sys.modules, (
    "constructor-only bootstrap loaded TimeSeries I/O"
)
assert "gwexpy.frequencyseries.io" not in sys.modules, (
    "constructor-only bootstrap loaded FrequencySeries I/O"
)
""")


def test_full_bootstrap_registers_constructors_and_io() -> None:
    _clean_python("""
import sys
import gwexpy
gwexpy.register_all()
from gwexpy.interop._registry import ConverterRegistry
assert ConverterRegistry.has_constructor("TimeSeries")
assert ConverterRegistry.has_constructor("Spectrogram")
assert "gwexpy.timeseries.io" in sys.modules
assert "gwexpy.frequencyseries.io" in sys.modules
""")


def test_public_io_entry_points_register_handlers_on_demand() -> None:
    _clean_python("""
import pathlib
import sys
import tempfile
import numpy as np
import gwexpy
from gwexpy import _bootstrap
assert _bootstrap._bootstrapped is False, "plain import marked bootstrap complete"
assert "gwexpy.timeseries.io" not in sys.modules, "plain import loaded TimeSeries I/O"
with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory) / "series.hdf5"
    original = gwexpy.TimeSeries(np.arange(4), sample_rate=1)
    original.write(path, format="hdf5", path="series")
    restored = gwexpy.TimeSeries.read(path, format="hdf5", path="series")
    assert restored.shape == original.shape
assert _bootstrap._bootstrapped is False
assert "gwexpy.timeseries.io" in sys.modules
""")


def test_public_bootstrap_is_idempotent() -> None:
    _clean_python("""
import gwexpy
from gwexpy.interop._registry import ConverterRegistry
gwexpy.register_all()
before = dict(ConverterRegistry._constructors)
gwexpy.register_all()
after = dict(ConverterRegistry._constructors)
assert before.keys() == after.keys()
assert all(before[name] is after[name] for name in before)
""")
