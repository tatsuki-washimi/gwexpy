"""Small harness checks that do not measure a GWexpy wheel."""

from __future__ import annotations

import io
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from benchmarks.copy_audit import bx_measure


class _Matrix:
    def __init__(self, values: np.ndarray) -> None:
        self.value = values
        self.t0 = SimpleNamespace(value=1_000_000_000.125)
        self.dt = SimpleNamespace(value=0.125)
        self.units = np.array([["V"]])

    def row_keys(self):
        return ["r0"]

    def col_keys(self):
        return ["c0"]


def test_nonfinite_preview_and_noncontiguous_sha_are_repeatable() -> None:
    """Nonfinite previews and strided C-order hashes are deterministic."""
    values = np.array([0.0, np.nan, np.inf, -np.inf, -0.0, 1.5])
    matrix = _Matrix(values.reshape(1, 1, -1))
    first = bx_measure._fingerprint_read("matrix_nan_inf", matrix, None, [])
    second = bx_measure._fingerprint_read("matrix_nan_inf", matrix, None, [])
    assert first == second
    assert first["outcome"]["first_values"] == [
        "0x0.0p+0",
        "nan",
        "inf",
        "-inf",
    ]
    assert first["outcome"]["last_values"][2] == "-0x0.0p+0"

    strided = np.arange(60, dtype=np.int64).reshape(6, 10)[:, ::2]
    assert not strided.flags.c_contiguous
    expected = bx_measure.hashlib.sha256(np.ascontiguousarray(strided).tobytes())
    assert bx_measure._numeric_sha256(strided) == expected.hexdigest()


def test_wall_clocks_stop_before_fingerprint(monkeypatch) -> None:
    """Warm reads precede clocks and hashing follows both stop clocks."""
    events = []

    def read(*args, **kwargs):
        events.append("read")
        return object()

    def wall_clock():
        events.append("wall_clock")
        return 100 if events.count("wall_clock") == 1 else 140

    def cpu_clock():
        events.append("cpu_clock")
        return 200 if events.count("cpu_clock") == 1 else 230

    def fingerprint(*args):
        events.append("fingerprint")
        return {"outcome": {"kind": "return"}, "warnings": []}

    fake_module = ModuleType("gwexpy.timeseries")
    fake_module.TimeSeries = SimpleNamespace(read=read)
    fake_module.TimeSeriesMatrix = SimpleNamespace(read=read)
    monkeypatch.setitem(bx_measure.sys.modules, "gwexpy.timeseries", fake_module)
    monkeypatch.setattr(bx_measure, "_fingerprint_read", fingerprint)
    monkeypatch.setattr(bx_measure.time, "perf_counter_ns", wall_clock)
    monkeypatch.setattr(bx_measure.time, "process_time_ns", cpu_clock)
    result = bx_measure._wall_read("ats32", Path("unused"))
    assert result["wall_ns"] == 40
    assert result["cpu_ns_parent"] == 30
    assert events == [
        "read",
        "wall_clock",
        "cpu_clock",
        "read",
        "wall_clock",
        "cpu_clock",
        "fingerprint",
    ]


def test_pss_worker_fingerprints_only_after_stop(monkeypatch, capsys) -> None:
    """The worker holds its result until the controller stops sampling."""
    events = []
    args = SimpleNamespace(
        fixtures=Path("unused"),
        scenario="ats32",
        wheel=Path("unused"),
        numpy_seterr="warn",
        mode="pss",
    )
    monkeypatch.setattr(bx_measure, "_fixture", lambda *_: {"path": "unused"})
    monkeypatch.setattr(bx_measure, "_wheel_audit", lambda *_: {})
    monkeypatch.setattr(bx_measure.sys, "stdin", io.StringIO("BX_GO\nBX_STOP\n"))
    monkeypatch.setattr(bx_measure.time, "sleep", lambda _: events.append("held"))

    def read(*args):
        events.append("read")
        return object(), None, [], {}

    monkeypatch.setattr(bx_measure, "_read_with_diagnostics", read)

    def fingerprint(*args):
        events.append("fingerprint")
        return {"outcome": {"kind": "return"}, "warnings": []}

    monkeypatch.setattr(bx_measure, "_fingerprint_read", fingerprint)
    bx_measure._worker(args)
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["BX_READY", "BX_READ_HELD"]
    assert events == ["read", "held", "fingerprint"]
