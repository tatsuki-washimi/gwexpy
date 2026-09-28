"""Small harness checks that do not measure a GWexpy wheel."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

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


@pytest.mark.parametrize("numpy_seterr", ["warn", "raise"])
@pytest.mark.parametrize(
    ("phase", "scenario", "tamper", "accepted", "error"),
    [
        ("historical", "matrix_object_strings", None, True, None),
        ("prex", "matrix_object_strings", None, False, "cross-arm"),
        ("candidate", "matrix_object_strings", None, False, "cross-arm"),
        ("historical", "matrix_nan_inf", None, False, "cross-arm"),
        (
            "historical",
            "matrix_object_strings",
            "wrong_a_source",
            False,
            "source/wheel",
        ),
        ("historical", "matrix_object_strings", "wrong_b_wheel", False, "source/wheel"),
        ("historical", "ats32", "same_public_wrong_a_source", False, "source/wheel"),
        ("historical", "ats32", "same_public_wrong_b_wheel", False, "source/wheel"),
        ("prex", "ats32", "wrong_a_source", False, "source/wheel"),
        ("prex", "ats32", "wrong_a_wheel", False, "source/wheel"),
        ("historical", "matrix_object_strings", "ignore", False, "numpy-seterr"),
        ("historical", "matrix_object_strings", "a_hash", False, "cross-arm"),
        ("historical", "matrix_object_strings", "a_unit", False, "cross-arm"),
        ("historical", "matrix_object_strings", "a_warning", False, "cross-arm"),
        ("historical", "matrix_object_strings", "b_warning", False, "cross-arm"),
        ("historical", "matrix_object_strings", "b_message", False, "cross-arm"),
        ("historical", "matrix_object_strings", "within_arm", False, "within-arm"),
    ],
)
def test_capture_allows_only_characterized_historical_delta(
    monkeypatch, tmp_path, phase, scenario, tamper, accepted, error, numpy_seterr
) -> None:
    """Only exact, source-bound B0/B1 results may differ across arms."""
    oracle = json.loads(bx_measure.HISTORICAL_OBJECT_ORACLE.read_text())
    (tmp_path / "manifest.json").write_text('{"generator_sha256": "fixture"}')
    monkeypatch.setattr(
        bx_measure,
        "_fixture",
        lambda *_: {
            "path": "fixture",
            "manifest_sha256": oracle["fixture_manifest_sha256"],
            "file_sha256": oracle["fixture_file_sha256"],
        },
    )
    counts = {"A": 0, "B": 0}

    def invoke(python, wheel, args, mode):
        arm = "A" if python == Path("python-a") else "B"
        audit = {
            "python": "3.12.12",
            "distributions": {"numpy": "1.26.4"},
            "wheel_sha256": oracle["arms"][arm]["wheel_sha256"],
        }
        if phase == "prex" and arm == "A":
            audit["wheel_sha256"] = oracle["arms"]["B"]["wheel_sha256"]
        if phase == "prex" and arm == "B":
            audit["wheel_sha256"] = "c" * 64
        if tamper in ("wrong_b_wheel", "same_public_wrong_b_wheel") and arm == "B":
            audit["wheel_sha256"] = "f" * 64
        if tamper == "wrong_a_wheel" and arm == "A":
            audit["wheel_sha256"] = "d" * 64
        if mode == "audit":
            return {"audit": audit}
        counts[arm] += 1
        public_arm = (
            "A"
            if tamper in ("same_public_wrong_a_source", "same_public_wrong_b_wheel")
            else arm
        )
        public = deepcopy(oracle["arms"][public_arm]["public_by_seterr"]["warn"])
        if tamper in ("a_hash", "within_arm") and arm == "A":
            if tamper == "a_hash" or counts[arm] > 1:
                public["outcome"]["values_sha256"] = "f" * 64
        if tamper == "a_unit" and arm == "A":
            public["outcome"]["units"][0][0] = "mV"
        if tamper == "a_warning" and arm == "A":
            public["warnings"][0]["message"] += " altered"
        if tamper == "b_warning" and arm == "B":
            public["warnings"][0]["message"] += " altered"
        if tamper == "b_message" and arm == "B":
            public["outcome"]["message"] += " altered"
        return {"audit": audit, "public": public}

    monkeypatch.setattr(bx_measure, "_invoke", invoke)
    args = SimpleNamespace(
        pre_x_sha=(
            oracle["arms"]["A"]["source_sha"]
            if phase == "candidate"
            else "f" * 40
            if phase == "prex"
            else oracle["arms"]["B"]["source_sha"]
        ),
        source_a=oracle["arms"]["B" if phase == "prex" else "A"]["source_sha"],
        source_b="f" * 40 if phase == "prex" else oracle["arms"]["B"]["source_sha"],
        samples=5,
        mode="public",
        scenario=scenario,
        sample_ms=10,
        phase=phase,
        fixtures=tmp_path,
        python_a=Path("python-a"),
        python_b=Path("python-b"),
        wheel_a=Path("wheel-a"),
        wheel_b=Path("wheel-b"),
        output=tmp_path / "capture",
        numpy_seterr=numpy_seterr,
    )
    if tamper in ("wrong_a_source", "same_public_wrong_a_source"):
        args.source_a = "e" * 40
    if tamper == "ignore":
        args.numpy_seterr = "ignore"
    if not accepted:
        exception = (
            ValueError if error in ("source/wheel", "numpy-seterr") else RuntimeError
        )
        with pytest.raises(exception, match=error):
            bx_measure._capture(args)
        return
    bx_measure._capture(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["within_arm_parity"] is True
    assert manifest["cross_arm_parity"] is False
    assert manifest["historical_public_delta_reason"]
    assert manifest["historical_public_oracle_sha256"] == bx_measure._sha256(
        bx_measure.HISTORICAL_OBJECT_ORACLE
    )
    assert manifest["public_fingerprints_by_arm"]["A"]["outcome"]["kind"] == "return"
    assert manifest["public_fingerprints_by_arm"]["B"]["outcome"]["kind"] == "error"
