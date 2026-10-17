#!/usr/bin/env python3
"""Runtime verification — inherited API priority batch 1 (10 identifiers)."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u

PIN = "c2898762ca68e20a1a241fffb3c3262b0a27290b"
REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

BATCH_APIS: list[str] = [
    "gwexpy.timeseries.timeseries.TimeSeries/override_unit",
    "gwexpy.timeseries.timeseries.TimeSeries/shift",
    "gwexpy.timeseries.timeseries.TimeSeries/pad",
    "gwexpy.frequencyseries.frequencyseries.FrequencySeries/crop",
    "gwexpy.spectrogram.spectrogram.Spectrogram/crop_frequencies",
    "gwexpy.types.plane2d.Plane2D/crop",
    "gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/copy",
    "gwexpy.types.array2d.Array2D/copy",
    "gwexpy.spectrogram.spectrogram.Spectrogram/read",
    "gwexpy.types.series.Series/write",
]


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(value) for value in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_import_provenance(
    callable_obj: Callable[..., Any],
    *,
    qualname: str | None = None,
    module_name: str | None = None,
) -> dict[str, Any]:
    import gwexpy

    unified = type(callable_obj).__name__.endswith(("Read", "Write")) or type(callable_obj).__name__ in {
        "UnifiedReadWriteMethod",
    }
    if unified and qualname and module_name:
        mod = importlib.import_module(module_name)
        mod_file = getattr(mod, "__file__", None)
        src_path = Path(mod_file) if mod_file else None
        repo_s = str(REPO.resolve())
        pkg_file = gwexpy.__file__ or ""
        return {
            "gwexpy_package_file": pkg_file,
            "callable_module": module_name,
            "callable_qualname": qualname,
            "target_module_file": mod_file,
            "target_source_file": mod_file,
            "target_source_sha256": file_sha256(src_path) if src_path else None,
            "loaded_from_repo_checkout": repo_s in pkg_file
            and (mod_file is None or repo_s in str(mod_file)),
        }

    mod = sys.modules.get(getattr(callable_obj, "__module__", None))
    mod_file = getattr(mod, "__file__", None) if mod else None
    src = inspect.getsourcefile(callable_obj) or mod_file
    src_path = Path(src) if src else None
    repo_s = str(REPO.resolve())
    pkg_file = gwexpy.__file__ or ""
    return {
        "gwexpy_package_file": pkg_file,
        "callable_module": callable_obj.__module__,
        "callable_qualname": getattr(callable_obj, "__qualname__", callable_obj.__name__),
        "target_module_file": mod_file,
        "target_source_file": str(src_path.resolve()) if src_path and src_path.is_file() else src,
        "target_source_sha256": file_sha256(src_path) if src_path else None,
        "loaded_from_repo_checkout": repo_s in pkg_file and (mod_file is None or repo_s in str(mod_file)),
    }


def capture_invoke_context(subject: Any) -> dict[str, Any]:
    cls = type(subject) if not inspect.isclass(subject) else subject
    return {
        "runtime_class": f"{cls.__module__}.{cls.__name__}",
        "mro": [f"{c.__module__}.{c.__name__}" for c in cls.mro()[:8]],
    }


def series_write_format_text(series_mod: Any) -> str:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        series_mod.Series.write.list_formats()
    return buf.getvalue()


def format_registered(format_text: str, fmt: str) -> bool:
    for line in format_text.splitlines():
        if line.strip().startswith(fmt):
            return "Yes" in line
    return False


def finalize_gwpy_comparison(rec: dict[str, Any], *, notes: str = "") -> None:
    if rec.get("trial_kind") != "gwpy_comparison":
        return
    if rec.get("execution_status") == "contract_pending":
        return
    if rec.get("outcome") != "returned":
        rec["gwpy_comparison"] = {
            "status": "trial_error",
            "notes": notes or f"comparison trial raised {rec.get('outcome')}",
        }
        return
    passed = rec.get("assertions", {}).get("passed")
    if passed is True:
        rec["gwpy_comparison"] = {"status": "match", "notes": notes}
    elif passed is False:
        rec["gwpy_comparison"] = {"status": "differ", "notes": notes}
    else:
        rec["gwpy_comparison"] = {"status": "incomparable", "notes": notes}


def run_trial(
    *,
    target_identifier: str,
    trial_kind: str,
    input_description: str,
    prov_fn: Callable[..., Any] | Any,
    fn: Callable[[], Any],
    prov_qualname: str | None = None,
    prov_module: str | None = None,
    expected: dict[str, Any] | None = None,
    validate: Callable[[Any], dict[str, Any]] | None = None,
    comparison_notes: str = "",
    related_issue: str | None = None,
    invoke_path: str | None = None,
    coverage_scope: str = "direct",
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "batch": "inherited_api_batch_1",
        "target_identifier": target_identifier,
        "trial_kind": trial_kind,
        "input_description": input_description,
        "expected": expected or {},
        "execution_status": "executed",
        "source_commit": git_head(),
        "audit_baseline_pin": PIN,
        "related_issue": related_issue,
        "invoke_path": invoke_path,
        "coverage_scope": coverage_scope,
        "import_provenance": capture_import_provenance(
            prov_fn, qualname=prov_qualname, module_name=prov_module
        ),
        "evidence": "runtime_observation_not_defect_confirmation",
    }
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        try:
            result = fn()
            rec["outcome"] = "returned"
            rec["actual"] = result if isinstance(result, dict) else {"repr": repr(result)[:400]}
            if isinstance(result, dict) and "invoke_context" in result:
                rec["invoke_context"] = result["invoke_context"]
            if validate is not None:
                rec["assertions"] = validate(result)
        except Exception as exc:
            rec["outcome"] = type(exc).__name__
            rec["actual"] = {"exception": str(exc)[:500]}
            if trial_kind == "abnormal_input":
                rec["assertions"] = {
                    "passed": True,
                    "checks": [{"check": "exception_observed", "passed": True}],
                }
            else:
                rec["assertions"] = {"passed": False, "checks": [{"check": "no_exception", "passed": False}]}
        rec["warnings"] = [
            {
                "category": w.category.__name__,
                "message": str(w.message)[:200],
                "filename": w.filename,
                "lineno": w.lineno,
            }
            for w in wrec
        ]
    finalize_gwpy_comparison(rec, notes=comparison_notes)
    return rec


def main() -> int:
    from gwexpy.frequencyseries import BifrequencyMap, FrequencySeries
    from gwexpy.spectrogram import Spectrogram
    from gwexpy.timeseries import TimeSeries
    import gwexpy.types.series as gwexpy_series_mod
    import gwpy.types.series as gwpy_series_mod
    from gwexpy.types.array2d import Array2D
    from gwexpy.types.plane2d import Plane2D
    import gwpy.frequencyseries as gwpy_fs
    import gwpy.spectrogram as gwpy_sg
    import gwpy.timeseries as gwpy_ts
    from gwpy.types import Array2D as GwpyArray2D

    trials: list[dict[str, Any]] = []

    # 1 override_unit
    ident = BATCH_APIS[0]
    base_ts = TimeSeries(np.array([1.0, 2.0, 3.0]), sample_rate=1.0, unit="m")

    def pos_override() -> dict[str, Any]:
        ts = TimeSeries(base_ts.value.copy(), sample_rate=base_ts.sample_rate, unit=base_ts.unit)
        ts.override_unit("cm")
        return {"unit": str(ts.unit), "values": ts.value.tolist()}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="override_unit('cm') on 3-sample series; values unchanged numerically",
            prov_fn=TimeSeries.override_unit,
            fn=pos_override,
            expected={"unit": "cm", "values_unchanged": True},
            validate=lambda r: {
                "passed": r["unit"] == "cm" and r["values"] == [1.0, 2.0, 3.0],
                "checks": [
                    {"check": "unit_is_cm", "passed": r["unit"] == "cm"},
                    {"check": "values_unchanged", "passed": r["values"] == [1.0, 2.0, 3.0]},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="invalid unit string",
            prov_fn=TimeSeries.override_unit,
            fn=lambda: TimeSeries(base_ts.value.copy(), sample_rate=1.0, unit="m").override_unit(
                "not_a_real_unit_xyz"
            ),
            expected={"exception": "ValueError"},
        )
    )

    def gwpy_override():
        gts = gwpy_ts.TimeSeries(base_ts.value.copy(), sample_rate=1.0, unit="m")
        gts.override_unit("cm")
        ts = TimeSeries(base_ts.value.copy(), sample_rate=1.0, unit="m")
        ts.override_unit("cm")
        return {"gwpy_unit": str(gts.unit), "gwexpy_unit": str(ts.unit)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="same override_unit('cm') on GWpy vs GWexpy TimeSeries",
            prov_fn=TimeSeries.override_unit,
            fn=gwpy_override,
            validate=lambda r: {
                "passed": r["gwpy_unit"] == r["gwexpy_unit"] == "cm",
                "checks": [{"check": "units_match", "passed": r["gwpy_unit"] == r["gwexpy_unit"]}],
            },
            comparison_notes="Unit metadata string matches; numeric values not re-scaled in either stack.",
        )
    )

    # 2 shift
    ident = BATCH_APIS[1]

    def pos_shift():
        ts = TimeSeries(np.array([1.0, 2.0]), sample_rate=1.0, unit="m", epoch=0)
        ts.shift(1 * u.s)
        return {"t0": float(ts.t0.value), "len": len(ts), "values": ts.value.tolist()}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="shift +1 s; epoch 0 -> t0 1 s",
            prov_fn=TimeSeries.shift,
            fn=pos_shift,
            validate=lambda r: {
                "passed": r["t0"] == 1.0 and r["len"] == 2,
                "checks": [
                    {"check": "t0_is_1s", "passed": r["t0"] == 1.0},
                    {"check": "length_preserved", "passed": r["len"] == 2},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="shift with plain string (non-Quantity)",
            prov_fn=TimeSeries.shift,
            fn=lambda: TimeSeries([1.0], sample_rate=1.0, unit="m").shift("not_a_delta"),
            expected={"exception": "observed"},
        )
    )

    def gwpy_shift():
        gts = gwpy_ts.TimeSeries([1.0, 2.0], sample_rate=1.0, unit="m", epoch=0)
        gts.shift(1 * u.s)
        ts = TimeSeries(np.array([1.0, 2.0]), sample_rate=1.0, unit="m", epoch=0)
        ts.shift(1 * u.s)
        return {"gwpy_t0": float(gts.t0.value), "gwexpy_t0": float(ts.t0.value)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="+1 s shift on identical data",
            prov_fn=TimeSeries.shift,
            fn=gwpy_shift,
            validate=lambda r: {
                "passed": r["gwpy_t0"] == r["gwexpy_t0"],
                "checks": [{"check": "t0_match", "passed": r["gwpy_t0"] == r["gwexpy_t0"]}],
            },
        )
    )

    # 3 pad
    ident = BATCH_APIS[2]

    def pos_pad():
        ts = TimeSeries(np.array([1.0, 2.0, 3.0]), sample_rate=1.0, unit="m")
        out = ts.pad(2)
        return {"len": len(out), "values": out.value.tolist(), "t0": float(out.t0.value)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="pad(2) default zeros; length 3 -> 7",
            prov_fn=TimeSeries.pad,
            fn=pos_pad,
            validate=lambda r: {
                "passed": r["len"] == 7 and r["values"] == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
                "checks": [
                    {"check": "length_7", "passed": r["len"] == 7},
                    {"check": "edge_zeros", "passed": r["values"][0] == 0.0 and r["values"][-1] == 0.0},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="pad(-1) negative width",
            prov_fn=TimeSeries.pad,
            fn=lambda: TimeSeries([1.0], sample_rate=1.0, unit="m").pad(-1),
            expected={"exception": "observed"},
        )
    )

    def gwpy_pad():
        gts = gwpy_ts.TimeSeries([1.0, 2.0, 3.0], sample_rate=1.0, unit="m")
        gout = gts.pad(2)
        ts = TimeSeries(np.array([1.0, 2.0, 3.0]), sample_rate=1.0, unit="m")
        out = ts.pad(2)
        return {"gwpy_len": len(gout), "gwexpy_len": len(out)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="pad(2) length comparison",
            prov_fn=TimeSeries.pad,
            fn=gwpy_pad,
            validate=lambda r: {
                "passed": r["gwpy_len"] == r["gwexpy_len"] == 7,
                "checks": [{"check": "length_match", "passed": r["gwpy_len"] == r["gwexpy_len"]}],
            },
        )
    )

    # 4 FrequencySeries/crop
    ident = BATCH_APIS[3]
    freqs = np.linspace(0, 4, 5)

    def pos_fs_crop():
        fs = FrequencySeries([1, 2, 3, 4, 5], frequencies=freqs, unit="strain")
        out = fs.crop(1, 3)
        return {
            "len": len(out),
            "f0": float(out.f0.value),
            "df": float(out.df.value),
            "f_end": float(out.frequencies[-1].value),
        }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="crop [1,3] Hz on 5-point FS",
            prov_fn=FrequencySeries.crop,
            fn=pos_fs_crop,
            validate=lambda r: {
                "passed": r["len"] == 2 and r["f0"] == 1.0 and r["df"] == 1.0,
                "checks": [
                    {"check": "len_2", "passed": r["len"] == 2},
                    {"check": "f0_1hz", "passed": r["f0"] == 1.0},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="crop with start > end (3, 1)",
            prov_fn=FrequencySeries.crop,
            fn=lambda: FrequencySeries([1, 2, 3], frequencies=[0, 1, 2], unit="strain").crop(3, 1),
            expected={"note": "observe empty or error"},
        )
    )

    def gwpy_fs_crop():
        gfs = gwpy_fs.FrequencySeries([1, 2, 3, 4, 5], frequencies=freqs, unit="strain")
        gout = gfs.crop(1, 3)
        fs = FrequencySeries([1, 2, 3, 4, 5], frequencies=freqs, unit="strain")
        out = fs.crop(1, 3)
        return {"gwpy_len": len(gout), "gwexpy_len": len(out), "gwpy_f0": float(gout.f0.value)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="crop [1,3] Hz GWpy vs GWexpy",
            prov_fn=FrequencySeries.crop,
            fn=gwpy_fs_crop,
            validate=lambda r: {
                "passed": r["gwpy_len"] == r["gwexpy_len"] == 2,
                "checks": [{"check": "len_match", "passed": r["gwpy_len"] == r["gwexpy_len"]}],
            },
        )
    )

    # 5 Spectrogram/crop_frequencies
    ident = BATCH_APIS[4]
    spec_data = np.arange(20, dtype=float).reshape(5, 4)

    def pos_spec_crop():
        sg = Spectrogram(spec_data, dt=0.5, f0=0, df=1, unit="strain**2/Hz")
        out = sg.crop_frequencies(1, 3)
        return {"shape": list(out.shape), "f0": float(out.f0.value), "df": float(out.df.value)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="crop_frequencies [1,3] Hz",
            prov_fn=Spectrogram.crop_frequencies,
            fn=pos_spec_crop,
            validate=lambda r: {
                "passed": r["shape"] == [5, 2] and r["f0"] == 1.0,
                "checks": [
                    {"check": "time_bins_5", "passed": r["shape"][0] == 5},
                    {"check": "freq_bins_2", "passed": r["shape"][1] == 2},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="crop_frequencies high < low (3, 1)",
            prov_fn=Spectrogram.crop_frequencies,
            fn=lambda: Spectrogram(spec_data, dt=0.5, f0=0, df=1, unit="strain**2/Hz").crop_frequencies(
                3, 1
            ),
            expected={"observe": "empty_or_error"},
        )
    )

    def gwpy_spec_crop():
        gsg = gwpy_sg.Spectrogram(spec_data, dt=0.5, f0=0, df=1, unit="strain**2/Hz")
        gout = gsg.crop_frequencies(1, 3)
        sg = Spectrogram(spec_data, dt=0.5, f0=0, df=1, unit="strain**2/Hz")
        out = sg.crop_frequencies(1, 3)
        return {"gwpy_shape": list(gout.shape), "gwexpy_shape": list(out.shape)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="crop_frequencies GWpy vs GWexpy",
            prov_fn=Spectrogram.crop_frequencies,
            fn=gwpy_spec_crop,
            validate=lambda r: {
                "passed": r["gwpy_shape"] == r["gwexpy_shape"],
                "checks": [{"check": "shape_match", "passed": r["gwpy_shape"] == r["gwexpy_shape"]}],
            },
        )
    )

    # 6 Plane2D/crop
    ident = BATCH_APIS[5]
    plane_data = np.arange(12.0).reshape(3, 4)

    def pos_plane_crop():
        p = Plane2D(plane_data, x0=0, dx=1, y0=10, dy=2, unit="m")
        out = p.crop(1, 3)
        return {"shape": list(out.shape), "x0": float(out.x0.value), "y0": float(out.y0.value)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="crop x-axis [1,3) on 3x4 Plane2D (observe axis semantics)",
            prov_fn=Plane2D.crop,
            fn=pos_plane_crop,
            validate=lambda r: {
                "passed": r["shape"] == [2, 4] and r["x0"] == 1.0 and r["y0"] == 10.0,
                "checks": [
                    {"check": "shape_2x4_observed", "passed": r["shape"] == [2, 4]},
                    {"check": "x0_is_1", "passed": r["x0"] == 1.0},
                    {"check": "y0_unchanged", "passed": r["y0"] == 10.0},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="crop start > end on x-axis",
            prov_fn=Plane2D.crop,
            fn=lambda: Plane2D(plane_data, x0=0, dx=1, y0=0, dy=1, unit="m").crop(5, 1),
            expected={"observe": "empty_or_error"},
        )
    )

    trials.append(
        {
            "batch": "inherited_api_batch_1",
            "target_identifier": ident,
            "trial_kind": "gwpy_comparison",
            "input_description": "GWpy has no Plane2D export; comparison not executed",
            "execution_status": "contract_pending",
            "source_commit": git_head(),
            "audit_baseline_pin": PIN,
            "import_provenance": capture_import_provenance(Plane2D.crop),
            "gwpy_comparison": {
                "status": "incomparable",
                "notes": "No gwpy.types.Plane2D in installed GWpy; GWexpy-only positive/abnormal trials still run.",
            },
            "assertions": {"passed": None, "checks": []},
            "evidence": "runtime_observation_not_defect_confirmation",
        }
    )

    # 7 BifrequencyMap/copy
    ident = BATCH_APIS[6]
    bfm_vals = np.arange(12.0, dtype=float).reshape(3, 4)

    def pos_bfm_copy():
        bfm = BifrequencyMap(bfm_vals.copy(), xindex=[10, 20, 40] * u.Hz, yindex=[1, 2, 4, 8] * u.Hz, unit="V")
        c = bfm.copy()
        c.value[0, 0] = -999.0
        return {
            "original_00": float(bfm.value[0, 0]),
            "copy_00": float(c.value[0, 0]),
            "same_buffer": bool(np.shares_memory(bfm.value, c.value)),
        }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="copy(); mutate copy; observe buffer sharing",
            prov_fn=BifrequencyMap.copy,
            fn=pos_bfm_copy,
            validate=lambda r: {
                "passed": r["original_00"] == 0.0 and r["copy_00"] == -999.0,
                "checks": [
                    {"check": "original_unmutated", "passed": r["original_00"] == 0.0},
                    {"check": "copy_mutated", "passed": r["copy_00"] == -999.0},
                    {"check": "shares_memory_observed", "passed": True, "value": r["same_buffer"]},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="copy() on minimal map (no abnormal path expected)",
            prov_fn=BifrequencyMap.copy,
            fn=lambda: BifrequencyMap.from_points(np.eye(2), [1, 2], [1, 2]).copy(),
            validate=lambda _: {"passed": True, "checks": [{"check": "copy_returns", "passed": True}]},
        )
    )

    def gwpy_bfm_copy():
        bfm = BifrequencyMap(bfm_vals.copy(), xindex=[10, 20, 40] * u.Hz, yindex=[1, 2, 4, 8] * u.Hz, unit="V")
        garr = GwpyArray2D(bfm_vals.copy(), xindex=[10, 20, 40] * u.Hz, yindex=[1, 2, 4, 8] * u.Hz, unit="V")
        bc = bfm.copy()
        gc = garr.copy()
        bc.value[0, 0] = 7.0
        gc.value[0, 0] = 7.0
        return {
            "gwexpy_shares": bool(np.shares_memory(bfm.value, bc.value)),
            "gwpy_shares": bool(np.shares_memory(garr.value, gc.value)),
        }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="copy buffer sharing GWexpy BifrequencyMap vs GWpy Array2D analogue",
            prov_fn=BifrequencyMap.copy,
            fn=gwpy_bfm_copy,
            comparison_notes="Both observe copy independence semantics via shares_memory flag (not defect verdict).",
            validate=lambda r: {
                "passed": r["gwexpy_shares"] == r["gwpy_shares"],
                "checks": [{"check": "sharing_flag_match", "passed": r["gwexpy_shares"] == r["gwpy_shares"]}],
            },
        )
    )

    # 8 Array2D/copy
    ident = BATCH_APIS[7]
    a2_data = np.ones((2, 3))

    def pos_a2_copy():
        a = Array2D(a2_data, x0=0, dx=1, y0=0, dy=1, unit="V")
        c = a.copy()
        c.value[0, 0] = 0.0
        return {
            "orig": float(a.value[0, 0]),
            "copy": float(c.value[0, 0]),
            "unit": str(c.unit),
            "shares_memory": bool(np.shares_memory(a.value, c.value)),
        }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="Array2D.copy unit and mutation isolation observation",
            prov_fn=Array2D.copy,
            fn=pos_a2_copy,
            validate=lambda r: {
                "passed": r["orig"] == 1.0 and r["copy"] == 0.0 and r["unit"] == "V",
                "checks": [
                    {"check": "unit_preserved", "passed": r["unit"] == "V"},
                    {"check": "mutation_observed", "passed": r["orig"] == 1.0 and r["copy"] == 0.0},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="copy empty Array2D",
            prov_fn=Array2D.copy,
            fn=lambda: Array2D(np.zeros((0, 2)), x0=0, dx=1, y0=0, dy=1, unit="V").copy(),
            validate=lambda _: {"passed": True, "checks": [{"check": "empty_copy_returns", "passed": True}]},
        )
    )

    def gwpy_a2_copy():
        a = Array2D(a2_data, x0=0, dx=1, y0=0, dy=1, unit="V")
        g = GwpyArray2D(a2_data, x0=0, dx=1, y0=0, dy=1, unit="V")
        ac = a.copy()
        gc = g.copy()
        return {
            "gwexpy_shares": bool(np.shares_memory(a.value, ac.value)),
            "gwpy_shares": bool(np.shares_memory(g.value, gc.value)),
        }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="Array2D.copy vs GWpy Array2D.copy sharing observation",
            prov_fn=Array2D.copy,
            fn=gwpy_a2_copy,
            validate=lambda r: {
                "passed": r["gwexpy_shares"] == r["gwpy_shares"],
                "checks": [{"check": "sharing_match", "passed": r["gwexpy_shares"] == r["gwpy_shares"]}],
            },
        )
    )

    # 9 Spectrogram/read
    ident = BATCH_APIS[8]
    sg_template = Spectrogram(np.ones((4, 6)), dt=0.25, f0=0, df=1, unit="strain**2/Hz")

    def pos_sg_read():
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sg.hdf5"
            sg_template.write(str(path), format="hdf5", path="spectrogram")
            out = Spectrogram.read(str(path), format="hdf5", path="spectrogram")
            values_match = bool(np.allclose(out.value, sg_template.value))
            return {
                "invoke_context": capture_invoke_context(out),
                "type": type(out).__name__,
                "shape": list(out.shape),
                "values_match": values_match,
                "unit_match": str(out.unit) == str(sg_template.unit),
                "dt_match": bool(out.dt == sg_template.dt),
                "path": str(path),
            }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="write hdf5 fixture then Spectrogram.read round-trip",
            prov_fn=Spectrogram.read,
            prov_module="gwexpy.spectrogram.spectrogram",
            prov_qualname="Spectrogram.read",
            fn=pos_sg_read,
            validate=lambda r: {
                "passed": r["type"] == "Spectrogram"
                and r["shape"] == [4, 6]
                and r["values_match"]
                and r["unit_match"]
                and r["dt_match"],
                "checks": [
                    {"check": "return_type", "passed": r["type"] == "Spectrogram"},
                    {"check": "shape_match", "passed": r["shape"] == [4, 6]},
                    {"check": "values_match", "passed": r["values_match"]},
                    {"check": "unit_match", "passed": r["unit_match"]},
                    {"check": "dt_match", "passed": r["dt_match"]},
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="read missing file",
            prov_fn=Spectrogram.read,
            prov_module="gwexpy.spectrogram.spectrogram",
            prov_qualname="Spectrogram.read",
            fn=lambda: Spectrogram.read("/tmp/gwexpy_missing_spectrogram.hdf5", format="hdf5"),
            expected={"exception": "observed"},
        )
    )

    def gwpy_sg_read():
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sg.hdf5"
            sg_template.write(str(path), format="hdf5", path="spectrogram")
            gout = gwpy_sg.Spectrogram.read(str(path), format="hdf5", path="spectrogram")
            out = Spectrogram.read(str(path), format="hdf5", path="spectrogram")
            return {"gwpy_shape": list(gout.shape), "gwexpy_shape": list(out.shape)}

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="same hdf5 file read via GWpy vs GWexpy Spectrogram.read",
            prov_fn=Spectrogram.read,
            prov_module="gwexpy.spectrogram.spectrogram",
            prov_qualname="Spectrogram.read",
            fn=gwpy_sg_read,
            validate=lambda r: {
                "passed": r["gwpy_shape"] == r["gwexpy_shape"],
                "checks": [{"check": "shape_match", "passed": r["gwpy_shape"] == r["gwexpy_shape"]}],
            },
        )
    )

    # 10 Series/write — gwexpy.types.series.Series instance (not FrequencySeries)
    ident = BATCH_APIS[9]
    series_write_fmt_text = series_write_format_text(gwexpy_series_mod)
    hdf5_on_series = format_registered(series_write_fmt_text, "hdf5")
    csv_on_series = format_registered(series_write_fmt_text, "csv")
    series_base = gwexpy_series_mod.Series([1.0, 2.0, 3.0], xindex=[0.0, 1.0, 2.0], unit="m")

    def pos_series_write():
        subject = gwexpy_series_mod.Series([1.0, 2.0, 3.0], xindex=[0.0, 1.0, 2.0], unit="m")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "series.csv"
            subject.write(str(path), format="csv")
            back = gwexpy_series_mod.Series.read(str(path), format="csv")
            values_match = bool(np.allclose(back.value, subject.value))
            unit_match = str(back.unit) == str(subject.unit)
            x0_match = float(back.x0.value) == float(subject.x0.value)
            dx_match = float(back.dx.value) == float(subject.dx.value)
            return {
                "invoke_context": capture_invoke_context(subject),
                "writer_registration": {
                    "csv_registered": csv_on_series,
                    "hdf5_registered": hdf5_on_series,
                },
                "format_used": "csv",
                "values_match": values_match,
                "unit_match": unit_match,
                "x0_match": x0_match,
                "dx_match": dx_match,
                "csv_roundtrip_ok": values_match and unit_match and x0_match and dx_match,
            }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="positive_control",
            input_description="gwexpy.types.series.Series.write/read csv round-trip (hdf5 not registered on Series)",
            prov_fn=gwexpy_series_mod.Series.write,
            prov_module="gwexpy.types.series",
            prov_qualname="Series.write",
            fn=pos_series_write,
            invoke_path="gwexpy.types.series.Series.write on gwexpy.types.series.Series instance",
            validate=lambda r: {
                "passed": r["writer_registration"]["csv_registered"]
                and r["values_match"]
                and r["x0_match"]
                and r["dx_match"],
                "checks": [
                    {"check": "csv_writer_registered", "passed": r["writer_registration"]["csv_registered"]},
                    {"check": "values_match", "passed": r["values_match"]},
                    {"check": "x0_dx_match", "passed": r["x0_match"] and r["dx_match"]},
                    {
                        "check": "unit_preserved_after_csv_roundtrip",
                        "passed": r["unit_match"],
                    },
                    {
                        "check": "hdf5_not_on_series_documented",
                        "passed": not r["writer_registration"]["hdf5_registered"],
                    },
                ],
            },
        )
    )
    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="abnormal_input",
            input_description="Series.write with unknown format identifier",
            prov_fn=gwexpy_series_mod.Series.write,
            prov_module="gwexpy.types.series",
            prov_qualname="Series.write",
            fn=lambda: series_base.write("/tmp/out.xyz", format="no_such_format_xyz"),
            invoke_path="gwexpy.types.series.Series.write",
            expected={"exception": "IORegistryError"},
        )
    )

    def gwpy_series_write():
        gwex = gwexpy_series_mod.Series([1.0, 2.0, 3.0], xindex=[0.0, 1.0, 2.0], unit="m")
        gwpy = gwpy_series_mod.Series([1.0, 2.0, 3.0], xindex=[0.0, 1.0, 2.0], unit="m")
        with tempfile.TemporaryDirectory() as td_g, tempfile.TemporaryDirectory() as td_e:
            pg = Path(td_g) / "gwpy.csv"
            pe = Path(td_e) / "gwexpy.csv"
            gwpy.write(str(pg), format="csv")
            gwex.write(str(pe), format="csv")
            gr = gwpy_series_mod.Series.read(str(pg), format="csv")
            er = gwexpy_series_mod.Series.read(str(pe), format="csv")
            return {
                "invoke_context": capture_invoke_context(gwex),
                "gwpy_values": gr.value.tolist(),
                "gwexpy_values": er.value.tolist(),
                "gwpy_unit": str(gr.unit),
                "gwexpy_unit": str(er.unit),
                "values_match": bool(np.allclose(gr.value, er.value)),
                "unit_match": str(gr.unit) == str(er.unit),
            }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="gwpy_comparison",
            input_description="csv round-trip GWpy.types.series.Series vs gwexpy.types.series.Series (separate temp files)",
            prov_fn=gwexpy_series_mod.Series.write,
            prov_module="gwexpy.types.series",
            prov_qualname="Series.write",
            fn=gwpy_series_write,
            invoke_path="gwexpy.types.series.Series.write",
            comparison_notes="Separate output paths; csv format because Series writer lacks hdf5.",
            validate=lambda r: {
                "passed": r["values_match"] and r["unit_match"],
                "checks": [
                    {"check": "values_match", "passed": r["values_match"]},
                    {"check": "unit_match", "passed": r["unit_match"]},
                ],
            },
        )
    )

    def ref_frequencyseries_hdf5():
        fs_ref = FrequencySeries([1.0, 2.0, 3.0], frequencies=[0, 1, 2], unit="m")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fs.hdf5"
            fs_ref.write(str(path), format="hdf5", path="data")
            back = FrequencySeries.read(str(path), format="hdf5", path="data")
            return {
                "invoke_context": capture_invoke_context(fs_ref),
                "values_match": bool(np.allclose(back.value, fs_ref.value)),
                "len": len(back),
            }

    trials.append(
        run_trial(
            target_identifier=ident,
            trial_kind="reference_subclass_io",
            coverage_scope="reference_only",
            input_description="Reference only: FrequencySeries hdf5 I/O (not Series/write direct evidence)",
            prov_fn=FrequencySeries.write,
            fn=ref_frequencyseries_hdf5,
            invoke_path="gwexpy.frequencyseries.FrequencySeries.write (reference)",
            validate=lambda r: {
                "passed": r["values_match"] and r["len"] == 3,
                "checks": [{"check": "reference_hdf5_roundtrip", "passed": r["values_match"]}],
            },
        )
    )

    jsonl_path = OUT_DIR / "c-inherited-batch1-runtime.jsonl"
    trials = [json_sanitize(t) for t in trials]
    jsonl_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trials), encoding="utf-8")

    from collections import Counter

    gwpy_cmp = [
        t.get("gwpy_comparison", {}).get("status")
        for t in trials
        if t.get("trial_kind") == "gwpy_comparison" and t.get("gwpy_comparison")
    ]
    contract_pending = sum(1 for t in trials if t.get("execution_status") == "contract_pending")
    direct_positive = [
        t
        for t in trials
        if t.get("trial_kind") == "positive_control"
        and t.get("coverage_scope", "direct") == "direct"
        and t.get("assertions", {}).get("passed")
    ]
    direct_verified_ids = {t["target_identifier"] for t in direct_positive}
    gwpy_assertion_pass = sum(
        1
        for t in trials
        if t.get("trial_kind") == "gwpy_comparison"
        and t.get("assertions", {}).get("passed") is True
    )
    summary = {
        "batch": "inherited_api_batch_1",
        "audit_baseline_pin": PIN,
        "source_commit": git_head(),
        "planned_api_count": len(BATCH_APIS),
        "planned_identifiers": BATCH_APIS,
        "trials_total": len(trials),
        "trials_core_batch": sum(1 for t in trials if t.get("coverage_scope", "direct") == "direct"),
        "reference_only_trials": sum(1 for t in trials if t.get("coverage_scope") == "reference_only"),
        "executed_unique_api_count": len({t["target_identifier"] for t in trials}),
        "direct_api_verified_count": len(direct_verified_ids),
        "direct_api_verified_identifiers": sorted(direct_verified_ids),
        "gwpy_comparison_assertions_passed": gwpy_assertion_pass,
        "gwpy_comparison_assertions_total": sum(
            1 for t in trials if t.get("trial_kind") == "gwpy_comparison"
        ),
        "positive_assertions_passed": sum(
            1
            for t in trials
            if t["trial_kind"] == "positive_control" and t.get("assertions", {}).get("passed")
        ),
        "positive_assertions_total": sum(1 for t in trials if t["trial_kind"] == "positive_control"),
        "abnormal_paths_observed": sum(
            1
            for t in trials
            if t["trial_kind"] == "abnormal_input"
            and (t.get("outcome") != "returned" or t.get("assertions", {}).get("passed"))
        ),
        "abnormal_trials_total": sum(1 for t in trials if t["trial_kind"] == "abnormal_input"),
        "gwpy_comparison_counts": dict(Counter(gwpy_cmp)),
        "contract_pending_or_not_run": contract_pending,
        "ledger_merge": "deferred_until_batch_review",
        "note": "Counts are contract observations, not product pass/fail scores.",
    }
    (OUT_DIR / "c-inherited-batch1-runtime-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md = f"""# Inherited API batch 1 — runtime verification

Pin: `{PIN}` · Source commit: `{summary['source_commit']}`

## Scope

10 planned inherited APIs (30 core trials + optional reference rows).

**Not merged** into `c-coverage-c2898762.jsonl` in this step.

## Aggregates

| Metric | Value |
| --- | ---: |
| Planned APIs | {summary['planned_api_count']} |
| Executed unique APIs | {summary['executed_unique_api_count']} |
| Positive assertions passed | {summary['positive_assertions_passed']} / {summary['positive_assertions_total']} |
| Abnormal paths observed | {summary['abnormal_paths_observed']} / {summary['abnormal_trials_total']} |
| GWpy comparison (status labels) | {json.dumps(summary['gwpy_comparison_counts'])} |
| GWpy comparison (assertions passed) | {summary['gwpy_comparison_assertions_passed']} / {summary['gwpy_comparison_assertions_total']} |
| Direct API verified (positive) | {summary['direct_api_verified_count']} / {summary['planned_api_count']} |
| Reference-only trials | {summary['reference_only_trials']} |

## Interpretation

- `override_unit` trials distinguish metadata unit assignment from physical value rescaling.
- `Series/write` uses `gwexpy.types.series.Series` instances; csv round-trip verified on values/axes; unit preservation recorded separately.
- `copy` trials record buffer sharing; no pre-judged correct deep-copy contract.
- GWpy comparison status is derived from trial outcome and assertions (not preset).
- Unified I/O uses measured read-back checks, not fixed `roundtrip=True` flags.

Artifacts: `c-inherited-batch1-runtime.jsonl`, `c-inherited-batch1-runtime-summary.json`.
"""
    (OUT_DIR / "2026-10-17-c-inherited-batch1-runtime.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
