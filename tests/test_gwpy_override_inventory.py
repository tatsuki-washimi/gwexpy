"""Contracts for the deterministic GWpy override inventory (issue #639)."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_gwpy_overrides.py"
MANIFEST = (
    ROOT
    / "docs/developers/plans/manifests/"
    / "audit-manifest-v0.2.3-gwpy-overrides.json"
)
WORKFLOW = ROOT / ".github/workflows/test-compat-gwpy.yml"
SUPPORTED_GWPY = ("4.0.1", "4.0.2")
SCHEMA = "gwexpy-v023-gwpy-override-inventory-v1"
IMPLEMENTATION_BASE = "a8085b71446d3ef3417a7e5b5ac8efb156368eac"
UPSTREAM_DEPENDENCY_PROVENANCE = (
    "GWpy providers retain package-relative source/line; inherited "
    "NumPy/Astropy providers retain normalized provider, member, kind, "
    "descriptor, and signature without source path or resolved version."
)
CASE_KEYS = {
    "case_key",
    "comparator",
    "counterpart_present",
    "evidence",
    "fixture",
    "gwpy_version",
    "implementation_group",
    "issues",
    "member",
    "member_id",
    "observations",
    "owner",
    "public_class",
    "state",
}
TEST_REFERENCE = {
    "reference": (
        "tests/test_gwpy_override_inventory.py"
        "::test_terminal_transition_with_case_derived_summary_is_valid"
    )
}
EXPECTED_ROOTS = {
    "gwexpy.fields.scalar.ScalarField": (
        "gwexpy.fields.scalar:ScalarField",
        "gwexpy.fields:ScalarField",
        "gwexpy:ScalarField",
    ),
    "gwexpy.frequencyseries.bifrequencymap.BifrequencyMap": (
        "gwexpy.frequencyseries:BifrequencyMap",
    ),
    "gwexpy.frequencyseries.frequencyseries.FrequencySeries": (
        "gwexpy.frequencyseries:FrequencySeries",
        "gwexpy:FrequencySeries",
    ),
    "gwexpy.plot.field.FieldPlot": ("gwexpy.plot.field:FieldPlot",),
    "gwexpy.plot.plot.Plot": ("gwexpy.plot.plot:Plot", "gwexpy.plot:Plot"),
    "gwexpy.plot.skymap.SkyMap": (
        "gwexpy.plot.skymap:SkyMap",
        "gwexpy.plot:SkyMap",
    ),
    "gwexpy.spectrogram.spectrogram.Spectrogram": (
        "gwexpy.spectrogram:Spectrogram",
        "gwexpy:Spectrogram",
    ),
    "gwexpy.timeseries.collections.TimeSeriesDict": (
        "gwexpy.timeseries:TimeSeriesDict",
        "gwexpy:TimeSeriesDict",
    ),
    "gwexpy.timeseries.collections.TimeSeriesList": (
        "gwexpy.timeseries:TimeSeriesList",
        "gwexpy:TimeSeriesList",
    ),
    "gwexpy.timeseries.timeseries.TimeSeries": (
        "gwexpy.timeseries.timeseries:TimeSeries",
        "gwexpy.timeseries:TimeSeries",
        "gwexpy:TimeSeries",
    ),
    "gwexpy.types.array.Array": ("gwexpy.types.array:Array", "gwexpy.types:Array"),
    "gwexpy.types.array2d.Array2D": (
        "gwexpy.types.array2d:Array2D",
        "gwexpy.types:Array2D",
    ),
    "gwexpy.types.array3d.Array3D": (
        "gwexpy.types.array3d:Array3D",
        "gwexpy.types:Array3D",
    ),
    "gwexpy.types.array4d.Array4D": (
        "gwexpy.types.array4d:Array4D",
        "gwexpy.types:Array4D",
    ),
    "gwexpy.types.plane2d.Plane2D": (
        "gwexpy.types.plane2d:Plane2D",
        "gwexpy.types:Plane2D",
    ),
    "gwexpy.types.series.Series": ("gwexpy.types.series:Series",),
}


def _load_audit_module() -> ModuleType:
    assert SCRIPT.is_file(), f"missing inventory CLI: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("audit_gwpy_overrides", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest() -> dict[str, Any]:
    assert MANIFEST.is_file(), f"missing inventory manifest: {MANIFEST}"
    loaded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _current_oracle_argument() -> str:
    current = version("gwpy")
    assert current in SUPPORTED_GWPY
    return f"{current}=@current"


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )


def _make_present_case_pending(
    manifest: dict[str, Any], case: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Turn one present terminal case into an exact synthetic pending case."""

    if case is None:
        case = next(
            item
            for item in manifest["cases"]
            if item["counterpart_present"] and item["state"] == "fixed"
        )
    case["state"] = "differential-required"
    case["fixture"] = "__pending_differential__"
    case["case_key"] = "/".join(
        (
            case["public_class"],
            case["member"],
            case["gwpy_version"],
            case["fixture"],
        )
    )
    case["comparator"] = {"name": "pending"}
    case["evidence"] = {
        "behavior": [],
        "oracle_projection_digest": manifest["oracle_projections"][
            case["gwpy_version"]
        ]["digest"],
    }
    case["observations"] = {
        "gwexpy": {"outcome": "pending"},
        "gwpy": {"outcome": "pending"},
    }
    case["issues"] = ["#639"]
    case["owner"] = "v0.2.3-compatibility-audit"
    return case


def _transition_pending_case(
    manifest: dict[str, Any],
    state: str = "no-finding",
    case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if case is None:
        case = _make_present_case_pending(manifest)
    elif case["state"] not in {"differential-required", "unreviewed"}:
        _make_present_case_pending(manifest, case)
    case["state"] = state
    case["fixture"] = "representative-behavior"
    case["case_key"] = "/".join(
        (
            case["public_class"],
            case["member"],
            case["gwpy_version"],
            case["fixture"],
        )
    )
    case["comparator"] = {"name": "exact"}
    case["evidence"] = {
        "behavior": [copy.deepcopy(TEST_REFERENCE)],
        "oracle_projection_digest": manifest["oracle_projections"][
            case["gwpy_version"]
        ]["digest"],
    }
    case["observations"] = {
        "gwexpy": {"outcome": "return", "value": "same"},
        "gwpy": {"outcome": "return", "value": "same"},
    }
    case["issues"] = ["#639"]
    case["owner"] = "v0.2.3-compatibility-audit"
    if state == "GWpy-fails":
        case["observations"]["gwpy"] = {
            "exception_class": "ValueError",
            "outcome": "exception",
        }
    elif state == "fixed":
        case["issues"].append("#640")
        case["evidence"].update(
            {
                "green_test": copy.deepcopy(TEST_REFERENCE),
                "pre_fix_mismatch": {
                    "gwexpy": {"outcome": "return", "value": "before"},
                    "gwpy": {"outcome": "return", "value": "upstream"},
                    "reference": TEST_REFERENCE["reference"],
                },
            }
        )
    return case


def _refresh_summary(audit: ModuleType, manifest: dict[str, Any]) -> None:
    manifest["summary"] = audit.calculate_summary(
        manifest["cases"], manifest["members"], manifest["oracle_projections"]
    )


def _stored_population(manifest: dict[str, Any]) -> dict[str, Any]:
    """Recreate the source-population input represented by a stored manifest."""

    return {
        "digest": manifest["population_digest"],
        "members": manifest["members"],
        "public_roots": manifest["public_roots"],
    }


def test_terminal_closure_catalog_covers_every_present_logical_member() -> None:
    audit = _load_audit_module()
    manifest = _load_manifest()
    present_member_ids = {
        item["member_id"]
        for item in manifest["oracle_projections"]["4.0.2"]["members"]
        if item["present"]
    }

    assert set(audit.TERMINAL_CLOSURES) == present_member_ids


def test_manifest_builder_expands_terminal_closure_across_both_oracles() -> None:
    audit = _load_audit_module()
    stored = _load_manifest()
    manifest = audit.build_manifest(
        _stored_population(stored), stored["oracle_projections"]
    )
    present_cases = [case for case in manifest["cases"] if case["counterpart_present"]]

    assert len(present_cases) == sum(
        item["present"] for item in stored["oracle_projections"]["4.0.2"]["members"]
    ) * len(SUPPORTED_GWPY)
    assert {case["state"] for case in present_cases} <= {
        "fixed",
        "no-finding",
        "GWpy-fails",
    }
    assert manifest["summary"]["differential-required"] == 0
    assert manifest["summary"]["unreviewed"] == 0
    audit.require_terminal_cases(manifest["cases"])
    audit.validate_manifest(manifest)


@pytest.mark.parametrize(
    ("member_id", "state"),
    [
        ("gwexpy.types.array.Array/__new__", "fixed"),
        (
            "gwexpy.frequencyseries.frequencyseries.FrequencySeries/__new__",
            "no-finding",
        ),
        ("gwexpy.timeseries.collections.TimeSeriesDict/prepend", "fixed"),
        ("gwexpy.timeseries.timeseries.TimeSeries/heterodyne", "fixed"),
        ("gwexpy.plot.plot.Plot/show", "fixed"),
        ("gwexpy.plot.plot.Plot/set", "no-finding"),
        ("gwexpy.types.array.Array/swapaxes", "fixed"),
        ("gwexpy.fields.scalar.ScalarField/transpose", "fixed"),
        ("gwexpy.types.array4d.Array4D/T", "no-finding"),
    ],
)
def test_terminal_closure_catalog_pins_representative_decisions(
    member_id: str, state: str
) -> None:
    audit = _load_audit_module()

    closure = audit.TERMINAL_CLOSURES[member_id]
    assert closure["state"] == state
    assert closure["fixture"] not in {audit.ABSENT_FIXTURE, audit.PENDING_FIXTURE}
    assert closure["behavior"]


def test_terminal_closure_catalog_uses_direct_row_differential_evidence() -> None:
    audit = _load_audit_module()
    transpose_reference = (
        "tests/test_gwpy_override_inventory.py"
        "::test_terminal_array_family_T_matches_gwpy"
    )
    show_reference = (
        "tests/test_gwpy_override_inventory.py"
        "::test_terminal_plot_show_lifecycle_matches_gwpy"
    )

    for public_class in (
        "gwexpy.fields.scalar.ScalarField",
        "gwexpy.types.array.Array",
        "gwexpy.types.array3d.Array3D",
        "gwexpy.types.array4d.Array4D",
    ):
        assert audit.TERMINAL_CLOSURES[f"{public_class}/T"]["behavior"] == (
            transpose_reference,
        )
    for public_class in (
        "gwexpy.plot.field.FieldPlot",
        "gwexpy.plot.plot.Plot",
        "gwexpy.plot.skymap.SkyMap",
    ):
        assert audit.TERMINAL_CLOSURES[f"{public_class}/show"]["behavior"] == (
            show_reference,
        )


@pytest.mark.parametrize(
    ("member_id", "fixture", "reference", "comparator"),
    [
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/crop",
            "third-positional-copy-rejection",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_crop_copy_is_keyword_only_like_gwpy",
            "exact-call-outcome-and-exception-class",
            id="timeseries-crop",
        ),
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/append",
            "gwpy-append-parameter-layout",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_append_parameter_layout_matches_gwpy",
            "exact-parameter-layout",
            id="timeseries-append",
        ),
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/t0",
            "none-epoch-setter",
            "tests/timeseries/test_exact_gps_epoch.py"
            "::test_epoch_setters_accept_none_and_clear_exact_authority[t0]",
            "exact-setter-outcome-and-metadata",
            id="timeseries-t0",
        ),
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/x0",
            "none-epoch-setter",
            "tests/timeseries/test_exact_gps_epoch.py"
            "::test_epoch_setters_accept_none_and_clear_exact_authority[x0]",
            "exact-setter-outcome-and-metadata",
            id="timeseries-x0",
        ),
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/spectrogram",
            "gwpy-spectrogram-parameter-layout",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_spectrogram_parameter_layout_matches_gwpy",
            "exact-parameter-layout",
            id="timeseries-spectrogram",
        ),
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/spectrogram2",
            "gwpy-spectrogram2-parameter-layout",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseries_spectrogram2_parameter_layout_matches_gwpy",
            "exact-parameter-layout",
            id="timeseries-spectrogram2",
        ),
        pytest.param(
            "gwexpy.timeseries.collections.TimeSeriesDict/crop",
            "third-positional-copy-rejection",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseriesdict_crop_copy_is_keyword_only_like_gwpy",
            "exact-call-outcome-and-exception-class",
            id="timeseriesdict-crop",
        ),
        pytest.param(
            "gwexpy.timeseries.collections.TimeSeriesDict/append",
            "invalid-copy-key-mapping",
            "tests/timeseries/test_gwpy_override_terminal_compat.py"
            "::test_timeseriesdict_append_invalid_copy_key_matches_gwpy_without_mutation",
            "exact-call-outcome-mutation-and-exception-class",
            id="timeseriesdict-append",
        ),
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/resample",
            "gwpy-resample-parameter-layout",
            "tests/timeseries/test_gwpy_audit_signal_compat.py"
            "::test_resample_signature_keeps_gwpy_positional_layout",
            "exact-parameter-layout",
            id="timeseries-resample",
        ),
    ],
)
def test_terminal_fixed_timeseries_evidence_uses_the_defect_fixture(
    member_id: str,
    fixture: str,
    reference: str,
    comparator: str,
) -> None:
    audit = _load_audit_module()
    closure = audit.TERMINAL_CLOSURES[member_id]

    assert closure["state"] == "fixed"
    assert closure["fixture"] == fixture
    assert closure["behavior"] == (reference,)
    assert closure["comparator"] == comparator


@pytest.mark.parametrize(
    ("member", "reference"),
    [
        pytest.param(
            member,
            "tests/timeseries/test_spectral_gwpy_phase3_compat.py"
            "::test_true_irregular_axis_preserves_parent_failure_class"
            f"[{member}-seconds]",
            id=member,
        )
        for member in ("fft", "psd", "asd", "coherence")
    ],
)
def test_terminal_spectral_rows_use_the_concrete_fixed_failure_fixture(
    member: str, reference: str
) -> None:
    audit = _load_audit_module()
    closure = audit.TERMINAL_CLOSURES[
        f"gwexpy.timeseries.timeseries.TimeSeries/{member}"
    ]
    fields = audit._terminal_case_fields(closure, "d" * 64)

    assert closure["state"] == "fixed"
    assert closure["fixture"] == "true-irregular-seconds"
    assert closure["behavior"] == (reference,)
    assert closure["comparator"] == "exact-failure-outcome-and-exception-class"
    assert fields["observations"] == {
        "gwexpy": {"exception_class": "AttributeError", "outcome": "exception"},
        "gwpy": {"exception_class": "AttributeError", "outcome": "exception"},
    }
    mismatch = fields["evidence"]["pre_fix_mismatch"]
    assert {
        side: {key: value for key, value in mismatch[side].items() if key != "detail"}
        for side in ("gwexpy", "gwpy")
    } == {
        "gwexpy": {"exception_class": "ValueError", "outcome": "exception"},
        "gwpy": {"exception_class": "AttributeError", "outcome": "exception"},
    }
    assert "upstream support" in mismatch["gwexpy"]["detail"]


@pytest.mark.parametrize(
    ("member_id", "current", "before"),
    [
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/crop",
            {
                "gwexpy": {"exception_class": "TypeError", "outcome": "exception"},
                "gwpy": {"exception_class": "TypeError", "outcome": "exception"},
            },
            {
                "gwexpy": {"outcome": "return"},
                "gwpy": {"exception_class": "TypeError", "outcome": "exception"},
            },
            id="timeseries-crop",
        ),
        pytest.param(
            "gwexpy.timeseries.collections.TimeSeriesDict/crop",
            {
                "gwexpy": {"exception_class": "TypeError", "outcome": "exception"},
                "gwpy": {"exception_class": "TypeError", "outcome": "exception"},
            },
            {
                "gwexpy": {"outcome": "return"},
                "gwpy": {"exception_class": "TypeError", "outcome": "exception"},
            },
            id="timeseriesdict-crop",
        ),
        pytest.param(
            "gwexpy.timeseries.collections.TimeSeriesDict/append",
            {
                "gwexpy": {"exception_class": "ValueError", "outcome": "exception"},
                "gwpy": {"exception_class": "ValueError", "outcome": "exception"},
            },
            {
                "gwexpy": {"outcome": "return"},
                "gwpy": {"exception_class": "ValueError", "outcome": "exception"},
            },
            id="timeseriesdict-append",
        ),
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/t0",
            {
                "gwexpy": {"outcome": "return"},
                "gwpy": {"outcome": "return"},
            },
            {
                "gwexpy": {"exception_class": "TypeError", "outcome": "exception"},
                "gwpy": {"outcome": "return"},
            },
            id="timeseries-t0",
        ),
        pytest.param(
            "gwexpy.timeseries.timeseries.TimeSeries/x0",
            {
                "gwexpy": {"outcome": "return"},
                "gwpy": {"outcome": "return"},
            },
            {
                "gwexpy": {"exception_class": "TypeError", "outcome": "exception"},
                "gwpy": {"outcome": "return"},
            },
            id="timeseries-x0",
        ),
    ],
)
def test_terminal_fixed_observations_describe_the_selected_behavior(
    member_id: str,
    current: dict[str, dict[str, str]],
    before: dict[str, dict[str, str]],
) -> None:
    audit = _load_audit_module()
    fields = audit._terminal_case_fields(audit.TERMINAL_CLOSURES[member_id], "d" * 64)

    assert fields["observations"] == current
    mismatch = fields["evidence"]["pre_fix_mismatch"]
    assert {
        side: {key: value for key, value in mismatch[side].items() if key != "detail"}
        for side in ("gwexpy", "gwpy")
    } == before


def test_terminal_cadence_fixture_name_matches_selected_evidence_scope() -> None:
    audit = _load_audit_module()

    for member in ("dt", "dx"):
        closure = audit.TERMINAL_CLOSURES[
            f"gwexpy.timeseries.timeseries.TimeSeries/{member}"
        ]
        assert closure["fixture"] == "cadence-set-copy-and-slice"


@pytest.mark.parametrize(
    ("module_name", "class_name", "shape"),
    [
        pytest.param(
            "gwexpy.fields.scalar", "ScalarField", (2, 2, 3, 4), id="ScalarField"
        ),
        pytest.param("gwexpy.types.array", "Array", (2, 3, 4), id="Array"),
        pytest.param("gwexpy.types.array3d", "Array3D", (2, 3, 4), id="Array3D"),
        pytest.param("gwexpy.types.array4d", "Array4D", (2, 2, 3, 4), id="Array4D"),
    ],
)
def test_terminal_array_family_T_matches_gwpy(
    module_name: str, class_name: str, shape: tuple[int, ...]
) -> None:
    import numpy as np
    from astropy import units as u
    from gwpy.types import Array as GwpyArray

    actual_type = vars(importlib.import_module(module_name))[class_name]
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    actual_input = actual_type(values.copy(), unit=u.m, name="transpose-source")
    expected_input = GwpyArray(values.copy(), unit=u.m, name="transpose-source")

    actual = actual_input.T
    expected = expected_input.T

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    np.testing.assert_array_equal(actual.value, expected.value)
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )


@pytest.mark.parametrize("class_name", ["Plot", "FieldPlot", "SkyMap"])
def test_terminal_plot_show_lifecycle_matches_gwpy(
    class_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from gwpy.plot import Plot as GwpyPlot
    from gwpy.timeseries import TimeSeries as GwpyTimeSeries
    from matplotlib.figure import Figure

    # Dynamic access is intentional in this behavior test; only the static
    # inventory scanner is prohibited from executing package ``__getattr__``.
    actual_type = getattr(importlib.import_module("gwexpy.plot"), class_name)
    source = GwpyTimeSeries(np.arange(4.0), t0=0, dt=1, name="show-source")
    actual = actual_type(source.copy())
    expected = GwpyPlot(source.copy())
    original_close = plt.close
    show_calls: list[tuple[Figure, bool]] = []
    close_calls: list[Figure | None] = []

    def record_show(figure: Figure, warn: bool = True) -> None:
        show_calls.append((figure, warn))

    monkeypatch.setattr(Figure, "show", record_show)
    monkeypatch.setattr(plt, "close", lambda figure=None: close_calls.append(figure))
    try:
        actual_result = actual.show(False, False)
        expected_result = expected.show(False, False)
        assert actual_result is expected_result is None
        assert show_calls == [(actual, False), (expected, False)]
        assert close_calls == []
    finally:
        original_close(actual)
        original_close(expected)


def test_manifest_has_terminal_structural_population_and_canonical_json() -> None:
    audit = _load_audit_module()
    manifest = _load_manifest()

    assert manifest["schema"] == SCHEMA
    assert manifest["policy"] == {
        "behavioral_owner": "v0.2.3-compatibility-audit",
        "fixture_key": ["public_class", "member", "gwpy_version", "fixture"],
        "implementation_base": IMPLEMENTATION_BASE,
        "member_walk_rule": (
            "first effective vars(owner) binding in the GWexpy MRO prefix before "
            "the first GWpy class; public callable/descriptors plus __new__/__init__"
        ),
        "oracle_versions": ["4.0.1", "4.0.2"],
        "pristine_oracle_rule": (
            "separate -I worker; sanitized PYTHONPATH/PYTHONHOME; no GWexpy import; "
            "exact GWpy 4.0.1/4.0.2"
        ),
        "provisional_states": ["unreviewed", "differential-required"],
        "public_root_rule": (
            "byte-sorted gwexpy Python paths; final literal top-level list/tuple "
            "__all__; all-source top-level class route index plus static "
            "vars(module) resolution; two-pass unique canonical-class-name lazy "
            "alias association; canonical GWexpy class identity; internal root "
            "exclusions"
        ),
        "terminal_states": ["fixed", "no-finding", "GWpy-fails", "GWexpy-only"],
        "upstream_dependency_provenance": UPSTREAM_DEPENDENCY_PROVENANCE,
    }
    assert manifest["summary"] == {
        "cases": 1150,
        "constructors": 11,
        "counterpart_absent_per_version": 441,
        "counterpart_implementation_groups": 70,
        "counterpart_present_per_version": 134,
        "differential-required": 0,
        "fixed": 224,
        "GWexpy-only": 882,
        "logical_members": 575,
        "no-finding": 44,
        "public_roots": 16,
        "GWpy-fails": 0,
        "unreviewed": 0,
    }
    assert MANIFEST.read_text(encoding="utf-8") == audit.canonical_manifest_json(
        manifest
    )
    assert not any(
        isinstance(value, str) and value.startswith("/")
        for value in audit.walk_manifest_values(manifest)
    )
    assert not re.search(r"0x[0-9a-fA-F]+", MANIFEST.read_text(encoding="utf-8"))
    assert {
        item["public_class"]: tuple(item["exports"])
        for item in manifest["public_roots"]
    } == EXPECTED_ROOTS


def test_manifest_matches_current_source_mro_population() -> None:
    audit = _load_audit_module()
    manifest = _load_manifest()

    current = audit.build_source_population(ROOT)
    audit.validate_manifest(manifest)
    audit.validate_source_population(manifest, current)
    assert len(current["public_roots"]) == 16
    assert len(current["members"]) == 575


def test_lazy_skymap_export_alias_is_associated_without_getattr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_audit_module()
    plot_module = importlib.import_module("gwexpy.plot")
    monkeypatch.delitem(vars(plot_module), "SkyMap", raising=False)

    def reject_dynamic_lookup(name: str) -> object:
        raise AssertionError(f"dynamic lookup executed for {name}")

    monkeypatch.setattr(plot_module, "__getattr__", reject_dynamic_lookup)
    discovered = {
        f"{value.__module__}.{value.__qualname__}": exports
        for value, exports in audit.discover_public_classes(ROOT)
    }

    assert discovered["gwexpy.plot.skymap.SkyMap"] == (
        "gwexpy.plot.skymap:SkyMap",
        "gwexpy.plot:SkyMap",
    )


def test_lazy_class_alias_routing_fails_closed_when_not_unique() -> None:
    audit = _load_audit_module()

    class First:
        pass

    class Second:
        pass

    First.__name__ = "Duplicate"
    Second.__name__ = "Duplicate"
    with pytest.raises(audit.InventoryError, match="missing lazy class export route"):
        audit._select_unique_lazy_class("Missing", [First])
    with pytest.raises(audit.InventoryError, match="ambiguous lazy class export route"):
        audit._select_unique_lazy_class("Duplicate", [First, Second])


def _write_synthetic_export_module(repository: Path, module: str, source: str) -> None:
    path = repository.joinpath(*module.split("."))
    path = path.with_suffix(".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _synthetic_gwpy_class(name: str, module: str) -> type[Any]:
    from gwpy.types import Array as GwpyArray

    return type(name, (GwpyArray,), {"__module__": module})


def test_lazy_only_class_is_found_in_full_explicit_module_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit_module()
    _write_synthetic_export_module(
        tmp_path,
        "gwexpy.public",
        '__all__ = ["Ghost", "helper", "ALL_CAPS"]\n',
    )
    _write_synthetic_export_module(
        tmp_path,
        "gwexpy.hidden",
        '__all__ = ["helper"]\nclass Ghost(Array):\n    pass\n',
    )
    public = ModuleType("gwexpy.public")
    setattr(public, "__all__", ["Ghost", "helper", "ALL_CAPS"])

    def reject_dynamic_lookup(name: str) -> object:
        raise AssertionError(f"dynamic lookup executed for {name}")

    setattr(public, "__getattr__", reject_dynamic_lookup)
    hidden = ModuleType("gwexpy.hidden")
    hidden.helper = object()  # type: ignore[attr-defined]
    ghost = _synthetic_gwpy_class("Ghost", "gwexpy.hidden")
    hidden.Ghost = ghost  # type: ignore[attr-defined]
    modules = {public.__name__: public, hidden.__name__: hidden}
    monkeypatch.setattr(audit.importlib, "import_module", modules.__getitem__)

    discovered = audit.discover_public_classes(tmp_path)

    assert discovered == [(ghost, ("gwexpy.public:Ghost",))]


def test_lazy_only_class_is_found_in_module_without_literal_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit_module()
    _write_synthetic_export_module(tmp_path, "gwexpy.public", '__all__ = ["Ghost"]\n')
    _write_synthetic_export_module(
        tmp_path,
        "gwexpy.hidden",
        "class Ghost(Array):\n    pass\n",
    )
    public = ModuleType("gwexpy.public")
    setattr(public, "__all__", ["Ghost"])

    def reject_dynamic_lookup(name: str) -> object:
        raise AssertionError(f"dynamic lookup executed for {name}")

    setattr(public, "__getattr__", reject_dynamic_lookup)
    hidden = ModuleType("gwexpy.hidden")
    ghost = _synthetic_gwpy_class("Ghost", "gwexpy.hidden")
    hidden.Ghost = ghost  # type: ignore[attr-defined]
    modules = {public.__name__: public, hidden.__name__: hidden}
    monkeypatch.setattr(audit.importlib, "import_module", modules.__getitem__)

    discovered = audit.discover_public_classes(tmp_path)

    assert discovered == [(ghost, ("gwexpy.public:Ghost",))]


def test_lazy_only_class_route_ambiguity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit_module()
    _write_synthetic_export_module(tmp_path, "gwexpy.public", '__all__ = ["Ghost"]\n')
    modules: dict[str, ModuleType] = {"gwexpy.public": ModuleType("gwexpy.public")}
    modules["gwexpy.public"].__all__ = ["Ghost"]  # type: ignore[attr-defined]
    for suffix in ("one", "two"):
        module_name = f"gwexpy.hidden_{suffix}"
        _write_synthetic_export_module(
            tmp_path,
            module_name,
            '__all__ = ["helper"]\nclass Ghost(Array):\n    pass\n',
        )
        module = ModuleType(module_name)
        module.helper = object()  # type: ignore[attr-defined]
        module.Ghost = _synthetic_gwpy_class("Ghost", module_name)  # type: ignore[attr-defined]
        modules[module_name] = module
    monkeypatch.setattr(audit.importlib, "import_module", modules.__getitem__)

    with pytest.raises(audit.InventoryError, match="ambiguous lazy class export route"):
        audit.discover_public_classes(tmp_path)


def test_missing_static_lazy_class_route_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit_module()
    _write_synthetic_export_module(tmp_path, "gwexpy.public", '__all__ = ["Ghost"]\n')
    _write_synthetic_export_module(
        tmp_path,
        "gwexpy.hidden",
        '__all__ = ["helper"]\nclass Ghost(Array):\n    pass\n',
    )
    public = ModuleType("gwexpy.public")
    setattr(public, "__all__", ["Ghost"])
    hidden = ModuleType("gwexpy.hidden")
    hidden.helper = object()  # type: ignore[attr-defined]
    modules = {public.__name__: public, hidden.__name__: hidden}
    monkeypatch.setattr(audit.importlib, "import_module", modules.__getitem__)

    with pytest.raises(audit.InventoryError, match="missing lazy class export route"):
        audit.discover_public_classes(tmp_path)


def test_missing_lazy_class_with_no_static_route_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit_module()
    _write_synthetic_export_module(tmp_path, "gwexpy.public", '__all__ = ["Ghost"]\n')
    public = ModuleType("gwexpy.public")
    setattr(public, "__all__", ["Ghost"])
    monkeypatch.setattr(
        audit.importlib, "import_module", {public.__name__: public}.__getitem__
    )

    with pytest.raises(audit.InventoryError, match="missing lazy class export route"):
        audit.discover_public_classes(tmp_path)


def test_present_nonclass_export_masks_a_lazy_class_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit_module()
    _write_synthetic_export_module(tmp_path, "gwexpy.public", '__all__ = ["Ghost"]\n')
    _write_synthetic_export_module(
        tmp_path,
        "gwexpy.hidden",
        '__all__ = ["helper"]\nclass Ghost(Array):\n    pass\n',
    )
    public = ModuleType("gwexpy.public")
    setattr(public, "__all__", ["Ghost"])
    public.Ghost = 42  # type: ignore[attr-defined]
    hidden = ModuleType("gwexpy.hidden")
    hidden.helper = object()  # type: ignore[attr-defined]
    hidden.Ghost = _synthetic_gwpy_class("Ghost", "gwexpy.hidden")  # type: ignore[attr-defined]
    modules = {public.__name__: public, hidden.__name__: hidden}
    monkeypatch.setattr(audit.importlib, "import_module", modules.__getitem__)

    assert audit.discover_public_classes(tmp_path) == []


def test_literal_all_uses_the_final_top_level_assignment() -> None:
    audit = _load_audit_module()
    tree = audit.ast.parse('__all__ = ["Old"]\n__all__ = ("New",)\n')

    assert audit._literal_all_names(tree) == ("New",)


def test_literal_all_annotation_without_value_does_not_rebind() -> None:
    audit = _load_audit_module()
    tree = audit.ast.parse('__all__ = ["Kept"]\n__all__: list[str]\n')

    assert audit._literal_all_names(tree) == ("Kept",)


def test_current_supported_version_pristine_projection_matches_manifest() -> None:
    result = _run_cli(
        "--check",
        "--manifest",
        str(MANIFEST),
        "--oracle-python",
        _current_oracle_argument(),
    )
    assert result.returncode == 0, result.stderr
    assert "inventory check passed" in result.stdout


def test_ordinary_and_terminal_checks_accept_closed_manifest() -> None:
    ordinary = _run_cli(
        "--check",
        "--manifest",
        str(MANIFEST),
        "--oracle-python",
        _current_oracle_argument(),
    )
    assert ordinary.returncode == 0, ordinary.stderr

    terminal = _run_cli(
        "--check",
        "--require-terminal",
        "--manifest",
        str(MANIFEST),
        "--oracle-python",
        _current_oracle_argument(),
    )
    assert terminal.returncode == 0, terminal.stderr
    assert "inventory check passed" in terminal.stdout


def test_terminal_cli_rejects_schema_valid_catalog_and_evidence_drift(
    tmp_path: Path,
) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = next(item for item in manifest["cases"] if item["state"] == "fixed")
    case["state"] = "no-finding"
    case["evidence"] = {
        "behavior": [
            {
                "reference": (
                    "tests/test_gwpy_override_inventory.py"
                    "::test_selector_that_does_not_exist"
                )
            }
        ],
        "oracle_projection_digest": manifest["oracle_projections"][
            case["gwpy_version"]
        ]["digest"],
    }
    _refresh_summary(audit, manifest)
    changed = tmp_path / "catalog-drift.json"
    changed.write_text(audit.canonical_manifest_json(manifest), encoding="utf-8")

    result = _run_cli(
        "--check",
        "--require-terminal",
        "--manifest",
        str(changed),
        "--oracle-python",
        _current_oracle_argument(),
    )

    assert result.returncode == 2
    assert "manifest/catalog drift" in result.stderr


def test_catalog_binding_compares_canonical_json_not_python_numeric_equality() -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    manifest["summary"]["cases"] = float(manifest["summary"]["cases"])

    with pytest.raises(audit.InventoryError, match="manifest/catalog drift"):
        audit.validate_catalog_binding(manifest, _stored_population(manifest))


def test_manifest_evidence_selector_set_is_frozen_and_recursion_safe() -> None:
    audit = _load_audit_module()

    selectors = audit.manifest_evidence_selectors(_load_manifest())

    assert len(selectors) == 58
    assert list(selectors) == sorted(selectors, key=lambda item: item.encode("utf-8"))
    assert {
        selector
        for selector in selectors
        if selector.startswith("tests/test_gwpy_override_inventory.py::")
    } == {
        "tests/test_gwpy_override_inventory.py"
        "::test_terminal_array_family_T_matches_gwpy",
        "tests/test_gwpy_override_inventory.py"
        "::test_terminal_plot_show_lifecycle_matches_gwpy",
    }


def test_evidence_selector_extraction_includes_each_distinct_reference_channel() -> (
    None
):
    audit = _load_audit_module()
    references = (
        "tests/frequencyseries/test_bifrequencymap_gwpy_compat.py"
        "::test_crop_common_route_matches_gwpy",
        "tests/frequencyseries/test_bifrequencymap_gwpy_compat.py"
        "::test_diagonal_common_numeric_routes_match_gwpy",
        "tests/frequencyseries/test_bifrequencymap_gwpy_compat.py"
        "::test_plot_common_routes_match_gwpy_artist_source_and_label",
    )
    manifest = {
        "cases": [
            {
                "evidence": {
                    "behavior": [{"reference": references[0]}],
                    "green_test": {"reference": references[1]},
                    "pre_fix_mismatch": {"reference": references[2]},
                }
            }
        ]
    }

    assert audit.manifest_evidence_selectors(manifest) == references


def test_evidence_collection_rejects_an_existing_file_with_a_missing_node() -> None:
    audit = _load_audit_module()
    missing = (
        "tests/test_gwpy_override_inventory.py::test_selector_that_does_not_exist",
    )

    with pytest.raises(audit.InventoryError, match="evidence collection failed"):
        audit.run_evidence_pytest(ROOT, missing, execute=False)


def test_evidence_collection_uses_bounded_argv_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_audit_module()
    selector = "tests/test_gwpy_override_inventory.py::test_terminal_array_family_T_matches_gwpy"
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="collected\n", stderr="")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    interpreter = str(ROOT / "synthetic-python")
    output = audit.run_evidence_pytest(
        ROOT, (selector,), execute=False, python_executable=interpreter
    )

    assert output == "collected\n"
    assert observed["command"] == [
        interpreter,
        "-m",
        "pytest",
        "--collect-only",
        "-qq",
        "--maxfail=1",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "--",
        selector,
    ]
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["cwd"] == ROOT
    assert isinstance(observed["kwargs"]["timeout"], int)
    assert "capture_output" not in observed["kwargs"]
    assert observed["kwargs"]["stdout"].closed is True
    assert observed["kwargs"]["stderr"].closed is True


def _write_evidence_junit(path: Path, *, cases: int, skipped: int = 0) -> None:
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for index in range(cases):
        testcase = ET.SubElement(
            suite, "testcase", classname="evidence", name=f"test_{index}"
        )
        if index < skipped:
            ET.SubElement(testcase, "skipped")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_evidence_junit_requires_exact_reviewed_count_and_no_skips(
    tmp_path: Path,
) -> None:
    audit = _load_audit_module()
    accepted = tmp_path / "accepted.xml"
    _write_evidence_junit(accepted, cases=383)
    audit.validate_evidence_junit(accepted)

    wrong_count = tmp_path / "wrong-count.xml"
    _write_evidence_junit(wrong_count, cases=382)
    with pytest.raises(audit.InventoryError, match="expected 383 cases, got 382"):
        audit.validate_evidence_junit(wrong_count)

    skipped = tmp_path / "skipped.xml"
    _write_evidence_junit(skipped, cases=383, skipped=1)
    with pytest.raises(audit.InventoryError, match="skipped=1"):
        audit.validate_evidence_junit(skipped)


def test_evidence_execution_rejects_recursive_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_audit_module()
    monkeypatch.setenv("GWEXPY_OVERRIDE_EVIDENCE_CHILD", "1")

    with pytest.raises(audit.InventoryError, match="recursive evidence execution"):
        audit.run_evidence_pytest(
            ROOT,
            (
                "tests/test_gwpy_override_inventory.py"
                "::test_terminal_array_family_T_matches_gwpy",
            ),
            execute=True,
        )


def test_execute_evidence_requires_terminal_check() -> None:
    result = _run_cli(
        "--check",
        "--execute-evidence",
        "--manifest",
        str(MANIFEST),
        "--oracle-python",
        _current_oracle_argument(),
    )

    assert result.returncode == 2
    assert "--execute-evidence requires --require-terminal" in result.stderr


def test_terminal_cli_can_execute_every_catalog_evidence_selector() -> None:
    result = _run_cli(
        "--check",
        "--require-terminal",
        "--execute-evidence",
        "--manifest",
        str(MANIFEST),
        "--oracle-python",
        _current_oracle_argument(),
    )

    assert result.returncode == 0, result.stderr
    assert "383 passed" in result.stdout
    assert "evidence execution passed: selectors=58, cases=383" in result.stdout


@pytest.mark.parametrize(
    "oracle_arguments, expected",
    [
        (("9.9.9=@current",), "unknown oracle version: 9.9.9"),
        (
            (f"{version('gwpy')}=@current", f"{version('gwpy')}=@current"),
            f"duplicate oracle version: {version('gwpy')}",
        ),
        (
            (
                f"{next(item for item in SUPPORTED_GWPY if item != version('gwpy'))}"
                "=@current",
            ),
            "oracle version mismatch",
        ),
    ],
)
def test_bad_oracle_arguments_and_version_mismatches_fail_closed(
    oracle_arguments: tuple[str, ...], expected: str
) -> None:
    command = ["--check", "--manifest", str(MANIFEST)]
    for argument in oracle_arguments:
        command.extend(("--oracle-python", argument))
    result = _run_cli(*command)
    assert result.returncode != 0
    assert expected in result.stderr


def test_write_requires_exactly_both_supported_oracles(tmp_path: Path) -> None:
    result = _run_cli(
        "--write",
        "--manifest",
        str(tmp_path / "inventory.json"),
        "--oracle-python",
        _current_oracle_argument(),
    )
    assert result.returncode != 0
    assert "--write requires exactly GWpy 4.0.1 and 4.0.2" in result.stderr


def test_pristine_worker_reports_all_four_collection_injections_absent() -> None:
    manifest = _load_manifest()
    injected = {
        ("gwexpy.timeseries.collections.TimeSeriesDict", "csd_matrix"),
        ("gwexpy.timeseries.collections.TimeSeriesDict", "coherence_matrix"),
        ("gwexpy.timeseries.collections.TimeSeriesList", "csd_matrix"),
        ("gwexpy.timeseries.collections.TimeSeriesList", "coherence_matrix"),
    }
    for oracle_version in SUPPORTED_GWPY:
        projection = manifest["oracle_projections"][oracle_version]
        observed = {
            (item["public_class"], item["member"]): item["present"]
            for item in projection["members"]
            if (item["public_class"], item["member"]) in injected
        }
        assert observed == {key: False for key in injected}


def test_golden_constructor_set_and_unified_io_counterparts_are_frozen() -> None:
    manifest = _load_manifest()
    constructors = {
        (item["public_class"], item["member"])
        for item in manifest["members"]
        if item["constructor"]
    }
    assert constructors == {
        ("gwexpy.fields.scalar.ScalarField", "__new__"),
        ("gwexpy.frequencyseries.frequencyseries.FrequencySeries", "__new__"),
        ("gwexpy.plot.field.FieldPlot", "__init__"),
        ("gwexpy.plot.plot.Plot", "__init__"),
        ("gwexpy.plot.skymap.SkyMap", "__init__"),
        ("gwexpy.timeseries.timeseries.TimeSeries", "__new__"),
        ("gwexpy.types.array.Array", "__new__"),
        ("gwexpy.types.array2d.Array2D", "__new__"),
        ("gwexpy.types.array3d.Array3D", "__new__"),
        ("gwexpy.types.array4d.Array4D", "__new__"),
        ("gwexpy.types.plane2d.Plane2D", "__new__"),
    }
    unified = {
        ("gwexpy.frequencyseries.frequencyseries.FrequencySeries", "read"),
        ("gwexpy.frequencyseries.frequencyseries.FrequencySeries", "write"),
        ("gwexpy.timeseries.collections.TimeSeriesDict", "read"),
        ("gwexpy.timeseries.collections.TimeSeriesDict", "write"),
        ("gwexpy.timeseries.timeseries.TimeSeries", "read"),
        ("gwexpy.timeseries.timeseries.TimeSeries", "write"),
    }
    for oracle_version in SUPPORTED_GWPY:
        observed = {
            (item["public_class"], item["member"]): item["kind"]
            for item in manifest["oracle_projections"][oracle_version]["members"]
            if (item["public_class"], item["member"]) in unified
        }
        assert observed == {key: "unified-read-write" for key in unified}


def test_live_worker_proves_isolation_and_exact_current_version() -> None:
    audit = _load_audit_module()
    population = audit.build_source_population(ROOT)
    projection = audit.run_pristine_oracle(
        SCRIPT,
        version("gwpy"),
        sys.executable,
        population["members"][:1],
    )
    assert projection["gwpy_version"] == version("gwpy")
    assert projection["isolation"] == {
        "cwd_matches_expected": True,
        "gwexpy_absent_at_end": True,
        "gwexpy_absent_at_start": True,
        "isolated_flag": True,
        "no_user_site": True,
    }


def test_oracle_first_non_callable_binding_masks_callable_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_audit_module()
    for name in tuple(sys.modules):
        if name == "gwexpy" or name.startswith("gwexpy."):
            monkeypatch.delitem(sys.modules, name)

    class CallableBase:
        def masked(self) -> None:
            return None

    class NonCallableMask(CallableBase):
        masked = None

    CallableBase.__module__ = "gwpy.synthetic_masking"
    NonCallableMask.__module__ = "gwpy.synthetic_masking"
    synthetic = ModuleType("gwpy.synthetic_masking")
    synthetic.NonCallableMask = NonCallableMask
    monkeypatch.setitem(sys.modules, synthetic.__name__, synthetic)

    projection = audit.build_oracle_projection(
        version("gwpy"),
        [
            {
                "counterpart_class": "gwpy.synthetic_masking.NonCallableMask",
                "member": "masked",
                "member_id": "gwexpy.synthetic.Public/masked",
                "public_class": "gwexpy.synthetic.Public",
            }
        ],
    )

    assert projection["members"][0]["present"] is False


@pytest.mark.parametrize("mutation", ["orphan", "duplicate", "unsorted"])
def test_manifest_reference_and_order_defects_fail_closed(mutation: str) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    if mutation == "orphan":
        manifest["cases"][0]["member_id"] = "missing/member"
    elif mutation == "duplicate":
        manifest["cases"][1] = copy.deepcopy(manifest["cases"][0])
    else:
        manifest["cases"][0], manifest["cases"][1] = (
            manifest["cases"][1],
            manifest["cases"][0],
        )
    with pytest.raises(audit.InventoryError):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize("mutation", ["summary", "digest", "presence", "group"])
def test_summary_digest_and_presence_inconsistency_fail_closed(mutation: str) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    if mutation == "summary":
        manifest["summary"]["logical_members"] += 1
    elif mutation == "digest":
        manifest["oracle_projections"]["4.0.1"]["digest"] = "0" * 64
    elif mutation == "presence":
        manifest["cases"][0]["counterpart_present"] = not manifest["cases"][0][
            "counterpart_present"
        ]
    else:
        case = next(item for item in manifest["cases"] if item["counterpart_present"])
        case["implementation_group"] = "implementation-wrong"
    with pytest.raises(audit.InventoryError):
        audit.validate_manifest(manifest)


def test_terminal_transition_with_case_derived_summary_is_valid() -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _make_present_case_pending(manifest)
    _refresh_summary(audit, manifest)
    assert manifest["summary"]["differential-required"] == 1

    _transition_pending_case(manifest, case=case)
    _refresh_summary(audit, manifest)

    audit.validate_manifest(manifest)
    assert manifest["summary"]["no-finding"] == 45
    assert manifest["summary"]["differential-required"] == 0


def test_multiple_fixtures_for_one_member_version_validate() -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    original = next(item for item in manifest["cases"] if item["counterpart_present"])
    additional = _transition_pending_case(manifest, case=copy.deepcopy(original))
    manifest["cases"].append(additional)
    manifest["cases"].sort(key=audit._case_sort_key)
    manifest["summary"] = audit.calculate_summary(
        manifest["cases"], manifest["members"], manifest["oracle_projections"]
    )

    audit.validate_manifest(manifest)
    assert (
        sum(
            case["member_id"] == original["member_id"]
            and case["gwpy_version"] == original["gwpy_version"]
            for case in manifest["cases"]
        )
        == 2
    )


def test_all_provisional_cases_can_reach_terminal_gate() -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    present_cases = [case for case in manifest["cases"] if case["counterpart_present"]]
    for case in present_cases:
        _make_present_case_pending(manifest, case)
    _refresh_summary(audit, manifest)
    assert manifest["summary"]["differential-required"] == len(present_cases) == 268

    for case in present_cases:
        _transition_pending_case(manifest, case=case)
    manifest["cases"].sort(key=audit._case_sort_key)
    _refresh_summary(audit, manifest)

    audit.validate_manifest(manifest)
    audit.require_terminal_cases(manifest["cases"])
    assert manifest["summary"]["differential-required"] == 0
    assert manifest["summary"]["unreviewed"] == 0


def test_fixed_terminal_schema_can_validate() -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _make_present_case_pending(manifest)
    _refresh_summary(audit, manifest)
    _transition_pending_case(manifest, state="fixed", case=case)
    _refresh_summary(audit, manifest)

    audit.validate_manifest(manifest)


def test_terminal_transition_with_stale_summary_fails_closed() -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _make_present_case_pending(manifest)
    _refresh_summary(audit, manifest)
    _transition_pending_case(manifest, case=case)

    with pytest.raises(audit.InventoryError, match="summary mismatch"):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize("mutation", ["extra", "missing", "non-object"])
def test_case_top_level_key_set_is_exact(mutation: str) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = manifest["cases"][0]
    assert set(case) == CASE_KEYS
    if mutation == "non-object":
        manifest["cases"][0] = True
    elif mutation == "extra":
        case["unexpected"] = True
    else:
        del case["issues"]

    with pytest.raises(audit.InventoryError, match="case schema mismatch"):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize("state", ["no-finding", "GWpy-fails"])
def test_behavioral_terminal_common_fields_are_strict(state: str) -> None:
    audit = _load_audit_module()
    mutations = [
        ("owner", None),
        ("owner", ""),
        ("owner", True),
        ("fixture", ""),
        ("fixture", True),
        ("fixture", audit.PENDING_FIXTURE),
        ("issues", []),
        ("issues", ["#640"]),
        ("issues", ["#639", True]),
        ("comparator", True),
        ("comparator", {}),
        ("comparator", {"name": ""}),
        ("comparator", {"name": True}),
    ]
    for field, value in mutations:
        manifest = copy.deepcopy(_load_manifest())
        case = _transition_pending_case(manifest, state=state)
        case[field] = value
        if field == "fixture":
            case["case_key"] = "/".join(
                (
                    case["public_class"],
                    case["member"],
                    case["gwpy_version"],
                    str(value),
                )
            )
        _refresh_summary(audit, manifest)
        with pytest.raises(audit.InventoryError):
            audit.validate_manifest(manifest)


@pytest.mark.parametrize(
    "comparator",
    [
        {"name": "exact", "rtol": 0.0},
        {"name": "approximate"},
        {"name": "approximate", "rtol": 0.0},
        {"name": "approximate", "rtol": -1.0, "atol": 0.0},
        {"name": "approximate", "rtol": 0.0, "atol": -1.0},
        {"name": "approximate", "rtol": True, "atol": 0.0},
        {"name": "approximate", "rtol": 0.0, "atol": False},
        {"name": "approximate", "rtol": float("inf"), "atol": 0.0},
        {"name": "approximate", "rtol": 0.0, "atol": float("nan")},
    ],
)
def test_terminal_comparator_schema_and_tolerances_are_strict(
    comparator: dict[str, object],
) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _transition_pending_case(manifest)
    case["comparator"] = comparator
    _refresh_summary(audit, manifest)

    with pytest.raises(audit.InventoryError):
        audit.validate_manifest(manifest)


def test_approximate_comparator_with_explicit_finite_tolerances_is_valid() -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _transition_pending_case(manifest)
    case["comparator"] = {"name": "approximate", "rtol": 1e-7, "atol": 0}
    manifest["summary"] = audit.calculate_summary(
        manifest["cases"], manifest["members"], manifest["oracle_projections"]
    )

    audit.validate_manifest(manifest)


@pytest.mark.parametrize(
    ("side", "observation"),
    [
        ("gwpy", True),
        ("gwpy", {}),
        ("gwpy", {"outcome": True}),
        ("gwpy", {"outcome": ""}),
        ("gwpy", {"outcome": "pending"}),
        ("gwpy", {"outcome": "exception"}),
        ("gwpy", {"outcome": "exception", "exception_class": ""}),
        ("gwpy", {"outcome": "exception", "exception_class": True}),
    ],
)
def test_terminal_observations_are_typed(side: str, observation: object) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _transition_pending_case(manifest)
    case["observations"][side] = observation
    _refresh_summary(audit, manifest)

    with pytest.raises(audit.InventoryError):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize("keys", [{"gwpy"}, {"gwexpy"}, {"gwpy", "gwexpy", "extra"}])
def test_terminal_observation_sides_are_exact(keys: set[str]) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _transition_pending_case(manifest)
    case["observations"] = {key: {"outcome": "return"} for key in keys}
    _refresh_summary(audit, manifest)

    with pytest.raises(audit.InventoryError, match="observation schema"):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize(
    "behavior",
    [
        [True],
        [{"reference": True}],
        [{"reference": "../outside.py::test_escape"}],
        [{"reference": "/tmp/outside.py::test_escape"}],
        [{"reference": "tests/does-not-exist.py::test_missing"}],
        [{"reference": "tests/test_gwpy_override_inventory.py"}],
        [{"reference": "tests/test_gwpy_override_inventory.py::"}],
        [{"reference": TEST_REFERENCE["reference"], "extra": True}],
    ],
)
def test_behavior_references_are_structured_safe_and_existing(
    behavior: list[object],
) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _transition_pending_case(manifest)
    case["evidence"]["behavior"] = behavior
    _refresh_summary(audit, manifest)

    with pytest.raises(audit.InventoryError, match="reference|behavior"):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize("state", ["no-finding", "GWpy-fails"])
def test_non_fixed_terminal_evidence_key_set_is_exact(state: str) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _transition_pending_case(manifest, state=state)
    case["evidence"]["unexpected"] = True
    _refresh_summary(audit, manifest)

    with pytest.raises(audit.InventoryError, match="evidence schema"):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-pre-fix",
        "missing-green",
        "extra",
        "green-bool",
        "green-missing-file",
        "pre-fix-bool",
        "pre-fix-extra",
        "pre-fix-missing-side",
        "pre-fix-bad-observation",
        "pre-fix-traversal",
        "equal-pre-fix",
    ],
)
def test_fixed_evidence_schema_is_structured_and_exact(mutation: str) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _transition_pending_case(manifest, state="fixed")
    if mutation == "missing-pre-fix":
        del case["evidence"]["pre_fix_mismatch"]
    elif mutation == "missing-green":
        del case["evidence"]["green_test"]
    elif mutation == "extra":
        case["evidence"]["unexpected"] = True
    elif mutation == "green-bool":
        case["evidence"]["green_test"] = True
    elif mutation == "green-missing-file":
        case["evidence"]["green_test"] = {"reference": "tests/missing.py::test_missing"}
    elif mutation == "pre-fix-bool":
        case["evidence"]["pre_fix_mismatch"] = True
    elif mutation == "pre-fix-extra":
        case["evidence"]["pre_fix_mismatch"]["extra"] = True
    elif mutation == "pre-fix-missing-side":
        del case["evidence"]["pre_fix_mismatch"]["gwpy"]
    elif mutation == "pre-fix-bad-observation":
        case["evidence"]["pre_fix_mismatch"]["gwpy"] = True
    elif mutation == "pre-fix-traversal":
        case["evidence"]["pre_fix_mismatch"]["reference"] = "../outside.py::test_escape"
    else:
        observation = {"outcome": "return", "value": "same"}
        case["evidence"]["pre_fix_mismatch"]["gwpy"] = observation
        case["evidence"]["pre_fix_mismatch"]["gwexpy"] = copy.deepcopy(observation)
    _refresh_summary(audit, manifest)

    with pytest.raises(audit.InventoryError):
        audit.validate_manifest(manifest)


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    audit = _load_audit_module()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"first","schema":"second"}\n', encoding="utf-8")
    with pytest.raises(audit.InventoryError, match="duplicate JSON key: schema"):
        audit.load_json_strict(duplicate)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_loader_rejects_non_finite_constants(
    tmp_path: Path, constant: str
) -> None:
    audit = _load_audit_module()
    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text(f'{{"value":{constant}}}\n', encoding="utf-8")
    with pytest.raises(audit.InventoryError, match="non-finite JSON constant"):
        audit.load_json_strict(non_finite)


@pytest.mark.parametrize("number", ["1e999", "-1e999"])
def test_strict_json_loader_rejects_overflowing_floats(
    tmp_path: Path, number: str
) -> None:
    audit = _load_audit_module()
    overflow = tmp_path / "overflow.json"
    overflow.write_text(f'{{"value":{number}}}\n', encoding="utf-8")
    with pytest.raises(audit.InventoryError, match="non-finite JSON float"):
        audit.load_json_strict(overflow)


def test_oracle_worker_stdin_uses_strict_json_decoder() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--oracle-worker"],
        cwd=ROOT,
        env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
        input='{"value":1e999}\n',
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "non-finite JSON float" in result.stderr


def test_oracle_worker_malformed_nested_query_exits_two_without_traceback() -> None:
    payload = {
        "expected_cwd": str(ROOT),
        "expected_version": version("gwpy"),
        "queries": [{}],
        "schema": "gwexpy-v023-gwpy-override-oracle-v1",
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--oracle-worker"],
        cwd=ROOT,
        env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "malformed oracle input" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_malformed_nested_manifest_exits_two_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    manifest["members"][0] = None
    malformed = tmp_path / "malformed.json"
    malformed.write_text(audit.canonical_manifest_json(manifest), encoding="utf-8")
    monkeypatch.setattr(
        audit,
        "build_source_population",
        lambda repository: _stored_population(manifest),
    )
    monkeypatch.setattr(
        audit,
        "run_pristine_oracle",
        lambda *args, **kwargs: manifest["oracle_projections"][version("gwpy")],
    )

    result = audit.main(
        [
            "--check",
            "--manifest",
            str(malformed),
            "--oracle-python",
            _current_oracle_argument(),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "member must be an object" in captured.err
    assert "Traceback" not in captured.err


def test_oracle_stdout_uses_strict_json_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_audit_module()

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"x":1e999}\n'
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    with pytest.raises(audit.InventoryError, match="non-finite JSON float"):
        audit.run_pristine_oracle(SCRIPT, version("gwpy"), sys.executable, [])


def test_relative_oracle_executable_is_resolved_before_temporary_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit_module()
    executable = tmp_path / "oracle-python"
    executable.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)

    parsed = audit.parse_oracle_arguments((f"{version('gwpy')}={executable.name}",))

    assert parsed == {version("gwpy"): str(executable.resolve())}


def test_oracle_subprocess_has_timeout_and_output_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_audit_module()

    def timeout(*args: object, **kwargs: object) -> None:
        assert kwargs["shell"] is False
        assert isinstance(kwargs["timeout"], int)
        raise subprocess.TimeoutExpired("synthetic-oracle", int(kwargs["timeout"]))

    monkeypatch.setattr(audit.subprocess, "run", timeout)
    with pytest.raises(audit.InventoryError, match="oracle .* timed out"):
        audit.run_pristine_oracle(SCRIPT, version("gwpy"), sys.executable, [])

    def oversized(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="x" * (audit.SUBPROCESS_OUTPUT_LIMIT + 1),
            stderr="",
        )

    monkeypatch.setattr(audit.subprocess, "run", oversized)
    with pytest.raises(audit.InventoryError, match="oracle .* exceeded output limit"):
        audit.run_pristine_oracle(SCRIPT, version("gwpy"), sys.executable, [])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_manifest_validation_rejects_recursive_non_finite_values(value: float) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    manifest["cases"][0]["observations"]["gwpy"]["value"] = value

    with pytest.raises(audit.InventoryError, match="non-finite float"):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize("state", ["fixed", "no-finding", "GWpy-fails"])
def test_write_refuses_to_overwrite_behavioral_terminal_evidence(
    tmp_path: Path, state: str
) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = next(item for item in manifest["cases"] if item["counterpart_present"])
    case["state"] = state
    path = tmp_path / "inventory.json"
    path.write_text(audit.canonical_manifest_json(manifest), encoding="utf-8")
    with pytest.raises(
        audit.InventoryError,
        match="refusing to overwrite existing fixed/no-finding/GWpy-fails",
    ):
        audit._refuse_behavioral_overwrite(path)


@pytest.mark.parametrize("state", ["fixed", "no-finding", "GWpy-fails"])
def test_behavioral_terminal_states_require_differential_evidence(state: str) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = next(item for item in manifest["cases"] if item["counterpart_present"])
    case["state"] = state
    case["fixture"] = "default"
    case["comparator"] = {"name": "exact"}
    case["evidence"]["behavior"] = []
    with pytest.raises(audit.InventoryError):
        audit.validate_manifest(manifest)


def test_fixed_state_requires_specific_issue_beyond_inventory_issue() -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = _transition_pending_case(manifest, state="fixed")
    case["issues"] = ["#639"]
    _refresh_summary(audit, manifest)

    with pytest.raises(
        audit.InventoryError, match="fixed case requires a specific issue reference"
    ):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize(
    ("state", "field", "value"),
    [
        ("differential-required", "comparator", {"name": "exact"}),
        (
            "differential-required",
            "observations",
            {"gwexpy": {"outcome": "return"}, "gwpy": {"outcome": "pending"}},
        ),
        ("differential-required", "issues", []),
        ("GWexpy-only", "issues", []),
    ],
)
def test_initial_states_require_exact_comparator_observations_and_issues(
    state: str, field: str, value: object
) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    if state == "differential-required":
        case = _make_present_case_pending(manifest)
    else:
        case = next(item for item in manifest["cases"] if item["state"] == state)
    case[field] = value

    with pytest.raises(audit.InventoryError):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize("state", ["differential-required", "GWexpy-only"])
def test_initial_states_reject_extra_evidence_fields(state: str) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    if state == "differential-required":
        case = _make_present_case_pending(manifest)
    else:
        case = next(item for item in manifest["cases"] if item["state"] == state)
    case["evidence"]["unexpected"] = True

    with pytest.raises(audit.InventoryError, match="evidence schema"):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("extra", "policy schema mismatch"),
        ("owner", "behavioral owner policy mismatch"),
        ("provenance", "upstream dependency provenance policy mismatch"),
    ],
)
def test_policy_shape_owner_and_dependency_provenance_are_exact(
    mutation: str, expected: str
) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    if mutation == "extra":
        manifest["policy"]["unexpected"] = True
    elif mutation == "owner":
        manifest["policy"]["behavioral_owner"] = "someone-else"
    else:
        manifest["policy"]["upstream_dependency_provenance"] = "unbounded"

    with pytest.raises(audit.InventoryError, match=expected):
        audit.validate_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("counterpart_present", True),
        ("fixture", "not-structural-absence"),
        ("comparator", {"name": "pending"}),
        ("evidence", {"behavior": [], "oracle_projection_digest": "wrong"}),
    ],
)
def test_gwexpy_only_requires_pristine_absence_evidence(
    field: str, value: object
) -> None:
    audit = _load_audit_module()
    manifest = copy.deepcopy(_load_manifest())
    case = next(item for item in manifest["cases"] if item["state"] == "GWexpy-only")
    case[field] = value
    with pytest.raises(audit.InventoryError):
        audit.validate_manifest(manifest)


def test_extractor_covers_bindings_inherited_aliases_and_constructors() -> None:
    audit = _load_audit_module()

    class GenericDescriptor:
        def __get__(self, instance: object, owner: type | None = None) -> object:
            raise RuntimeError("descriptor execution is forbidden")

        def __set__(self, instance: object, value: object) -> None:
            return None

    class Upstream:
        def inherited(self) -> None:
            return None

        def aliased(self) -> None:
            return None

    Upstream.__module__ = "gwpy.synthetic"

    class SyntheticMixin:
        def inherited(self) -> None:
            return None

        @classmethod
        def from_value(cls) -> None:
            return None

        @staticmethod
        def utility() -> None:
            return None

        @property
        def read_only(self) -> int:
            return 1

        def _get_read_write(self) -> int:
            return 1

        def _set_read_write(self, value: int) -> None:
            return None

        read_write = property(_get_read_write, _set_read_write)
        generic = GenericDescriptor()

        def aliased(self) -> None:
            return None

        alias = aliased

    SyntheticMixin.__module__ = "gwexpy.synthetic"

    class PublicSynthetic(SyntheticMixin, Upstream):
        def __init__(self) -> None:
            return None

    PublicSynthetic.__module__ = "gwexpy.synthetic"

    members = audit.extract_members_for_classes(
        [(PublicSynthetic, ("gwexpy.synthetic:PublicSynthetic",))], ROOT
    )
    by_name = {item["member"]: item for item in members}
    assert by_name["inherited"]["resolution"] == "inherited-mixin"
    assert by_name["from_value"]["kind"] == "classmethod"
    assert by_name["utility"]["kind"] == "staticmethod"
    assert by_name["read_only"]["descriptor"]["accessors"] == ["get"]
    assert by_name["read_write"]["descriptor"]["accessors"] == ["get", "set"]
    assert by_name["generic"]["kind"] == "generic-descriptor"
    assert by_name["__init__"]["constructor"] is True
    assert by_name["alias"]["alias_group"] == by_name["aliased"]["alias_group"]

    class MaskedPublic(PublicSynthetic):
        inherited = None

    MaskedPublic.__module__ = "gwexpy.synthetic"
    masked = audit.extract_members_for_classes(
        [(MaskedPublic, ("gwexpy.synthetic:MaskedPublic",))], ROOT
    )
    assert "inherited" not in {item["member"] for item in masked}


def test_callable_numpy_generic_descriptor_records_stable_call_evidence() -> None:
    import numpy as np

    audit = _load_audit_module()
    raw = vars(np.ndarray)["swapaxes"]

    assert audit.raw_binding_kind(raw) == "generic-descriptor"
    descriptor = audit._oracle_descriptor(raw, "generic-descriptor", np.ndarray)
    assert descriptor["accessors"] == ["get"]
    assert descriptor["details"] == {
        "call": {
            "signature": {"available": False, "error": "ValueError"},
            "source": None,
        }
    }


def test_callable_generic_descriptor_projection_never_invokes_get() -> None:
    audit = _load_audit_module()
    accesses: list[str] = []

    class ExplosiveDescriptor:
        __signature__ = inspect.Signature(
            [inspect.Parameter("value", inspect.Parameter.POSITIONAL_ONLY)]
        )

        def __get__(self, instance: object, owner: type | None = None) -> object:
            accesses.append("get")
            raise AssertionError("descriptor execution is forbidden")

        def __call__(self, value: object, /) -> object:
            return value

    raw = ExplosiveDescriptor()

    assert audit.raw_binding_kind(raw) == "generic-descriptor"
    descriptor = audit._descriptor_projection(raw, "generic-descriptor", ROOT)
    assert accesses == []
    assert descriptor["accessors"] == ["get"]
    assert descriptor["details"]["call"]["signature"] == {
        "available": True,
        "parameters": [
            {
                "annotation": {"kind": "empty"},
                "default": {"kind": "empty"},
                "kind": "POSITIONAL_ONLY",
                "name": "value",
            }
        ],
        "return_annotation": {"kind": "empty"},
    }
    assert descriptor["details"]["call"]["source"]["path"] == (
        "tests/test_gwpy_override_inventory.py"
    )


def test_unavailable_signature_records_only_stable_exception_class() -> None:
    audit = _load_audit_module()

    class Uninspectable:
        @property
        def __signature__(self) -> object:
            raise ValueError("message with unstable object 0xdeadbeef")

        def __call__(self) -> None:
            return None

    assert audit.normalize_signature(Uninspectable()) == {
        "available": False,
        "error": "ValueError",
    }


def test_source_population_bytes_are_hash_seed_independent() -> None:
    code = (
        "import importlib.util,pathlib,sys;"
        f"p=pathlib.Path({str(SCRIPT)!r});"
        "s=importlib.util.spec_from_file_location('audit_seed',p);"
        "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
        "s.loader.exec_module(m);"
        f"print(m.canonical_compact_json(m.build_source_population(pathlib.Path({str(ROOT)!r}))))"
    )
    outputs = []
    for seed in ("1", "987654"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=os.environ
            | {
                "PYTHONHASHSEED": seed,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(ROOT),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_workflow_runs_ordinary_inventory_check_in_each_existing_matrix_cell() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert workflow["jobs"]["gwpy-compat"]["strategy"]["matrix"]["gwpy"] == [
        "4.0.1",
        "4.0.2",
    ]
    paths = set(workflow["on"]["pull_request"]["paths"])
    assert {
        "gwexpy/**",
        "tests/**",
        "scripts/audit_gwpy_overrides.py",
        "tests/test_gwpy_override_inventory.py",
        "docs/developers/plans/manifests/audit-manifest-v0.2.3-gwpy-overrides.json",
    } <= paths
    steps = workflow["jobs"]["gwpy-compat"]["steps"]
    inventory = next(
        step for step in steps if step.get("name") == "Check GWpy override inventory"
    )
    assert inventory["timeout-minutes"] == "12"
    command = " ".join(inventory["run"].split())
    assert "scripts/audit_gwpy_overrides.py --check" in command
    assert '--oracle-python "$GWPY_VERSION=@current"' in command
    assert "--require-terminal" in command
    assert "--execute-evidence" in command
    provision = next(
        step
        for step in steps
        if step.get("name") == "Provision compatibility environment"
    )
    assert "PyYAML" in provision["run"]
