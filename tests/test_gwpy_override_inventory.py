"""Contracts for the deterministic GWpy override inventory (issue #639)."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
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
    "gwexpy.plot.skymap.SkyMap": ("gwexpy.plot.skymap:SkyMap",),
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


def test_manifest_has_initial_structural_population_and_canonical_json() -> None:
    audit = _load_audit_module()
    manifest = _load_manifest()

    assert manifest["schema"] == SCHEMA
    assert manifest["policy"]["implementation_base"] == IMPLEMENTATION_BASE
    assert manifest["policy"]["public_root_rule"] == (
        "byte-sorted gwexpy Python paths; literal top-level list/tuple __all__; "
        "static vars(module) exports; canonical GWexpy class identity; internal "
        "root exclusions"
    )
    assert manifest["policy"]["member_walk_rule"] == (
        "first effective vars(owner) binding in the GWexpy MRO prefix before "
        "the first GWpy class; public callable/descriptors plus __new__/__init__"
    )
    assert manifest["policy"]["pristine_oracle_rule"] == (
        "separate -I worker; sanitized PYTHONPATH/PYTHONHOME; no GWexpy import; "
        "exact GWpy 4.0.1/4.0.2"
    )
    assert manifest["summary"] == {
        "cases": 1146,
        "constructors": 11,
        "counterpart_absent_per_version": 441,
        "counterpart_implementation_groups": 66,
        "counterpart_present_per_version": 132,
        "differential-required": 264,
        "GWexpy-only": 882,
        "logical_members": 573,
        "public_roots": 16,
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
    assert len(current["members"]) == 573


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


def test_ordinary_check_permits_provisional_but_terminal_check_rejects_counts() -> None:
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
    assert terminal.returncode != 0
    assert (
        "provisional states remain: differential-required=264, unreviewed=0"
        in terminal.stderr
    )


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


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    audit = _load_audit_module()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"first","schema":"second"}\n', encoding="utf-8")
    with pytest.raises(audit.InventoryError, match="duplicate JSON key: schema"):
        audit.load_json_strict(duplicate)


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
        "scripts/audit_gwpy_overrides.py",
        "tests/test_gwpy_override_inventory.py",
        "docs/developers/plans/manifests/audit-manifest-v0.2.3-gwpy-overrides.json",
    } <= paths
    steps = workflow["jobs"]["gwpy-compat"]["steps"]
    inventory = next(
        step for step in steps if step.get("name") == "Check GWpy override inventory"
    )
    command = " ".join(inventory["run"].split())
    assert "scripts/audit_gwpy_overrides.py --check" in command
    assert '--oracle-python "$GWPY_VERSION=@current"' in command
    assert "--require-terminal" not in command
