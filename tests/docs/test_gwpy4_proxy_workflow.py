"""Keep the exact-GWpy compatibility matrix wired to proxy contracts."""

import json
import re
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

from scripts.ci.audit_gwosc_timeseries import _validate_candidate_cases

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/test-compat-gwpy.yml"
GWOSC_INVENTORY = ROOT / "tests/docs/gwosc_timeseries_inventory.json"
GWOSC_PHASE_A2 = ROOT / "tests/docs/gwosc_timeseries_phase_a2_category_c.json"


def test_exact_gwpy_matrix_proxy_gate_is_wired() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    pull_request = workflow[True]["pull_request"]
    assert set(pull_request["branches"]) >= {"main", "maint/0.2"}
    assert "paths" not in pull_request
    assert "paths-ignore" not in pull_request

    job = workflow["jobs"]["gwpy-compat"]
    strategy = job["strategy"]
    assert strategy["fail-fast"] is False
    assert strategy["matrix"]["gwpy"] == ["4.0.1", "4.0.2"]
    assert "matrix.gwpy" in job["name"]
    assert job["env"]["GWPY_VERSION"] == "${{ matrix.gwpy }}"
    assert job["env"]["EXPECTED_SOURCE_SHA"] == (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )

    steps = job["steps"]
    by_name = {step["name"]: step for step in steps if "name" in step}
    checkout_step = by_name["Checkout repository"]
    assert checkout_step["with"]["ref"] == "${{ env.EXPECTED_SOURCE_SHA }}"
    source_step = by_name["Verify checked-out source"]
    assert steps.index(source_step) == steps.index(checkout_step) + 1
    normalized_source_command = " ".join(source_step["run"].split())
    assert 'actual_source_sha="$(git rev-parse HEAD)"' in normalized_source_command
    assert 'test "$actual_source_sha" = "$EXPECTED_SOURCE_SHA"' in (
        normalized_source_command
    )

    provisioning = by_name["Provision compatibility environment"]["run"]
    assert "python -m pip install lalsuite" in provisioning
    assert '"gwpy==$GWPY_VERSION"' in provisioning
    assert '"certifi>=2026.10.19"' in provisioning

    for lock_path in (
        ROOT / "requirements-dev.txt",
        ROOT / "requirements" / "release-build.txt",
    ):
        match = re.search(
            r"(?m)^certifi==([^\s;\\]+)",
            lock_path.read_text(encoding="utf-8"),
        )
        assert match is not None, f"certifi missing from {lock_path}"
        assert Version(match.group(1)) >= Version("2026.10.19")

    release_requirements = ROOT / "requirements" / "release-build.in"
    pip_input = re.search(
        r"(?m)^pip==([^\s;\\]+)",
        release_requirements.read_text(encoding="utf-8"),
    )
    pip_lock = re.search(
        r"(?m)^pip==([^\s;\\]+)",
        (ROOT / "requirements" / "release-build.txt").read_text(encoding="utf-8"),
    )
    assert pip_input is not None
    assert pip_lock is not None
    assert Version(pip_input.group(1)) >= Version("26.2.2")
    assert Version(pip_lock.group(1)) >= Version("26.2.2")
    assert pip_input.group(1) == pip_lock.group(1)

    version_step = by_name["Record compatibility versions"]
    provisioning_index = steps.index(by_name["Provision compatibility environment"])
    assert steps.index(version_step) == provisioning_index + 1
    normalized_version_command = " ".join(version_step["run"].split())
    assert 'expected_gwpy = os.environ["GWPY_VERSION"]' in normalized_version_command
    assert 'actual_gwpy = version("gwpy")' in normalized_version_command
    assert "if actual_gwpy != expected_gwpy:" in normalized_version_command
    assert "raise SystemExit(" in normalized_version_command
    assert 'version("lalsuite")' in normalized_version_command
    assert '"certifi"' in normalized_version_command

    old_focused = by_name["Run focused compatibility tests"]["run"]
    for test_path in (
        "tests/timeseries/test_transfer_function_compat.py",
        "tests/timeseries/test_collections_spectral_compat.py",
        "tests/timeseries/test_fft_param_compat.py",
    ):
        assert test_path in old_focused

    proxy_step = by_name["Run GWpy 4 proxy compatibility tests"]
    expected_proxy_command = " ".join(
        (
            "pytest -q",
            "tests/test_gwpy4_proxy_contract.py",
            "tests/table/test_table.py",
            "tests/interop/test_interop_lal.py",
        )
    )
    proxy_command = " ".join(proxy_step["run"].replace("\\\n", " ").split())
    proxy_command = proxy_command.replace(
        '--junitxml="${RUNNER_TEMP}/gwpy-compat-evidence/proxy.xml" ', ""
    )
    assert proxy_command == (
        expected_proxy_command
        + ' 2>&1 | tee "${RUNNER_TEMP}/gwpy-compat-evidence/proxy.log"'
    )
    full_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Run full timeseries suite"
    )
    assert steps.index(proxy_step) < full_index
    audit_step = by_name["Audit GWOSC live test selection"]
    audit_command = " ".join(audit_step["run"].replace("\\\n", " ").split())
    assert "scripts/ci/audit_gwosc_timeseries.py" in audit_command
    assert "gwosc-timeseries-audit.json" in audit_command
    assert "gwosc-timeseries-audit.log" in audit_command
    assert steps.index(audit_step) < full_index
    full_command = " ".join(steps[full_index]["run"].replace("\\\n", " ").split())
    assert full_command == (
        'pytest -q -m "not upstream_gwosc_live" '
        '--junitxml="${RUNNER_TEMP}/gwpy-compat-evidence/timeseries.xml" '
        'tests/timeseries 2>&1 | tee "${RUNNER_TEMP}/gwpy-compat-evidence/timeseries.log"'
    )
    assert '"not network"' not in full_command
    assert by_name["Retain compatibility evidence"]["if"] == "always()"


def test_gwosc_live_gate_is_limited_to_audited_upstream_node_ids() -> None:
    inventory = json.loads(GWOSC_INVENTORY.read_text(encoding="utf-8"))
    assert inventory["schema"] == "gwexpy-gwosc-timeseries-candidate-inventory-v2"
    assert inventory["gwpy_versions"] == ["4.0.1", "4.0.2"]
    cases = {entry["nodeid"]: entry for entry in inventory["candidate_cases"]}
    live_ids = set(inventory["live_marker_ids"])
    assert len(cases) == 92
    assert len(live_ids) == 89
    inherited_test_ids = set(inventory["inherited_upstream_test_ids"])
    assert len(inherited_test_ids) == 256
    assert set(cases) <= inherited_test_ids
    assert {case["category"] for case in cases.values()} == {"A", "C"}
    assert sum(case["category"] == "A" for case in cases.values()) == 8
    assert sum(case["category"] == "C" for case in cases.values()) == 84
    assert live_ids <= set(cases)
    expected_live_ids = {
        nodeid
        for nodeid, case in cases.items()
        if case["category"] == "A"
        or (
            case["category"] == "C"
            and set(case["fixtures"]).intersection({"gw150914", "gw150914_16384"})
        )
    }
    assert live_ids == expected_live_ids

    retained_h1_l1 = {
        nodeid
        for nodeid, case in cases.items()
        if case["category"] == "C"
        and set(case["fixtures"]).intersection({"gw150914_h1_32", "gw150914_l1_32"})
    }
    assert len(retained_h1_l1) == 3
    assert not (retained_h1_l1 & live_ids)

    audit_source = (ROOT / "scripts/ci/audit_gwosc_timeseries.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ.get("EXPECTED_SOURCE_SHA")' in audit_source
    assert "live_marker_ids" in audit_source
    assert "unknown_fixture_ids" in audit_source
    assert "unknown_network_ids" in audit_source

    phase_a2 = json.loads(GWOSC_PHASE_A2.read_text(encoding="utf-8"))
    assert phase_a2["audit_base"] == "da3391d12d97a793c9710107d03d78e9f41d8e3b"
    assert phase_a2["gwpy_versions"] == ["4.0.1", "4.0.2"]
    category_c = {nodeid for nodeid, case in cases.items() if case["category"] == "C"}
    phase_a2_cases = {case["nodeid"]: case for case in phase_a2["cases"]}
    assert set(phase_a2_cases) == category_c
    assert len(phase_a2_cases) == 84
    assert {case["phase_a2_classification"] for case in phase_a2_cases.values()} == {
        "upstream-only",
        "gwexpy-influenced",
    }
    assert (
        sum(
            case["phase_a2_classification"] == "upstream-only"
            for case in phase_a2_cases.values()
        )
        == 83
    )
    assert (
        sum(
            case["phase_a2_classification"] == "gwexpy-influenced"
            for case in phase_a2_cases.values()
        )
        == 1
    )
    influenced = next(
        case
        for case in phase_a2_cases.values()
        if case["phase_a2_classification"] == "gwexpy-influenced"
    )
    assert influenced["nodeid"].endswith("test_spectrogram_multiprocessing")
    assert "test_timeseries_spectrogram_multiprocessing_matches_serial" in (
        ROOT / "tests/timeseries/test_gwpy_override_terminal_compat.py"
    ).read_text(encoding="utf-8")


def test_gwosc_audit_rejects_fixture_and_classification_metadata_drift() -> None:
    inventory = json.loads(GWOSC_INVENTORY.read_text(encoding="utf-8"))
    cases = inventory["candidate_cases"]
    _validate_candidate_cases(cases, cases)

    changed = [dict(case) for case in cases]
    changed[0]["fixtures"] = ["new_fixture"]
    with pytest.raises(SystemExit, match="metadata mismatch"):
        _validate_candidate_cases(changed, cases)

    changed = [dict(case) for case in cases]
    changed[0]["category"] = "unclassified"
    with pytest.raises(SystemExit, match="metadata mismatch"):
        _validate_candidate_cases(changed, cases)
