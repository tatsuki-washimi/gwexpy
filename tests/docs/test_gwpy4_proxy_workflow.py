"""Keep the latest-GWpy compatibility workflow wired to proxy contracts."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/test-compat-gwpy.yml"


def test_latest_gwpy_proxy_gate_is_wired() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    paths = set(workflow[True]["pull_request"]["paths"])
    required_existing = {
        "gwexpy/timeseries/**",
        "gwexpy/frequencyseries/**",
        "gwexpy/spectrogram/**",
        "gwexpy/signal/**",
        "gwexpy/types/**",
        "gwexpy/interop/**",
        "gwexpy/io/**",
        "gwexpy/fitting/**",
        "gwexpy/plot/**",
        "gwexpy/utils/**",
        "tests/timeseries/**",
        "pyproject.toml",
        "requirements*.txt",
        "environment.yml",
        ".github/workflows/test-compat-gwpy.yml",
    }
    required_new = {
        "gwexpy/table/**",
        "tests/table/**",
        "tests/interop/**",
        "tests/test_gwpy4_proxy_contract.py",
        "docs/developers/contracts/public_io_contract.*",
    }
    assert required_existing | required_new <= paths

    job = workflow["jobs"]["gwpy-compat"]
    strategy = job["strategy"]
    assert strategy["fail-fast"] is False
    assert strategy["matrix"]["gwpy"] == ["4.0.1", "4.0.2"]
    assert "matrix.gwpy" in job["name"]
    assert job["env"]["GWPY_VERSION"] == "${{ matrix.gwpy }}"

    steps = job["steps"]
    by_name = {step["name"]: step for step in steps if "name" in step}
    provisioning = by_name["Provision compatibility environment"]["run"]
    assert "python -m pip install lalsuite" in provisioning
    assert '"gwpy==$GWPY_VERSION"' in provisioning

    version_step = by_name["Record compatibility versions"]
    provisioning_index = steps.index(by_name["Provision compatibility environment"])
    assert steps.index(version_step) == provisioning_index + 1
    normalized_version_command = " ".join(version_step["run"].split())
    assert 'version("gwpy")' in normalized_version_command
    assert 'version("lalsuite")' in normalized_version_command

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
    assert " ".join(proxy_step["run"].split()) == expected_proxy_command
    full_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Run full timeseries suite"
    )
    assert steps.index(proxy_step) < full_index
    assert steps[full_index]["run"].strip() == "pytest -q tests/timeseries"
