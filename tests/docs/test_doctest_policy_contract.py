"""Contract tests for the raw-doctest policy."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
POLICY = ROOT / "docs/developers/contracts/doctest-policy.md"
DEVELOPERS_INDEX = ROOT / "docs/developers/index.rst"
CONFIG_FILES = ("pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml")
CONFTEST_FILES = ("conftest.py", "tests/conftest.py")
LANE_MODULES = (
    "gwexpy/analysis/bruco.py",
    "gwexpy/fields/scalar.py",
    "gwexpy/fields/signal.py",
    "gwexpy/fields/tensor.py",
    "gwexpy/fitting/gls.py",
    "gwexpy/fitting/highlevel.py",
    "gwexpy/histogram/histogram.py",
    "gwexpy/plot/field.py",
    "gwexpy/plot/pairplot.py",
    "gwexpy/plot/plot.py",
    "gwexpy/signal/preprocessing/ml.py",
    "gwexpy/table/segment_plot.py",
)
SOURCE_ROOT = ROOT / "gwexpy"
AUDIT_MODULES = (
    "gwexpy/io/utils.py",
    "gwexpy/timeseries/_signal.py",
    "gwexpy/timeseries/_spectral_special.py",
    "gwexpy/timeseries/io/_registration.py",
)


def test_doctest_policy_defines_the_four_required_categories() -> None:
    policy = POLICY.read_text(encoding="utf-8")
    headings = re.findall(r"^### ([1-4])\.\s", policy, flags=re.MULTILINE)
    assert headings == ["1", "2", "3", "4"]
    required_phrases = (
        "must pass",
        "illustrative fragment",
        "not a doctest",
        "local reason",
        "optional/external",
        "outside this lane",
        "stale repr/output",
        "API/semantic",
        "code block",
        "dedicated tests",
        "no `+SKIP`",
    )
    assert all(phrase in policy for phrase in required_phrases)


def test_doctest_policy_preserves_raw_collection_and_forbids_blanket_exclusions() -> (
    None
):
    policy = POLICY.read_text(encoding="utf-8")
    assert "--doctest-modules -q gwexpy/" in policy
    assert "No blanket" in policy
    assert "global pytest exclusion" in policy
    assert "module-wide skip" in policy

    for relative_path in CONFIG_FILES:
        path = ROOT / relative_path
        if path.exists():
            _assert_no_global_doctest_exclusion(
                path.read_text(encoding="utf-8"), relative_path
            )

    for relative_path in CONFTEST_FILES:
        path = ROOT / relative_path
        if path.exists():
            _assert_no_global_doctest_exclusion(
                path.read_text(encoding="utf-8"), relative_path
            )


def test_lane_has_no_undocumented_doctest_skip_directives() -> None:
    for relative_path in LANE_MODULES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        _assert_lane_source_is_doctest_safe(source, relative_path)


def test_all_gwexpy_source_has_no_raw_doctest_skip_directives() -> None:
    skipped = []
    for path in SOURCE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "# doctest: +SKIP" in source:
            skipped.append(path.relative_to(ROOT).as_posix())

    assert skipped == []


def test_audited_optional_and_schematic_examples_use_reasoned_rendering() -> None:
    sources = {
        relative_path: (ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in AUDIT_MODULES
    }
    utils = sources["gwexpy/io/utils.py"]
    signal = sources["gwexpy/timeseries/_signal.py"]
    spectral = sources["gwexpy/timeseries/_spectral_special.py"]
    registration = sources["gwexpy/timeseries/io/_registration.py"]

    assert 'ensure_dependency("json", import_name="json")' in utils
    assert 'ensure_dependency("xarray")' in utils
    assert 'ensure_dependency("nptdms"' in utils
    assert 'extract_audio_metadata("song.mp3")' in utils
    assert "dedicated tests" in utils.lower()
    assert "rtol=2e-3" not in signal
    setup_match = re.search(
        r"np\.arange\(\s*1000\s*\)\s*\*\s*0\.001.*?"
        r"TimeSeries\(.*?dt\s*=\s*0\.001",
        signal,
        flags=re.DOTALL,
    )
    assert setup_match is not None
    assertion_match = re.search(
        r"np\.testing\.assert_allclose\(.*?"
        r"np\.median\(f_inst\.value\[100:-100\]\).*?50\.0.*?"
        r"rtol\s*=\s*([0-9.eE+-]+).*?"
        r"atol\s*=\s*([0-9.eE+-]+)",
        signal,
        flags=re.DOTALL,
    )
    assert assertion_match is not None
    assert float(assertion_match.group(1)) <= 1e-12
    assert float(assertion_match.group(2)) <= 1e-12
    assert "reference.transfer_function(test" in signal
    assert ".. code-block:: python" in signal
    assert "ts.emd(method='eemd'" in spectral
    assert "ts.hht(emd_method='eemd'" in spectral
    assert ".. code-block:: python" in spectral
    assert "register_timeseries_format(" in registration
    assert ".. code-block:: python" in registration

    for relative_path, source in sources.items():
        _assert_no_prompts_inside_python_code_blocks(source, relative_path)


def test_prompt_to_code_block_mutation_is_rejected() -> None:
    with pytest.raises(AssertionError):
        _assert_no_prompts_inside_python_code_blocks(
            ".. code-block:: python\n\n   >>> value = 1\n",
            "prompt mutation",
        )


def test_lane_source_validation_rejects_module_skip_mutation() -> None:
    relative_path = LANE_MODULES[0]
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    mutated_source = (
        "pytestmark = pytest.mark.skip(reason='hidden doctests')\n" + source
    )
    with pytest.raises(AssertionError):
        _assert_lane_source_is_doctest_safe(mutated_source, relative_path)


def _assert_lane_source_is_doctest_safe(source: str, label: str) -> None:
    _assert_no_global_doctest_exclusion(source, label)
    assert "# doctest: +SKIP" not in source, label


def _assert_no_global_doctest_exclusion(source: str, label: str) -> None:
    assert not re.search(
        r"--ignore(?:\s*=\s*|\s+)gwexpy(?:[/\\]|(?=\s|$))",
        source,
        flags=re.IGNORECASE,
    ), label
    assert not re.search(
        r"^\s*(?:collect_ignore(?:_glob)?|pytestmark)\s*=.*"
        r"(?:gwexpy|doctest|pytest\.mark\.skip)",
        source,
        flags=re.IGNORECASE | re.MULTILINE,
    ), label
    for hook_name in ("pytest_ignore_collect", "pytest_collection_modifyitems"):
        for hook in re.finditer(rf"^\s*def {hook_name}\b", source, flags=re.MULTILINE):
            next_function = re.search(
                r"^\s*def\s+", source[hook.end() :], flags=re.MULTILINE
            )
            end = hook.end() + next_function.start() if next_function else len(source)
            body = source[hook.end() : end]
            if hook_name == "pytest_ignore_collect":
                assert not re.search(r"(?:gwexpy|doctest)", body), label
            else:
                assert not (
                    re.search(r"(?:gwexpy|doctest)", body)
                    and re.search(r"pytest\.mark\.skip", body)
                ), label


def _assert_code_blocks_have_local_reason(source: str, label: str) -> None:
    for match in re.finditer(r"^\s*\.\. code-block:: python", source, re.MULTILINE):
        local_context = source[max(0, match.start() - 500) : match.start()]
        assert "illustrative" in local_context.lower(), label
        assert "not a doctest" in local_context.lower(), label


def _assert_no_prompts_inside_python_code_blocks(source: str, label: str) -> None:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != ".. code-block:: python":
            continue
        directive_indent = len(line) - len(line.lstrip())
        body = []
        cursor = index + 1
        while cursor < len(lines):
            candidate = lines[cursor]
            if candidate:
                candidate_indent = len(candidate) - len(candidate.lstrip())
                if candidate_indent <= directive_indent:
                    break
            body.append(candidate)
            cursor += 1
        assert ">>>" not in "\n".join(body), label


def test_policy_is_discoverable_in_developers_toctree() -> None:
    index = DEVELOPERS_INDEX.read_text(encoding="utf-8")
    assert "contracts/doctest-policy" in index
    pytest_config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"testpaths\s*=\s*\[\s*['\"]tests['\"]", pytest_config)
    assert (ROOT / "tests/docs/test_doctest_policy_contract.py").exists()


def test_global_exclusion_mutations_are_rejected() -> None:
    for option in (
        "--ignore=gwexpy",
        "--ignore gwexpy",
        "--ignore=gwexpy/",
        "--ignore gwexpy/",
    ):
        with pytest.raises(AssertionError):
            _assert_no_global_doctest_exclusion(f"[pytest]\naddopts = {option}", option)

    mutations = (
        'collect_ignore = ["gwexpy"]',
        'collect_ignore_glob = ["gwexpy/**"]',
        "def pytest_ignore_collect(path):\n    return 'gwexpy' in str(path)",
        "pytestmark = pytest.mark.skip(reason='doctest')",
        "def pytest_collection_modifyitems(config, items):\n"
        "    if 'doctest' in item.nodeid:\n"
        "        item.add_marker(pytest.mark.skip)",
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            _assert_no_global_doctest_exclusion(mutation, mutation)


def test_code_block_conversion_requires_local_rationale() -> None:
    for relative_path in LANE_MODULES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        _assert_code_blocks_have_local_reason(source, relative_path)

    with pytest.raises(AssertionError):
        _assert_code_blocks_have_local_reason(
            "Examples\n--------\n.. code-block:: python\n\n   value = 1\n",
            "mutation without rationale",
        )


def test_scalar_time_space_plot_example_remains_executable() -> None:
    import gwexpy

    gwexpy.register_all()
    from gwexpy.fields import ScalarField

    source = inspect.getdoc(ScalarField.plot_time_space_map)
    assert source is not None
    assert '>>> fig, ax = sf.plot_time_space_map("x")' in source
    assert ">>> len(ax.collections)" in source
    assert "# doctest: +SKIP" not in source

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy import units as u

    sf = ScalarField(np.ones((100, 4, 1, 1)), axis1=np.arange(4) * u.m)
    fig, ax = sf.plot_time_space_map("x")
    try:
        assert len(ax.collections) == 1
        assert ax.get_xlabel() == "x [m]"
        assert ax.get_ylabel() == "t []"
    finally:
        plt.close(fig)
