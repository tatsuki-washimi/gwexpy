# GWpy 4 Legacy Proxy Compatibility Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove four obsolete GWpy developer-utility proxies, repair five runtime proxies against stable GWpy 4 owners, and make the explicit FrameL proxy lazy so the required module-doctest gate can collect successfully.

**Architecture:** Replace accidental upstream namespace mirroring with exact, identity-tested symbol maps. Keep canonical GWF availability independent from the optional FrameL backend, and verify the contract in both the frozen GWpy 4.0.1 environment and the existing latest-4.x compatibility workflow. User-visible removals are recorded under `CHANGELOG.md` `[Unreleased]` and the paired EN/JA migration guides; no release metadata is advanced.

**Tech Stack:** Python 3.11/3.12, GWpy 4.x, Astropy, LALSuite, pytest, Ruff, MyPy, Sphinx, setuptools/build, GitHub Actions YAML.

**Status:** Proposed; specification approved at `4d3f12e`, implementation awaits human approval of this plan.

**Specification:** `docs/superpowers/specs/2026-08-25-gwpy4-legacy-proxy-design.md`

---

## File map

### Production modules

- Modify `gwexpy/table/filter.py`: exact GWpy table-filter exports.
- Modify `gwexpy/table/table.py`: exact EventTable/Table and filter exports from maintained owners.
- Modify `gwexpy/timeseries/core.py`: exact twelve-symbol compatibility contract, including corrected `ChannelList` owner.
- Modify `gwexpy/utils/lal.py`: exact LAL conversion surface without `LAL_UNIT_INDEX`.
- Modify `gwexpy/utils/misc.py`: exact four semantic utility exports.
- Modify `gwexpy/timeseries/io/gwf/framel.py`: deterministic lazy FrameL proxy.
- Delete `gwexpy/utils/shell.py`.
- Delete `gwexpy/utils/sphinx/__init__.py`.
- Delete `gwexpy/utils/sphinx/ex2rst.py`.
- Delete `gwexpy/utils/sphinx/zenodo.py`.
- Preserve `gwexpy/io/gwf.py`: production implementation is out of scope; its current public surface is boundary-tested.

### Contracts, docs, and CI

- Modify `docs/developers/contracts/public_io_contract.json`: distinguish canonical GWF availability from explicit optional FrameL aliases in notes.
- Modify `docs/developers/contracts/public_io_contract.md`: document the same distinction.
- Modify `docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md`: breaking-removal and migration guidance.
- Modify `docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md`: paired Japanese guidance.
- Modify `CHANGELOG.md`: `[Unreleased]` breaking-removal and compatibility entry.
- Modify `.github/workflows/test-compat-gwpy.yml`: latest-4.x triggers, LAL provisioning, and focused proxy gate.
- Preserve `pyproject.toml`: keep `gwexpy.utils.sphinx*` excluded and verify it remains so.
- Preserve `MANIFEST.in`: keep `prune gwexpy/utils/sphinx` and verify the sdist policy remains effective.
- Create `docs/developers/plans/manifests/audit-manifest-v020-gwpy4-proxy-compat.yaml`: command/result evidence and final boundaries.

### Tests

- Create `tests/test_gwpy4_proxy_contract.py`: exact exports, identities, removed names/modules, lazy FrameL, and generic GWF boundary.
- Create `tests/docs/test_gwpy4_proxy_workflow.py`: workflow path/provisioning/command contract.
- Create `tests/docs/test_gwpy4_proxy_docs.py`: changelog and EN/JA navigation/migration contract.
- Modify `tests/io/test_io_contract.py`: alias-specific GWF availability assertions.
- Modify `tests/io/test_io_docs_contract_sync.py`: canonical GWF versus explicit FrameL documentation contract.
- Modify `tests/timeseries/test_io_gwf_timeseriesdict.py`: default-skip and `GWEXPY_REQUIRE_GWF_FRAMEL=1` required-gate behavior.
- Reuse `tests/table/test_table.py`, `tests/interop/test_interop_lal.py`, `tests/timeseries/test_io_gwf_timeseriesdict.py`, and the broader timeseries suite for behavioral regression coverage.

## Models, skills, and effort

- **Implementation:** GPT-5.6-Terra. Required skills: `superpowers:subagent-driven-development`, `superpowers:test-driven-development`, `gwexpy-knowledge-base`, `gwexpy_conda_jobs`, `audit_io_backends`, `manage_docs`, `lint_check`, and `finalize_work`.
- **Specification review:** GPT-5.6-Luna in read-only/plan mode after implementation.
- **Quality/adversarial review:** GPT-5.6-Sol, read-only, only after Luna has no Critical or Important findings.
- **Estimated total wall-clock time:** 90–150 minutes.
- **Estimated quota:** High, driven by two independent reviews, full doctest/timeseries/static gates, package builds, and potentially bounded Sphinx retries.
- **Breakdown:** baseline and RED tests 15–25 min; proxy implementation 20–35 min; FrameL/GWF contracts 15–25 min; removals/docs/CI 15–25 min; validation/audit/reviews 25–40 min.
- **Main uncertainty:** the full module-doctest and EN/JA Sphinx gates may reveal pre-existing failures or exceed the 180-second harness bound. Record these truthfully; do not hide or relabel them as passes.

### Command policy

Every shell invocation is prefixed by RTK. Use `rtk <command>` for a single
command and `rtk run -c '<shell composition>'` when environment assignments,
pipes, variables, timeouts, or heredocs must share one shell. These forms were
checked against the installed RTK interface; the plan does not depend on an
RTK-specific `env`, `python`, `mktemp`, `timeout`, or `sha256sum` subcommand.

---

### Task 1: Capture the baseline and add failing compatibility contracts

**Files:**
- Create: `tests/test_gwpy4_proxy_contract.py`
- Create: `tests/docs/test_gwpy4_proxy_workflow.py`

- [ ] **Step 1: Verify the isolated integration baseline**

Run:

```bash
rtk git status --short
rtk git rev-parse HEAD
rtk git remote -v
rtk git config --get-regexp '^remote\.'
rtk run -c 'git -C /home/washimi/work/gwexpy status --porcelain=v1 | sha256sum'
```

Expected: integration status is empty; HEAD contains this committed plan; original dirty-worktree hash remains `ba596e14b056730df2ec95a41920b8a2845d52ee42e930e0cc73811fc2b98dfc`.

- [ ] **Step 2: Write exact proxy-surface and owner tests**

Create `tests/test_gwpy4_proxy_contract.py` with exact constants, not `dir()`-derived expectations:

```python
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

EXPECTED_EXPORTS = {
    "gwexpy.table.filter": (
        "DELIM_REGEX", "OPERATORS", "OPERATORS_INV", "QUOTE_REGEX",
        "filter_table", "generate_tokens", "is_filter_tuple",
        "parse_column_filter", "parse_column_filters", "parse_operator",
    ),
    "gwexpy.table.table": (
        "DEFAULT_GWOSC_URL", "TIME_LIKE_COLUMN_NAMES", "EventTable", "Table",
        "filter_table", "parse_operator",
    ),
    "gwexpy.timeseries.core": (
        "GWOSC_DEFAULT_HOST", "Channel", "ChannelList", "LIGOTimeGPS",
        "SegmentList", "Series", "Time", "TimeSeriesBase",
        "TimeSeriesBaseDict", "TimeSeriesBaseList", "to_gps", "units",
    ),
    "gwexpy.utils.lal": (
        "LAL_DETECTORS", "LAL_NUMPY_FROM_TYPE_STR", "LAL_TYPE_FROM_NUMPY",
        "LAL_TYPE_FROM_STR", "LAL_TYPE_REGEX", "LAL_TYPE_STR",
        "LAL_TYPE_STR_FROM_NUMPY", "find_typed_function", "from_lal_type",
        "from_lal_unit", "gwpy_units", "to_gps", "to_lal_ligotimegps",
        "to_lal_type_str", "to_lal_unit",
    ),
    "gwexpy.utils.misc": (
        "if_not_none", "property_alias", "round_to_power", "unique",
    ),
}

REMOVED_MODULES = (
    "gwexpy.utils.shell",
    "gwexpy.utils.sphinx",
    "gwexpy.utils.sphinx.ex2rst",
    "gwexpy.utils.sphinx.zenodo",
)

REMOVED_NAMES = {
    "gwexpy.table.filter": ("OrderedDict", "StringIO", "numpy", "operator", "re", "token"),
    "gwexpy.table.table": ("attrgetter", "ceil", "gps_types", "inherit_io_registrations", "io_read_multi", "registry", "vstack", "wraps"),
    "gwexpy.timeseries.core": ("OrderedDict", "as_series_dict_class", "ceil", "gps_types", "io_registry", "property_alias"),
    "gwexpy.utils.lal": ("LAL_UNIT_INDEX",),
    "gwexpy.utils.misc": ("OrderedDict", "nullcontext"),
}

def _import_name(module_name: str, name: str) -> None:
    exec(f"from {module_name} import {name}", {})

@pytest.mark.parametrize("module_name, expected", EXPECTED_EXPORTS.items())
def test_curated_proxy_exports_are_exact(module_name: str, expected: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)
    assert tuple(module.__all__) == expected

@pytest.mark.parametrize(
    ("module_name", "name"),
    [(module, name) for module, names in REMOVED_NAMES.items() for name in names],
)
def test_removed_proxy_leaks_raise_import_error(module_name: str, name: str) -> None:
    with pytest.raises(ImportError):
        _import_name(module_name, name)

@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_deleted_proxy_source_paths_are_absent(module_name: str) -> None:
    relative = Path(*module_name.split("."))
    assert not relative.with_suffix(".py").exists()
    assert not (relative / "__init__.py").exists()
```

The source-tree test uses path absence because an ignored `__pycache__`
directory can temporarily make the deleted `gwexpy.utils.sphinx` directory a
namespace package in a developer checkout. The built-wheel smoke in Task 5 is
the authoritative assertion that all four imports raise normal
`ModuleNotFoundError` in a clean distributable environment.

Add explicit identity tests against every owner listed in the approved specification, including:

```python
def test_timeseries_core_uses_maintained_owners() -> None:
    from astropy import units
    from gwosc.api import DEFAULT_URL
    from gwpy.detector.channel import Channel, ChannelList
    from gwpy.segments import SegmentList
    from gwpy.time import LIGOTimeGPS, Time, to_gps
    from gwpy.timeseries.core import TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList
    from gwpy.types import Series
    from gwexpy.timeseries import core

    assert core.Channel is Channel
    assert core.ChannelList is ChannelList
    assert core.LIGOTimeGPS is LIGOTimeGPS
    assert core.SegmentList is SegmentList
    assert core.Series is Series
    assert core.Time is Time
    assert core.TimeSeriesBase is TimeSeriesBase
    assert core.TimeSeriesBaseDict is TimeSeriesBaseDict
    assert core.TimeSeriesBaseList is TimeSeriesBaseList
    assert core.to_gps is to_gps
    assert core.units is units
    assert core.GWOSC_DEFAULT_HOST == DEFAULT_URL
```

- [ ] **Step 3: Write FrameL and generic-GWF boundary tests**

In the same test file, define the static FrameL exports and the thirteen-name generic GWF boundary. Monkeypatch `importlib.import_module` so module import, `__all__`, and `dir()` prove they do not load `gwpy.timeseries.io.gwf.framel`; then assert access to `read` preserves the original `ModuleNotFoundError`. When `framel` is installed, assert forwarded identity against the upstream module.

```python
FRAME_EXPORTS = (
    "FRAME_LIBRARY", "Segment", "TimeSeries", "file_list", "file_path",
    "framel", "read", "warnings", "write",
)
GWF_EXPORTS = (
    "BACKENDS", "backend", "channel_exists", "core", "data_segments",
    "get_backend", "get_backend_function", "get_channel_names",
    "get_channel_type", "identify_gwf", "import_backend",
    "iter_channel_names", "num_channels",
)

def test_generic_gwf_boundary_is_frozen() -> None:
    from gwexpy.io import gwf
    assert tuple(gwf.__all__) == GWF_EXPORTS
```

- [ ] **Step 4: Write the failing workflow contract**

Create `tests/docs/test_gwpy4_proxy_workflow.py` using `yaml.safe_load`. Require
all pre-existing and new path filters, `lalsuite` in the provisioning command,
all three focused test paths in one dedicated run step, that step's ordering
before the full timeseries step, and the unchanged pre-existing focused tests.

```python
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/test-compat-gwpy.yml"

def test_latest_gwpy_proxy_gate_is_wired() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    paths = set(workflow[True]["pull_request"]["paths"])
    required_existing = {
        "gwexpy/timeseries/**", "gwexpy/frequencyseries/**",
        "gwexpy/spectrogram/**", "gwexpy/signal/**", "gwexpy/types/**",
        "gwexpy/interop/**", "gwexpy/io/**", "gwexpy/fitting/**",
        "gwexpy/plot/**", "gwexpy/utils/**", "tests/timeseries/**",
        "pyproject.toml", "requirements*.txt", "environment.yml",
        ".github/workflows/test-compat-gwpy.yml",
    }
    required_new = {
        "gwexpy/table/**", "tests/table/**", "tests/interop/**",
        "tests/test_gwpy4_proxy_contract.py",
        "docs/developers/contracts/public_io_contract.*",
    }
    assert required_existing | required_new <= paths

    steps = workflow["jobs"]["gwpy-compat"]["steps"]
    by_name = {step["name"]: step for step in steps if "name" in step}
    provisioning = by_name["Provision compatibility environment"]["run"]
    assert "python -m pip install lalsuite" in provisioning

    old_focused = by_name["Run focused compatibility tests"]["run"]
    for test_path in (
        "tests/timeseries/test_transfer_function_compat.py",
        "tests/timeseries/test_collections_spectral_compat.py",
        "tests/timeseries/test_fft_param_compat.py",
    ):
        assert test_path in old_focused

    proxy_step = by_name["Run GWpy 4 proxy compatibility tests"]
    expected_proxy_paths = (
        "tests/test_gwpy4_proxy_contract.py",
        "tests/table/test_table.py",
        "tests/interop/test_interop_lal.py",
    )
    assert proxy_step["run"].count("pytest -q") == 1
    assert all(path in proxy_step["run"] for path in expected_proxy_paths)

    full_index = next(
        index for index, step in enumerate(steps)
        if step.get("name") == "Run full timeseries suite"
    )
    proxy_index = steps.index(proxy_step)
    assert proxy_index < full_index
    assert steps[full_index]["run"].strip() == "pytest -q tests/timeseries"
```

- [ ] **Step 5: Run RED tests**

Run:

```bash
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/test_gwpy4_proxy_contract.py tests/docs/test_gwpy4_proxy_workflow.py'
```

Expected: FAIL because the five stale proxies do not import, the four deleted modules still exist, FrameL loads eagerly, and the workflow lacks the new gate.

---

### Task 2: Curate the table and miscellaneous proxies

**Files:**
- Modify: `gwexpy/table/filter.py`
- Modify: `gwexpy/table/table.py`
- Modify: `gwexpy/utils/misc.py`
- Test: `tests/test_gwpy4_proxy_contract.py`
- Test: `tests/table/test_table.py`

- [ ] **Step 1: Replace `table.filter` with exact imports**

Retain only the ten approved symbols from `gwpy.table.filter` and set `__all__` to the exact tuple used by the contract test. Do not import standard-library or NumPy implementation helpers.

- [ ] **Step 2: Replace `table.table` with maintained owners**

Use:

```python
from astropy.table import Table
from gwosc.api import DEFAULT_URL as DEFAULT_GWOSC_URL
from gwpy.table.filter import filter_table, parse_operator
from gwpy.table.table import EventTable, TIME_LIKE_COLUMN_NAMES
```

Set `__all__` to the exact six-symbol tuple from Task 1.

- [ ] **Step 3: Replace `utils.misc` with semantic utilities**

Import only `if_not_none`, `property_alias`, `round_to_power`, and `unique` from `gwpy.utils.misc`; set the exact four-name `__all__`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/test_gwpy4_proxy_contract.py -k "table or misc"'
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/table/test_table.py'
```

Expected: table/misc proxy tests and behavioral table tests PASS; unrelated still-unfixed proxy rows may be deselected.

- [ ] **Step 5: Commit the table/misc slice**

```bash
rtk git add gwexpy/table/filter.py gwexpy/table/table.py gwexpy/utils/misc.py tests/test_gwpy4_proxy_contract.py
rtk git commit -m "fix(compat): curate GWpy 4 table proxies"
```

---

### Task 3: Curate TimeSeries core and LAL interoperability proxies

**Files:**
- Modify: `gwexpy/timeseries/core.py`
- Modify: `gwexpy/utils/lal.py`
- Test: `tests/test_gwpy4_proxy_contract.py`
- Test: `tests/interop/test_interop_lal.py`

- [ ] **Step 1: Implement the exact TimeSeries core owner map**

Use only these maintained imports:

```python
from astropy import units
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST
from gwpy.detector.channel import Channel, ChannelList
from gwpy.segments import SegmentList
from gwpy.time import LIGOTimeGPS, Time, to_gps
from gwpy.timeseries.core import TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList
from gwpy.types import Series
```

Set `__all__` to the exact twelve-name tuple from Task 1. Do not preserve `as_series_dict_class`, `property_alias`, or removed implementation imports.

- [ ] **Step 2: Implement the exact LAL map without private unit indices**

Import `gwpy_units` with `from gwpy.detector import units as gwpy_units`,
`to_gps` from `gwpy.time`, and the approved maps/regex/converters from
`gwpy.utils.lal`. Omit `LAL_UNIT_INDEX` entirely and set the exact fifteen-name
`__all__` from Task 1. The owner map is decision-complete:

| Owner | Symbols |
|---|---|
| `gwpy.utils.lal` | `LAL_DETECTORS`, `LAL_NUMPY_FROM_TYPE_STR`, `LAL_TYPE_FROM_NUMPY`, `LAL_TYPE_FROM_STR`, `LAL_TYPE_REGEX`, `LAL_TYPE_STR`, `LAL_TYPE_STR_FROM_NUMPY`, `find_typed_function`, `from_lal_type`, `from_lal_unit`, `to_lal_ligotimegps`, `to_lal_type_str`, `to_lal_unit` |
| `gwpy.time` | `to_gps` |
| `gwpy.detector.units` module | `gwpy_units` |

- [ ] **Step 3: Add real LAL owner and behavior assertions**

Extend the contract test to assert every retained LAL object is identical to its maintained owner. Keep `tests/interop/test_interop_lal.py` as the public-operation regression; do not rewrite its mock boundary.

- [ ] **Step 4: Run focused tests**

```bash
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/test_gwpy4_proxy_contract.py -k "timeseries_core or lal or removed_proxy_leaks"'
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/interop/test_interop_lal.py'
```

Expected: all selected tests PASS with LALSuite present.

- [ ] **Step 5: Commit the core/LAL slice**

```bash
rtk git add gwexpy/timeseries/core.py gwexpy/utils/lal.py tests/test_gwpy4_proxy_contract.py tests/interop/test_interop_lal.py
rtk git commit -m "fix(compat): stabilize GWpy 4 core proxies"
```

---

### Task 4: Make FrameL lazy and reconcile the public GWF contract

**Files:**
- Modify: `gwexpy/timeseries/io/gwf/framel.py`
- Modify: `docs/developers/contracts/public_io_contract.json`
- Modify: `docs/developers/contracts/public_io_contract.md`
- Modify: `tests/io/test_io_docs_contract_sync.py`
- Test: `tests/test_gwpy4_proxy_contract.py`
- Test: `tests/timeseries/test_io_gwf_timeseriesdict.py`

- [ ] **Step 1: Implement a deterministic lazy proxy**

Use this structure, with full type annotations:

```python
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_UPSTREAM = "gwpy.timeseries.io.gwf.framel"
__all__ = (
    "FRAME_LIBRARY", "Segment", "TimeSeries", "file_list", "file_path",
    "framel", "read", "warnings", "write",
)
_module: ModuleType | None = None

def _load() -> ModuleType:
    global _module
    if _module is None:
        _module = import_module(_UPSTREAM)
    return _module

def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_load(), name)
    globals()[name] = value
    return value

_DIRECTORY = tuple(sorted(__all__))

def __dir__() -> list[str]:
    return list(_DIRECTORY)
```

Do not catch or rewrite `ModuleNotFoundError`; missing `framel` must retain its original type, name, message, cause, and context.

- [ ] **Step 2: Make lazy tests independent of the host backend**

The missing-backend test must monkeypatch the proxy's `import_module` call and
clear the proxy module cache. Assert `dir(proxy) == sorted(FRAME_EXPORTS)` before
and after accessing a symbol, so caching cannot grow the directory surface or
expose `_load`, `_module`, `import_module`, `ModuleType`, or `Any`. The
available-backend identity test uses `pytest.importorskip("framel")`, reloads
the proxy, compares requested objects to `gwpy.timeseries.io.gwf.framel`, and
reasserts the identical fixed `dir()` result after access.

- [ ] **Step 3: Update the GWF contract notes and executable sync test**

Keep `optional_dependencies: []` and `available_in_base_install` for canonical
`gwf`. Add notes stating `framel`/`gwf.framel` require optional
`python-framel`, default tests skip unavailable explicit-backend rows, and
`GWEXPY_REQUIRE_GWF_FRAMEL=1` makes absence a required-gate failure. Mirror this
in the Markdown contract and assert the strings in
`tests/io/test_io_docs_contract_sync.py`.

In `tests/io/test_io_contract.py`, add a contract test that selects the
canonical `gwf` entry and requires:

```python
assert entry["optional_dependencies"] == []
assert entry["unavailable_behavior"] == {
    "read": "available_in_base_install",
    "write": "available_in_base_install",
}
assert {"framel", "gwf.framel"} <= set(entry["aliases"])
assert any("explicit FrameL" in note and "python-framel" in note for note in entry["notes"])
```

In `tests/timeseries/test_io_gwf_timeseriesdict.py`, add a test-only helper
that checks `has_gwf_backend("framel")`. When unavailable it calls
`pytest.skip` by default and `pytest.fail` when
`os.environ.get("GWEXPY_REQUIRE_GWF_FRAMEL") == "1"`. Use that helper in the
real explicit `format="gwf.framel"` read test. Add unit tests proving all three
branches: default missing backend skips, required missing backend fails, and
available backend continues. No production environment-variable branch is
added.

- [ ] **Step 4: Run focused and adjacent GWF tests**

```bash
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/test_gwpy4_proxy_contract.py -k "framel or generic_gwf"'
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/io/test_io_contract.py tests/io/test_io_docs_contract_sync.py'
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/timeseries/test_io_gwf_timeseriesdict.py tests/timeseries/test_io_gwf_framel.py'
rtk run -c 'PYTHONPATH=$PWD GWEXPY_REQUIRE_GWF_FRAMEL=1 pytest -q tests/timeseries/test_io_gwf_timeseriesdict.py -k framel'
```

Expected: lazy import/boundary, helper-unit, and contract tests PASS. Explicit
real FrameL tests PASS when installed or SKIP under the default policy. The
required-gate command PASSes when FrameL is installed; when absent, it returns
the intentional required-gate failure and that nonzero result is recorded as
contract evidence, not relabeled as a passing release gate.

- [ ] **Step 5: Commit the FrameL/GWF slice**

```bash
rtk git add gwexpy/timeseries/io/gwf/framel.py docs/developers/contracts/public_io_contract.json docs/developers/contracts/public_io_contract.md tests/test_gwpy4_proxy_contract.py tests/io/test_io_contract.py tests/io/test_io_docs_contract_sync.py tests/timeseries/test_io_gwf_timeseriesdict.py
rtk git commit -m "fix(io): load the FrameL proxy lazily"
```

---

### Task 5: Delete obsolete developer utilities and prove packaging exclusion

**Files:**
- Delete: `gwexpy/utils/shell.py`
- Delete: `gwexpy/utils/sphinx/__init__.py`
- Delete: `gwexpy/utils/sphinx/ex2rst.py`
- Delete: `gwexpy/utils/sphinx/zenodo.py`
- Preserve: `pyproject.toml`
- Preserve: `MANIFEST.in`
- Test: `tests/test_gwpy4_proxy_contract.py`
- Test: `tests/test_check_release_artifacts_script.py`

- [ ] **Step 1: Delete the four approved files without stubs**

Delete only the four paths above. Do not delete `gwexpy/utils/__init__.py`, and
do not add deprecated modules that merely raise custom errors. After deletion,
verify `gwexpy/utils/sphinx` contains no files. If an ignored `__pycache__`
exists, move it to a task-specific directory under `/tmp` and then remove the
now-empty `gwexpy/utils/sphinx` directory with `rmdir`; never use recursive
deletion.

- [ ] **Step 2: Add package-policy regression coverage**

In `tests/test_gwpy4_proxy_contract.py`, parse `pyproject.toml` with `tomllib`
and require `gwexpy.utils.sphinx*` in
`tool.setuptools.packages.find.exclude`. Read `MANIFEST.in` and require
`prune gwexpy/utils/sphinx`. Require the source directory itself to be absent,
not only its `.py` files. Extend the existing release-artifact tests only if
needed to assert both wheel and sdist reject `gwexpy/utils/sphinx/**`; do not
change general release policy.

- [ ] **Step 3: Run deletion and artifact-policy tests**

```bash
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/test_gwpy4_proxy_contract.py -k "deleted or package"'
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/test_check_release_artifacts_script.py'
rtk run -c '
PYTHONPATH=$PWD python -P - <<'"'"'PY'"'"'
import importlib

for name in (
    "gwexpy.utils.shell",
    "gwexpy.utils.sphinx",
    "gwexpy.utils.sphinx.ex2rst",
    "gwexpy.utils.sphinx.zenodo",
):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        continue
    raise AssertionError(f"removed source module imported: {name}")
PY
'
```

Expected: deleted imports raise normal `ModuleNotFoundError` from the isolated
source checkout; source/package-policy tests PASS.

- [ ] **Step 4: Build wheel/sdist outside the worktree and inspect contents**

Run the build, archive inspection, and isolated wheel import in one supervised
shell so the generated path is unambiguous:

```bash
rtk run -c '
set -eu
GWEXPY_PROXY_DIST="$(mktemp -d --tmpdir gwexpy-v020-proxy-dist.XXXXXXXX)"
python -m build --sdist --wheel --no-isolation --outdir "$GWEXPY_PROXY_DIST"
python scripts/check_release_artifacts.py "$GWEXPY_PROXY_DIST"
python -P - "$GWEXPY_PROXY_DIST" <<'"'"'PY'"'"'
from __future__ import annotations
import importlib
import sys
import tarfile
import zipfile
from pathlib import Path

dist = Path(sys.argv[1])
wheel, = dist.glob("*.whl")
sdist, = dist.glob("*.tar.gz")
forbidden = (
    "gwexpy/utils/shell.py",
    "gwexpy/utils/sphinx/__init__.py",
    "gwexpy/utils/sphinx/ex2rst.py",
    "gwexpy/utils/sphinx/zenodo.py",
)
with zipfile.ZipFile(wheel) as archive:
    wheel_names = set(archive.namelist())
with tarfile.open(sdist, "r:*") as archive:
    sdist_names = {"/".join(Path(name).parts[1:]) for name in archive.getnames()}
assert not (set(forbidden) & wheel_names)
assert not (set(forbidden) & sdist_names)

sys.path.insert(0, str(wheel))
for name in (
    "gwexpy.utils.shell",
    "gwexpy.utils.sphinx",
    "gwexpy.utils.sphinx.ex2rst",
    "gwexpy.utils.sphinx.zenodo",
):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        continue
    raise AssertionError(f"removed module imported from wheel: {name}")
PY
'
```

Expected: one wheel and one sdist, artifact hygiene PASS, zero removed proxy
paths, and four normal import failures without custom stubs. The `-P` isolated
process prepends the wheel itself, so the source checkout and any editable
installation cannot satisfy the removed imports.

- [ ] **Step 5: Commit the removals**

```bash
rtk git add -A -- gwexpy/utils/shell.py gwexpy/utils/sphinx tests/test_gwpy4_proxy_contract.py tests/test_check_release_artifacts_script.py
rtk git commit -m "refactor(utils): remove obsolete GWpy developer proxies"
```

---

### Task 6: Wire latest-GWpy CI and publish migration guidance

**Files:**
- Modify: `.github/workflows/test-compat-gwpy.yml`
- Modify: `tests/docs/test_gwpy4_proxy_workflow.py`
- Modify: `CHANGELOG.md`
- Modify: `docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md`
- Modify: `docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md`

- [ ] **Step 1: Add exact workflow triggers and LAL provisioning**

Add these PR paths: `gwexpy/table/**`, `tests/table/**`, `tests/interop/**`, `tests/test_gwpy4_proxy_contract.py`, and `docs/developers/contracts/public_io_contract.*`. Add `python -m pip install lalsuite` after latest-GWpy provisioning.

- [ ] **Step 2: Add the exact focused command**

Add a dedicated step named `Run GWpy 4 proxy compatibility tests` before the
existing full timeseries step, with exactly one pytest invocation:

```bash
pytest -q \
  tests/test_gwpy4_proxy_contract.py \
  tests/table/test_table.py \
  tests/interop/test_interop_lal.py
```

Do not remove the existing `pytest -q tests/timeseries` step.
Also preserve the workflow's pre-existing focused transfer-function,
collection-spectral, and FFT-parameter compatibility tests.

- [ ] **Step 3: Make the workflow contract GREEN**

```bash
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/docs/test_gwpy4_proxy_workflow.py'
```

Expected: PASS and YAML parse succeeds.

- [ ] **Step 4: Add paired migration notes and Unreleased changelog entry**

Under `CHANGELOG.md` `[Unreleased]`, add a `Removed (breaking)` entry listing
the four deleted imports and replacements, followed by a compatibility fix
entry for the five curated surfaces and lazy FrameL behavior. Add the same
facts and before/after import examples to the existing EN/JA GWpy migration
guides. Do not create `release_notes/v0.2.0.md`, bump a version, or alter
release metadata.

Create `tests/docs/test_gwpy4_proxy_docs.py` to assert the changelog names all
four removed paths and both migration guides describe the removals and lazy
FrameL behavior. Assert the canonical pages remain explicitly linked from
both locale roots and reference indexes:

```python
assert "user_guide/gwexpy_for_gwpy_users_en" in EN_INDEX.read_text()
assert "../user_guide/gwexpy_for_gwpy_users_en" in EN_REFERENCE.read_text()
assert "user_guide/gwexpy_for_gwpy_users_ja" in JA_INDEX.read_text()
assert "../user_guide/gwexpy_for_gwpy_users_ja" in JA_REFERENCE.read_text()
```

- [ ] **Step 5: Verify docs navigation and release contracts**

```bash
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/docs/test_gwpy4_proxy_docs.py tests/docs/test_docs_conf_runtime.py tests/test_gen_release_notes.py tests/test_release_contracts.py'
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/io/test_io_docs_contract_sync.py'
```

Expected: PASS; existing EN/JA migration pages remain linked and `Unreleased` remains the source of truth.

- [ ] **Step 6: Commit CI and migration docs**

```bash
rtk git add .github/workflows/test-compat-gwpy.yml tests/docs/test_gwpy4_proxy_workflow.py tests/docs/test_gwpy4_proxy_docs.py CHANGELOG.md docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md
rtk git commit -m "docs(compat): publish GWpy 4 proxy migration"
```

---

### Task 7: Run full verification and record the audit manifest

**Files:**
- Create: `docs/developers/plans/manifests/audit-manifest-v020-gwpy4-proxy-compat.yaml`
- Modify only if a verified failure is attributable to this scope.

- [ ] **Step 1: Run the complete focused compatibility matrix in GWpy 4.0.1**

```bash
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/test_gwpy4_proxy_contract.py tests/table/test_table.py tests/interop/test_interop_lal.py tests/docs/test_gwpy4_proxy_workflow.py tests/docs/test_gwpy4_proxy_docs.py tests/io/test_io_contract.py tests/io/test_io_docs_contract_sync.py'
rtk run -c 'PYTHONPATH=$PWD pytest -q tests/timeseries'
```

Expected: the default lane PASSes with only documented optional-backend skips.
The separate `GWEXPY_REQUIRE_GWF_FRAMEL=1` command has the conditional result
defined in Task 4 and is recorded separately from this default passing lane.

- [ ] **Step 2: Run the required full module-doctest gate**

```bash
rtk run -c 'PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/gwexpy-v020-proxy-doctest-mpl timeout 180 pytest -q --doctest-modules gwexpy'
```

Expected: collection succeeds with none of the original ten proxy errors. Any later failure is classified, reproduced at the integration parent if applicable, and recorded rather than hidden.

- [ ] **Step 3: Run static checks**

```bash
rtk ruff check gwexpy tests
rtk ruff format --check gwexpy tests
rtk mypy gwexpy
rtk git diff --check
```

Expected: Ruff and MyPy PASS. If the unchanged `tests/docs/test_root_roadmap_contract.py` formatting discrepancy remains, record it as the known baseline and additionally require all changed files to pass format checking.

- [ ] **Step 4: Run bounded EN/JA Sphinx gates**

```bash
rtk run -c 'PYTHONPATH=$PWD timeout 180 python -m sphinx -W --keep-going -b html -D language=en docs /tmp/gwexpy-v020-proxy-docs-en'
rtk run -c 'PYTHONPATH=$PWD timeout 180 python -m sphinx -W --keep-going -b html -D language=ja docs /tmp/gwexpy-v020-proxy-docs-ja'
```

If either times out, retry once with `-j 1` and a distinct `/tmp` output directory. Record pass/fail/timeout and the exact reason; do not claim an uncompleted build.

- [ ] **Step 5: Record latest-4.x evidence truthfully**

Run the workflow-equivalent environment only if already available without installing into the shared environment. Otherwise record local latest-4.x evidence as unavailable and require `.github/workflows/test-compat-gwpy.yml` before merge. Record exact `gwpy.__version__` and LALSuite version for every executed lane.

- [ ] **Step 6: Write and validate the audit manifest**

Record baseline hash, RED/GREEN results, versions, optional dependency conditions, package contents, every test/static/docs gate, latest-4.x disposition, prohibited mutations, and pre-audit status. Parse it:

```bash
rtk run -c 'PYTHONPATH=$PWD python -c "import pathlib,yaml; yaml.safe_load(pathlib.Path('"'"'docs/developers/plans/manifests/audit-manifest-v020-gwpy4-proxy-compat.yaml'"'"').read_text())"'
```

Expected: exit 0.

- [ ] **Step 7: Commit the audit record**

```bash
rtk git add docs/developers/plans/manifests/audit-manifest-v020-gwpy4-proxy-compat.yaml
rtk git commit -m "docs(audit): record GWpy 4 proxy compatibility"
```

- [ ] **Step 8: Verify final local state and original-worktree preservation**

```bash
rtk git status --short
rtk git log -1 --oneline
rtk git remote -v
rtk git config --get-regexp '^remote\.'
rtk run -c 'git -C /home/washimi/work/gwexpy status --porcelain=v1 | sha256sum'
```

Expected: integration worktree clean; original dirty-worktree hash unchanged;
remote listings byte-for-byte match the baseline evidence; no push, tag,
release, version, remote, shared-environment package installation, or GitHub
mutation. The manifest explicitly records GitHub state as **not inspected** and
states that no `gh`, web-write, or GitHub mutation tool was invoked; it must not
claim external-state equality without inspection.

---

### Task 8: Independent review and remediation loop

**Files:**
- Modify only files required by verified findings.

- [ ] **Step 1: Request Luna specification review**

Provide the approved specification, this plan, exact implementation range, audit manifest, and test evidence. Require a read-only Critical/Important/Minor report. If Critical or Important findings exist, return them to the same Terra implementer, use `superpowers:receiving-code-review`, add RED regressions, fix, commit, and re-review. Maximum three loops before escalating to the user.

- [ ] **Step 2: Request Sol adversarial quality review after Luna approval**

Require independent checks of namespace leakage, symbol identity, hostile/missing optional imports, package contents, workflow trigger completeness, full doctest collection, docs truthfulness, and original-worktree preservation. Apply the same TDD remediation loop for verified Critical/Important findings.

- [ ] **Step 3: Perform verification-before-completion**

Use `superpowers:verification-before-completion` and `finalize_work`. Re-run changed-file Ruff/format, MyPy, focused tests, `git diff --check`, YAML parse, status, and original-worktree hash after the final review commit.

- [ ] **Step 4: Hand off without release mutation**

Report exact HEAD, commit list, tests, baseline/remaining concerns, review verdicts, and human gates. Do not merge, push, open/update a PR, tag, bump the version, publish, or alter issue/release state without separate authorization.
