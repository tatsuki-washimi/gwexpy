"""Contract checks for the generated HDF5 provenance tutorials."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import pytest

from scripts import fix_tutorial_notebooks
from scripts.dev_tools import make_provenance_tutorial

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {
    "en": ROOT / "docs/web/en/user_guide/tutorials/case_hdf5_provenance.ipynb",
    "ja": ROOT / "docs/web/ja/user_guide/tutorials/case_hdf5_provenance.ipynb",
}
GENERATOR = ROOT / "scripts/dev_tools/make_provenance_tutorial.py"

EXPECTED_T0_NS = 1234567890123456789
EXPECTED_METADATA = {
    "channel": "K1:TEST-STRAIN",
    "analysis": {"sample_rate_hz": 4.0, "window": "hann"},
    "labels": ["synthetic", "tutorial"],
}
EXPECTED_PROVENANCE = {
    "pipeline": {"name": "hdf5-provenance-tutorial", "version": 1},
    "inputs": ["deterministic synthetic samples"],
    "parameters": {"sample_rate_hz": 4.0, "calibration": "not applied"},
}

_RELEASE_CLAIM_RE = re.compile(
    r"(?:\bv0\.2\.0\b[^\n.]{0,100}\b(?:released|published|shipped|available)\b"
    r"|\b(?:released|published|shipped|available)\b[^\n.]{0,100}\bv0\.2\.0\b)",
    re.IGNORECASE,
)
_STANDALONE_RELEASE_CLAIM_RE = re.compile(
    r"\b(?:published|released|shipped)\b",
    re.IGNORECASE,
)
_NEUTRAL_RELEASE_WORDING_RE = re.compile(
    r"(?:\bnot\s+(?:yet\s+)?(?:published|released|shipped)\b"
    r"|\bresources?\s+are\s+(?:published|released|shipped)\s+by\s+"
    r"(?:a\s+)?context\s+manager\b)",
    re.IGNORECASE,
)
_JAPANESE_RELEASE_CLAIM_RE = re.compile(
    r"v0\.2\.0[^\n。]{0,40}(?:リリース|公開|出荷|提供開始)|"
    r"(?:リリース|公開|出荷|提供開始)[^\n。]{0,40}v0\.2\.0"
)


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sources(nb: dict, cell_type: str) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell["cell_type"] == cell_type
    ]


def _code_sources(nb: dict) -> list[str]:
    return _sources(nb, "code")


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _call_has_hdf5_format(node: ast.Call) -> bool:
    return any(
        keyword.arg == "format"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "hdf5"
        for keyword in node.keywords
    )


def _fail(context: str, message: str) -> None:
    raise AssertionError(f"{context}: {message}")


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        _fail(context, message)


def _has_release_claim(text: str) -> bool:
    neutralized = _NEUTRAL_RELEASE_WORDING_RE.sub(" ", text)
    return bool(
        _RELEASE_CLAIM_RE.search(neutralized)
        or _STANDALONE_RELEASE_CLAIM_RE.search(neutralized)
        or _JAPANESE_RELEASE_CLAIM_RE.search(text)
    )


def _json_literal(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ValueError("not a JSON literal")
    if isinstance(node, ast.List):
        return [_json_literal(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        result: dict[str, object] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                key is None
                or not isinstance(key, ast.Constant)
                or not isinstance(key.value, str)
            ):
                raise ValueError("JSON object keys must be string literals")
            result[key.value] = _json_literal(value)
        return result
    raise ValueError("not a JSON literal")


def _literal_bindings(tree: ast.AST) -> dict[str, object]:
    bindings: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            bindings[target.id] = _json_literal(node.value)
        except ValueError:
            continue
    return bindings


def _resolved_json_literal(node: ast.AST, bindings: dict[str, object]) -> object:
    if isinstance(node, ast.Name):
        if node.id not in bindings:
            raise ValueError(f"unresolved literal binding {node.id!r}")
        return bindings[node.id]
    return _json_literal(node)


def _has_chain(node: ast.AST, *parts: str) -> bool:
    return _attribute_chain(node) == parts


def _is_integer_literal(node: ast.AST, expected: int) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value == expected
    )


def _assert_no_forbidden_code(tree: ast.AST, source: str, context: str) -> None:
    forbidden_modules = {
        "aiohttp",
        "datetime",
        "ftplib",
        "glob",
        "http",
        "httpx",
        "os",
        "paramiko",
        "platform",
        "random",
        "requests",
        "secrets",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "telnetlib",
        "time",
        "urllib",
    }
    forbidden_names = {
        "delattr",
        "h5py",
        "hdf5_sidecar",
        "getattr",
        "random",
        "requests",
        "setattr",
        "socket",
        "subprocess",
        "urlopen",
        "urlretrieve",
    }
    forbidden_attributes = {
        "_gwexpy_sidecar_json_v1",
        "attrs",
        "create_dataset",
        "cwd",
        "delattr",
        "from_hdf5",
        "getattr",
        "home",
        "setattr",
        "to_hdf5",
    }
    forbidden_calls = {
        "delattr",
        "eval",
        "exec",
        "getattr",
        "input",
        "open",
        "setattr",
    }

    _require("_gwexpy_sidecar_json_v1" not in source, context, "manual sidecar use")
    _require("h5py" not in source, context, "h5py is unnecessary")
    _require(
        "gwexpy.io.hdf5_sidecar" not in source,
        context,
        "internal HDF5 sidecar use",
    )
    _require(
        not _has_release_claim(source),
        context,
        "release claim",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".", 1)[0]
                _require(
                    module_root not in forbidden_modules,
                    context,
                    f"forbidden import {alias.name}",
                )
                _require(alias.name != "h5py", context, "h5py is unnecessary")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            module_root = module.split(".", 1)[0]
            _require(
                module_root not in forbidden_modules,
                context,
                f"forbidden import {module}",
            )
            _require(
                not module.startswith("gwexpy.io.hdf5_sidecar"),
                context,
                "internal HDF5 sidecar import",
            )
            if module == "gwexpy.io":
                _require(
                    all(alias.name != "hdf5_sidecar" for alias in node.names),
                    context,
                    "internal HDF5 sidecar import",
                )
        elif isinstance(node, ast.Name):
            _require(
                node.id not in forbidden_names, context, f"forbidden name {node.id}"
            )
        elif isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            _require(
                not any(part in forbidden_attributes for part in chain),
                context,
                f"forbidden attribute {'.'.join(chain)}",
            )
            _require(
                not any(part in {"hdf5_sidecar", "random"} for part in chain),
                context,
                f"forbidden API {'.'.join(chain)}",
            )
        elif isinstance(node, ast.Call):
            call_name = node.func.id if isinstance(node.func, ast.Name) else ""
            _require(
                call_name not in forbidden_calls,
                context,
                f"forbidden call {call_name}",
            )
            chain = _attribute_chain(node.func)
            _require(
                not any(part in {"system", "popen", "run", "Popen"} for part in chain),
                context,
                f"forbidden shell/external call {'.'.join(chain)}",
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            _require(
                not re.search(r"(?:https?|ftp)://", node.value, re.IGNORECASE),
                context,
                "URL-like input",
            )


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == name for child in ast.walk(node)
    )


def _target_is_name(target: ast.AST, name: str) -> bool:
    return isinstance(target, ast.Name) and target.id == name


def _target_roots_at_name(target: ast.AST, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Attribute, ast.Subscript)):
        return _target_roots_at_name(target.value, name)
    if isinstance(target, (ast.List, ast.Tuple)):
        return any(_target_roots_at_name(element, name) for element in target.elts)
    return False


def _is_deterministic_archive_path_expression(node: ast.AST) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    if not (
        isinstance(node.left, ast.Call)
        and _has_chain(node.left.func, "Path")
        and len(node.left.args) == 1
        and isinstance(node.left.args[0], ast.Name)
        and node.left.args[0].id == "temporary_directory"
    ):
        return False
    return isinstance(node.right, ast.Constant) and node.right.value == "provenance.h5"


def _is_direct_read_assignment(node: ast.AST, path_name: str) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "restored"
        and isinstance(node.value, ast.Call)
        and _has_chain(node.value.func, "gwexpy", "TimeSeries", "read")
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == path_name
        and _call_has_hdf5_format(node.value)
    )


class _PathFlowVisitor(ast.NodeVisitor):
    def __init__(self, path_name: str) -> None:
        self.path_name = path_name
        self.parents: list[ast.AST] = []
        self.control_depth = 0
        self.events: list[tuple[int, int, str, ast.AST, bool]] = []

    def visit(self, node: ast.AST) -> None:
        self.parents.append(node)
        super().visit(node)
        self.parents.pop()

    def _record(self, kind: str, node: ast.AST) -> None:
        self.events.append(
            (
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
                kind,
                node,
                self.control_depth > 0,
            )
        )

    def _visit_control(self, node: ast.AST) -> None:
        self._record("control", node)
        self.control_depth += 1
        self.generic_visit(node)
        self.control_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_control(node)

    def visit_If(self, node: ast.If) -> None:
        self._visit_control(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._visit_control(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._visit_control(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_control(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_control(node)

    def visit_With(self, node: ast.With) -> None:
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._visit_control(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if _target_is_name(target, self.path_name):
                self._record("binding", node)
            elif _target_roots_at_name(target, self.path_name):
                self._record("mutation", node)
        if not _is_direct_read_assignment(node, self.path_name):
            if not any(
                _target_roots_at_name(target, self.path_name) for target in node.targets
            ) and _contains_name(node.value, self.path_name):
                self._record("alias", node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _target_is_name(node.target, self.path_name):
            self._record("binding", node)
        elif _target_roots_at_name(node.target, self.path_name):
            self._record("mutation", node)
        elif node.value is not None and _contains_name(node.value, self.path_name):
            self._record("alias", node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if _target_is_name(node.target, self.path_name):
            self._record("binding", node)
        elif _target_roots_at_name(node.target, self.path_name):
            self._record("mutation", node)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if _target_is_name(node.target, self.path_name):
            self._record("binding", node)
        elif _target_roots_at_name(node.target, self.path_name):
            self._record("mutation", node)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if _target_roots_at_name(target, self.path_name):
                self._record("binding", node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        parent = self.parents[-2] if len(self.parents) >= 2 else None
        is_write = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "write"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ts"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and _call_has_hdf5_format(node)
            and isinstance(parent, ast.Expr)
        )
        is_read = (
            isinstance(parent, ast.Assign)
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
            and parent.targets[0].id == "restored"
            and _has_chain(node.func, "gwexpy", "TimeSeries", "read")
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and _call_has_hdf5_format(node)
        )
        if is_write:
            self._record("write", node)
        elif is_read:
            self._record("read", node)
        elif _contains_name(node, self.path_name):
            self._record("mutation", node)
        self.generic_visit(node)


def _path_flow_events(
    tree: ast.AST, path_name: str
) -> list[tuple[int, int, str, ast.AST, bool]]:
    visitor = _PathFlowVisitor(path_name)
    visitor.visit(tree)
    unique_events: dict[tuple[str, int], tuple[int, int, str, ast.AST, bool]] = {}
    for event in visitor.events:
        unique_events[(event[2], id(event[3]))] = event
    return sorted(
        unique_events.values(), key=lambda event: (event[0], event[1], event[2])
    )


def _validate_tutorial_code(source: str, *, context: str = "tutorial code") -> None:
    tree = ast.parse(source, filename=context)
    _assert_no_forbidden_code(tree, source, context)
    bindings = _literal_bindings(tree)

    _require(
        any(
            isinstance(node, ast.Call)
            and _has_chain(node.func, "gwexpy", "register_all")
            for node in ast.walk(tree)
        ),
        context,
        "missing gwexpy.register_all()",
    )

    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "ts"
        and isinstance(node.value, ast.Call)
        and _has_chain(node.value.func, "gwexpy", "TimeSeries")
    ]
    _require(
        len(constructors) == 1, context, "missing unique ts = gwexpy.TimeSeries(...)"
    )
    constructor = constructors[0].value
    assert isinstance(constructor, ast.Call)
    t0_keywords = [
        keyword for keyword in constructor.keywords if keyword.arg == "t0_ns"
    ]
    _require(len(t0_keywords) == 1, context, "t0_ns must be specified once")
    _require(
        _is_integer_literal(t0_keywords[0].value, EXPECTED_T0_NS),
        context,
        "t0_ns must be the expected integer literal",
    )

    mapping_assignments: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "ts"
            and target.attr in {"metadata", "provenance"}
        ):
            continue
        try:
            mapping_assignments[target.attr] = _resolved_json_literal(
                node.value, bindings
            )
        except ValueError as exc:
            _fail(context, f"{target.attr} is not a JSON-literal-safe mapping: {exc}")
    _require(
        mapping_assignments
        == {
            "metadata": EXPECTED_METADATA,
            "provenance": EXPECTED_PROVENANCE,
        },
        context,
        "ts metadata/provenance assignments are not the expected mappings",
    )

    write_candidates = [
        event
        for event in _path_flow_events(tree, "__unknown_archive_path__")
        if event[2] == "write"
    ]
    _require(
        len(write_candidates) == 1,
        context,
        "missing ts.write(path, format='hdf5')",
    )
    write_call = write_candidates[0][3]
    assert isinstance(write_call, ast.Call)
    assert isinstance(write_call.args[0], ast.Name)
    path_name = write_call.args[0].id

    events = _path_flow_events(tree, path_name)
    write_indexes = [index for index, event in enumerate(events) if event[2] == "write"]
    read_indexes = [index for index, event in enumerate(events) if event[2] == "read"]
    _require(len(write_indexes) == 1, context, "missing unique ts.write call")
    _require(
        len(read_indexes) == 1,
        context,
        "missing restored = gwexpy.TimeSeries.read(same_path, format='hdf5')",
    )
    write_index = write_indexes[0]
    read_index = read_indexes[0]
    _require(write_index < read_index, context, "write must precede read")

    path_bindings = [event for event in events if event[2] == "binding"]
    _require(
        len(path_bindings) == 1,
        context,
        "archive path must have one uninterrupted binding",
    )
    path_binding = path_bindings[0][3]
    if isinstance(path_binding, ast.Assign):
        path_expression = path_binding.value
    elif isinstance(path_binding, ast.AnnAssign) and path_binding.value is not None:
        path_expression = path_binding.value
    else:
        _fail(context, "archive path binding must be a deterministic assignment")
    _require(
        _is_deterministic_archive_path_expression(path_expression),
        context,
        "archive path must use the deterministic local provenance path",
    )
    binding_index = events.index(path_bindings[0])
    _require(binding_index < write_index, context, "write path binding must be current")

    for event in events[binding_index + 1 : read_index]:
        _require(
            event[2] not in {"binding", "alias", "mutation", "control"},
            context,
            "archive path binding is rebound or invalidated before read",
        )

    required_assertions: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        comparison = node.test
        if not (
            len(comparison.ops) == 1
            and isinstance(comparison.ops[0], ast.Eq)
            and isinstance(comparison.left, ast.Attribute)
            and isinstance(comparison.left.value, ast.Name)
            and comparison.left.value.id == "restored"
            and comparison.left.attr in {"t0_gps_ns", "metadata", "provenance"}
            and len(comparison.comparators) == 1
        ):
            continue
        field = comparison.left.attr
        _require(
            field not in required_assertions,
            context,
            f"duplicate assertion for {field}",
        )
        try:
            value = _resolved_json_literal(comparison.comparators[0], bindings)
        except ValueError as exc:
            _fail(context, f"assertion for restored.{field} is not exact: {exc}")
        required_assertions[field] = value

    _require(
        required_assertions
        == {
            "t0_gps_ns": EXPECTED_T0_NS,
            "metadata": EXPECTED_METADATA,
            "provenance": EXPECTED_PROVENANCE,
        },
        context,
        "required restored comparisons must be ast.Assert statements with exact values",
    )


def _validate_tutorial_markdown(
    markdown: str, *, context: str = "tutorial markdown"
) -> None:
    _require(
        not _has_release_claim(markdown),
        context,
        "release claim",
    )


def test_tutorial_uses_public_hdf5_provenance_roundtrip_contract() -> None:
    for path in NOTEBOOKS.values():
        code = "\n".join(_code_sources(_read_notebook(path)))
        _validate_tutorial_code(code, context=str(path))


def test_tutorial_explains_root_sidecar_without_teaching_manual_attribute_edits() -> (
    None
):
    forbidden = (
        "to_hdf5",
        "from_hdf5",
        "h5py.File",
        ".attrs[",
        "create_dataset",
        "dset.attrs",
    )
    for path in NOTEBOOKS.values():
        notebook = _read_notebook(path)
        markdown = "\n".join(_sources(notebook, "markdown"))
        code = "\n".join(_code_sources(notebook))

        assert "_gwexpy_sidecar_json_v1" in markdown
        assert "_gwexpy_sidecar_json_v1" not in code
        assert "manually" in markdown.lower() or "手動" in markdown
        assert all(token not in code for token in forbidden)
        _validate_tutorial_markdown(markdown, context=str(path))


def test_generated_roundtrip_code_executes_and_proves_exact_state() -> None:
    notebook = _read_notebook(NOTEBOOKS["en"])
    code = "\n".join(_code_sources(notebook))
    namespace = {"__name__": "__hdf5_provenance_contract__"}

    exec(compile(code, str(NOTEBOOKS["en"]), "exec"), namespace, namespace)


def test_hdf5_fixer_is_isolated_idempotent_and_generator_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_root = tmp_path / "isolated-repository"
    tutorial_root = isolated_root / "docs/web"
    authoritative: dict[str, bytes] = {}

    for locale, cells in (
        ("en", make_provenance_tutorial.EN_CELLS),
        ("ja", make_provenance_tutorial.JA_CELLS),
    ):
        path = (
            tutorial_root / locale / "user_guide/tutorials/case_hdf5_provenance.ipynb"
        )
        path.parent.mkdir(parents=True)
        make_provenance_tutorial.write_nb(cells, path)
        authoritative[locale] = path.read_bytes()

    shared_before = {locale: path.read_bytes() for locale, path in NOTEBOOKS.items()}
    monkeypatch.setattr(fix_tutorial_notebooks, "ROOT", isolated_root)

    for locale in ("en", "ja"):
        fix_tutorial_notebooks._fix_case_hdf5(locale)
        first = (
            isolated_root
            / "docs/web"
            / locale
            / "user_guide/tutorials/case_hdf5_provenance.ipynb"
        ).read_bytes()
        fix_tutorial_notebooks._fix_case_hdf5(locale)
        second = (
            isolated_root
            / "docs/web"
            / locale
            / "user_guide/tutorials/case_hdf5_provenance.ipynb"
        ).read_bytes()

        assert first == second
        assert first == authoritative[locale]

    assert {
        locale: path.read_bytes() for locale, path in NOTEBOOKS.items()
    } == shared_before


def test_bilingual_tutorials_have_matching_cell_structure_and_code_contract() -> None:
    english = _read_notebook(NOTEBOOKS["en"])
    japanese = _read_notebook(NOTEBOOKS["ja"])

    assert [cell["cell_type"] for cell in english["cells"]] == [
        cell["cell_type"] for cell in japanese["cells"]
    ]
    assert _sources(english, "code") == _sources(japanese, "code")
    assert len(_sources(english, "markdown")) == len(_sources(japanese, "markdown"))


def test_generated_notebooks_equal_generator_output(tmp_path: Path) -> None:
    generated = {
        "en": tmp_path / "en.ipynb",
        "ja": tmp_path / "ja.ipynb",
    }
    make_provenance_tutorial.write_nb(
        make_provenance_tutorial.EN_CELLS, generated["en"]
    )
    make_provenance_tutorial.write_nb(
        make_provenance_tutorial.JA_CELLS, generated["ja"]
    )

    for locale, path in NOTEBOOKS.items():
        assert generated[locale].read_bytes() == path.read_bytes()

    second = tmp_path / "second-en.ipynb"
    make_provenance_tutorial.write_nb(make_provenance_tutorial.EN_CELLS, second)
    assert generated["en"].read_bytes() == second.read_bytes()


def test_generator_has_no_unseeded_builtin_hash_for_notebook_ids() -> None:
    source = GENERATOR.read_text(encoding="utf-8")
    assert "hash(source)" not in source


def test_contract_helpers_reject_adversarial_lookalikes() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE
    mutations = {
        "unrelated_write": source.replace("ts.write", "other.write"),
        "unrelated_read": source.replace(
            "restored = gwexpy.TimeSeries.read", "restored = other.read"
        ),
        "decorative_metadata_compare": source.replace(
            "    assert restored.metadata == metadata",
            "    restored.metadata == metadata\n    assert True",
        ),
        "wrong_receiver": source.replace(
            "ts.metadata = metadata", "other.metadata = metadata"
        ),
        "internal_sidecar": source.replace(
            '    ts.write(archive_path, format="hdf5")',
            "    gwexpy.io.hdf5_sidecar.write_sidecar(ts, archive_path)\n"
            '    ts.write(archive_path, format="hdf5")',
        ),
        "imported_internal_sidecar": source.replace(
            "import tempfile",
            "from gwexpy.io.hdf5_sidecar import read_sidecar\nimport tempfile",
        ),
        "h5py": source.replace("import tempfile", "import h5py\nimport tempfile"),
        "dynamic_attrs": source.replace(
            '    ts.write(archive_path, format="hdf5")',
            '    getattr(ts, "attrs")\n    ts.write(archive_path, format="hdf5")',
        ),
        "environment_input": source.replace(
            "import tempfile", "import os\nimport tempfile"
        ),
        "unseeded_random": source.replace(
            "import tempfile", "import random\nimport tempfile"
        ),
        "stale_hdf5_method": source.replace(
            '    ts.write(archive_path, format="hdf5")',
            '    ts.to_hdf5(archive_path)\n    ts.write(archive_path, format="hdf5")',
        ),
        "external_network": source.replace(
            "import tempfile", "import requests\nimport tempfile"
        ),
    }
    for name, mutated in mutations.items():
        with pytest.raises(AssertionError, match=name):
            _validate_tutorial_code(mutated, context=name)


def test_contract_helper_rejects_static_manual_sidecar_attribute_forms() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE
    mutations = {
        "static_ts_attrs_assignment": source.replace(
            "ts.metadata = metadata",
            'ts.attrs["_gwexpy_sidecar_json_v1"] = "{}"\nts.metadata = metadata',
        ),
        "static_restored_attrs_access": source.replace(
            "assert restored.t0_gps_ns == 1234567890123456789",
            'restored.attrs["_gwexpy_sidecar_json_v1"]\n'
            "    assert restored.t0_gps_ns == 1234567890123456789",
        ),
        "static_file_attrs_assignment": source.replace(
            'archive_path = Path(temporary_directory) / "provenance.h5"',
            'file.attrs["_gwexpy_sidecar_json_v1"] = "{}"\n'
            '    archive_path = Path(temporary_directory) / "provenance.h5"',
        ),
    }

    for name, mutated in mutations.items():
        with pytest.raises(AssertionError, match=name):
            _validate_tutorial_code(mutated, context=name)


def test_contract_helper_rejects_subprocess_and_shell_process_variants() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE
    mutations = {
        "subprocess_import": source.replace(
            "import tempfile", "import subprocess\nimport tempfile"
        ),
        "subprocess_from_import": source.replace(
            "import tempfile", "from subprocess import run\nimport tempfile"
        ),
        "subprocess_run": source.replace(
            "import tempfile",
            'import subprocess\nsubprocess.run(["echo", "blocked"])\nimport tempfile',
        ),
        "subprocess_popen": source.replace(
            "import tempfile",
            'import subprocess\nsubprocess.Popen(["echo", "blocked"])\nimport tempfile',
        ),
        "shell_system_call": source.replace(
            "import tempfile", 'shell.system("echo blocked")\nimport tempfile'
        ),
        "shell_popen_call": source.replace(
            "import tempfile", 'shell.popen("echo blocked")\nimport tempfile'
        ),
        "process_run_call": source.replace(
            "import tempfile", 'process.run(["echo", "blocked"])\nimport tempfile'
        ),
        "process_popen_call": source.replace(
            "import tempfile",
            'process.Popen(["echo", "blocked"])\nimport tempfile',
        ),
    }

    for name, mutated in mutations.items():
        with pytest.raises(AssertionError, match=name):
            _validate_tutorial_code(mutated, context=name)


def test_contract_helper_rejects_archive_path_reassignment_between_io_calls() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE.replace(
        '    ts.write(archive_path, format="hdf5")',
        '    ts.write(archive_path, format="hdf5")\n'
        '    archive_path = Path(temporary_directory) / "other.h5"',
    )

    with pytest.raises(AssertionError, match="simple_reassignment"):
        _validate_tutorial_code(source, context="simple_reassignment")


def test_contract_helper_rejects_annotated_archive_path_reassignment() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE.replace(
        '    ts.write(archive_path, format="hdf5")',
        '    ts.write(archive_path, format="hdf5")\n'
        '    archive_path: Path = Path(temporary_directory) / "other.h5"',
    )

    with pytest.raises(AssertionError, match="annotated_reassignment"):
        _validate_tutorial_code(source, context="annotated_reassignment")


def test_contract_helper_rejects_augmented_archive_path_reassignment() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE.replace(
        '    ts.write(archive_path, format="hdf5")',
        '    ts.write(archive_path, format="hdf5")\n    archive_path /= "other.h5"',
    )

    with pytest.raises(AssertionError, match="augmented_reassignment"):
        _validate_tutorial_code(source, context="augmented_reassignment")


def test_contract_helper_rejects_walrus_archive_path_reassignment() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE.replace(
        '    ts.write(archive_path, format="hdf5")',
        '    ts.write(archive_path, format="hdf5")\n'
        '    marker = (archive_path := Path(temporary_directory) / "other.h5")',
    )

    with pytest.raises(AssertionError, match="walrus_reassignment"):
        _validate_tutorial_code(source, context="walrus_reassignment")


def test_contract_helper_rejects_deleted_archive_path() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE.replace(
        '    ts.write(archive_path, format="hdf5")',
        '    ts.write(archive_path, format="hdf5")\n    del archive_path',
    )

    with pytest.raises(AssertionError, match="deleted_archive_path"):
        _validate_tutorial_code(source, context="deleted_archive_path")


def test_contract_helper_rejects_mutated_archive_path() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE.replace(
        '    ts.write(archive_path, format="hdf5")',
        '    ts.write(archive_path, format="hdf5")\n    archive_path.touch()',
    )

    with pytest.raises(AssertionError, match="mutated_archive_path"):
        _validate_tutorial_code(source, context="mutated_archive_path")


def test_contract_helper_rejects_alias_to_different_archive_path() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE.replace(
        '    ts.write(archive_path, format="hdf5")',
        '    other_path = Path(temporary_directory) / "other.h5"\n'
        '    ts.write(other_path, format="hdf5")',
    ).replace(
        'gwexpy.TimeSeries.read(archive_path, format="hdf5")',
        'gwexpy.TimeSeries.read(other_path, format="hdf5")',
    )

    with pytest.raises(AssertionError, match="alias_to_different_path"):
        _validate_tutorial_code(source, context="alias_to_different_path")


def test_contract_helper_rejects_archive_path_rebinding_in_control_flow() -> None:
    source = make_provenance_tutorial.ROUNDTRIP_CODE.replace(
        '    ts.write(archive_path, format="hdf5")',
        '    ts.write(archive_path, format="hdf5")\n'
        "    if True:\n"
        '        archive_path = Path(temporary_directory) / "other.h5"',
    )

    with pytest.raises(AssertionError, match="control_flow_rebinding"):
        _validate_tutorial_code(source, context="control_flow_rebinding")


def test_contract_helper_rejects_v020_release_claim_in_markdown() -> None:
    markdown = "The tutorial is compatible with v0.2.0, which has been released."

    with pytest.raises(AssertionError, match="release"):
        _validate_tutorial_markdown(markdown)


@pytest.mark.parametrize(
    "markdown",
    (
        "v0.2.0 is published.",
        "This release is released.",
        "The tutorial is now shipped.",
        "Published.",
        "Released.",
    ),
)
def test_contract_helper_rejects_standalone_publication_claims(markdown: str) -> None:
    with pytest.raises(AssertionError, match="release"):
        _validate_tutorial_markdown(markdown)


@pytest.mark.parametrize(
    "markdown",
    (
        "Unreleased implementation documentation.",
        "This tutorial is not released.",
        "Resources are released by a context manager.",
    ),
)
def test_contract_helper_allows_neutral_release_wording(markdown: str) -> None:
    _validate_tutorial_markdown(markdown)
