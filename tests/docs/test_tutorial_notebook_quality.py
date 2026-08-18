import ast
import json
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from tests.docs import test_docs_notebooks as docs_notebooks

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_ROOT = ROOT / "docs" / "web"
FORBIDDEN_OUTPUT_PATTERNS = [
    re.compile(r"/home/"),
    re.compile(r"/tmp/"),
    re.compile(r"\bUserWarning\b"),
    re.compile(r"\bDeprecationWarning\b"),
    re.compile(r"\bConvergenceWarning\b"),
]
FORBIDDEN_PUBLIC_DOC_LINK_PATTERNS = [
    re.compile(r"docs/developers/"),
    re.compile(r"docs_internal/"),
    re.compile(r"API_MAPPING\.md"),
]
STALE_TUTORIAL_CODE_SNIPPETS = [
    "plt.gca().get_images()",
    "plt.gca().collections[-1]",
    "plt.gca().get_children()",
    "hasattr(c, 'get_clim')",
    "spec = data.hht(",
]
COLAB_BADGE_IMAGE_PATTERN = re.compile(
    r"!\[([^\]]+)\]\(https://colab\.research\.google\.com/assets/colab-badge\.svg\)"
)
COLAB_BADGE_LABELS = {"en": "Open In Colab", "ja": "Colab で開く"}
COLAB_BADGE_GENERATOR_LABEL_COUNTS = {
    Path("scripts/fix_tutorial_notebooks.py"): {"en": 1, "ja": 0},
    Path("scripts/make_bruco_advanced_notebook.py"): {"en": 1, "ja": 1},
    Path("scripts/make_bruco_ica_notebook.py"): {"en": 1, "ja": 1},
    Path("scripts/make_peak_tracking_notebook.py"): {"en": 1, "ja": 1},
    Path("scripts/make_schumann_notebook.py"): {"en": 1, "ja": 1},
    Path("scripts/make_spectrogram_processing_notebook.py"): {"en": 1, "ja": 1},
    Path("scripts/make_violin_mode_notebook.py"): {"en": 1, "ja": 1},
    Path("scripts/notebook_gen/make_arima_burst_notebook.py"): {"en": 0, "ja": 1},
}
COLAB_BADGE_GENERATOR_SECTION_MARKERS = {
    Path("scripts/make_bruco_advanced_notebook.py"): {
        "en": ("# English cells", "# Japanese cells"),
        "ja": ("# Japanese cells", None),
    },
    Path("scripts/make_bruco_ica_notebook.py"): {
        "en": ("# English notebook", "# Japanese notebook"),
        "ja": ("# Japanese notebook", None),
    },
    Path("scripts/make_peak_tracking_notebook.py"): {
        "en": ("# English cells", "# Japanese cells"),
        "ja": ("# Japanese cells", None),
    },
    Path("scripts/make_schumann_notebook.py"): {
        "en": ("# English cells", "# Japanese cells"),
        "ja": ("# Japanese cells", None),
    },
    Path("scripts/make_spectrogram_processing_notebook.py"): {
        "en": ("# English cells", "# Japanese cells"),
        "ja": ("# Japanese cells", None),
    },
    Path("scripts/make_violin_mode_notebook.py"): {
        "en": ("# English cells", "# Japanese cells"),
        "ja": ("# Japanese cells", None),
    },
}


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _localized_tutorial_path(relative_path: Path) -> Path:
    path = TUTORIAL_ROOT / relative_path
    if path.exists():
        return path
    parts = relative_path.parts
    if parts and parts[0] == "ja":
        return TUTORIAL_ROOT / Path("en", *parts[1:])
    return path


def _read_tutorial_notebook(relative_path: Path) -> dict:
    return _read_notebook(_localized_tutorial_path(relative_path))


def _public_tutorial_notebooks() -> list[Path]:
    return sorted(TUTORIAL_ROOT.glob("*/user_guide/tutorials/*.ipynb"))


def _public_tutorial_markdown_files() -> list[Path]:
    return sorted(TUTORIAL_ROOT.glob("*/user_guide/tutorials/*.md"))


def _notebook_locale(relative_path: Path) -> str:
    return relative_path.parts[0]


def _code_cell_source_containing(nb: dict, text: str) -> str:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if text in source:
            return source
    raise AssertionError(f"Could not find code cell containing {text!r}")


@contextmanager
def _pushd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _iter_output_texts(nb: dict):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            chunks: list[str] = []
            text = output.get("text")
            if isinstance(text, list):
                chunks.extend(text)
            elif isinstance(text, str):
                chunks.append(text)
            for mime, payload in output.get("data", {}).items():
                if not mime.startswith("text/"):
                    continue
                if isinstance(payload, list):
                    chunks.extend(payload)
                elif isinstance(payload, str):
                    chunks.append(payload)
            joined = "".join(chunks)
            if joined:
                yield joined


def _markdown_texts(nb: dict) -> list[str]:
    texts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        texts.append("".join(source) if isinstance(source, list) else str(source))
    return texts


def _colab_image_alternative_labels(nb: dict) -> list[str]:
    return [
        label
        for markdown in _markdown_texts(nb)
        for label in COLAB_BADGE_IMAGE_PATTERN.findall(markdown)
    ]


def _localized_colab_image_alternative_labels(nb: dict, locale: str) -> list[str]:
    wanted_tag = f"lang-{locale}"
    return [
        label
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "markdown"
        and wanted_tag in cell.get("metadata", {}).get("tags", [])
        for label in COLAB_BADGE_IMAGE_PATTERN.findall("".join(cell.get("source", [])))
    ]


def _colab_badge_label_counts(text: str) -> dict[str, int]:
    labels = Counter(COLAB_BADGE_IMAGE_PATTERN.findall(text))
    return {
        locale: labels[expected_label]
        for locale, expected_label in COLAB_BADGE_LABELS.items()
    }


def _generator_cell_section(
    text: str, start_marker: str, end_marker: str | None
) -> str:
    section = text.split(start_marker, maxsplit=1)[1]
    return section if end_marker is None else section.split(end_marker, maxsplit=1)[0]


def test_english_tutorial_colab_badge_labels_are_unique_within_notebook():
    tutorial_root = TUTORIAL_ROOT / "en" / "user_guide" / "tutorials"
    duplicate_labels_by_notebook = {
        path.relative_to(ROOT): sorted(
            label
            for label, count in Counter(
                _colab_image_alternative_labels(_read_notebook(path))
            ).items()
            if count > 1
        )
        for path in sorted(tutorial_root.glob("*.ipynb"))
    }
    duplicate_labels_by_notebook = {
        path: labels for path, labels in duplicate_labels_by_notebook.items() if labels
    }

    assert not duplicate_labels_by_notebook, (
        "Colab image alternative labels must be unique within each source "
        f"notebook: {duplicate_labels_by_notebook}"
    )


def test_canonical_tutorial_colab_badges_have_locale_aware_labels():
    canonical_badge_labels = {
        path.relative_to(ROOT): {
            locale: _localized_colab_image_alternative_labels(
                _read_notebook(path), locale
            )
            for locale in COLAB_BADGE_LABELS
        }
        for path in sorted(
            (TUTORIAL_ROOT / "en" / "user_guide" / "tutorials").glob("*.ipynb")
        )
        if any(
            _localized_colab_image_alternative_labels(_read_notebook(path), locale)
            for locale in COLAB_BADGE_LABELS
        )
    }

    assert len(canonical_badge_labels) == 40
    assert all(
        labels
        == {
            locale: [expected_label]
            for locale, expected_label in COLAB_BADGE_LABELS.items()
        }
        for labels in canonical_badge_labels.values()
    )


def test_tutorial_colab_badge_generators_have_localized_labels_in_cell_sections():
    generator_sources = {
        path: (ROOT / path).read_text() for path in COLAB_BADGE_GENERATOR_LABEL_COUNTS
    }
    assert {
        path: _colab_badge_label_counts(source)
        for path, source in generator_sources.items()
    } == COLAB_BADGE_GENERATOR_LABEL_COUNTS
    assert {
        path: {
            locale: _colab_badge_label_counts(
                _generator_cell_section(generator_sources[path], *markers)
            )
            for locale, markers in sections.items()
        }
        for path, sections in COLAB_BADGE_GENERATOR_SECTION_MARKERS.items()
    } == {
        path: {
            locale: {
                label_locale: int(label_locale == locale)
                for label_locale in COLAB_BADGE_LABELS
            }
            for locale in sections
        }
        for path, sections in COLAB_BADGE_GENERATOR_SECTION_MARKERS.items()
    }


def _localized_markdown_texts(nb: dict, locale: str) -> list[str]:
    localized: list[str] = []
    wanted_tag = f"lang-{locale}"
    other_i18n_tags = {"lang-en", "lang-ja"} - {wanted_tag}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []))
        if wanted_tag in tags or not (tags & other_i18n_tags):
            source = cell.get("source", [])
            localized.append(
                "".join(source) if isinstance(source, list) else str(source)
            )
    return localized


def _code_text(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _code_cell_sources(nb: dict) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


CONSTRUCTOR_BOOTSTRAP_CASES = {
    Path("en/user_guide/tutorials/intro_noise.ipynb"),
    Path("en/user_guide/tutorials/intro_fitting.ipynb"),
}


@dataclass(frozen=True)
class _NotebookEvent:
    cell_index: int
    kind: str
    qualified_name: tuple[str, ...] | None = None


def _parse_notebook_code_cell(source: str, cell_index: int) -> ast.Module:
    normalized = "\n".join(
        f"#{line}" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )
    try:
        return ast.parse(normalized)
    except SyntaxError as exc:
        raise AssertionError(
            f"code cell {cell_index} is not statically parseable"
        ) from exc


class _NotebookDataFlow(ast.NodeVisitor):
    def __init__(self) -> None:
        self.cell_index = 0
        self.bindings: dict[str, tuple[str, ...]] = {}
        self.origins: dict[str, str] = {}
        self.static_root_aliases: set[str] = set()
        self.events: list[_NotebookEvent] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            qualified = tuple(alias.name.split("."))
            if alias.name == "gwexpy":
                name = alias.asname or "gwexpy"
                self.bindings[name] = qualified
                self.static_root_aliases.add(name)
            elif alias.name.startswith("gwexpy."):
                name = alias.asname or alias.name.split(".")[0]
                self.bindings[name] = qualified if alias.asname else ("gwexpy",)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None or not node.module.startswith("gwexpy"):
            return
        module = tuple(node.module.split("."))
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            self.bindings[name] = module + (alias.name,)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        origin = self._origin(node.value)
        for target in node.targets:
            self._assign_target(target, origin, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            origin = self._origin(node.value)
            self._assign_target(node.target, origin, node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._assign_target(node.target, self._origin(node.value), node.value)

    def visit_Call(self, node: ast.Call) -> None:
        self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

        kind = self._call_kind(node)
        if kind is not None:
            self.events.append(
                _NotebookEvent(
                    self.cell_index,
                    kind,
                    self._qualified_name(node.func),
                )
            )

    def _assign_target(
        self,
        target: ast.AST,
        origin: str | None,
        value: ast.AST,
    ) -> None:
        if isinstance(target, ast.Name):
            self.origins.pop(target.id, None)
            self.static_root_aliases.discard(target.id)
            qualified = self._qualified_name(value)
            if qualified is not None:
                self.bindings[target.id] = qualified
                if qualified == ("gwexpy",):
                    self.static_root_aliases.add(target.id)
            else:
                self.bindings.pop(target.id, None)
            if origin is not None:
                self.origins[target.id] = origin
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._assign_target(element, origin, value)

    def _qualified_name(self, node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            parent = self._qualified_name(node.value)
            return None if parent is None else parent + (node.attr,)
        return None

    def _origin(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in self.origins:
                return self.origins[node.id]
            qualified = self.bindings.get(node.id)
            return "callable" if qualified and qualified[0] == "gwexpy" else None
        if isinstance(node, ast.Attribute):
            if self._is_gwexpy_value(node.value):
                return "callable"
            qualified = self._qualified_name(node)
            return "callable" if qualified and qualified[0] == "gwexpy" else None
        if isinstance(node, ast.Subscript):
            return self._origin(node.value)
        if isinstance(node, ast.Call) and self._call_kind(node) == "dependent":
            return "value"
        return None

    def _is_gwexpy_value(self, node: ast.AST) -> bool:
        return self._origin(node) == "value" or (
            (qualified := self._qualified_name(node)) is not None
            and qualified[0] == "gwexpy"
        )

    def _call_kind(self, node: ast.Call) -> str | None:
        qualified = self._qualified_name(node.func)
        if self._is_exact_bootstrap(node, qualified):
            return "bootstrap"
        if qualified is not None and qualified[0] == "gwexpy":
            return "dependent"
        if (
            isinstance(node.func, ast.Name)
            and self.origins.get(node.func.id) == "callable"
        ):
            return "dependent"
        if isinstance(node.func, ast.Attribute) and self._is_gwexpy_value(
            node.func.value
        ):
            return "dependent"
        return None

    def _is_exact_bootstrap(
        self, node: ast.Call, qualified: tuple[str, ...] | None
    ) -> bool:
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.static_root_aliases
            and qualified == ("gwexpy", "register_all")
            and not node.args
            and len(node.keywords) == 1
            and node.keywords[0].arg == "include_io"
            and isinstance(node.keywords[0].value, ast.Constant)
            and node.keywords[0].value.value is False
        )


def _notebook_structural_events(nb: dict) -> list[_NotebookEvent]:
    events: list[_NotebookEvent] = []
    analyzer = _NotebookDataFlow()
    for cell_index, source in enumerate(_code_cell_sources(nb)):
        analyzer.cell_index = cell_index
        event_start = len(analyzer.events)
        analyzer.visit(_parse_notebook_code_cell(source, cell_index))
        events.extend(analyzer.events[event_start:])
    return events


def _constructor_bootstrap_violations(nb: dict) -> list[str]:
    events = _notebook_structural_events(nb)
    bootstrap_indices = [
        index for index, event in enumerate(events) if event.kind == "bootstrap"
    ]
    violations = []
    if len(bootstrap_indices) != 1:
        violations.append("expected exactly one explicit constructor bootstrap")
        return violations

    first_bootstrap = bootstrap_indices[0]
    violations.extend(
        f"cell {event.cell_index}: dependent operation before bootstrap"
        for event in events[:first_bootstrap]
        if event.kind == "dependent"
    )
    return violations


def _notebook_with_code(*sources: str) -> dict:
    return {
        "cells": [
            {"cell_type": "code", "source": source, "metadata": {}, "outputs": []}
            for source in sources
        ]
    }


def test_constructor_dependent_notebooks_bootstrap_registry_before_use():
    for relative_path in CONSTRUCTOR_BOOTSTRAP_CASES:
        nb = _read_tutorial_notebook(relative_path)
        assert _constructor_bootstrap_violations(nb) == []


@pytest.mark.parametrize(
    "source_before_bootstrap",
    [
        (
            "import gwexpy as gp\n"
            "from gwexpy.noise import wave\n"
            "noise = wave.pink_noise(duration=1, sample_rate=8, amplitude=1)\n"
            "gp.register_all(include_io=False)\n"
        ),
        (
            "from gwexpy.fitting.highlevel import fit_bootstrap_spectrum as fit\n"
            "fit(data, model_fn=model, plot=True)\n"
            "import gwexpy\n"
            "gwexpy.register_all(include_io=False)\n"
        ),
        (
            "from gwexpy.frequencyseries import FrequencySeries as FS\n"
            "spectrum = FS(values, frequencies=freqs)\n"
            "import gwexpy\n"
            "gwexpy.register_all(include_io=False)\n"
        ),
        (
            "import gwexpy\n"
            "from gwexpy.noise import wave\n"
            "noise = wave.pink_noise(duration=1, sample_rate=8, amplitude=1)\n"
            "plot = noise.plot\n"
            "plot()\n"
            "gwexpy.register_all(include_io=False)\n"
        ),
    ],
)
def test_constructor_bootstrap_guard_catches_alias_and_same_cell_operations(
    source_before_bootstrap: str,
):
    violations = _constructor_bootstrap_violations(
        _notebook_with_code(source_before_bootstrap)
    )
    assert violations, "relevant operations before bootstrap must be rejected"


def test_constructor_bootstrap_guard_ignores_text_that_is_not_executed():
    nb = _notebook_with_code(
        'description = "noise_ts.plot() fit_bootstrap_spectrum("\n'
        "import gwexpy\n"
        "gwexpy.register_all(include_io=False)\n"
    )
    assert _constructor_bootstrap_violations(nb) == []


def test_constructor_bootstrap_guard_tracks_data_flow_across_code_cells():
    nb = _notebook_with_code(
        "import gwexpy as gp\nfrom gwexpy.noise import wave\n",
        "noise = wave.pink_noise(duration=1, sample_rate=8, amplitude=1)\n",
        "plot = noise.plot\n",
        "plot()\n",
        "gp.register_all(include_io=False)\n",
    )
    violations = _constructor_bootstrap_violations(nb)
    assert any("dependent operation" in violation for violation in violations)


def test_notebook_kernel_uses_current_interpreter_and_gate_environment():
    if (
        os.environ.get("PYTHONNOUSERSITE") != "1"
        or os.environ.get("PYTHONPATH") != str(docs_notebooks.REPO_ROOT)
        or sys.version_info[:2] != (3, 11)
    ):
        pytest.skip("requires the pinned docs-notebook gate environment")
    assert docs_notebooks._kernel_spec_argv()[0] == sys.executable
    assert sys.version_info[:2] == (3, 11)
    environment = docs_notebooks._notebook_environment()
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"] == str(docs_notebooks.REPO_ROOT)
    assert environment["PATH"] == os.environ["PATH"]


def _call_function_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_mappable_plot_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_function_name(node.func) in {
        "imshow",
        "pcolormesh",
    }


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for element in target.elts for name in _assigned_names(element)}
    return set()


def _explicit_colorbar_mappables_from_plot_assignments(nb: dict) -> list[str]:
    sources = _code_cell_sources(nb)
    parsed_cells: list[ast.Module] = []
    unparsed_colorbar_cells: list[int] = []

    for index, source in enumerate(sources, start=1):
        try:
            parsed_cells.append(ast.parse(source))
        except SyntaxError:
            if "colorbar(" in source:
                unparsed_colorbar_cells.append(index)

    assert not unparsed_colorbar_cells, (
        "Could not parse code cells containing colorbar calls: "
        + ", ".join(str(index) for index in unparsed_colorbar_cells)
    )

    class ColorbarMappableVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.valid_mappables: set[str] = set()
            self.explicit_mappables: list[str] = []
            self.invalid_colorbars: list[str] = []

        def visit_Assign(self, node: ast.Assign) -> None:
            self.visit(node.value)
            target_names = {
                name for target in node.targets for name in _assigned_names(target)
            }
            self._record_assignment(target_names, _is_mappable_plot_call(node.value))

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.annotation is not None:
                self.visit(node.annotation)
            if node.value is None:
                return
            self.visit(node.value)
            self._record_assignment(
                _assigned_names(node.target), _is_mappable_plot_call(node.value)
            )

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            self.visit(node.value)
            self._record_assignment(
                _assigned_names(node.target), _is_mappable_plot_call(node.value)
            )

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self.visit(node.target)
            self.visit(node.value)
            self.valid_mappables.difference_update(_assigned_names(node.target))

        def visit_For(self, node: ast.For) -> None:
            self.visit(node.iter)
            self.valid_mappables.difference_update(_assigned_names(node.target))
            for statement in node.body:
                self.visit(statement)
            for statement in node.orelse:
                self.visit(statement)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.visit(node.iter)
            self.valid_mappables.difference_update(_assigned_names(node.target))
            for statement in node.body:
                self.visit(statement)
            for statement in node.orelse:
                self.visit(statement)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            self.visit(node.iter)
            self.valid_mappables.difference_update(_assigned_names(node.target))
            for condition in node.ifs:
                self.visit(condition)

        def visit_Call(self, node: ast.Call) -> None:
            if not (
                isinstance(node, ast.Call)
                and _call_function_name(node.func) == "colorbar"
            ):
                self.generic_visit(node)
                return

            mappable = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "mappable"
                ),
                None,
            )
            if mappable is None:
                self.invalid_colorbars.append(
                    f"line {node.lineno}: colorbar call lacks mappable="
                )
            elif not isinstance(mappable, ast.Name):
                self.invalid_colorbars.append(
                    f"line {node.lineno}: mappable= is not a simple assigned name"
                )
            elif mappable.id not in self.valid_mappables:
                self.invalid_colorbars.append(
                    f"line {node.lineno}: mappable={mappable.id} is not assigned "
                    "from imshow/pcolormesh"
                )
            else:
                self.explicit_mappables.append(mappable.id)

            self.generic_visit(node)

        def _record_assignment(
            self, target_names: set[str], is_mappable_assignment: bool
        ) -> None:
            if is_mappable_assignment:
                self.valid_mappables.update(target_names)
            else:
                self.valid_mappables.difference_update(target_names)

    visitor = ColorbarMappableVisitor()
    for tree in parsed_cells:
        visitor.visit(tree)

    assert not visitor.invalid_colorbars, "Invalid colorbar calls:\n" + "\n".join(
        visitor.invalid_colorbars
    )
    return visitor.explicit_mappables


def _synthetic_notebook(*sources: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": source,
            }
            for source in sources
        ]
    }


def test_colorbar_mappable_guard_rejects_use_before_assignment():
    nb = _synthetic_notebook(
        "plt.colorbar(mappable=mesh)\n",
        "mesh = ax.pcolormesh(x, y, z)\n",
    )

    with pytest.raises(AssertionError, match="mappable=mesh is not assigned"):
        _explicit_colorbar_mappables_from_plot_assignments(nb)


def test_colorbar_mappable_guard_rejects_stale_reassignment():
    nb = _synthetic_notebook(
        "mesh = ax.pcolormesh(x, y, z)\n",
        "mesh = None\n",
        "plt.colorbar(mappable=mesh)\n",
    )

    with pytest.raises(AssertionError, match="mappable=mesh is not assigned"):
        _explicit_colorbar_mappables_from_plot_assignments(nb)


def test_colorbar_mappable_guard_accepts_current_pcolormesh_assignment():
    nb = _synthetic_notebook(
        "mesh = ax.pcolormesh(x, y, z)\n",
        "plt.colorbar(mappable=mesh)\n",
    )

    assert _explicit_colorbar_mappables_from_plot_assignments(nb) == ["mesh"]


def test_tutorial_outputs_do_not_expose_local_paths_or_raw_warnings():
    notebooks = sorted(TUTORIAL_ROOT.glob("*/user_guide/tutorials/*.ipynb"))
    offenders: list[str] = []

    for path in notebooks:
        nb = _read_notebook(path)
        for text in _iter_output_texts(nb):
            hit = next(
                (pat.pattern for pat in FORBIDDEN_OUTPUT_PATTERNS if pat.search(text)),
                None,
            )
            if hit:
                offenders.append(f"{path.relative_to(ROOT)} -> {hit}")
                break

    assert not offenders, "Forbidden notebook output found:\n" + "\n".join(offenders)


def test_public_tutorial_markdown_does_not_link_internal_docs_surfaces():
    offenders: list[str] = []

    for path in _public_tutorial_notebooks():
        nb = _read_notebook(path)
        for markdown in _markdown_texts(nb):
            hit = next(
                (
                    pattern.pattern
                    for pattern in FORBIDDEN_PUBLIC_DOC_LINK_PATTERNS
                    if pattern.search(markdown)
                ),
                None,
            )
            if hit:
                offenders.append(f"{path.relative_to(ROOT)} -> {hit}")
                break

    for path in _public_tutorial_markdown_files():
        markdown = path.read_text()
        hit = next(
            (
                pattern.pattern
                for pattern in FORBIDDEN_PUBLIC_DOC_LINK_PATTERNS
                if pattern.search(markdown)
            ),
            None,
        )
        if hit:
            offenders.append(f"{path.relative_to(ROOT)} -> {hit}")

    assert not offenders, "Forbidden internal-doc link found:\n" + "\n".join(offenders)


def test_public_tutorial_code_does_not_use_known_stale_hht_or_colorbar_patterns():
    offenders: list[str] = []

    for path in _public_tutorial_notebooks():
        joined = _code_text(_read_notebook(path))
        hit = next(
            (snippet for snippet in STALE_TUTORIAL_CODE_SNIPPETS if snippet in joined),
            None,
        )
        if hit:
            offenders.append(f"{path.relative_to(ROOT)} -> {hit}")

    for path in _public_tutorial_markdown_files():
        markdown = path.read_text()
        hit = next(
            (
                snippet
                for snippet in STALE_TUTORIAL_CODE_SNIPPETS
                if snippet in markdown
            ),
            None,
        )
        if hit:
            offenders.append(f"{path.relative_to(ROOT)} -> {hit}")

    assert not offenders, "Stale tutorial pattern found:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/intro_interop.ipynb"),
        Path("ja/user_guide/tutorials/intro_interop.ipynb"),
    ],
)
def test_intro_interop_uses_explicit_axes_for_pandas_plot(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    source = _code_cell_source_containing(nb, 's_pd = ts.to_pandas(index="datetime")')

    assert "fig, ax = plt.subplots()" in source
    assert 's_pd.plot(ax=ax, title="Pandas Series")' in source
    assert "plt.close(fig)" in source


def test_intro_interop_explains_all_extra_scope():
    nb = _read_tutorial_notebook(Path("en/user_guide/tutorials/intro_interop.ipynb"))
    markdown = " ".join("\n".join(_markdown_texts(nb)).split())

    assert "`gwexpy[all]` installs the declared GWexpy extras" in markdown
    assert "does not install every public interop backend" in markdown


def test_example_intro_interop_uses_explicit_axes_for_pandas_plot():
    nb = _read_notebook(ROOT / "examples" / "basic-new-methods" / "intro_Interop.ipynb")
    source = _code_cell_source_containing(nb, 's_pd = ts.to_pandas(index="datetime")')

    assert "fig, ax = plt.subplots()" in source
    assert 's_pd.plot(ax=ax, title="Pandas Series")' in source
    assert "plt.close(fig)" in source


def test_ja_advanced_coupling_mentions_frequency_range_restriction():
    relative_path = Path("ja/user_guide/tutorials/advanced_coupling.ipynb")
    nb = _read_tutorial_notebook(relative_path)
    joined = "\n".join(_localized_markdown_texts(nb, _notebook_locale(relative_path)))
    assert "周波数帯域" in joined or "frange" in joined


def test_ja_case_seismic_obspy_includes_multichannel_section():
    relative_path = Path("ja/user_guide/tutorials/case_seismic_obspy.ipynb")
    nb = _read_tutorial_notebook(relative_path)
    joined = "\n".join(_localized_markdown_texts(nb, _notebook_locale(relative_path)))
    assert "マルチチャンネル" in joined or "3成分" in joined


def test_en_case_arima_burst_search_is_actually_english():
    relative_path = Path("en/user_guide/tutorials/case_arima_burst_search.ipynb")
    nb = _read_tutorial_notebook(relative_path)
    first_markdown = _localized_markdown_texts(nb, _notebook_locale(relative_path))[0]
    assert "# ARIMA-Based Burst Detection" in first_markdown
    assert "## Introduction" in first_markdown


def test_en_case_arima_burst_search_has_markdown_sections_not_code():
    nb = _read_tutorial_notebook(
        Path("en/user_guide/tutorials/case_arima_burst_search.ipynb")
    )
    code_texts = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    assert all("[![Open In Colab]" not in text for text in code_texts)
    assert all(
        not text.lstrip().startswith("## 1. Generate detector noise")
        for text in code_texts
    )


def test_advanced_arima_notebooks_are_tagged_ci_heavy():
    for relative_path in (
        Path("en/user_guide/tutorials/advanced_bruco.ipynb"),
        Path("en/user_guide/tutorials/advanced_arima.ipynb"),
        Path("en/user_guide/tutorials/advanced_correlation.ipynb"),
        Path("en/user_guide/tutorials/advanced_fitting.ipynb"),
        Path("en/user_guide/tutorials/advanced_peak_tracking.ipynb"),
        Path("en/user_guide/tutorials/advanced_spectrogram_processing.ipynb"),
        Path("en/user_guide/tutorials/case_bootstrap_gls_fitting.ipynb"),
        Path("en/user_guide/tutorials/case_gbd_format.ipynb"),
        Path("en/user_guide/tutorials/case_transfer_function.ipynb"),
        Path("en/user_guide/tutorials/intro_interop.ipynb"),
        Path("en/user_guide/tutorials/intro_plotting.ipynb"),
        Path("en/user_guide/tutorials/intro_timeseries.ipynb"),
        Path("en/user_guide/tutorials/matrix_frequencyseries.ipynb"),
        Path("en/user_guide/tutorials/matrix_spectrogram.ipynb"),
        Path("en/user_guide/tutorials/matrix_timeseries.ipynb"),
        Path("en/user_guide/tutorials/rayleigh_gauch_tutorial.ipynb"),
        Path("en/user_guide/tutorials/advanced_decomposition.ipynb"),
    ):
        nb = _read_tutorial_notebook(relative_path)
        tags = nb.get("cells", [{}])[0].get("metadata", {}).get("tags", [])
        assert "ci-heavy" in tags


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/intro_table.ipynb"),
        Path("ja/user_guide/tutorials/intro_table.ipynb"),
    ],
)
def test_intro_table_sample_csv_resolves_from_repo_root(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    source = _code_cell_source_containing(nb, "sample_segment_data.csv")

    namespace: dict[str, object] = {}
    with _pushd(ROOT):
        exec(source, namespace)

    sample_csv = cast(Path, namespace["sample_csv"])
    assert (
        sample_csv.resolve()
        == (ROOT / "docs" / "_static" / "samples" / "sample_segment_data.csv").resolve()
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/case_bootstrap_gls_fitting.ipynb"),
        Path("ja/user_guide/tutorials/case_bootstrap_gls_fitting.ipynb"),
    ],
)
def test_case_bootstrap_gls_fitting_uses_explicit_mappables_for_colorbars(
    relative_path: Path,
):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "plt.gca().get_images()" not in joined
    assert "plt.gca().collections[-1]" not in joined
    assert "plt.colorbar(mappable=im, ax=ax1" in joined
    assert "plt.colorbar(mappable=im3, ax=ax3" in joined
    assert "plt.colorbar(mappable=im4, ax=ax4" in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/advanced_hht.ipynb"),
        Path("ja/user_guide/tutorials/advanced_hht.ipynb"),
    ],
)
def test_advanced_hht_uses_explicit_mappables_for_colorbars(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "plt.gca().get_images()" not in joined
    assert "plt.gca().collections[-1]" not in joined
    assert 'plt.colorbar(mappable=mesh, ax=ax1, label="Power")' in joined
    assert "sc = None" in joined
    assert "if sc is not None:" in joined
    assert "cbar = plt.colorbar(mappable=sc, ax=ax2)" in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/advanced_hht.ipynb"),
        Path("ja/user_guide/tutorials/advanced_hht.ipynb"),
    ],
)
def test_advanced_hht_spectrogram_example_calls_hht_on_timeseries(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "spec = data.hht(" not in joined
    assert "spec = ts_norm.hht(" in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/advanced_control_basics.ipynb"),
        Path("ja/user_guide/tutorials/advanced_control_basics.ipynb"),
        Path("en/user_guide/tutorials/advanced_control_discretization.ipynb"),
        Path("ja/user_guide/tutorials/advanced_control_discretization.ipynb"),
        Path("en/user_guide/tutorials/advanced_control_modeling.ipynb"),
        Path("ja/user_guide/tutorials/advanced_control_modeling.ipynb"),
        Path("en/user_guide/tutorials/case_coupling_analysis.ipynb"),
        Path("ja/user_guide/tutorials/case_coupling_analysis.ipynb"),
        Path("en/user_guide/tutorials/case_lockin_detection.ipynb"),
        Path("ja/user_guide/tutorials/case_lockin_detection.ipynb"),
        Path("en/user_guide/tutorials/case_signal_extraction.ipynb"),
        Path("ja/user_guide/tutorials/case_signal_extraction.ipynb"),
        Path("en/user_guide/tutorials/case_wiener_filter.ipynb"),
        Path("ja/user_guide/tutorials/case_wiener_filter.ipynb"),
        Path("ja/user_guide/tutorials/advanced_hht.ipynb"),
    ],
)
def test_non_fitting_tutorials_keep_source_notebooks_clean(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    assert not any(
        cell.get("cell_type") == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
        for cell in nb.get("cells", [])
    ), f"Expected clean committed notebook source in {relative_path}"


def test_ja_advanced_hht_keeps_note_in_markdown_not_code():
    relative_path = Path("ja/user_guide/tutorials/advanced_hht.ipynb")
    nb = _read_tutorial_notebook(relative_path)
    first_code = next(
        cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"
    )

    first_code_source = "".join(first_code.get("source", []))
    first_markdown_source = _localized_markdown_texts(
        nb, _notebook_locale(relative_path)
    )[0]

    assert "ワークフロー重視" not in first_code_source
    assert "ワークフロー重視" in first_markdown_source


def test_ja_advanced_hht_spectrogram_cell_keeps_inline_kwargs():
    nb = _read_tutorial_notebook(Path("ja/user_guide/tutorials/advanced_hht.ipynb"))
    joined = _code_text(nb)

    assert "emd_kwargs=emd_kwargs" not in joined
    assert "hilbert_kwargs=hilbert_kwargs" not in joined
    assert '"eemd_trials": 10' in joined
    assert '"pad": 200' in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/intro_frequencyseries.ipynb"),
        Path("ja/user_guide/tutorials/intro_frequencyseries.ipynb"),
    ],
)
def test_intro_frequencyseries_avoids_slow_plot_wrappers(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "ts.plot(title=ts.name)" not in joined
    assert "red_ts.plot(" not in joined
    assert "ax.plot(ts.times.value, ts.value" in joined
    assert "axes[1].plot(red_ts.times.value, red_ts.value" in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/case_seismic_obspy.ipynb"),
        Path("ja/user_guide/tutorials/case_seismic_obspy.ipynb"),
    ],
)
def test_case_seismic_obspy_avoids_slow_plot_wrappers(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "ts_seismic.plot(" not in joined
    assert "ax.plot(ts_seismic.times.value, ts_seismic.value" in joined
    assert "plot = sg.plot()" not in joined
    assert "mesh = ax.pcolormesh(" in joined


@pytest.mark.parametrize(
    ("relative_path", "minimum_explicit_colorbars"),
    [
        (
            Path("en/user_guide/tutorials/advanced_correlation.ipynb"),
            1,
        ),
        (
            Path("en/user_guide/tutorials/time_frequency_analysis_comparison.ipynb"),
            2,
        ),
        (
            Path("en/user_guide/tutorials/case_gbd_format.ipynb"),
            1,
        ),
        (
            Path("en/user_guide/tutorials/rayleigh_gauch_tutorial.ipynb"),
            3,
        ),
    ],
)
def test_public_tutorial_colorbars_use_explicit_mappables(
    relative_path: Path,
    minimum_explicit_colorbars: int,
):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "plt.gca().get_images()" not in joined
    assert "plt.gca().collections[-1]" not in joined
    assert "plt.gca().get_children()" not in joined
    assert "hasattr(c, 'get_clim')" not in joined

    explicit_mappables = _explicit_colorbar_mappables_from_plot_assignments(nb)
    assert len(explicit_mappables) >= minimum_explicit_colorbars


def test_case_gbd_format_spectrogram_uses_auto_gps_xscale():
    nb = _read_tutorial_notebook(Path("en/user_guide/tutorials/case_gbd_format.ipynb"))
    source = _code_cell_source_containing(nb, "ts_ch0.spectrogram")
    tree = ast.parse(source)

    assert "sg = ts_ch0.spectrogram" in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_xscale"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "auto-gps"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    "path",
    [
        TUTORIAL_ROOT / "en/user_guide/tutorials/case_calibration_pipeline.ipynb",
        ROOT / "docs_redesign/how-to/case-studies/case_calibration_pipeline.ipynb",
    ],
)
def test_calibration_notebook_formats_a_scalar_rms(path: Path):
    nb = _read_notebook(path)
    source = _code_cell_source_containing(nb, "RMS (counts)")

    assert "ts_raw.rms().value" not in source
    assert "np.sqrt(np.mean(ts_raw.value**2))" in source
