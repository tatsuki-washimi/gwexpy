from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_release_metadata.py"
    )
    spec = importlib.util.spec_from_file_location("check_release_metadata", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cff_version_parser_does_not_consume_following_fields(
    tmp_path: Path, monkeypatch
):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)
    Path("CITATION.cff").write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                "title: GWexpy",
                "version: 0.1.1",
                "date-released: 2026-04-28",
                "url: https://example.invalid/gwexpy",
            ]
        ),
        encoding="utf-8",
    )

    assert module.get_version_from_cff() == "0.1.1"


def test_cff_version_parser_handles_quoted_values(tmp_path: Path, monkeypatch):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)

    Path("CITATION.cff").write_text("version: '0.2.0'\n", encoding="utf-8")
    assert module.get_version_from_cff() == "0.2.0"

    Path("CITATION.cff").write_text('version: "0.3.0"\n', encoding="utf-8")
    assert module.get_version_from_cff() == "0.3.0"


def test_cff_version_parser_handles_unquoted_inline_comment(
    tmp_path: Path, monkeypatch
):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)
    Path("CITATION.cff").write_text(
        "version: 0.4.0  # planned release\n",
        encoding="utf-8",
    )

    assert module.get_version_from_cff() == "0.4.0"


def test_cff_version_parser_ignores_nested_version(tmp_path: Path, monkeypatch):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)
    Path("CITATION.cff").write_text(
        "\n".join(
            [
                "preferred-citation:",
                "  version: 9.9.9",
                "version: 0.5.0",
            ]
        ),
        encoding="utf-8",
    )

    assert module.get_version_from_cff() == "0.5.0"


def test_cff_version_missing_file_warns_and_returns_none(
    tmp_path: Path, monkeypatch, capsys
):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)

    assert module.get_version_from_cff() is None
    assert "Warning: CITATION.cff not found" in capsys.readouterr().out


def test_cff_version_parser_rejects_comment_only_value(
    tmp_path: Path, monkeypatch, capsys
):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)
    Path("CITATION.cff").write_text("version: # missing\n", encoding="utf-8")

    assert module.get_version_from_cff() == ""
    assert "empty value" in capsys.readouterr().out


def test_cff_version_parser_rejects_quoted_empty_value(
    tmp_path: Path, monkeypatch, capsys
):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)
    Path("CITATION.cff").write_text('version: ""\n', encoding="utf-8")

    assert module.get_version_from_cff() == ""
    assert "empty value" in capsys.readouterr().out


def test_cff_version_parser_rejects_unterminated_quote(
    tmp_path: Path, monkeypatch, capsys
):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)
    Path("CITATION.cff").write_text('version: "0.1.1\n', encoding="utf-8")

    assert module.get_version_from_cff() == ""
    assert "unterminated quote" in capsys.readouterr().out


def test_main_fails_when_cff_version_is_malformed(tmp_path: Path, monkeypatch, capsys):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)

    Path("gwexpy").mkdir()
    Path("gwexpy/_version.py").write_text(
        '__version__ = "0.1.1"\n',
        encoding="utf-8",
    )
    Path("CITATION.cff").write_text("version: # missing\n", encoding="utf-8")
    Path(".zenodo.json").write_text(
        json.dumps({"version": "0.1.1"}),
        encoding="utf-8",
    )
    Path("CHANGELOG.md").write_text("## [0.1.1]\n", encoding="utf-8")

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("malformed CITATION.cff version should fail")

    output = capsys.readouterr().out
    assert "Malformed CITATION.cff version" in output
    assert "Version mismatch in CITATION.cff" in output


def test_main_passes_when_release_metadata_matches(tmp_path: Path, monkeypatch, capsys):
    module = load_script_module()
    monkeypatch.chdir(tmp_path)

    Path("gwexpy").mkdir()
    Path("gwexpy/_version.py").write_text(
        '__version__ = "0.6.0"\n',
        encoding="utf-8",
    )
    Path("CITATION.cff").write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                "version: 0.6.0 # metadata checker smoke",
                "date-released: 2026-04-28",
                "url: https://example.invalid/gwexpy",
            ]
        ),
        encoding="utf-8",
    )
    Path(".zenodo.json").write_text(
        json.dumps({"version": "0.6.0"}),
        encoding="utf-8",
    )
    Path("CHANGELOG.md").write_text(
        "# Changelog\n\n## [0.6.0] - 2026-04-28\n\n- Release metadata check.\n",
        encoding="utf-8",
    )

    module.main()

    assert "Metadata consistency check passed!" in capsys.readouterr().out
