"""Regression tests for published workflow verification (9 logical pages)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from scripts.check_public_docs import (
    ENTRY_PAGES,
    LEGACY_ANCHORS,
    WORKFLOW_NOTEBOOK_PAGES,
    WORKFLOW_PAGES,
    check,
    check_remote,
    has_japanese_text,
    is_notebook_payload,
    workflow_page_errors,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs_redesign"
REVISION = "9" * 40
BASEURL = "https://docs.example.test/"


def _page_html(name: str, language: str, *, body: str = "") -> str:
    counterpart = ("ja/" if not language else "") + name
    switch = BASEURL + counterpart
    ids = "".join(
        f'<div id="{anchor}"></div>' for anchor in LEGACY_ANCHORS.get(name, ())
    )
    text = "BODY " * 200 + body
    if language == "ja/":
        text += "日本語の説明文 " * 20
    return (
        "<html><body>"
        f"<aside class='gwexpy-build-status'>{REVISION[:8]}</aside>"
        f'{ids}<a href="{switch}">switch</a><p>{text}</p>'
        "</body></html>"
    )


def _build_tree(root: Path) -> None:
    for language in ("", "ja/"):
        lang_dir = root / language
        (lang_dir / "how-to/monitoring").mkdir(parents=True, exist_ok=True)
        info = {
            "source_revision": REVISION,
            "language_baseurl": BASEURL,
            "notebook_sources": {"how-to/case-studies/case_x.ipynb": "x"},
            "notebook_execution": {
                "how-to/case-studies/case_x": {"succeeded": True, "runtime": 1.0}
            },
        }
        (lang_dir / "build-info.json").write_text(json.dumps(info))
        for name in ENTRY_PAGES:
            page = lang_dir / name
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(_page_html(name, language))
        case_page = lang_dir / "how-to/case-studies/case_x.html"
        case_page.parent.mkdir(parents=True, exist_ok=True)
        case_page.write_text(
            f"<aside class='gwexpy-build-status'>{REVISION[:8]}</aside>"
            "<p class='gwexpy-case-status'>ok</p>"
        )
        static_img = lang_dir / "_static/images/quickstart-asd.png"
        static_img.parent.mkdir(parents=True, exist_ok=True)
        static_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 1200)
        analysis_img = root / "_images/workflow-figure.png"
        analysis_img.parent.mkdir(parents=True, exist_ok=True)
        analysis_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 1200)
        for wp, nb_name in WORKFLOW_NOTEBOOK_PAGES.items():
            wp_file = lang_dir / wp
            sources = lang_dir / "_sources"
            sources.mkdir(exist_ok=True)
            (sources / nb_name).write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "markdown",
                                "metadata": {},
                                "source": ["workflow notebook"],
                            }
                        ],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 5,
                    }
                )
            )
            html = wp_file.read_text(encoding="utf-8")
            html = html.replace(
                "</body></html>",
                f'<img src="/gwexpy/docs/_images/workflow-figure.png">'
                f'<a href="../../_sources/{nb_name}">notebook</a></body></html>',
            )
            wp_file.write_text(html)


def _workflow_errors(errors: list[str]) -> list[str]:
    return [
        e
        for e in errors
        if "workflow" in e
        or "notebook download link" in e
        or "Japanese translation" in e
    ]


def test_workflow_pages_present_in_both_languages(tmp_path) -> None:
    _build_tree(tmp_path)
    assert _workflow_errors(check(tmp_path, REVISION)) == []


def test_missing_workflow_page_is_reported(tmp_path) -> None:
    _build_tree(tmp_path)
    (tmp_path / WORKFLOW_PAGES[1]).unlink()
    errors = check(tmp_path, REVISION)
    assert any("long_term_trend" in e for e in _workflow_errors(errors))


def test_missing_ja_workflow_page_is_reported(tmp_path) -> None:
    _build_tree(tmp_path)
    (tmp_path / "ja" / WORKFLOW_PAGES[1]).unlink()
    errors = check(tmp_path, REVISION)
    assert any("ja/how-to/monitoring/long_term_trend" in e for e in errors)


def test_ja_page_without_japanese_is_reported(tmp_path) -> None:
    _build_tree(tmp_path)
    target = tmp_path / "ja" / WORKFLOW_PAGES[0]
    html = target.read_text(encoding="utf-8").replace("日本語の説明文", "English text")
    target.write_text(html)
    errors = check(tmp_path, REVISION)
    assert any("lacks Japanese translation" in e for e in errors)


def test_language_switch_mismatch_is_reported(tmp_path) -> None:
    _build_tree(tmp_path)
    target = tmp_path / WORKFLOW_PAGES[0]
    html = target.read_text(encoding="utf-8").replace(BASEURL + "ja/", BASEURL + "fr/")
    target.write_text(html)
    errors = check(tmp_path, REVISION)
    assert any("Missing language switch" in e for e in errors)


@pytest.mark.parametrize(
    "page_name",
    ["how-to/monitoring/index.html", "how-to/calibration/index.html"],
)
def test_landing_pages_do_not_require_analysis_figure(page_name: str) -> None:
    html = _page_html(page_name, "")
    assert (
        workflow_page_errors(
            page_html=html,
            language="",
            expected_revision=REVISION,
            counterpart_url=BASEURL + "ja/" + page_name,
            page_name=page_name,
        )
        == []
    )


def test_missing_notebook_download_link_is_reported(tmp_path) -> None:
    _build_tree(tmp_path)
    for language in ("", "ja/"):
        target = tmp_path / language / next(iter(WORKFLOW_NOTEBOOK_PAGES))
        target.write_text(_page_html(next(iter(WORKFLOW_NOTEBOOK_PAGES)), language))
    errors = check(tmp_path, REVISION)
    assert any("Missing notebook download link" in e for e in errors)


def test_build_revision_mismatch_is_reported(tmp_path) -> None:
    _build_tree(tmp_path)
    errors = check(tmp_path, "0" * 40)
    assert any("Wrong source revision" in e for e in errors)


def test_workflow_page_content_helper() -> None:
    good_en = (
        f"<aside class='gwexpy-build-status'>{REVISION[:8]}</aside>"
        '<a href="https://x.test/ja/a.html">JA</a>'
    )
    assert (
        workflow_page_errors(
            page_html=good_en,
            language="",
            expected_revision=REVISION,
            counterpart_url="https://x.test/ja/a.html",
        )
        == []
    )
    assert workflow_page_errors(
        page_html="<p>no marker</p>",
        language="",
        expected_revision=REVISION,
        counterpart_url="https://x.test/ja/a.html",
    )
    ja_good = (
        f"<aside class='gwexpy-build-status'>{REVISION[:8]}</aside>"
        '<a href="https://x.test/a.html">EN</a>'
        "<p>日本語</p>"
    )
    assert (
        workflow_page_errors(
            page_html=ja_good,
            language="ja/",
            expected_revision=REVISION,
            counterpart_url="https://x.test/a.html",
        )
        == []
    )
    assert workflow_page_errors(
        page_html=good_en,
        language="ja/",
        expected_revision=REVISION,
        counterpart_url="https://x.test/a.html",
    )


def test_notebook_payload_helper() -> None:
    valid = json.dumps(
        {
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["text"]}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    ).encode()
    assert is_notebook_payload(valid)
    assert not is_notebook_payload(b'{"cells": [], "nbformat": 4}')
    assert not is_notebook_payload(
        b'{"cells": [123], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    )
    assert not is_notebook_payload(b"<html>not a notebook</html>")
    assert not is_notebook_payload(b"\x89PNG\r\n\x1a\n")
    assert has_japanese_text("日本語")
    assert not has_japanese_text("English only")


# --- Remote readback with a mocked deployment -------------------------------

_MOCK_ROUTES = (
    "index.html",
    "tutorials/quickstart.html",
    "tutorials/first_analysis.html",
    "tutorials/first_analysis.html#for-gw-experimentalists",
    "tutorials/commissioner.html",
    "tutorials/scientific_python.html",
    "explanation/gwexpy_for_gwpy_users.html",
    "how-to/index.html",
)


def _mock_response(defect: str | None):
    def response(request, timeout):
        url = request.full_url
        if "build-info.json" in url:
            return io.BytesIO(
                json.dumps(
                    {
                        "source_revision": "0" * 40
                        if defect == "revision"
                        else REVISION,
                        "dirty": False,
                    }
                ).encode()
            )
        if url.split("?", 1)[0].endswith(".png"):
            if defect == "image" and "_images/workflow" in url:
                return io.BytesIO(b"")
            if defect == "image_html" and "_images/workflow" in url:
                return io.BytesIO(b"<html><body>404</body></html>")
            return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 1200)
        if ".ipynb" in url:
            if defect == "notebook":
                return io.BytesIO(b"<html>not a notebook</html>")
            if defect == "notebook_empty":
                return io.BytesIO(
                    b'{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
                )
            if defect == "notebook_bad_cell":
                return io.BytesIO(
                    b'{"cells": [123], "metadata": {}, "nbformat": 4, '
                    b'"nbformat_minor": 5}'
                )
            return io.BytesIO(
                b'{"cells": [{"cell_type": "markdown", "metadata": {}, '
                b'"source": ["text"]}], "metadata": {}, "nbformat": 4, '
                b'"nbformat_minor": 5}'
            )
        if "/downloads/" in url:
            filename = url.split("?", 1)[0].rsplit("/", 1)[1]
            return io.BytesIO((DOCS / "_static/downloads" / filename).read_bytes())
        rel = url.split("docs.example.test/", 1)[1].split("?", 1)[0]
        is_ja_request = rel.startswith("ja/")
        rel = rel[3:] if is_ja_request else rel
        is_workflow = rel in WORKFLOW_PAGES
        page = ""
        if defect != "identity" or not is_workflow:
            page += f"<aside class='gwexpy-build-status'>{REVISION[:8]}</aside>"
        en_link = f"https://docs.example.test/{rel}"
        ja_link = f"https://docs.example.test/ja/{rel}"
        if defect == "lang_switch" and is_workflow:
            # Serve only the self link, like a page whose switch is broken.
            page += f'<a href="{ja_link if is_ja_request else en_link}">self</a>'
        else:
            page += f'<a href="{en_link}">EN</a><a href="{ja_link}">JA</a>'
        page += "".join(f'<a href="{route}">route</a>' for route in _MOCK_ROUTES)
        page += '<div id="for-gw-experimentalists"></div>'
        if is_workflow:
            t1 = "how-to/monitoring/long_term_trend.html"
            t7 = "how-to/interop/root_to_python_migration.html"
            if not (defect == "analysis_figure_missing" and rel == t1):
                page += '<img src="_images/workflow-fig.png">'
            if rel in WORKFLOW_NOTEBOOK_PAGES:
                notebook_name = WORKFLOW_NOTEBOOK_PAGES[rel]
                if defect == "notebook_wrong_target" and rel == t1:
                    notebook_name = WORKFLOW_NOTEBOOK_PAGES[t7]
                page += f'<a href="_sources/{notebook_name}">notebook</a>'
        if "/ja/" in url and defect != "ja_text":
            page += "<p>日本語の説明文</p>"
        if defect == "unreachable" and is_workflow and "/ja/" not in url:
            raise URLError("mock connection failure")
        return io.BytesIO(page.encode())

    return response


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        (None, []),
        ("identity", ["missing expected build identity"]),
        ("lang_switch", ["missing language switch link"]),
        ("ja_text", ["lacks Japanese translation"]),
        ("notebook", ["not a notebook"]),
        ("notebook_empty", ["not a notebook"]),
        ("notebook_bad_cell", ["not a notebook"]),
        ("notebook_wrong_target", ["missing notebook download link"]),
        ("image", ["invalid image"]),
        ("image_html", ["invalid image"]),
        ("analysis_figure_missing", ["missing analysis figure"]),
        ("revision", ["deployed revision is not the clean expected commit"]),
        ("unreachable", ["mock connection failure"]),
    ],
)
def test_remote_workflow_readback(monkeypatch, defect, expected) -> None:
    monkeypatch.setattr("scripts.check_public_docs.urlopen", _mock_response(defect))
    errors = check_remote("https://docs.example.test/", REVISION)
    if defect is None:
        assert errors == []
    else:
        assert any(expected[0] in e for e in errors), errors
