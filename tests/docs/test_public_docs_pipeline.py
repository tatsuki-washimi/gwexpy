"""Behavioral checks for reproducible public lessons and source preparation."""

from __future__ import annotations

import io
import json
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from scripts.check_public_docs import (
    AUDIENCE_ROUTES,
    WORKFLOW_NOTEBOOK_PAGES,
    check_remote,
)
from scripts.prepare_public_docs import code_cells, prepare

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs_redesign"


def test_all_public_notebooks_embed_clean_execution_cells() -> None:
    for path in DOCS.rglob("*.ipynb"):
        if "_build" in path.parts:
            continue
        cells = code_cells(json.loads(path.read_text()))
        assert cells, path
        for cell in cells:
            assert cell.get("outputs", []) == [], path
            assert cell.get("execution_count") is None, path


def test_quickstart_generates_reproducible_voltage_spectra(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runpy.run_path(str(DOCS / "_static/downloads/quickstart.py"))
    spectra = list(result["spectra"].values())
    for spectrum in spectra:
        assert spectrum.unit == u.V / u.Hz**0.5
        assert spectrum.df == 0.5 * u.Hz
        assert spectrum.frequencies[np.argmax(spectrum.value)] == 40 * u.Hz
    assert (tmp_path / "asd.png").stat().st_size > 1000
    repeated = runpy.run_path(str(DOCS / "_static/downloads/quickstart.py"))
    for name, channel in result["channels"].items():
        np.testing.assert_array_equal(channel.value, repeated["channels"][name].value)


def test_commissioner_records_reproducible_data_and_settings(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runpy.run_path(str(DOCS / "_static/downloads/commissioner.py"))
    saved = json.loads(
        (tmp_path / "commissioner-output/analysis-parameters.json").read_text()
    )
    assert saved["crop_end_gps_s"] - saved["crop_start_gps_s"] == 24
    for name, channel in result["channels"].items():
        np.testing.assert_array_equal(channel.value, result["loaded"][name].value)
        assert result["segment"][name].duration == 24 * u.s
        assert result["segment"][name].unit == u.V
    coherence = result["coherence"]
    assert 0.95 < coherence.value[np.argmin(abs(coherence.frequencies.value - 40))] <= 1
    for filename in ("asd.png", "coherence.png"):
        assert (tmp_path / "commissioner-output" / filename).stat().st_size > 1000


def test_preparation_preserves_changelog_and_records_source_commit(tmp_path) -> None:
    repository = tmp_path / "repository"
    source = repository / "docs_redesign"
    source.mkdir(parents=True)
    (source / "index.md").write_text("Docs")
    (repository / "CHANGELOG.md").write_text("Canonical release history")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Docs test",
            "-c",
            "user.email=docs@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        cwd=repository,
        check=True,
    )
    output = tmp_path / "prepared/docs_redesign"
    prepare(output, root=repository)
    assert (output.parent / "CHANGELOG.md").read_text() == "Canonical release history"
    identity = json.loads((output / "_build_identity.json").read_text())
    assert len(identity["source_revision"]) == 40
    assert identity["dirty"] is False
    with pytest.raises(FileExistsError):
        prepare(output, root=repository)
    with pytest.raises(ValueError, match="outside"):
        prepare(repository / "generated", root=repository)


def test_preparation_uses_redesign_notebook_code_without_legacy_docs_tree(tmp_path):
    repository = tmp_path / "repository"
    source = repository / "docs_redesign" / "tutorials"
    source.mkdir(parents=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["Translated lesson"]},
            {
                "cell_type": "code",
                "source": ["value = 42\n", "print(value)"],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["stale output\n"],
                    }
                ],
                "execution_count": 3,
                "metadata": {"execution": {"iopub.execute_input": "stale"}},
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = source / "lesson.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    (repository / "CHANGELOG.md").write_text("Release history", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Docs test",
            "-c",
            "user.email=docs@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        cwd=repository,
        check=True,
    )

    output = tmp_path / "prepared/docs_redesign"
    prepare(output, root=repository)

    prepared = json.loads((output / "tutorials/lesson.ipynb").read_text())
    assert prepared["cells"][0]["source"] == ["Translated lesson"]
    assert prepared["cells"][1]["source"] == ["value = 42\n", "print(value)"]
    assert prepared["cells"][1]["outputs"] == []
    assert prepared["cells"][1]["execution_count"] is None
    assert "execution" not in prepared["cells"][1]["metadata"]
    identity = json.loads((output / "_build_identity.json").read_text())
    assert "notebook_sources" not in identity
    assert identity["public_notebooks"] == ["tutorials/lesson.ipynb"]


@pytest.mark.parametrize(
    "defect", [None, "revision", "figure", "route", "anchor", "download"]
)
def test_deployment_readback_rejects_stale_or_incomplete_publication(
    monkeypatch, defect
):
    revision = "a" * 40
    requests = []

    def response(request, timeout):
        requests.append(request.full_url)
        if "build-info.json" in request.full_url:
            data = json.dumps(
                {
                    "source_revision": "b" * 40 if defect == "revision" else revision,
                    "dirty": False,
                }
            ).encode()
        elif ".ipynb" in request.full_url:
            data = (
                b'{"cells": [{"cell_type": "markdown", "metadata": {}, '
                b'"source": ["text"]}], "metadata": {}, "nbformat": 4, '
                b'"nbformat_minor": 5}'
            )
        elif ".png" in request.full_url:
            data = (
                b"not an image"
                if defect == "figure"
                else b"\x89PNG\r\n\x1a\n" + b"0" * 1200
            )
        elif "/downloads/" in request.full_url:
            filename = request.full_url.split("?", 1)[0].rsplit("/", 1)[1]
            data = (DOCS / "_static/downloads" / filename).read_bytes()
            if defect == "download":
                data = b"stale download"
        else:
            page = f"<aside class='gwexpy-build-status'>{revision[:8]}</aside>"
            routes = AUDIENCE_ROUTES[:-1] if defect == "route" else AUDIENCE_ROUTES
            page += "".join(f'<a href="{route}">route</a>' for route in routes)
            if defect != "anchor":
                page += '<div id="for-gw-experimentalists"></div>'
            # Workflow readback fixtures: language switch, figure, download.
            rel = request.full_url.split("docs.example.test/", 1)[1].split("?", 1)[0]
            rel = rel[3:] if rel.startswith("ja/") else rel
            notebook_name = WORKFLOW_NOTEBOOK_PAGES.get(rel)
            page += (
                f'<a href="https://docs.example.test/{rel}">EN</a>'
                f'<a href="https://docs.example.test/ja/{rel}">JA</a>'
                '<img src="_images/fig.png">'
            )
            if notebook_name:
                page += f'<a href="_sources/{notebook_name}">notebook</a>'
            if "/ja/" in request.full_url:
                page += "<p>日本語の説明文</p>"
            data = page.encode()
        return io.BytesIO(data)

    monkeypatch.setattr("scripts.check_public_docs.urlopen", response)
    errors = check_remote("https://docs.example.test/", revision)
    assert bool(errors) == (defect is not None)
    assert any("/ja/build-info.json" in request for request in requests)
    assert any("/ja/_static/downloads/commissioner.xml" in item for item in requests)


def _lesson_cell(relative: str, marker: str) -> str:
    notebook = json.loads((DOCS / relative).read_text())
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


def test_noise_budget_example_converts_native_witness_units_to_output_units(
    monkeypatch,
) -> None:
    import matplotlib.pyplot as plt

    from gwexpy.timeseries import TimeSeries, TimeSeriesDict

    monkeypatch.setattr(plt, "show", lambda: None)
    values = np.random.default_rng(120).normal(size=16384)
    channels = TimeSeriesDict(
        {
            "MAIN": TimeSeries(2e-21 * values, sample_rate=512, unit=""),
            "AUX_SEIS": TimeSeries(values, sample_rate=512, unit="um"),
            "AUX_MAG": TimeSeries(values, sample_rate=512, unit="nT"),
            "AUX_ELEC": TimeSeries(values, sample_rate=512, unit="V"),
        }
    )
    namespace = {"tsd": channels, "plt": plt, "np": np}
    try:
        code = _lesson_cell("how-to/case-studies/case_noise_budget.ipynb", "tf_seis =")
        exec(compile(code, "noise-budget-transfer-cell", "exec"), namespace)
        for key, name in (
            ("tf_seis", "AUX_SEIS"),
            ("tf_mag", "AUX_MAG"),
            ("tf_elec", "AUX_ELEC"),
        ):
            transfer = namespace[key]
            assert transfer.unit == channels["MAIN"].unit / channels[name].unit
            np.testing.assert_allclose(transfer.value, 2e-21, rtol=1e-12, atol=1e-34)
    finally:
        plt.close("all")


def test_wiener_example_recovers_gain_and_phase_with_output_input_units(
    monkeypatch,
) -> None:
    import matplotlib.pyplot as plt

    from gwexpy.timeseries import TimeSeries, TimeSeriesDict

    monkeypatch.setattr(plt, "show", lambda: None)
    rng = np.random.default_rng(321)
    first, second = rng.normal(size=(2, 65536))
    channels = TimeSeriesDict(
        {
            "MAIN": TimeSeries(
                2e-22 * first + 5e-22 * np.roll(second, 3), sample_rate=512, unit=""
            ),
            "AUX1": TimeSeries(first, sample_rate=512, unit="V"),
            "AUX2": TimeSeries(second, sample_rate=512, unit="V"),
        }
    )
    namespace = {
        "tsd": channels,
        "TimeSeriesDict": TimeSeriesDict,
        "np": np,
        "plt": plt,
    }
    try:
        code = _lesson_cell(
            "how-to/case-studies/case_wiener_filter.ipynb", "H_lowres ="
        )
        exec(compile(code, "wiener-estimation-cell", "exec"), namespace)
        transfer = namespace["H_lowres"]
        index = int(40 / transfer.df.value)
        assert transfer[0, 1].unit == u.V**-1
        np.testing.assert_allclose(transfer[0, 0].value[index], 2e-22, rtol=0.02)
        expected = 5e-22 * np.exp(-2j * np.pi * 40 * 3 / 512)
        np.testing.assert_allclose(transfer[0, 1].value[index], expected, rtol=0.02)
    finally:
        plt.close("all")
