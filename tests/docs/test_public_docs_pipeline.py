"""Behavioral checks for reproducible public lessons and source preparation."""

from __future__ import annotations

import copy
import io
import json
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from scripts.check_public_docs import check_remote
from scripts.prepare_public_docs import canonicalize, code_cells, prepare

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs_redesign"


def test_canonical_code_replaces_stale_execution_without_changing_prose() -> None:
    public = {
        "cells": [
            {"cell_type": "markdown", "source": ["Translated lesson identity"]},
            {
                "cell_type": "code",
                "source": ["old()"],
                "outputs": ["stale"],
                "execution_count": 8,
            },
        ]
    }
    canonical = {
        "cells": [
            {"cell_type": "code", "source": ["%pip install legacy-bootstrap"]},
            {"cell_type": "code", "source": ["corrected()"]},
        ]
    }
    result = canonicalize(copy.deepcopy(public), canonical)
    assert result["cells"][0] == public["cells"][0]
    assert code_cells(result)[0]["source"] == ["corrected()"]
    assert code_cells(result)[0]["outputs"] == []
    assert code_cells(result)[0]["execution_count"] is None


def test_mismatched_lesson_structure_stops_preparation() -> None:
    with pytest.raises(ValueError, match="reconcile lesson structure"):
        canonicalize(
            {"cells": []}, {"cells": [{"cell_type": "code", "source": ["x = 1"]}]}
        )


def test_all_public_notebook_execution_cells_have_one_canonical_source() -> None:
    for path in DOCS.rglob("*.ipynb"):
        if "_build" in path.parts:
            continue
        canonical = ROOT / "docs/web/en/user_guide/tutorials" / path.name
        actual = code_cells(json.loads(path.read_text()))
        expected = code_cells(json.loads(canonical.read_text()))
        assert len(actual) == len(expected), path
        assert ["".join(c["source"]) for c in actual] == [
            "".join(c["source"]) for c in expected
        ], path


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


@pytest.mark.parametrize(
    "stale,broken_figure", [(False, False), (True, False), (False, True)]
)
def test_deployment_readback_rejects_stale_commits_and_missing_figures(
    monkeypatch,
    stale,
    broken_figure,
) -> None:
    revision = "a" * 40
    requests = []

    def response(request, timeout):
        requests.append(request.full_url)
        if "build-info.json" in request.full_url:
            data = json.dumps(
                {"source_revision": "b" * 40 if stale else revision, "dirty": False}
            ).encode()
        elif ".png" in request.full_url:
            data = (
                b"not an image" if broken_figure else b"\x89PNG\r\n\x1a\n" + b"0" * 1200
            )
        else:
            data = f"<aside class='gwexpy-build-status'>{revision[:8]}</aside>".encode()
        return io.BytesIO(data)

    monkeypatch.setattr("scripts.check_public_docs.urlopen", response)
    errors = check_remote("https://docs.example.test/", revision)
    assert bool(errors) == (stale or broken_figure)
    assert any("/ja/build-info.json" in request for request in requests)


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
