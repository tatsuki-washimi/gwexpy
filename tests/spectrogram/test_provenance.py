"""Public provenance contract for statistical Spectrogram results (#508)."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
from gwpy.spectrogram import Spectrogram as GwpySpectrogram

from gwexpy.spectrogram import Spectrogram, SpectrogramList
from gwexpy.statistics.gauch import compute_gauch
from gwexpy.statistics.rayleigh_test import rayleigh_pvalue
from gwexpy.statistics.student_t_indicator import compute_student_t_nu
from gwexpy.timeseries import TimeSeries


def _spectrogram() -> Spectrogram:
    return Spectrogram(
        np.arange(12.0).reshape(3, 4),
        times=np.arange(3.0),
        frequencies=np.arange(10.0, 14.0),
        name="provenance",
    )


def _provenance() -> dict[str, object]:
    return {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": 1,
        "analysis": {
            "method": "example",
            "parameters": {"n_monte_carlo": 20, "seed": 7},
        },
    }


def test_provenance_is_versioned_json_mapping_with_detached_state() -> None:
    spec = _spectrogram()
    supplied = _provenance()

    spec.provenance = supplied
    supplied["analysis"]["parameters"]["seed"] = 99  # type: ignore[index]

    observed = spec.provenance
    assert observed == _provenance()
    assert json.loads(json.dumps(observed)) == observed

    observed["analysis"]["parameters"]["seed"] = 100  # type: ignore[index]
    assert spec.provenance == _provenance()


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"schema": "gwexpy.spectrogram.provenance"},
        {"schema_version": 1},
        {"schema": "other", "schema_version": 1},
        {"schema": "gwexpy.spectrogram.provenance", "schema_version": 2},
        {
            "schema": "gwexpy.spectrogram.provenance",
            "schema_version": 1,
            "analysis": {"bad": {1, 2}},
        },
        {
            "schema": "gwexpy.spectrogram.provenance",
            "schema_version": 1,
            "analysis": {"bad": float("nan")},
        },
        {
            "schema": "gwexpy.spectrogram.provenance",
            "schema_version": 1,
            "analysis": {1: "non-string-key"},
        },
    ],
)
def test_provenance_rejects_invalid_or_ambiguous_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _spectrogram().provenance = value  # type: ignore[assignment]


def test_legacy_spectrogram_has_no_provenance_value() -> None:
    assert _spectrogram().provenance is None


def test_provenance_survives_copy_slice_and_arithmetic_without_aliasing() -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()

    results = [spec.copy(), spec[:2], spec + 1]
    for result in results:
        assert isinstance(result, Spectrogram)
        assert result.provenance == _provenance()
        changed = result.provenance
        changed["analysis"]["parameters"]["seed"] = 100  # type: ignore[index]
        assert result.provenance == _provenance()


def test_provenance_survives_pickle_and_hdf5_roundtrips(tmp_path) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()

    pickled = pickle.loads(pickle.dumps(spec))
    assert pickled.provenance == _provenance()

    path = tmp_path / "provenance.hdf5"
    spec.write(path, format="hdf5")
    # The GWexpy sidecar must not become an unsupported GWpy dataset
    # attribute: a GWpy-only consumer remains able to read the native data.
    assert isinstance(GwpySpectrogram.read(path, format="hdf5"), GwpySpectrogram)
    restored = Spectrogram.read(path, format="hdf5")
    assert isinstance(restored, Spectrogram)
    assert restored.provenance == _provenance()


def test_provenance_survives_hdf5_collection_roundtrip(tmp_path) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / "provenance-list.hdf5"

    SpectrogramList([spec]).write(path, format="hdf5")
    restored = SpectrogramList().read(path, format="hdf5")

    assert restored[0].provenance == _provenance()


def test_statistics_publish_consistent_versioned_provenance() -> None:
    rayleigh = rayleigh_pvalue(
        _spectrogram(), n_samples=8, n_monte_carlo=12, nfft=16, seed=7
    )
    assert rayleigh.provenance == {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": 1,
        "analysis": {
            "method": "rayleigh_pvalue",
            "parameters": {"n_samples": 8, "n_monte_carlo": 12, "nfft": 16},
            "random": {"seed": 7, "rng_provided": False, "seed_unused": False},
        },
    }

    ts = TimeSeries(np.random.default_rng(10).normal(size=512), sample_rate=128)
    gauch = compute_gauch(ts, fftlength=0.25, window=8, n_monte_carlo=12, seed=7)
    for result in (gauch.pvalue_map, gauch.statistic_map):
        assert result.provenance == {
            "schema": "gwexpy.spectrogram.provenance",
            "schema_version": 1,
            "analysis": {
                "method": "compute_gauch",
                "parameters": {
                    "fftlength": 0.25,
                    "stride": 0.25,
                    "window": 8,
                    "overlap": None,
                    "n_monte_carlo": 12,
                },
                "random": {
                    "seed": 7,
                    "rng_provided": False,
                    "seed_unused": False,
                },
            },
        }

    student = compute_student_t_nu(ts, fftlength=0.25, window=8)
    assert student.provenance == {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": 1,
        "analysis": {
            "method": "compute_student_t_nu",
            "parameters": {
                "fftlength": 0.25,
                "stride": 0.25,
                "window": 8,
                "overlap": None,
                "frange": None,
            },
        },
    }


def test_rng_provenance_is_a_safe_descriptor_not_a_live_generator() -> None:
    result = rayleigh_pvalue(
        _spectrogram(),
        n_samples=8,
        n_monte_carlo=12,
        rng=np.random.default_rng(7),
    )

    assert result.provenance["analysis"]["random"] == {
        "seed": None,
        "rng_provided": True,
        "seed_unused": False,
    }
    assert "Generator" not in json.dumps(result.provenance)
