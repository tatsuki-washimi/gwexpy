from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from gwexpy.analysis.bruco import BrucoResult, _resolve_block_size
from gwexpy.analysis.coupling_result import CouplingResult, CouplingResultCollection
from gwexpy.analysis.response import ResponseFunctionResult
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectrogram import Spectrogram


def _frequency_series(
    values: list[float] | np.ndarray,
    freqs: np.ndarray,
    *,
    unit: u.UnitBase | str = u.dimensionless_unscaled,
    name: str | None = None,
) -> FrequencySeries:
    return FrequencySeries(
        np.asarray(values, dtype=float),
        frequencies=freqs * u.Hz,
        unit=unit,
        name=name,
    )


def test_bruco_result_preserves_metadata_and_exports_ranked_topn(monkeypatch) -> None:
    freqs = np.array([10.0, 20.0, 30.0, 40.0])
    target_psd = np.array([4.0, 9.0, 16.0, 25.0])
    metadata = {"fftlength": 4, "overlap": 2, "run": "contract"}

    monkeypatch.setenv("GWEXPY_BRUCO_BLOCK_BYTES", str(8 * len(freqs) * (2 + 17)))
    assert _resolve_block_size("auto", n_bins=len(freqs), top_n=2) == 17

    result = BrucoResult(
        freqs,
        "TGT",
        target_psd,
        top_n=2,
        metadata=metadata,
        block_size="auto",
    )
    metadata["fftlength"] = 8

    result.update_batch(
        ["AUX_A", "AUX_B", "AUX_C"],
        np.array(
            [
                [0.25, 0.81, 0.10, 0.16],
                [0.36, 0.49, 0.64, 0.04],
                [0.10, 0.90, 0.20, 0.09],
            ]
        ),
    )

    assert result.metadata == {"fftlength": 4, "overlap": 2, "run": "contract"}
    assert result.block_size == 17
    np.testing.assert_array_equal(
        result.top_channels,
        np.array(
            [
                ["AUX_B", "AUX_A"],
                ["AUX_C", "AUX_A"],
                ["AUX_B", "AUX_C"],
                ["AUX_A", "AUX_C"],
            ],
            dtype=object,
        ),
    )

    exported = result.to_dataframe(ranks=[0], asd=True)
    assert exported["frequency"].to_list() == freqs.tolist()
    assert exported["rank"].to_list() == [1, 1, 1, 1]
    assert exported["channel"].to_list() == ["AUX_B", "AUX_C", "AUX_B", "AUX_A"]
    np.testing.assert_allclose(
        exported["coherence"].to_numpy(),
        np.sqrt([0.36, 0.90, 0.64, 0.16]),
    )
    np.testing.assert_allclose(
        exported["projection"].to_numpy(),
        np.sqrt(target_psd * np.array([0.36, 0.90, 0.64, 0.16])),
    )


def test_coupling_result_frequency_units_and_summary_export_contract(tmp_path) -> None:
    freqs = np.array([5.0, 10.0, 20.0])
    cf = _frequency_series([0.1, 0.2, np.nan], freqs, unit=u.m / u.V, name="CF")
    cf_ul = _frequency_series([0.3, 0.4, 0.5], freqs, unit=u.m / u.V, name="CF UL")
    psd_wit_inj = _frequency_series([4.0, 9.0, 16.0], freqs, unit=u.V**2 / u.Hz)
    psd_wit_bkg = _frequency_series([1.0, 4.0, 9.0], freqs, unit=u.V**2 / u.Hz)
    psd_tgt_inj = _frequency_series([0.25, 0.36, 0.49], freqs, unit=u.m**2 / u.Hz)
    psd_tgt_bkg = _frequency_series([0.16, 0.25, 0.36], freqs, unit=u.m**2 / u.Hz)

    result = CouplingResult(
        cf=cf,
        cf_ul=cf_ul,
        psd_witness_inj=psd_wit_inj,
        psd_witness_bkg=psd_wit_bkg,
        psd_target_inj=psd_tgt_inj,
        psd_target_bkg=psd_tgt_bkg,
        valid_mask=np.array([True, True, False]),
        witness_name="WIT",
        target_name="TGT",
        fftlength=8.0,
        overlap=4.0,
    )

    assert result.frequencies is cf.xindex
    assert result.cf.unit == u.m / u.V
    assert result.psd_witness_inj.unit == u.V**2 / u.Hz
    assert result.psd_target_inj.unit == u.m**2 / u.Hz
    np.testing.assert_array_equal(result.valid_mask, [True, True, False])
    assert result.fftlength == 8.0
    assert result.overlap == 4.0

    summary = tmp_path / "summary.csv"
    CouplingResultCollection({"WIT-TGT": result, "ignored": object()}).to_summary_csv(
        summary
    )

    with summary.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert [row["channel_pair"] for row in rows] == ["WIT-TGT"] * len(freqs)
    np.testing.assert_allclose([float(row["frequency"]) for row in rows], freqs)
    np.testing.assert_allclose([float(row["cf_ul"]) for row in rows], cf_ul.value)
    np.testing.assert_allclose([float(row["inj_asd"]) for row in rows], [2.0, 3.0, 4.0])
    np.testing.assert_allclose([float(row["bkg_asd"]) for row in rows], [1.0, 2.0, 3.0])


def test_response_function_result_plot_sorts_without_mutating_metadata() -> None:
    freqs = np.array([8.0, 16.0, 32.0, 64.0])
    times = np.array([100.0, 110.0, 120.0])
    unit = u.m / u.Hz**0.5
    spectrogram_inj = Spectrogram(
        np.array(
            [
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
                [3.0, 3.1, 3.2, 3.3],
            ]
        ),
        times=times,
        frequencies=freqs * u.Hz,
        unit=unit,
        name="target injection ASD",
    )
    spectrogram_bkg = Spectrogram(
        np.ones((3, 4)),
        times=times,
        frequencies=freqs * u.Hz,
        unit=unit,
        name="target background ASD",
    )
    injected_freqs = np.array([30.0, 10.0, 20.0])
    coupling_factors = np.array([3.0e-4, 1.0e-4, 2.0e-4])

    result = ResponseFunctionResult(
        spectrogram_inj=spectrogram_inj,
        spectrogram_bkg=spectrogram_bkg,
        injected_freqs=injected_freqs.copy(),
        step_times=times.copy(),
        coupling_factors=coupling_factors.copy(),
        witness_name="WIT",
        target_name="TGT",
    )

    ax = result.plot()
    line = ax.lines[0]

    np.testing.assert_allclose(line.get_xdata(), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(line.get_ydata(), [1.0e-4, 2.0e-4, 3.0e-4])
    np.testing.assert_allclose(result.injected_freqs, injected_freqs)
    np.testing.assert_allclose(result.coupling_factors, coupling_factors)
    assert result.spectrogram_inj.unit == unit
    np.testing.assert_allclose(result.spectrogram_inj.frequencies.value, freqs)
    assert result.witness_name == "WIT"
    assert result.target_name == "TGT"

    plt.close("all")
