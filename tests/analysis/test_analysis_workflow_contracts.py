from __future__ import annotations

import csv
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from gwpy.segments import Segment
from gwpy.timeseries import TimeSeries

from gwexpy.analysis import coupling as coupling_mod
from gwexpy.analysis.bruco import (
    Bruco,
    BrucoMetadataValue,
    BrucoResult,
    _resolve_block_size,
)
from gwexpy.analysis.coupling_result import CouplingResult, CouplingResultCollection
from gwexpy.analysis.response import (
    ResponseFunctionAnalysis,
    ResponseFunctionResult,
    _compute_response_row,
)
from gwexpy.analysis.threshold import ThresholdStrategy
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeriesDict


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
    metadata: dict[str, BrucoMetadataValue] = {
        "fftlength": 4,
        "overlap": 2,
        "run": "contract",
    }

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


def test_bruco_compute_records_auto_block_size_and_exports_metadata(
    monkeypatch,
) -> None:
    monkeypatch.delenv("GWEXPY_BRUCO_BLOCK_BYTES", raising=False)
    monkeypatch.delenv("GWEXPY_BRUCO_BLOCK_SIZE", raising=False)

    sample_rate = 64.0
    times = np.arange(int(sample_rate * 4.0)) / sample_rate
    target = TimeSeries(
        np.sin(2.0 * np.pi * 8.0 * times),
        sample_rate=sample_rate * u.Hz,
        name="TGT",
    )

    result = Bruco("TGT", aux_channels=[]).compute(
        target_data=target,
        aux_data=TimeSeriesDict(),
        fftlength=1.0,
        overlap=0.5,
        parallel=1,
        batch_size=7,
        top_n=1,
        block_size="auto",
    )

    expected_block_size = _resolve_block_size(
        "auto", n_bins=len(result.frequencies), top_n=1
    )
    assert result.block_size == expected_block_size
    assert result.metadata["block_size_requested"] == "auto"
    assert result.metadata["block_size"] == expected_block_size
    assert result.metadata["n_frequency_bins"] == len(result.frequencies)
    assert result.metadata["frequency_resolution"] == pytest.approx(1.0)

    exported = result.to_dataframe(
        ranks=[0],
        stride=len(result.frequencies) + 1,
        include_metadata=True,
    )
    assert exported["metadata_fftlength"].to_list() == [1.0]
    assert exported["metadata_overlap"].to_list() == [0.5]
    assert exported["metadata_block_size"].to_list() == [expected_block_size]


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


class _StaticPsdSeries:
    def __init__(self, psd: FrequencySeries, unit: u.UnitBase) -> None:
        self._psd = psd
        self.unit = unit

    def psd(self, **kwargs: object) -> FrequencySeries:
        del kwargs
        return self._psd


class _AllTrueThreshold(ThresholdStrategy):
    def check(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        del psd_bkg, raw_bkg, kwargs
        return np.ones_like(psd_inj.value, dtype=bool)

    def threshold(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        del psd_inj, raw_bkg, kwargs
        return psd_bkg.value * 2.0


def test_coupling_skips_target_when_target_injection_background_grids_mismatch() -> (
    None
):
    witness_freqs = np.array([10.0, 20.0, 30.0])
    shifted_target_freqs = np.array([10.0, 20.0, 31.0])
    psd_wit_inj = _frequency_series([4.0, 5.0, 6.0], witness_freqs, unit=u.V**2 / u.Hz)
    psd_wit_bkg = _frequency_series([1.0, 1.0, 1.0], witness_freqs, unit=u.V**2 / u.Hz)
    psd_tgt_inj = _frequency_series([4.0, 5.0, 6.0], witness_freqs, unit=u.m**2 / u.Hz)
    psd_tgt_bkg = _frequency_series(
        [1.0, 1.0, 1.0], shifted_target_freqs, unit=u.m**2 / u.Hz
    )

    with pytest.warns(UserWarning, match="Skipping coupling target TGT"):
        result = coupling_mod._process_single_target(
            "TGT",
            cast(TimeSeries, _StaticPsdSeries(psd_tgt_inj, u.m)),
            cast(TimeSeries, _StaticPsdSeries(psd_tgt_bkg, u.m)),
            {"fftlength": 1.0},
            psd_wit_inj,
            psd_wit_bkg,
            np.ones(3, dtype=bool),
            psd_wit_inj.value - psd_wit_bkg.value,
            "WIT",
            cast(TimeSeries, _StaticPsdSeries(psd_wit_inj, u.V)),
            cast(TimeSeries, _StaticPsdSeries(psd_wit_bkg, u.V)),
            _AllTrueThreshold(),
            {},
            1.0,
            0.0,
            None,
        )

    assert result is None


def test_coupling_uses_common_prefix_when_target_nyquist_is_lower() -> None:
    witness_freqs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    target_freqs = np.array([0.0, 10.0, 20.0])
    psd_wit_inj = _frequency_series(
        [2.0, 5.0, 10.0, 17.0, 26.0], witness_freqs, unit=u.V**2 / u.Hz
    )
    psd_wit_bkg = _frequency_series(
        [1.0, 1.0, 1.0, 1.0, 1.0], witness_freqs, unit=u.V**2 / u.Hz
    )
    psd_tgt_inj = _frequency_series([2.0, 5.0, 10.0], target_freqs, unit=u.m**2 / u.Hz)
    psd_tgt_bkg = _frequency_series([1.0, 1.0, 1.0], target_freqs, unit=u.m**2 / u.Hz)

    result = coupling_mod._process_single_target(
        "TGT",
        cast(TimeSeries, _StaticPsdSeries(psd_tgt_inj, u.m)),
        cast(TimeSeries, _StaticPsdSeries(psd_tgt_bkg, u.m)),
        {"fftlength": 1.0},
        psd_wit_inj,
        psd_wit_bkg,
        np.ones(len(witness_freqs), dtype=bool),
        psd_wit_inj.value - psd_wit_bkg.value,
        "WIT",
        cast(TimeSeries, _StaticPsdSeries(psd_wit_inj, u.V)),
        cast(TimeSeries, _StaticPsdSeries(psd_wit_bkg, u.V)),
        _AllTrueThreshold(),
        {},
        1.0,
        0.0,
        None,
    )

    assert result is not None
    _, coupling_result = result
    np.testing.assert_allclose(coupling_result.cf.xindex.value, target_freqs)
    np.testing.assert_allclose(coupling_result.cf.value, [1.0, 1.0, 1.0])
    assert len(coupling_result.psd_witness_inj.value) == len(target_freqs)
    assert len(coupling_result.psd_target_inj.value) == len(target_freqs)


def test_coupling_rejects_nonfinite_excess_power_bins() -> None:
    freqs = np.array([10.0, 20.0, 30.0, 40.0])
    psd_wit_inj = _frequency_series([4.0, 4.0, 4.0, 4.0], freqs, unit=u.V**2 / u.Hz)
    psd_wit_bkg = _frequency_series([1.0, 1.0, 1.0, 1.0], freqs, unit=u.V**2 / u.Hz)
    psd_tgt_inj = _frequency_series(
        [4.0, np.inf, np.nan, 1.0], freqs, unit=u.m**2 / u.Hz
    )
    psd_tgt_bkg = _frequency_series([1.0, 1.0, 1.0, 2.0], freqs, unit=u.m**2 / u.Hz)

    result = coupling_mod._process_single_target(
        "TGT",
        cast(TimeSeries, _StaticPsdSeries(psd_tgt_inj, u.m)),
        cast(TimeSeries, _StaticPsdSeries(psd_tgt_bkg, u.m)),
        {"fftlength": 1.0},
        psd_wit_inj,
        psd_wit_bkg,
        np.ones(4, dtype=bool),
        psd_wit_inj.value - psd_wit_bkg.value,
        "WIT",
        cast(TimeSeries, _StaticPsdSeries(psd_wit_inj, u.V)),
        cast(TimeSeries, _StaticPsdSeries(psd_wit_bkg, u.V)),
        _AllTrueThreshold(),
        {},
        1.0,
        0.0,
        None,
    )

    assert result is not None
    _, coupling_result = result
    np.testing.assert_array_equal(
        coupling_result.valid_mask, [True, False, False, False]
    )
    assert coupling_result.cf.value[0] == pytest.approx(1.0)
    assert np.all(np.isnan(coupling_result.cf.value[1:]))


def test_response_row_mismatched_background_grid_returns_nan_cf(
    monkeypatch,
) -> None:
    ts_wit = TimeSeries(np.ones(512), t0=0.0, dt=1.0 / 256, name="W")
    ts_tgt = TimeSeries(np.ones(512), t0=0.0, dt=1.0 / 256, name="T")
    inj_asd = _frequency_series([1.0, 2.0, 4.0], np.array([10.0, 20.0, 30.0]))
    bkg_asd_short = _frequency_series([1.0, 1.0], np.array([10.0, 20.0]))
    bkg_asd = _frequency_series([1.0, 1.0, 1.0], np.array([10.0, 20.0, 30.0]))

    monkeypatch.setattr(TimeSeries, "asd", lambda self, **kwargs: inj_asd)

    row = _compute_response_row(
        witness=ts_wit,
        target=ts_tgt,
        segment=Segment(0.0, 1.0),
        injected_freq=30.0,
        fftlength=1.0,
        overlap=0.0,
        kwargs={},
        master_asd_wit_bkg=bkg_asd_short,
        master_asd_tgt_bkg=bkg_asd,
    )

    assert np.isnan(row["cf"])


def test_response_row_preserves_target_high_frequency_grid_when_witness_is_shorter(
    monkeypatch,
) -> None:
    ts_wit = TimeSeries(np.ones(512), t0=0.0, dt=1.0 / 128, name="W")
    ts_tgt = TimeSeries(np.ones(2048), t0=0.0, dt=1.0 / 512, name="T")
    wit_inj = _frequency_series([1.0, 3.0, 5.0], np.array([0.0, 32.0, 64.0]))
    wit_bkg = _frequency_series([1.0, 1.0, 1.0], np.array([0.0, 32.0, 64.0]))
    tgt_inj = _frequency_series(
        [1.0, 4.0, 9.0, 16.0, 25.0], np.array([0.0, 32.0, 64.0, 96.0, 128.0])
    )
    tgt_bkg = _frequency_series(
        [1.0, 1.0, 1.0, 1.0, 1.0], np.array([0.0, 32.0, 64.0, 96.0, 128.0])
    )
    asd_by_name = {
        "W": wit_inj,
        "T": tgt_inj,
    }

    def fake_asd(self, **kwargs):
        del kwargs
        return asd_by_name[self.name]

    monkeypatch.setattr(TimeSeries, "asd", fake_asd)

    row = _compute_response_row(
        witness=ts_wit,
        target=ts_tgt,
        segment=Segment(0.0, 4.0),
        injected_freq=32.0,
        fftlength=1.0,
        overlap=0.0,
        kwargs={},
        master_asd_wit_bkg=wit_bkg,
        master_asd_tgt_bkg=tgt_bkg,
    )

    np.testing.assert_allclose(row["tgt_asd_inj"].xindex.value, tgt_inj.xindex.value)
    assert row["cf"] == pytest.approx(np.sqrt((4.0**2 - 1.0**2) / (3.0**2 - 1.0**2)))


def test_response_compute_keeps_target_spectrogram_above_witness_nyquist(
    monkeypatch,
) -> None:
    ts_wit = TimeSeries(np.ones(512), t0=0.0, dt=1.0 / 128, name="W")
    ts_tgt = TimeSeries(np.ones(2048), t0=0.0, dt=1.0 / 512, name="T")
    wit_asd = _frequency_series([1.0, 3.0, 5.0], np.array([0.0, 32.0, 64.0]))
    tgt_asd = _frequency_series(
        [1.0, 4.0, 9.0, 16.0, 25.0], np.array([0.0, 32.0, 64.0, 96.0, 128.0])
    )
    asd_by_name = {
        "W": wit_asd,
        "T": tgt_asd,
    }

    def fake_asd(self, **kwargs):
        del kwargs
        return asd_by_name[self.name]

    monkeypatch.setattr(TimeSeries, "asd", fake_asd)

    result = ResponseFunctionAnalysis().compute(
        witness=ts_wit,
        target=ts_tgt,
        segments=[(0.0, 4.0, 32.0)],
        fftlength=1.0,
        auto_detect=False,
    )

    np.testing.assert_allclose(
        result.spectrogram_inj.yindex.value, tgt_asd.xindex.value
    )
    assert result.spectrogram_inj.value.shape == (1, len(tgt_asd.value))


def test_coupling_rejects_mismatched_witness_psd_frequency_grids(
    monkeypatch,
) -> None:
    freqs = np.array([10.0, 20.0, 30.0])
    psd_wit_inj = _frequency_series([4.0, 5.0, 6.0], freqs, unit=u.V**2 / u.Hz)
    psd_wit_bkg = _frequency_series(
        [1.0, 1.0, 1.0], np.array([10.0, 20.0, 31.0]), unit=u.V**2 / u.Hz
    )
    psd_tgt = _frequency_series([2.0, 2.0, 2.0], freqs, unit=u.m**2 / u.Hz)

    sample_rate = 64.0
    times = np.arange(int(sample_rate * 4.0)) / sample_rate
    witness_inj = TimeSeries(
        np.sin(2.0 * np.pi * 8.0 * times),
        sample_rate=sample_rate * u.Hz,
        name="WIT",
    )
    witness_bkg = TimeSeries(
        np.sin(2.0 * np.pi * 8.0 * times),
        sample_rate=sample_rate * u.Hz,
        name="WIT",
    )
    target = TimeSeries(
        np.sin(2.0 * np.pi * 8.0 * times),
        sample_rate=sample_rate * u.Hz,
        name="TGT",
    )

    psd_by_id = {
        id(witness_inj): psd_wit_inj,
        id(witness_bkg): psd_wit_bkg,
        id(target): psd_tgt,
    }

    def fake_psd(self, **kwargs):
        del kwargs
        return psd_by_id[id(self)]

    monkeypatch.setattr(TimeSeries, "psd", fake_psd)

    with pytest.raises(ValueError, match="Witness.*frequency grids"):
        coupling_mod.CouplingFunctionAnalysis().compute(
            data_inj=TimeSeriesDict({"WIT": witness_inj, "TGT": target}),
            data_bkg=TimeSeriesDict({"WIT": witness_bkg, "TGT": target}),
            fftlength=1.0,
            witness="WIT",
        )


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

    np.testing.assert_allclose(np.asarray(line.get_xdata()), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(np.asarray(line.get_ydata()), [1.0e-4, 2.0e-4, 3.0e-4])
    np.testing.assert_allclose(result.injected_freqs, injected_freqs)
    np.testing.assert_allclose(result.coupling_factors, coupling_factors)
    assert result.spectrogram_inj.unit == unit
    np.testing.assert_allclose(result.spectrogram_inj.frequencies.value, freqs)
    assert result.witness_name == "WIT"
    assert result.target_name == "TGT"

    plt.close("all")
