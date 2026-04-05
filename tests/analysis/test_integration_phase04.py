"""
Phase 4 統合テスト。

- Coupling / Response の E2E フロー
- Time-window API と collection 集約
- Wrapper parameter forwarding
"""

from __future__ import annotations

import csv
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gwexpy.analysis.coupling import (
    CouplingFunctionAnalysis,
    PercentileThreshold,
    estimate_coupling,
)
from gwexpy.analysis.coupling_result import CouplingResult, CouplingResultCollection
from gwexpy.analysis.response import detect_step_segments, estimate_response_function
from gwexpy.analysis.stats import SpectralStats
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

FS = 128.0
FFTLENGTH = 1.0
DURATION_BKG = 10.0
DURATION_INJ = 8.0
T_BKG_START = 1_000_000_000.0
T_BKG_END = T_BKG_START + DURATION_BKG
T_INJ_START = T_BKG_END + 2.0
T_INJ_END = T_INJ_START + DURATION_INJ


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None, None, None]:
    """各テストの前後で Figure を確実に解放する。"""
    plt.close("all")
    yield
    plt.close("all")


def _make_full_data(rng: np.random.Generator) -> TimeSeriesDict:
    """背景区間と注入区間を含む Coupling 用の全データを作る。"""
    n_bkg = int(DURATION_BKG * FS)
    n_gap = int(2.0 * FS)
    n_inj = int(DURATION_INJ * FS)
    t_inj = np.arange(n_inj) / FS
    witness_inj = 8.0 * np.sin(2 * np.pi * 12.0 * t_inj) + rng.normal(0, 0.5, n_inj)
    target_inj = 2.0 * np.sin(2 * np.pi * 12.0 * t_inj) + rng.normal(0, 0.2, n_inj)

    wit_data = np.concatenate(
        [
            rng.normal(0, 0.2, n_bkg),
            rng.normal(0, 0.2, n_gap),
            witness_inj,
        ]
    )
    tgt_data = np.concatenate(
        [
            rng.normal(0, 0.1, n_bkg),
            rng.normal(0, 0.1, n_gap),
            target_inj,
        ]
    )
    return TimeSeriesDict(
        {
            "WIT": TimeSeries(wit_data, sample_rate=FS, t0=T_BKG_START, name="WIT"),
            "TGT": TimeSeries(tgt_data, sample_rate=FS, t0=T_BKG_START, name="TGT"),
        }
    )


def _make_split_dicts(
    rng: np.random.Generator,
) -> tuple[TimeSeriesDict, TimeSeriesDict]:
    """estimate_coupling legacy mode 用の注入/背景 dict を作る。"""
    n = int(DURATION_INJ * FS)
    t = np.arange(n) / FS
    wit_bkg = TimeSeries(
        rng.normal(0, 0.15, n), sample_rate=FS, t0=T_BKG_START, name="WIT"
    )
    tgt_bkg = TimeSeries(
        rng.normal(0, 0.08, n), sample_rate=FS, t0=T_BKG_START, name="TGT"
    )
    wit_inj = TimeSeries(
        8.0 * np.sin(2 * np.pi * 12.0 * t) + rng.normal(0, 0.3, n),
        sample_rate=FS,
        t0=T_INJ_START,
        name="WIT",
    )
    tgt_inj = TimeSeries(
        2.0 * np.sin(2 * np.pi * 12.0 * t) + rng.normal(0, 0.1, n),
        sample_rate=FS,
        t0=T_INJ_START,
        name="TGT",
    )
    return (
        TimeSeriesDict({"WIT": wit_inj, "TGT": tgt_inj}),
        TimeSeriesDict({"WIT": wit_bkg, "TGT": tgt_bkg}),
    )


def _make_response_inputs(
    rng: np.random.Generator,
) -> tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
    """Response 用の背景 + 注入データを作る。"""
    sample_rate = 256.0
    dt = 1.0 / sample_rate
    bkg_duration = 12.0
    inj_duration = 24.0
    gap_duration = 2.0

    n_bkg = int(bkg_duration * sample_rate)
    n_gap = int(gap_duration * sample_rate)
    n_inj = int(inj_duration * sample_rate)
    t = np.arange(n_inj) * dt

    witness_bkg_data = rng.normal(0, 0.01, n_bkg)
    target_bkg_data = rng.normal(0, 0.005, n_bkg)
    witness_inj_data = np.sin(2 * np.pi * 18.0 * t) + rng.normal(0, 0.01, n_inj)
    target_inj_data = 0.4 * np.sin(2 * np.pi * 18.0 * t) + rng.normal(0, 0.005, n_inj)

    t0 = 0.0
    witness = TimeSeries(
        np.concatenate([witness_bkg_data, np.zeros(n_gap), witness_inj_data]),
        sample_rate=sample_rate,
        t0=t0,
        name="WIT",
    )
    target = TimeSeries(
        np.concatenate([target_bkg_data, np.zeros(n_gap), target_inj_data]),
        sample_rate=sample_rate,
        t0=t0,
        name="TGT",
    )
    witness_bkg = witness.crop(t0, t0 + bkg_duration)
    target_bkg = target.crop(t0, t0 + bkg_duration)
    return witness, target, witness_bkg, target_bkg


class TestCouplingE2E:
    """Coupling Function Analysis の統合フロー。"""

    def test_estimate_to_csv_roundtrip(self, tmp_path: Any) -> None:
        """estimate_coupling() → to_csv() → from_csv() を往復できる。"""
        rng = np.random.default_rng(10)
        data_inj, data_bkg = _make_split_dicts(rng)

        result = estimate_coupling(
            data_inj,
            data_bkg,
            fftlength=FFTLENGTH,
        )
        assert isinstance(result, CouplingResult)

        csv_path = tmp_path / "coupling.csv"
        result.to_csv(csv_path)
        restored = CouplingResult.from_csv(csv_path)

        np.testing.assert_allclose(restored.cf.value, result.cf.value)
        np.testing.assert_allclose(restored.cf.xindex.value, result.cf.xindex.value)

    def test_estimate_to_txt_roundtrip(self, tmp_path: Any) -> None:
        """estimate_coupling() → to_txt() → from_txt() を往復できる。"""
        rng = np.random.default_rng(11)
        data_inj, data_bkg = _make_split_dicts(rng)

        result = estimate_coupling(
            data_inj,
            data_bkg,
            fftlength=FFTLENGTH,
        )
        assert isinstance(result, CouplingResult)

        txt_path = tmp_path / "coupling.txt"
        result.to_txt(txt_path)
        restored = CouplingResult.from_txt(txt_path)

        np.testing.assert_allclose(restored.cf.value, result.cf.value)

    def test_estimate_with_percentile_threshold_keeps_timeseries_for_plots(self) -> None:
        """PercentileThreshold 経路でも plot_asdgram / plot_snrgram まで流せる。"""
        rng = np.random.default_rng(12)
        data_inj, data_bkg = _make_split_dicts(rng)

        result = estimate_coupling(
            data_inj,
            data_bkg,
            fftlength=FFTLENGTH,
            threshold_witness=PercentileThreshold(percentile=50.0, factor=1.0),
            threshold_target=4.0,
        )
        assert isinstance(result, CouplingResult)
        # 既存 compute() が注入/背景時系列を保持していることを検証
        assert result.ts_witness_inj is not None
        assert result.ts_witness_bkg is not None
        fig_asd = result.plot_asdgram()
        fig_snr = result.plot_snrgram(snrmax=5.0)
        assert fig_asd is not None
        assert fig_snr is not None

    def test_spectral_stats_not_nan(self) -> None:
        """result.spectral_stats() が SpectralStats を返し、主要配列が NaN で埋まらない。"""
        rng = np.random.default_rng(13)
        data_inj, data_bkg = _make_split_dicts(rng)

        result = estimate_coupling(data_inj, data_bkg, fftlength=FFTLENGTH)
        assert isinstance(result, CouplingResult)
        stats = result.spectral_stats()

        assert isinstance(stats, SpectralStats)
        assert np.all(np.isfinite(stats.mean.value))
        assert np.all(np.isfinite(stats.sigma.value))
        assert stats.n_avg >= 1

    def test_all_plot_methods_return_figure(self) -> None:
        """CouplingResult の主要 plot 群が Figure/Plot を返して完走する。"""
        rng = np.random.default_rng(14)
        data_inj, data_bkg = _make_split_dicts(rng)

        result = estimate_coupling(data_inj, data_bkg, fftlength=FFTLENGTH)
        assert isinstance(result, CouplingResult)
        assert result.plot_cf() is not None
        assert result.plot() is not None
        assert result.plot_significance() is not None
        assert result.plot_asdgram() is not None
        assert result.plot_snrgram() is not None

    def test_valid_mask_dtype(self) -> None:
        """valid_mask は bool ndarray で保持される。"""
        rng = np.random.default_rng(15)
        data_inj, data_bkg = _make_split_dicts(rng)

        result = estimate_coupling(data_inj, data_bkg, fftlength=FFTLENGTH)
        assert isinstance(result, CouplingResult)
        assert isinstance(result.valid_mask, np.ndarray)
        assert result.valid_mask.dtype == bool


class TestResponseE2E:
    """Response Function Analysis の統合フロー。"""

    def test_detect_and_estimate_full_flow(self) -> None:
        """detect_step_segments() → estimate_response_function() の一連フローが完走する。"""
        rng = np.random.default_rng(20)
        witness, target, witness_bkg, target_bkg = _make_response_inputs(rng)

        segments = detect_step_segments(
            witness,
            fftlength=1.0,
            snr_threshold=4.0,
            min_duration=3.0,
            trim_edge=1.0,
            freq_tolerance=5.0,
        )
        assert segments

        result = estimate_response_function(
            witness=witness,
            target=target,
            segments=segments,
            fftlength=1.0,
            witness_bkg=witness_bkg,
            target_bkg=target_bkg,
        )
        assert len(result.coupling_factors) == len(segments)

    def test_all_plot_methods_return_figure(self) -> None:
        """ResponseResult の plot 群が例外なく完走する。"""
        rng = np.random.default_rng(21)
        witness, target, witness_bkg, target_bkg = _make_response_inputs(rng)
        segments = detect_step_segments(
            witness,
            fftlength=1.0,
            snr_threshold=4.0,
            min_duration=3.0,
            trim_edge=1.0,
            freq_tolerance=5.0,
        )
        result = estimate_response_function(
            witness=witness,
            target=target,
            segments=segments,
            fftlength=1.0,
            witness_bkg=witness_bkg,
            target_bkg=target_bkg,
        )
        assert result.plot() is not None
        assert result.plot_map() is not None
        assert result.plot_snapshot(step_index=0) is not None
        assert result.plot_projection_summary() is not None
        assert result.plot_response_matrix() is not None

    def test_spectrogram_shape_consistency(self) -> None:
        """結果の注入/背景 spectrogram 形状が一致する。"""
        rng = np.random.default_rng(22)
        witness, target, witness_bkg, target_bkg = _make_response_inputs(rng)
        segments = detect_step_segments(
            witness,
            fftlength=1.0,
            snr_threshold=4.0,
            min_duration=3.0,
            trim_edge=1.0,
            freq_tolerance=5.0,
        )
        result = estimate_response_function(
            witness=witness,
            target=target,
            segments=segments,
            fftlength=1.0,
            witness_bkg=witness_bkg,
            target_bkg=target_bkg,
        )
        assert result.spectrogram_inj.shape == result.spectrogram_bkg.shape


class TestTimeWindowsE2E:
    """時間ウィンドウ API と collection の統合。"""

    def test_from_time_windows_single(self) -> None:
        """from_time_windows() が CouplingResult を返す。"""
        rng = np.random.default_rng(30)
        data = _make_full_data(rng)
        result = CouplingFunctionAnalysis.from_time_windows(
            data,
            bkg_window=(T_BKG_START, T_BKG_END),
            inj_window=(T_INJ_START, T_INJ_END),
            fftlength=FFTLENGTH,
        )
        assert isinstance(result, CouplingResult)

    def test_from_time_windows_batch_consistency(self) -> None:
        """single と batch 1 要素の結果が周波数軸・CF で一致する。"""
        rng = np.random.default_rng(31)
        data = _make_full_data(rng)
        single = CouplingFunctionAnalysis.from_time_windows(
            data,
            bkg_window=(T_BKG_START, T_BKG_END),
            inj_window=(T_INJ_START, T_INJ_END),
            fftlength=FFTLENGTH,
        )
        assert isinstance(single, CouplingResult)
        batch = CouplingFunctionAnalysis.from_time_windows_batch(
            data,
            bkg_window=(T_BKG_START, T_BKG_END),
            inj_windows=[(T_INJ_START, T_INJ_END)],
            fftlength=FFTLENGTH,
        )
        assert len(batch) == 1
        assert isinstance(batch[0], CouplingResult)
        np.testing.assert_allclose(batch[0].cf.value, single.cf.value)
        np.testing.assert_allclose(batch[0].cf.xindex.value, single.cf.xindex.value)

    def test_collection_iteration(self) -> None:
        """CouplingResultCollection の iteration と len が整合する。"""
        rng = np.random.default_rng(32)
        r1, r2 = [
            CouplingFunctionAnalysis.from_time_windows(
                _make_full_data(rng),
                bkg_window=(T_BKG_START, T_BKG_END),
                inj_window=(T_INJ_START, T_INJ_END),
                fftlength=FFTLENGTH,
            )
            for _ in range(2)
        ]
        assert isinstance(r1, CouplingResult)
        assert isinstance(r2, CouplingResult)
        collection = CouplingResultCollection({"a": r1, "b": r2})
        assert len(collection) == 2
        assert list(collection.keys()) == ["a", "b"]
        assert list(collection.values()) == [r1, r2]


class TestCollectionAggregation:
    """Collection の集約 CSV と stats 補助。"""

    def test_summary_csv_row_count(self, tmp_path: Any) -> None:
        """summary CSV の行数が全要素の周波数ビン数合計に一致する。"""
        rng = np.random.default_rng(40)
        data1_inj, data1_bkg = _make_split_dicts(rng)
        data2_inj, data2_bkg = _make_split_dicts(rng)
        r1 = estimate_coupling(data1_inj, data1_bkg, fftlength=FFTLENGTH)
        r2 = estimate_coupling(data2_inj, data2_bkg, fftlength=FFTLENGTH)
        assert isinstance(r1, CouplingResult)
        assert isinstance(r2, CouplingResult)
        collection = CouplingResultCollection({"p1": r1, "p2": r2})

        out = tmp_path / "summary.csv"
        collection.to_summary_csv(out)

        with open(out, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(r1.cf.value) + len(r2.cf.value)

    def test_spectral_stats_keys(self) -> None:
        """SpectralStats.to_dict() のキーが仕様通り。"""
        rng = np.random.default_rng(41)
        data_inj, data_bkg = _make_split_dicts(rng)
        result = estimate_coupling(data_inj, data_bkg, fftlength=FFTLENGTH)
        assert isinstance(result, CouplingResult)
        stats_dict = result.spectral_stats().to_dict()
        assert set(stats_dict) == {"mean", "sigma", "n_avg"}


class TestWrapperForwarding:
    """Wrapper parameter forwarding を Phase 4 側でも固定する。"""

    def test_estimate_coupling_forwards_params(self) -> None:
        """estimate_coupling() が主要パラメータを compute() に転送する。"""
        captured: dict[str, object] = {}
        data_inj = TimeSeriesDict()
        data_bkg = TimeSeriesDict()

        def fake_compute(self, data_inj, data_bkg, fftlength, **kwargs):
            captured["data_inj"] = data_inj
            captured["data_bkg"] = data_bkg
            captured["fftlength"] = fftlength
            captured.update(kwargs)
            return "ok"

        with patch.object(CouplingFunctionAnalysis, "compute", fake_compute):
            result = estimate_coupling(
                data_inj=data_inj,
                data_bkg=data_bkg,
                fftlength=1.5,
                overlap=0.25,
                percentile_factor=3.1,
                bkg_stride=2.0,
                memory_limit=123456,
            )
        assert cast(str, result) == "ok"
        assert captured["overlap"] == 0.25
        assert captured["percentile_factor"] == 3.1
        assert captured["bkg_stride"] == 2.0
        assert captured["memory_limit"] == 123456

    def test_estimate_response_forwards_params(self) -> None:
        """estimate_response_function() が主要パラメータを compute() に転送する。"""
        captured: dict[str, object] = {}
        witness = TimeSeries(np.zeros(256), sample_rate=256.0, t0=0.0)
        target = TimeSeries(np.zeros(256), sample_rate=256.0, t0=0.0)

        def fake_compute(self, **kwargs):
            captured.update(kwargs)
            return "ok"

        with patch("gwexpy.analysis.response.ResponseFunctionAnalysis.compute", fake_compute):
            result = estimate_response_function(
                witness=witness,
                target=target,
                segments=[(1.0, 3.0, 10.0)],
                fftlength=2.0,
                bkg_window=(0.0, 1.0),
                n_jobs=4,
                memory_limit=987654,
            )
        assert cast(str, result) == "ok"
        assert captured["bkg_window"] == (0.0, 1.0)
        assert captured["n_jobs"] == 4
        assert captured["memory_limit"] == 987654
