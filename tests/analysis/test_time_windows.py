"""
Phase 1 時間ウィンドウ API テスト

- test_bkg_window_override_auto_detect
- test_from_time_windows_basic
- test_from_time_windows_batch_processing
- test_window_vs_auto_detection_comparison
- test_estimate_coupling_bkg_window
- test_response_bkg_window
"""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.analysis.coupling import CouplingFunctionAnalysis, estimate_coupling
from gwexpy.analysis.coupling_result import CouplingResult
from gwexpy.analysis.response import (
    ResponseFunctionAnalysis,
    estimate_response_function,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = 100.0
DURATION_BKG = 10.0
DURATION_INJ = 5.0
T_BKG_START = 1_000_000_000.0
T_BKG_END = T_BKG_START + DURATION_BKG
T_INJ_START = T_BKG_END + 2.0
T_INJ_END = T_INJ_START + DURATION_INJ
TOTAL_DURATION = T_INJ_END - T_BKG_START


def _make_full_data(rng: np.random.Generator) -> TimeSeriesDict:
    """背景 + 注入区間を含む全時間系列データを作成する。"""
    n_bkg = int(DURATION_BKG * FS)
    n_gap = int(2.0 * FS)
    n_inj = int(DURATION_INJ * FS)
    wit_data = np.concatenate([
        rng.normal(0, 1, n_bkg),
        rng.normal(0, 1, n_gap),
        rng.normal(0, 10, n_inj),  # 注入区間: 10x 振幅
    ])
    tgt_data = np.concatenate([
        rng.normal(0, 1, n_bkg),
        rng.normal(0, 1, n_gap),
        rng.normal(0, 5, n_inj),   # 注入区間: 5x 振幅
    ])

    t0 = T_BKG_START
    wit = TimeSeries(wit_data, sample_rate=FS, t0=t0, name="WIT")
    tgt = TimeSeries(tgt_data, sample_rate=FS, t0=t0, name="TGT")
    return TimeSeriesDict({"WIT": wit, "TGT": tgt})


def _make_split_dicts(rng: np.random.Generator) -> tuple[TimeSeriesDict, TimeSeriesDict]:
    """背景・注入をそれぞれ別の TimeSeriesDict として作成する。"""
    n = int(DURATION_INJ * FS)  # 同一長で比較
    wit_bkg = TimeSeries(rng.normal(0, 1, n), sample_rate=FS, t0=T_BKG_START, name="WIT")
    tgt_bkg = TimeSeries(rng.normal(0, 1, n), sample_rate=FS, t0=T_BKG_START, name="TGT")
    wit_inj = TimeSeries(rng.normal(0, 10, n), sample_rate=FS, t0=T_INJ_START, name="WIT")
    tgt_inj = TimeSeries(rng.normal(0, 5, n), sample_rate=FS, t0=T_INJ_START, name="TGT")
    data_bkg = TimeSeriesDict({"WIT": wit_bkg, "TGT": tgt_bkg})
    data_inj = TimeSeriesDict({"WIT": wit_inj, "TGT": tgt_inj})
    return data_inj, data_bkg


# ---------------------------------------------------------------------------
# CouplingFunctionAnalysis テスト
# ---------------------------------------------------------------------------


def test_from_time_windows_basic():
    """from_time_windows() が単一 CouplingResult を返す。"""
    rng = np.random.default_rng(42)
    data = _make_full_data(rng)

    result = CouplingFunctionAnalysis.from_time_windows(
        data,
        bkg_window=(T_BKG_START, T_BKG_END),
        inj_window=(T_INJ_START, T_INJ_END),
        fftlength=1.0,
    )

    assert isinstance(result, CouplingResult), f"Expected CouplingResult, got {type(result)}"
    assert result.witness_name == "WIT"
    assert result.target_name == "TGT"
    assert len(result.cf.value) > 0
    # CF 値は有限であるべき (NaN のみではない)
    assert np.any(np.isfinite(result.cf.value))


def test_from_time_windows_batch_processing():
    """from_time_windows_batch() で複数結果のリストを返す。"""
    rng = np.random.default_rng(123)
    data = _make_full_data(rng)

    # 同じ区間を2回渡してバッチ処理をテスト
    inj_windows = [
        (T_INJ_START, T_INJ_END),
        (T_INJ_START, T_INJ_END),
    ]

    results = CouplingFunctionAnalysis.from_time_windows_batch(
        data,
        bkg_window=(T_BKG_START, T_BKG_END),
        inj_windows=inj_windows,
        fftlength=1.0,
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for res in results:
        assert isinstance(res, CouplingResult)


def test_from_time_windows_batch_empty_raises():
    """inj_windows が空リストの場合に ValueError を送出する。"""
    rng = np.random.default_rng(0)
    data = _make_full_data(rng)
    with pytest.raises(ValueError, match="inj_windows"):
        CouplingFunctionAnalysis.from_time_windows_batch(
            data,
            bkg_window=(T_BKG_START, T_BKG_END),
            inj_windows=[],
            fftlength=1.0,
        )


def test_bkg_window_invalid_range_raises():
    """bkg_window の終端 <= 始端のとき ValueError を送出する。"""
    rng = np.random.default_rng(0)
    data = _make_full_data(rng)
    with pytest.raises(ValueError, match="bkg_window end"):
        CouplingFunctionAnalysis.from_time_windows(
            data,
            bkg_window=(T_BKG_END, T_BKG_START),  # 逆順
            inj_window=(T_INJ_START, T_INJ_END),
            fftlength=1.0,
        )


def test_window_vs_auto_detection_comparison():
    """from_time_windows() と compute() が同一データで同等の結果を返す。"""
    rng = np.random.default_rng(7)
    data_inj, data_bkg = _make_split_dicts(rng)

    # 方法 A: from_time_windows (全データを一つの TimeSeriesDict に結合)
    from gwexpy.timeseries import TimeSeries, TimeSeriesDict
    combined = TimeSeriesDict({
        k: TimeSeries(
            np.concatenate([data_bkg[k].value, data_inj[k].value]),
            sample_rate=FS,
            t0=T_BKG_START,
            name=k,
        )
        for k in data_inj
    })
    bkg_end = T_BKG_START + DURATION_INJ
    inj_start = bkg_end
    inj_end = inj_start + DURATION_INJ

    res_a = CouplingFunctionAnalysis.from_time_windows(
        combined,
        bkg_window=(T_BKG_START, bkg_end),
        inj_window=(inj_start, inj_end),
        fftlength=1.0,
    )

    # 方法 B: 直接 compute
    analysis = CouplingFunctionAnalysis()
    res_b = analysis.compute(data_inj, data_bkg, fftlength=1.0)

    # 周波数グリッドが一致すること
    assert np.allclose(res_a.cf.xindex.value, res_b.cf.xindex.value)
    # valid_mask の形状が一致すること
    assert res_a.valid_mask.shape == res_b.valid_mask.shape


# ---------------------------------------------------------------------------
# estimate_coupling() bkg_window テスト
# ---------------------------------------------------------------------------


def test_estimate_coupling_bkg_window():
    """estimate_coupling() に bkg_window + inj_window を渡すと正常に動作する。"""
    rng = np.random.default_rng(99)
    data = _make_full_data(rng)

    result = estimate_coupling(
        data_inj=data,
        bkg_window=(T_BKG_START, T_BKG_END),
        inj_window=(T_INJ_START, T_INJ_END),
        fftlength=1.0,
    )
    assert isinstance(result, CouplingResult)


def test_estimate_coupling_bkg_window_without_inj_raises():
    """bkg_window だけで inj_window なしの場合に ValueError を送出する。"""
    rng = np.random.default_rng(0)
    data = _make_full_data(rng)
    with pytest.raises(ValueError, match="inj_window"):
        estimate_coupling(
            data_inj=data,
            bkg_window=(T_BKG_START, T_BKG_END),
            fftlength=1.0,
        )


def test_estimate_coupling_legacy_requires_data_bkg():
    """bkg_window なしで data_bkg も渡さない場合に ValueError を送出する。"""
    rng = np.random.default_rng(0)
    data = _make_full_data(rng)
    with pytest.raises(ValueError, match="data_bkg"):
        estimate_coupling(
            data_inj=data,
            fftlength=1.0,
        )


def test_estimate_coupling_forwards_legacy_parameters(monkeypatch: pytest.MonkeyPatch):
    """estimate_coupling() が legacy mode の主要パラメータを compute() に転送する。"""
    rng = np.random.default_rng(1234)
    data_inj, data_bkg = _make_split_dicts(rng)
    captured: dict[str, object] = {}

    def fake_compute(self, data_inj, data_bkg, fftlength, **kwargs):
        captured["data_inj"] = data_inj
        captured["data_bkg"] = data_bkg
        captured["fftlength"] = fftlength
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(CouplingFunctionAnalysis, "compute", fake_compute)

    result = estimate_coupling(
        data_inj=data_inj,
        data_bkg=data_bkg,
        fftlength=1.5,
        overlap=0.25,
        percentile_factor=3.1,
        bkg_stride=2.0,
        memory_limit=123456,
        n_jobs=2,
    )

    assert result == "ok"
    assert captured["data_inj"] is data_inj
    assert captured["data_bkg"] is data_bkg
    assert captured["fftlength"] == 1.5
    assert captured["overlap"] == 0.25
    assert captured["percentile_factor"] == 3.1
    assert captured["bkg_stride"] == 2.0
    assert captured["memory_limit"] == 123456
    assert captured["n_jobs"] == 2


def test_estimate_coupling_forwards_time_window_parameters(
    monkeypatch: pytest.MonkeyPatch,
):
    """estimate_coupling() が time-window mode の主要パラメータを転送する。"""
    rng = np.random.default_rng(5678)
    data = _make_full_data(rng)
    captured: dict[str, object] = {}

    @classmethod
    def fake_from_time_windows(
        cls,
        data,
        bkg_window,
        inj_window,
        **kwargs,
    ):
        captured["data"] = data
        captured["bkg_window"] = bkg_window
        captured["inj_window"] = inj_window
        captured.update(kwargs)
        return "time-window-ok"

    monkeypatch.setattr(
        CouplingFunctionAnalysis,
        "from_time_windows",
        fake_from_time_windows,
    )

    result = estimate_coupling(
        data_inj=data,
        fftlength=1.25,
        bkg_window=(T_BKG_START, T_BKG_END),
        inj_window=(T_INJ_START, T_INJ_END),
        overlap=0.5,
        percentile_factor=2.9,
        bkg_stride=1.5,
        memory_limit=654321,
        n_jobs=3,
    )

    assert result == "time-window-ok"
    assert captured["data"] is data
    assert captured["bkg_window"] == (T_BKG_START, T_BKG_END)
    assert captured["inj_window"] == (T_INJ_START, T_INJ_END)
    assert captured["fftlength"] == 1.25
    assert captured["overlap"] == 0.5
    assert captured["percentile_factor"] == 2.9
    assert captured["bkg_stride"] == 1.5
    assert captured["memory_limit"] == 654321
    assert captured["n_jobs"] == 3


# ---------------------------------------------------------------------------
# ResponseFunctionAnalysis bkg_window テスト
# ---------------------------------------------------------------------------


def _make_sine_ts(freq: float, duration: float, t0: float = 0.0, fs: float = 256.0) -> TimeSeries:
    """単一正弦波の TimeSeries を作成する。"""
    n = int(duration * fs)
    t = np.arange(n) / fs
    data = np.sin(2 * np.pi * freq * t)
    return TimeSeries(data, sample_rate=fs, t0=t0, name="CH")


def test_response_bkg_window():
    """ResponseFunctionAnalysis.compute() の bkg_window が正しく機能する。

    bkg_window を指定した場合と witness_bkg / target_bkg を直接指定した場合が
    同一結果を返すことを確認する。
    """
    rng = np.random.default_rng(55)
    fs = 256.0
    duration_bkg = 10.0
    duration_inj = 8.0
    inj_freq = 10.0

    # 背景データ（小さい振幅）
    n_bkg = int(duration_bkg * fs)
    bkg_data = rng.normal(0, 0.01, n_bkg)

    # 注入区間は正弦波 + 小さいノイズ
    n_inj = int(duration_inj * fs)
    t = np.arange(n_inj) / fs
    inj_data = np.sin(2 * np.pi * inj_freq * t) + rng.normal(0, 0.01, n_inj)

    t0_bkg = 1_000_000_000.0
    t0_inj = t0_bkg + duration_bkg + 1.0

    wit_full = TimeSeries(
        np.concatenate([bkg_data, np.zeros(int(fs)), inj_data]),
        sample_rate=fs, t0=t0_bkg, name="WIT",
    )
    tgt_full = TimeSeries(
        np.concatenate([bkg_data * 0.5, np.zeros(int(fs)), inj_data * 0.5]),
        sample_rate=fs, t0=t0_bkg, name="TGT",
    )

    # A: bkg_window 指定
    segments = [(t0_inj, t0_inj + duration_inj - 1.0, inj_freq)]
    analysis = ResponseFunctionAnalysis()
    res_a = analysis.compute(
        wit_full,
        tgt_full,
        segments=segments,
        fftlength=2.0,
        bkg_window=(t0_bkg, t0_bkg + duration_bkg),
    )

    # B: witness_bkg / target_bkg 直接指定
    wit_bkg = wit_full.crop(t0_bkg, t0_bkg + duration_bkg)
    tgt_bkg = tgt_full.crop(t0_bkg, t0_bkg + duration_bkg)
    res_b = analysis.compute(
        wit_full,
        tgt_full,
        segments=segments,
        fftlength=2.0,
        witness_bkg=wit_bkg,
        target_bkg=tgt_bkg,
    )

    # 結合係数の差が十分小さいこと
    np.testing.assert_allclose(
        res_a.coupling_factors, res_b.coupling_factors,
        rtol=1e-10, atol=1e-10,
        err_msg="bkg_window result should match explicit witness_bkg/target_bkg result.",
    )


def test_response_bkg_window_invalid_raises():
    """bkg_window の終端 <= 始端のとき ValueError を送出する。"""
    fs = 256.0
    n = int(10 * fs)
    rng = np.random.default_rng(0)
    ts = TimeSeries(rng.normal(0, 1, n), sample_rate=fs, t0=0.0)
    analysis = ResponseFunctionAnalysis()
    with pytest.raises(ValueError, match="bkg_window end"):
        analysis.compute(
            ts, ts,
            segments=[(5.0, 9.0, 10.0)],
            fftlength=1.0,
            bkg_window=(5.0, 1.0),  # 逆順
        )


def test_estimate_response_function_forwards_wrapper_parameters(
    monkeypatch: pytest.MonkeyPatch,
):
    """estimate_response_function() が wrapper パラメータを compute() に転送する。"""
    ts = _make_sine_ts(10.0, duration=8.0, t0=1_000_000_000.0)
    captured: dict[str, object] = {}

    def fake_compute(self, witness, target, **kwargs):
        captured["witness"] = witness
        captured["target"] = target
        captured.update(kwargs)
        return "response-ok"

    monkeypatch.setattr(ResponseFunctionAnalysis, "compute", fake_compute)

    result = estimate_response_function(
        witness=ts,
        target=ts,
        segments=[(1_000_000_002.0, 1_000_000_006.0, 10.0)],
        fftlength=2.0,
        bkg_window=(1_000_000_000.0, 1_000_000_004.0),
        n_jobs=4,
        memory_limit=987654,
    )

    assert result == "response-ok"
    assert captured["witness"] is ts
    assert captured["target"] is ts
    assert captured["segments"] == [(1_000_000_002.0, 1_000_000_006.0, 10.0)]
    assert captured["fftlength"] == 2.0
    assert captured["bkg_window"] == (1_000_000_000.0, 1_000_000_004.0)
    assert captured["n_jobs"] == 4
    assert captured["memory_limit"] == 987654
