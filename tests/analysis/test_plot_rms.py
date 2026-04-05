"""
CouplingResult.plot_rms() および _compute_rms_timeseries() のテスト。

テストケース:
1. test_plot_rms_both_channels           — 基本動作: 2パネル, axvspan, グリッド
2. test_plot_rms_frange                  — fmin/fmax 帯域制限でエラーなし
3. test_plot_rms_missing_timeseries      — TimeSeries なしで ValueError
4. test_compute_rms_matches_manual_trapz — trapz 手計算との一致確認
5. test_plot_rms_channels_param          — channels="witness"/"target" の1パネル動作
6. test_compute_rms_overlap_is_seconds   — overlap が秒単位で扱われることを確認
7. test_compute_rms_preserves_nan        — NaN が極小値に潰れず伝播することを確認
8. test_plot_rms_ignores_reserved_kwargs — 予約済み plot kwargs を安全に無視
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gwexpy.analysis.coupling_result import CouplingResult, _compute_rms_timeseries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries

FS = 256.0
DURATION = 8.0
FFTLENGTH = 2.0
N_FREQS = 20


def _make_fs(vals: np.ndarray, f0: float = 1.0, df: float = 1.0) -> FrequencySeries:
    return FrequencySeries(vals, f0=f0, df=df)


def _make_ts(rng: np.random.Generator, *, t0: float = 0.0, scale: float = 1.0) -> TimeSeries:
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    data = np.sin(2 * np.pi * 10.0 * t) * scale + rng.normal(0.0, scale * 0.1, n)
    return TimeSeries(data, sample_rate=FS, t0=t0, unit="m")


def _make_coupling_result(
    rng: np.random.Generator,
    *,
    with_ts: bool = True,
    with_target_ts: bool = True,
) -> CouplingResult:
    freqs = np.linspace(1.0, float(N_FREQS), N_FREQS)
    df = freqs[1] - freqs[0]

    psd_bkg = np.ones(N_FREQS) * 1e-10
    psd_inj = psd_bkg * 2.0
    cf_vals = rng.uniform(0.01, 0.5, N_FREQS)

    ts_witness_bkg = _make_ts(rng, t0=0.0) if with_ts else None
    ts_witness_inj = _make_ts(rng, t0=100.0, scale=2.0) if with_ts else None
    ts_target_bkg = _make_ts(rng, t0=0.0) if with_target_ts else None
    ts_target_inj = _make_ts(rng, t0=100.0, scale=1.5) if with_target_ts else None

    return CouplingResult(
        cf=_make_fs(cf_vals, f0=freqs[0], df=df),
        psd_witness_inj=_make_fs(psd_inj, f0=freqs[0], df=df),
        psd_witness_bkg=_make_fs(psd_bkg, f0=freqs[0], df=df),
        psd_target_inj=_make_fs(psd_inj * 1.1, f0=freqs[0], df=df),
        psd_target_bkg=_make_fs(psd_bkg * 1.1, f0=freqs[0], df=df),
        valid_mask=np.ones(N_FREQS, dtype=bool),
        witness_name="WIT",
        target_name="TGT",
        fftlength=FFTLENGTH,
        ts_witness_bkg=ts_witness_bkg,
        ts_witness_inj=ts_witness_inj,
        ts_target_bkg=ts_target_bkg,
        ts_target_inj=ts_target_inj,
    )


# ------------------------------------------------------------------
# 1. 基本動作テスト
# ------------------------------------------------------------------


def test_plot_rms_both_channels():
    """channels='both' で 2 パネルが生成され、axvspan と grid が描画される。"""
    rng = np.random.default_rng(42)
    result = _make_coupling_result(rng)

    fig = result.plot_rms(fftlength=FFTLENGTH)

    try:
        assert len(fig.axes) == 2, "2 パネルであること"

        for ax in fig.axes:
            # grid が設定されている
            assert ax.xaxis.get_gridlines() or ax.yaxis.get_gridlines()
            # axvspan は Polygon として描画される — 少なくとも 2 個 (bkg + inj)
            n_spans = sum(
                1 for patch in ax.patches if hasattr(patch, "get_xy")
            )
            assert n_spans >= 2, f"axvspan が少なすぎます: {n_spans}"
    finally:
        plt.close(fig)


# ------------------------------------------------------------------
# 2. 帯域制限テスト
# ------------------------------------------------------------------


def test_plot_rms_frange():
    """fmin/fmax を指定してもエラーなく描画できる。"""
    rng = np.random.default_rng(0)
    result = _make_coupling_result(rng)

    fig = result.plot_rms(fmin=5.0, fmax=50.0, fftlength=FFTLENGTH)

    try:
        assert len(fig.axes) == 2
        # タイトルに周波数範囲が含まれる
        title = fig.axes[0].get_title()
        assert "5.0" in title and "50.0" in title
    finally:
        plt.close(fig)


# ------------------------------------------------------------------
# 3. TimeSeries なしのエラー確認
# ------------------------------------------------------------------


def test_plot_rms_missing_witness_raises():
    """ts_witness_bkg が None の場合、ValueError が発生する。"""
    rng = np.random.default_rng(1)
    result = _make_coupling_result(rng, with_ts=False)

    with pytest.raises(ValueError, match="ts_witness_bkg"):
        result.plot_rms(channels="witness", fftlength=FFTLENGTH)


def test_plot_rms_missing_target_raises():
    """ts_target_bkg が None の場合、ValueError が発生する。"""
    rng = np.random.default_rng(1)
    result = _make_coupling_result(rng, with_target_ts=False)

    with pytest.raises(ValueError, match="ts_target_bkg"):
        result.plot_rms(channels="target", fftlength=FFTLENGTH)


# ------------------------------------------------------------------
# 4. _compute_rms_timeseries の手計算一致確認
# ------------------------------------------------------------------


def test_compute_rms_matches_manual_trapz():
    """
    _compute_rms_timeseries の出力が、手計算（Spectrogram → trapz → sqrt）と一致する。
    """
    rng = np.random.default_rng(7)
    ts = _make_ts(rng)

    rms_ts = _compute_rms_timeseries(ts, FFTLENGTH, 0.0, None, None)

    # 手計算: 同じパラメータで Spectrogram を作り trapz → sqrt
    spec = ts.spectrogram(
        stride=FFTLENGTH,
        fftlength=FFTLENGTH,
        overlap=0.0,
        method="welch",
        window="hann",
    )
    freqs = spec.frequencies.value
    psd_matrix = spec.value
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    expected = np.sqrt(_trapz(psd_matrix, freqs, axis=1))

    np.testing.assert_allclose(rms_ts.value, expected, rtol=1e-6)


def test_compute_rms_overlap_is_seconds():
    """overlap は割合ではなく秒単位で spectrogram() に渡される。"""
    rng = np.random.default_rng(11)
    ts = _make_ts(rng)

    overlap_seconds = 0.5
    rms_ts = _compute_rms_timeseries(ts, FFTLENGTH, overlap_seconds, None, None)

    spec = ts.spectrogram(
        stride=FFTLENGTH,
        fftlength=FFTLENGTH,
        overlap=overlap_seconds,
        method="welch",
        window="hann",
    )
    expected = np.sqrt((getattr(np, "trapezoid", None) or getattr(np, "trapz"))(
        spec.value,
        spec.frequencies.value,
        axis=1,
    ))

    np.testing.assert_allclose(rms_ts.value, expected, rtol=1e-6)


def test_compute_rms_preserves_nan(monkeypatch: pytest.MonkeyPatch):
    """積分結果が NaN の時間ビンは RMS でも NaN のまま残る。"""
    rng = np.random.default_rng(12)
    ts = _make_ts(rng)
    spec = ts.spectrogram(
        stride=FFTLENGTH,
        fftlength=FFTLENGTH,
        overlap=0.0,
        method="welch",
        window="hann",
    )
    spec.value[0, :] = np.nan

    def _fake_spectrogram(*args, **kwargs):
        return spec

    monkeypatch.setattr(ts, "spectrogram", _fake_spectrogram)
    rms_ts = _compute_rms_timeseries(ts, FFTLENGTH, 0.0, None, None)

    assert np.isnan(rms_ts.value[0])


def test_plot_rms_ignores_reserved_kwargs():
    """color/label/linewidth を渡しても内部スタイルと衝突せず描画できる。"""
    rng = np.random.default_rng(13)
    result = _make_coupling_result(rng)

    fig = result.plot_rms(
        fftlength=FFTLENGTH,
        color="tab:green",
        label="user-label",
        linewidth=5.0,
        alpha=0.4,
    )

    try:
        lines = fig.axes[0].lines
        assert len(lines) == 2
        assert lines[0].get_color() == "black"
        assert lines[1].get_color() == "tab:red"
        assert lines[0].get_label() == "Background"
        assert lines[1].get_label() == "Injection"
        assert lines[0].get_alpha() == pytest.approx(0.4)
        assert lines[1].get_alpha() == pytest.approx(0.4)
    finally:
        plt.close(fig)


# ------------------------------------------------------------------
# 5. channels パラメータの動作確認
# ------------------------------------------------------------------


def test_plot_rms_channels_witness_single_panel():
    """channels='witness' で 1 パネルのみ生成される。"""
    rng = np.random.default_rng(3)
    result = _make_coupling_result(rng)

    fig = result.plot_rms(channels="witness", fftlength=FFTLENGTH)

    try:
        assert len(fig.axes) == 1
        title = fig.axes[0].get_title()
        assert "WIT" in title
    finally:
        plt.close(fig)


def test_plot_rms_channels_target_single_panel():
    """channels='target' で 1 パネルのみ生成される。"""
    rng = np.random.default_rng(4)
    result = _make_coupling_result(rng)

    fig = result.plot_rms(channels="target", fftlength=FFTLENGTH)

    try:
        assert len(fig.axes) == 1
        title = fig.axes[0].get_title()
        assert "TGT" in title
    finally:
        plt.close(fig)


def test_plot_rms_invalid_channels_raises():
    """channels に無効な値を指定すると ValueError が発生する。"""
    rng = np.random.default_rng(5)
    result = _make_coupling_result(rng)

    with pytest.raises(ValueError, match="channels"):
        result.plot_rms(channels="invalid", fftlength=FFTLENGTH)
