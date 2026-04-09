"""
Phase 3 可視化拡張テスト。

- CouplingResult.plot_significance()
- CouplingResult.plot_asdgram()
- CouplingResult.plot_snrgram()
- ResponseFunctionResult.plot_projection_summary()
- ResponseFunctionResult.plot_response_matrix()
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh

from gwexpy.analysis.coupling_result import CouplingResult
from gwexpy.analysis.response import ResponseFunctionResult
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeries

FS = 256.0
DURATION = 8.0
FFTLENGTH = 2.0
N_FREQS = 20
N_STEPS = 4


def _make_fs(vals: np.ndarray, f0: float = 1.0, df: float = 1.0) -> FrequencySeries:
    """指定した値から FrequencySeries を作成する。"""
    return FrequencySeries(vals, f0=f0, df=df)


def _make_timeseries(
    rng: np.random.Generator,
    scale: float,
    *,
    seed_offset: float = 0.0,
) -> TimeSeries:
    """再現性のある TimeSeries を作成する。"""
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    signal = np.sin(2 * np.pi * 12.0 * t + seed_offset)
    noise = rng.normal(0.0, scale, n)
    return TimeSeries(signal + noise, sample_rate=FS, t0=0.0)


def _make_coupling_result(
    rng: np.random.Generator,
    *,
    ts_witness_inj: TimeSeries | None = None,
    ts_witness_bkg: TimeSeries | None = None,
    fftlength: float | None = FFTLENGTH,
) -> CouplingResult:
    """可視化テスト用の CouplingResult を構成する。"""
    freqs = np.linspace(1.0, float(N_FREQS), N_FREQS)
    df = freqs[1] - freqs[0]

    psd_bkg = np.linspace(1.0, 2.0, N_FREQS) * 1e-10
    psd_inj = psd_bkg * np.linspace(1.5, 3.0, N_FREQS)
    cf_vals = rng.uniform(0.01, 0.5, N_FREQS)
    cf_ul_vals = cf_vals * 1.5

    return CouplingResult(
        cf=_make_fs(cf_vals, f0=freqs[0], df=df),
        cf_ul=_make_fs(cf_ul_vals, f0=freqs[0], df=df),
        psd_witness_inj=_make_fs(psd_inj, f0=freqs[0], df=df),
        psd_witness_bkg=_make_fs(psd_bkg, f0=freqs[0], df=df),
        psd_target_inj=_make_fs(psd_inj * 1.1, f0=freqs[0], df=df),
        psd_target_bkg=_make_fs(psd_bkg * 1.1, f0=freqs[0], df=df),
        valid_mask=np.ones(N_FREQS, dtype=bool),
        witness_name="WIT",
        target_name="TGT",
        fftlength=fftlength,
        ts_witness_inj=ts_witness_inj,
        ts_witness_bkg=ts_witness_bkg,
    )


def _make_response_result(rng: np.random.Generator) -> ResponseFunctionResult:
    """可視化テスト用の ResponseFunctionResult を構成する。"""
    freq_axis = np.linspace(1.0, float(N_FREQS), N_FREQS)
    step_times = np.linspace(1000.0, 1000.0 + N_STEPS - 1, N_STEPS)
    injected_freqs = np.array([5.0, 7.0, 11.0, 17.0])
    coupling_factors = rng.uniform(0.05, 0.5, N_STEPS)

    data_inj = rng.uniform(1e-10, 1e-8, (N_STEPS, N_FREQS))
    data_bkg = rng.uniform(1e-11, 1e-9, (N_STEPS, N_FREQS))

    sg_inj = Spectrogram(
        data_inj,
        times=step_times,
        frequencies=freq_axis,
        unit="1/Hz**0.5",
    )
    sg_bkg = Spectrogram(
        data_bkg,
        times=step_times,
        frequencies=freq_axis,
        unit="1/Hz**0.5",
    )

    return ResponseFunctionResult(
        spectrogram_inj=sg_inj,
        spectrogram_bkg=sg_bkg,
        injected_freqs=injected_freqs,
        step_times=step_times,
        coupling_factors=coupling_factors,
        witness_name="WIT",
        target_name="TGT",
    )


def _find_horizontal_lines(ax: plt.Axes, y_value: float) -> list[plt.Line2D]:
    """指定 y 値の水平線候補を抽出する。"""
    matches: list[plt.Line2D] = []
    for line in ax.get_lines():
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size >= 2 and np.allclose(ydata, y_value):
            matches.append(line)
    return matches


def test_coupling_result_plot_significance_has_threshold_line():
    """threshold > 0 のとき水平閾値線が描画され、x 軸は log になる。"""
    rng = np.random.default_rng(0)
    result = _make_coupling_result(rng)
    threshold = 3.0

    fig = result.plot_significance(threshold=threshold)

    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert len(_find_horizontal_lines(ax, threshold)) == 1

    sig_expected = (
        np.sqrt(np.abs(result.psd_witness_inj.value))
        - np.sqrt(np.abs(result.psd_witness_bkg.value))
    ) / np.sqrt(np.abs(result.psd_witness_bkg.value))
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), sig_expected)
    plt.close(fig)


def test_coupling_result_plot_significance_without_threshold_line():
    """threshold <= 0 のとき水平閾値線が追加されない。"""
    rng = np.random.default_rng(1)
    result = _make_coupling_result(rng)

    fig = result.plot_significance(threshold=0.0)

    ax = fig.axes[0]
    assert len(ax.get_lines()) == 1
    assert _find_horizontal_lines(ax, 0.0) == []
    plt.close(fig)


def test_coupling_result_plot_significance_respects_frequency_limits():
    """plot_significance() が freq_min / freq_max を x 軸範囲に反映する。"""
    rng = np.random.default_rng(2)
    result = _make_coupling_result(rng)

    fig = result.plot_significance(freq_min=3.0, freq_max=9.0)

    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(9.0)
    plt.close(fig)


def test_coupling_result_plot_asdgram_requires_ts_witness_inj():
    """plot_asdgram() は ts_witness_inj がないと ValueError を送出する。"""
    rng = np.random.default_rng(3)
    result = _make_coupling_result(
        rng,
        ts_witness_inj=None,
        ts_witness_bkg=_make_timeseries(rng, 0.05),
    )

    with pytest.raises(ValueError, match="ts_witness_inj is required"):
        result.plot_asdgram()


def test_coupling_result_plot_asdgram_requires_ts_witness_bkg():
    """plot_asdgram() は ts_witness_bkg がないと ValueError を送出する。"""
    rng = np.random.default_rng(4)
    result = _make_coupling_result(
        rng,
        ts_witness_inj=_make_timeseries(rng, 0.2),
        ts_witness_bkg=None,
    )

    with pytest.raises(ValueError, match="ts_witness_bkg is required"):
        result.plot_asdgram()


def test_coupling_result_plot_asdgram_requires_fftlength():
    """plot_asdgram() は fftlength がないと ValueError を送出する。"""
    rng = np.random.default_rng(5)
    result = _make_coupling_result(
        rng,
        ts_witness_inj=_make_timeseries(rng, 0.2),
        ts_witness_bkg=_make_timeseries(rng, 0.05),
        fftlength=None,
    )

    with pytest.raises(ValueError, match="fftlength is required"):
        result.plot_asdgram()


def test_coupling_result_plot_asdgram_layout_and_percentile_overlays():
    """plot_asdgram() は 2 列レイアウトと twiny パーセンタイル線を作る。"""
    rng = np.random.default_rng(6)
    result = _make_coupling_result(
        rng,
        ts_witness_inj=_make_timeseries(rng, 0.3),
        ts_witness_bkg=_make_timeseries(rng, 0.1),
    )

    fig = result.plot_asdgram(freq_min=2.0, freq_max=40.0)

    main_axes = [
        ax
        for ax in fig.axes
        if ax.get_xlabel() == "Time [s]" and ax.get_title() in {"Injection: WIT", "Background: WIT"}
    ]
    twin_axes = [
        ax for ax in fig.axes if ax.get_xlabel().startswith("ASD [") and len(ax.get_lines()) == 3
    ]
    assert len(main_axes) == 2
    assert len(twin_axes) == 2

    for ax in main_axes:
        assert ax.get_yscale() == "log"
        lo, hi = sorted(ax.get_ylim())
        assert 1.5 <= lo <= 2.5
        assert 35.0 <= hi <= 40.5

    for ax_twin in twin_axes:
        assert ax_twin.get_xscale() == "log"
        assert len(ax_twin.get_lines()) == 3

    plt.close(fig)


def test_coupling_result_plot_snrgram_requires_ts_witness_inj():
    """plot_snrgram() は ts_witness_inj がないと ValueError を送出する。"""
    rng = np.random.default_rng(7)
    result = _make_coupling_result(
        rng,
        ts_witness_inj=None,
        ts_witness_bkg=_make_timeseries(rng, 0.05),
    )

    with pytest.raises(ValueError, match="ts_witness_inj is required"):
        result.plot_snrgram()


def test_coupling_result_plot_snrgram_requires_ts_witness_bkg():
    """plot_snrgram() は ts_witness_bkg がないと ValueError を送出する。"""
    rng = np.random.default_rng(8)
    result = _make_coupling_result(
        rng,
        ts_witness_inj=_make_timeseries(rng, 0.3),
        ts_witness_bkg=None,
    )

    with pytest.raises(ValueError, match="ts_witness_bkg is required"):
        result.plot_snrgram()


def test_coupling_result_plot_snrgram_requires_fftlength():
    """plot_snrgram() は fftlength がないと ValueError を送出する。"""
    rng = np.random.default_rng(9)
    result = _make_coupling_result(
        rng,
        ts_witness_inj=_make_timeseries(rng, 0.3),
        ts_witness_bkg=_make_timeseries(rng, 0.1),
        fftlength=None,
    )

    with pytest.raises(ValueError, match="fftlength is required"):
        result.plot_snrgram()


def test_coupling_result_plot_snrgram_clips_to_snrmax():
    """plot_snrgram() の pcolormesh 値が [-snrmax, snrmax] に収まる。"""
    rng = np.random.default_rng(10)
    result = _make_coupling_result(
        rng,
        ts_witness_inj=_make_timeseries(rng, 0.8, seed_offset=0.3),
        ts_witness_bkg=_make_timeseries(rng, 0.01, seed_offset=0.0),
    )
    snrmax = 2.5

    fig = result.plot_snrgram(freq_min=2.0, freq_max=50.0, snrmax=snrmax)

    ax = fig.axes[0]
    mesh = next(
        collection for collection in ax.collections if isinstance(collection, QuadMesh)
    )
    mesh_data = np.asarray(mesh.get_array(), dtype=float)
    assert np.nanmin(mesh_data) >= -snrmax - 1e-9
    assert np.nanmax(mesh_data) <= snrmax + 1e-9
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_response_function_plot_projection_overlay_count_and_colors():
    """plot_projection_summary() はステップ数分の線と tab10 色を持つ。"""
    rng = np.random.default_rng(11)
    result = _make_response_result(rng)

    fig = result.plot_projection_summary(freq_min=2.0, freq_max=10.0)

    ax = fig.axes[0]
    lines = [line for line in ax.get_lines() if len(line.get_xdata()) > 1]
    assert len(lines) == N_STEPS
    expected_colors = [tuple(plt.get_cmap("tab10")(i % 10)) for i in range(N_STEPS)]
    actual_colors = [tuple(line.get_color()) for line in lines]
    assert actual_colors == expected_colors
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(10.0)
    plt.close(fig)


def test_response_function_plot_response_matrix_has_expected_layout():
    """plot_response_matrix() は 3 パネル構成で main 軸に QuadMesh を持つ。"""
    rng = np.random.default_rng(12)
    result = _make_response_result(rng)

    fig = result.plot_response_matrix(freq_min=2.0, freq_max=10.0)

    assert len(fig.axes) == 4
    ax_main = fig.axes[0]
    meshes = [collection for collection in ax_main.collections if isinstance(collection, QuadMesh)]
    assert len(meshes) == 1

    mesh_array = np.asarray(meshes[0].get_array()).reshape(-1)
    freq_mask = (result.spectrogram_inj.frequencies.value >= 2.0) & (
        result.spectrogram_inj.frequencies.value <= 10.0
    )
    n_freqs = int(np.count_nonzero(freq_mask))
    assert mesh_array.size == N_STEPS * n_freqs
    plt.close(fig)
