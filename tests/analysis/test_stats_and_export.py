"""
Phase 2 統計レポート機能テスト

- test_spectral_stats_significance_calculation
- test_coupling_result_to_csv_roundtrip
- test_coupling_result_to_txt_roundtrip
- test_coupling_result_collection_summary_csv
"""
from __future__ import annotations

import numpy as np
import pytest

from gwexpy.analysis.coupling_result import CouplingResult, CouplingResultCollection
from gwexpy.analysis.stats import SpectralStats
from gwexpy.frequencyseries import FrequencySeries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 100.0
FFTLENGTH = 1.0
N_FREQS = 50


def _make_fs(vals: np.ndarray, f0: float = 0.0, df: float = 1.0) -> FrequencySeries:
    return FrequencySeries(vals, f0=f0, df=df)


def _make_coupling_result(rng: np.random.Generator) -> CouplingResult:
    """テスト用の CouplingResult を生成する。"""
    freqs = np.linspace(1.0, 50.0, N_FREQS)
    df = freqs[1] - freqs[0]

    cf_vals = rng.uniform(0.01, 1.0, N_FREQS)
    cf_ul_vals = cf_vals * rng.uniform(1.1, 2.0, N_FREQS)
    psd_inj = rng.uniform(1e-10, 1e-8, N_FREQS)
    psd_bkg = rng.uniform(1e-11, 1e-9, N_FREQS)

    return CouplingResult(
        cf=_make_fs(cf_vals, f0=freqs[0], df=df),
        cf_ul=_make_fs(cf_ul_vals, f0=freqs[0], df=df),
        psd_witness_inj=_make_fs(psd_inj, f0=freqs[0], df=df),
        psd_witness_bkg=_make_fs(psd_bkg, f0=freqs[0], df=df),
        psd_target_inj=_make_fs(psd_inj, f0=freqs[0], df=df),
        psd_target_bkg=_make_fs(psd_bkg, f0=freqs[0], df=df),
        valid_mask=np.ones(N_FREQS, dtype=bool),
        witness_name="WIT",
        target_name="TGT",
        fftlength=FFTLENGTH,
    )


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------


def test_spectral_stats_significance_calculation():
    """SpectralStats.significance() が (μ_inj - μ_bkg) / σ_bkg を正しく計算する。"""
    rng = np.random.default_rng(42)
    n = 20
    freqs = np.linspace(1.0, 20.0, n)
    df = freqs[1] - freqs[0]

    mean_vals = rng.uniform(1.0, 2.0, n)
    sigma_vals = rng.uniform(0.1, 0.5, n)
    mu_inj_vals = mean_vals + sigma_vals * 3.0  # SNR = 3 になるように設定

    stats = SpectralStats(
        mean=_make_fs(mean_vals, f0=freqs[0], df=df),
        sigma=_make_fs(sigma_vals, f0=freqs[0], df=df),
        n_avg=8,
    )
    mu_inj = _make_fs(mu_inj_vals, f0=freqs[0], df=df)
    sig = stats.significance(mu_inj)

    # 各点で SNR ≈ 3 であることを確認
    np.testing.assert_allclose(sig.value, 3.0, rtol=1e-10)


def test_coupling_result_to_csv_roundtrip(tmp_path):
    """to_csv() → from_csv() で周波数・CF・ASD 値が一致する。"""
    rng = np.random.default_rng(7)
    result = _make_coupling_result(rng)

    csv_path = tmp_path / "coupling.csv"
    result.to_csv(csv_path)

    restored = CouplingResult.from_csv(csv_path)

    np.testing.assert_allclose(
        restored.cf.value,
        result.cf.value,
        rtol=1e-10,
        err_msg="CF values must survive CSV roundtrip",
    )
    np.testing.assert_allclose(
        restored.cf.xindex.value,
        result.cf.xindex.value,
        rtol=1e-10,
        err_msg="Frequency axis must survive CSV roundtrip",
    )
    # cf_ul が存在する場合は一致すること
    assert restored.cf_ul is not None
    np.testing.assert_allclose(
        restored.cf_ul.value,
        result.cf_ul.value,
        rtol=1e-10,
        err_msg="CF upper limit must survive CSV roundtrip",
    )


def test_coupling_result_to_txt_roundtrip(tmp_path):
    """to_txt() → from_txt() で CF・cf_ul 値が一致する。"""
    rng = np.random.default_rng(13)
    result = _make_coupling_result(rng)

    txt_path = tmp_path / "coupling.txt"
    result.to_txt(txt_path)

    restored = CouplingResult.from_txt(txt_path)

    np.testing.assert_allclose(
        restored.cf.value,
        result.cf.value,
        rtol=1e-10,
        err_msg="CF values must survive TXT roundtrip",
    )
    # cf_ul = cf + uncertainty として復元される
    assert restored.cf_ul is not None
    expected_ul = result.cf_ul.value  # 元の cf_ul と一致するはず
    np.testing.assert_allclose(
        restored.cf_ul.value,
        expected_ul,
        rtol=1e-10,
        err_msg="CF upper limit must survive TXT roundtrip",
    )


def test_coupling_result_collection_summary_csv(tmp_path):
    """CouplingResultCollection.to_summary_csv() が全ペア分の行を書き出す。"""
    rng = np.random.default_rng(99)
    r1 = _make_coupling_result(rng)
    r2 = _make_coupling_result(rng)

    col = CouplingResultCollection({"WIT-TGT1": r1, "WIT-TGT2": r2})
    csv_path = tmp_path / "summary.csv"
    col.to_summary_csv(csv_path)

    import csv as _csv

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    # 各結果が N_FREQS 行ずつ出力されること
    assert len(rows) == N_FREQS * 2, f"Expected {N_FREQS * 2} rows, got {len(rows)}"
    pairs = {r["channel_pair"] for r in rows}
    assert pairs == {"WIT-TGT1", "WIT-TGT2"}
