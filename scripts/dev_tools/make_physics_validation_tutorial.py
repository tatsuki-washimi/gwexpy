"""Generate physical validity checking tutorial notebooks (EN + JA)."""

import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "id": f"md_{abs(hash(source))%10**8:08x}",
            "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None,
            "id": f"cd_{abs(hash(source))%10**8:08x}",
            "metadata": {}, "outputs": [], "source": source}


EN_CELLS = [
    md("""\
# Physical Validity Checking: Units, Floors, and Sanity Tests

One of gwexpy's design principles is **numerical safety for GW-scale
quantities**.  Gravitational-wave strain is ~10⁻²³ — close to the limits
of double-precision floating point — so naive computations can silently
produce wrong answers.

gwexpy provides `gwexpy.numerics.constants` and `gwexpy.numerics.scaling`
to handle:
- **Numerical floors** that prevent log(0) and division-by-zero
- **Dtype-aware epsilons** that adapt to float32 / float64
- **Safe log-scale visualisation** with a physical dynamic range

This tutorial demonstrates these tools through a series of **validation
exercises** that every GW analyst should perform on their data pipelines.

**What this tutorial covers:**
1. Why strain-scale quantities need special numerical care
2. Using `SAFE_FLOOR`, `EPS_PSD`, and `EPS_COHERENCE`
3. Unit-consistent ASD / PSD operations with `astropy.units`
4. Safe log-scale plotting with `safe_log_scale`
5. A checklist of physics sanity tests for an ASD pipeline
"""),

    md("## Setup"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.numerics.constants import (
    SAFE_FLOOR, SAFE_FLOOR_STRAIN,
    EPS_PSD, EPS_VARIANCE, EPS_COHERENCE,
    eps_for_dtype,
)
from gwexpy.numerics.scaling import safe_log_scale, safe_epsilon

print("Numerical floors:")
print(f"  SAFE_FLOOR        = {SAFE_FLOOR:.2e}  (absolute floor for log ops)")
print(f"  EPS_PSD           = {EPS_PSD:.2e}  (Welch estimator regularisation)")
print(f"  EPS_VARIANCE      = {EPS_VARIANCE:.2e}  (variance / power floor)")
print(f"  EPS_COHERENCE     = {EPS_COHERENCE:.2e}  (coherence denominator clip)")
print()
print(f"Machine epsilon (float64): {eps_for_dtype(np.float64):.2e}")
print(f"Machine epsilon (float32): {eps_for_dtype(np.float32):.2e}")
"""),

    md("""\
## 1. Why GW Strain Needs Special Care

Double-precision floating point has a machine epsilon of ~2.2×10⁻¹⁶.
GW strain is ~10⁻²³ — only 7 orders of magnitude above machine epsilon.
This means that **squaring strain to compute PSD, then taking log for
plotting, can lose significant bits without careful flooring**.
"""),

    code("""\
# Demonstrate the problem: strain-scale PSD near machine precision
h_typical = 1e-23   # typical strain amplitude
h_tiny    = 1e-30   # below any physical source

psd_typical = h_typical**2   # 1e-46
psd_tiny    = h_tiny**2      # 1e-60

print(f"Typical PSD : {psd_typical:.2e} Hz⁻¹")
print(f"Tiny PSD    : {psd_tiny:.2e} Hz⁻¹")
print(f"SAFE_FLOOR  : {SAFE_FLOOR:.2e}")
print()

# Safe log: floor first, then log
def safe_log10(x, floor=SAFE_FLOOR):
    return np.log10(np.maximum(x, floor))

val_typical = safe_log10(psd_typical)
val_tiny    = safe_log10(psd_tiny)
val_zero    = safe_log10(0.0)    # would be -inf without floor

print(f"safe_log10(psd_typical) = {val_typical:.2f}")
print(f"safe_log10(psd_tiny)    = {val_tiny:.2f}  "
      f"(floored to {safe_log10(SAFE_FLOOR):.1f} dB)")
print(f"safe_log10(0.0)         = {val_zero:.2f}  (not -inf!)")
"""),

    md("""\
## 2. Numerical Floors in ASD Computation

gwexpy's Welch estimator uses `EPS_PSD` internally to prevent zero PSD
values.  Here we show explicitly what happens without vs. with flooring.
"""),

    code("""\
fs   = 4096.0
T    = 16.0
N    = int(T * fs)
rng  = np.random.default_rng(1)

# Create strain-scale data
freqs_n = np.fft.rfftfreq(N, 1.0/fs)[1:]
asd_model = 1e-23 * np.where(freqs_n < 100, (100/freqs_n)**2, 1.0)
fft = asd_model * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(freqs_n)))
noise = np.fft.irfft(np.concatenate([[0.0], fft]), n=N)

ts = TimeSeries(noise, t0=1_300_000_000, sample_rate=fs,
                name="K1:LSC-DARM_STRAIN", unit="strain")

# Compute ASD
asd = ts.asd(fftlength=4.0, method="median")
freqs = asd.frequencies.value

print(f"ASD range: [{asd.value.min():.2e}, {asd.value.max():.2e}] strain/rtHz")
print(f"Any zeros in ASD? {np.any(asd.value == 0)}")
print(f"Any NaN in ASD?   {np.any(np.isnan(asd.value))}")

# Safe ASD magnitude for log plot
asd_safe = np.maximum(asd.value, SAFE_FLOOR_STRAIN)

fig, ax = plt.subplots(figsize=(9, 4))
ax.loglog(freqs[1:], asd.value[1:], color="steelblue", lw=1.5, label="ASD")
ax.loglog(freqs[1:], asd_model[:(len(freqs)-1)], "--", color="tomato",
          lw=1.5, label="True ASD model")
ax.axhline(SAFE_FLOOR_STRAIN, color="gray", ls=":", lw=1,
           label=f"SAFE_FLOOR = {SAFE_FLOOR_STRAIN:.0e}")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [strain/√Hz]")
ax.set_title("Strain-scale ASD with numerical floor")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 3. Unit-Consistent Operations with astropy.units

gwexpy uses `astropy.units` throughout.  Physical validation means
checking that unit arithmetic is self-consistent: ASD² = PSD, etc.
"""),

    code("""\
# Build unit-aware FrequencySeries
asd_u = FrequencySeries(
    asd.value * (u.strain / u.Hz**0.5),
    frequencies=freqs,
    name="DARM ASD",
)

# PSD = ASD^2
psd_u = asd_u**2
print(f"ASD unit : {asd_u.unit}")
print(f"PSD unit : {psd_u.unit}")

# Integrated power = integral of PSD over frequency = variance
df = freqs[1] - freqs[0]
integrated_power = (psd_u.value * df).sum()
rms_from_psd     = np.sqrt(integrated_power)
rms_direct       = ts.std().value

print(f"\\nRMS from PSD  : {rms_from_psd:.4e} strain")
print(f"RMS from data  : {rms_direct:.4e} strain")
print(f"Relative error : {abs(rms_from_psd - rms_direct)/rms_direct * 100:.2f}%")
assert abs(rms_from_psd - rms_direct) / rms_direct < 0.05, \
    "PSD-derived RMS should agree with direct RMS to within 5%"
print("PASS: PSD integral is consistent with direct RMS")
"""),

    md("""\
## 4. Safe Log-Scale Plotting

`safe_log_scale()` converts an array to decibels with a configurable
dynamic range, preventing -inf from appearing in plots.
"""),

    code("""\
# Compute coherence (some bins will be near zero)
ts2 = TimeSeries(noise * 0.6 + rng.normal(0, 1e-24, N),
                 t0=ts.t0.value, sample_rate=fs, name="witness")
coh = ts.coherence(ts2, fftlength=4.0)

# Raw log transform (problem: near-zero coherence → -inf)
with np.errstate(divide="ignore", invalid="ignore"):
    log_raw = 10 * np.log10(np.maximum(coh.value, 0))

# Safe version using gwexpy
log_safe = safe_log_scale(coh.value, dynamic_range_db=60.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax1.semilogx(coh.frequencies.value[1:], log_raw[1:],
             color="tomato", lw=1.2, alpha=0.8, label="raw 10·log₁₀")
ax1.set_ylabel("Coherence [dB]")
ax1.set_title("Coherence in dB — raw vs. safe")
ax1.set_ylim(-200, 5)
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)

ax2.semilogx(coh.frequencies.value[1:], log_safe[1:],
             color="steelblue", lw=1.2, label="safe_log_scale (60 dB range)")
ax2.set_ylabel("Coherence [dB]")
ax2.set_xlabel("Frequency [Hz]")
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
print(f"safe_log_scale range: [{log_safe.min():.1f}, {log_safe.max():.1f}] dB")
"""),

    md("""\
## 5. Physics Sanity Checklist

A systematic set of tests that every ASD pipeline should pass before
results are trusted or shared.
"""),

    code("""\
def check_asd_pipeline(ts_data: TimeSeries, label: str = "ASD") -> dict:
    results = {}

    asd_check = ts_data.asd(fftlength=4.0, method="median")
    freqs_check = asd_check.frequencies.value
    vals = asd_check.value

    # --- Test 1: No NaN or Inf ---
    no_nan = not np.any(np.isnan(vals))
    no_inf = not np.any(np.isinf(vals))
    results["no_nan"]  = ("PASS" if no_nan  else "FAIL", no_nan)
    results["no_inf"]  = ("PASS" if no_inf  else "FAIL", no_inf)

    # --- Test 2: All values >= SAFE_FLOOR ---
    above_floor = np.all(vals >= SAFE_FLOOR_STRAIN)
    results["above_floor"] = ("PASS" if above_floor else "FAIL", above_floor)

    # --- Test 3: PSD integral gives finite RMS ---
    df_check = freqs_check[1] - freqs_check[0]
    rms_psd = np.sqrt((vals**2 * df_check).sum())
    finite_rms = np.isfinite(rms_psd) and rms_psd > 0
    results["finite_rms"] = ("PASS" if finite_rms else "FAIL", finite_rms)

    # --- Test 4: RMS from PSD ≈ RMS from data (within 10%) ---
    rms_data = ts_data.std().value
    rms_agree = abs(rms_psd - rms_data) / (rms_data + SAFE_FLOOR) < 0.10
    results["rms_agree"] = ("PASS" if rms_agree else "FAIL", rms_agree)

    # --- Test 5: ASD is monotonically plausible (no > 10x jump bin-to-bin) ---
    ratio = vals[1:] / np.maximum(vals[:-1], SAFE_FLOOR_STRAIN)
    no_spike = np.all((ratio < 20.0) & (ratio > 0.05))
    results["no_spike"] = ("PASS" if no_spike else "FAIL", no_spike)

    # --- Test 6: Nyquist bin is not dominant (no aliasing) ---
    nyq_idx = len(vals) - 1
    peak_idx = np.argmax(vals)
    no_nyq_alias = peak_idx != nyq_idx
    results["no_nyq_alias"] = ("PASS" if no_nyq_alias else "FAIL", no_nyq_alias)

    # Summary
    n_pass = sum(1 for v in results.values() if v[1])
    print(f"\\n=== Physics Sanity Checklist: {label} ===")
    for test, (status, _) in results.items():
        print(f"  [{status}] {test}")
    print(f"\\n  Score: {n_pass}/{len(results)}")

    return results

res = check_asd_pipeline(ts, label="K1:LSC-DARM_STRAIN")

# Demonstrate a failing case: ASD computed from near-zero data
ts_bad = TimeSeries(np.zeros(N) + 1e-310, t0=ts.t0.value,
                    sample_rate=fs, name="near-zero data", unit="strain")
res_bad = check_asd_pipeline(ts_bad, label="near-zero data (expected failures)")
"""),

    md("""\
## Summary

| Tool | Purpose | When to use |
|------|---------|-------------|
| `SAFE_FLOOR` | Absolute floor (1e-50) for log ops | Before any `log10(x)` call |
| `EPS_PSD` | Regularises Welch PSD denominator | Internal to `ts.asd()` |
| `EPS_COHERENCE` | Clips coherence denominator | Internal to coherence estimation |
| `eps_for_dtype(dtype)` | Machine epsilon for the working dtype | When mixing float32/float64 |
| `safe_log_scale(x, db)` | dB conversion with bounded dynamic range | All dB / log plots |
| `safe_epsilon(x)` | Data-relative epsilon | Adaptive thresholding |

**Physics sanity checklist for ASD pipelines:**
- [ ] No NaN or Inf in the output
- [ ] All bins ≥ `SAFE_FLOOR_STRAIN` (1e-50 strain/√Hz)
- [ ] PSD integral gives finite, positive RMS
- [ ] PSD-derived RMS ≈ data RMS within 10%
- [ ] No >20x amplitude jump between adjacent bins
- [ ] Nyquist bin is not the dominant peak (no aliasing)
"""),
]


# ---------------------------------------------------------------------------
# Japanese notebook
# ---------------------------------------------------------------------------

JA_CELLS = [
    md("""\
# 物理妥当性検証：単位・数値床・健全性テスト

gwexpy の設計原則のひとつは **GW スケール量に対する数値的安全性** です。
重力波歪みは ~10⁻²³ — 倍精度浮動小数点のマシンイプシロン（~2.2×10⁻¹⁶）の
わずか 7 桁上にあります。つまり、歪みを二乗して PSD を計算し、
ログ変換してプロットする操作が、適切なフロア処理なしには
**サイレントに誤った結果を生じさせる**可能性があります。

gwexpy の `gwexpy.numerics.constants` と `gwexpy.numerics.scaling` は
以下を提供します：
- **数値床**：log(0) とゼロ除算を防ぐ
- **dtype 対応イプシロン**：float32 / float64 に適応
- **安全な対数スケール可視化**：物理的な動的範囲で制限

**このチュートリアルで学ぶこと：**
1. 歪みスケール量が特別な数値的注意を必要とする理由
2. `SAFE_FLOOR`、`EPS_PSD`、`EPS_COHERENCE` の使い方
3. `astropy.units` による単位一貫性の確認
4. `safe_log_scale` による安全な対数スケールプロット
5. ASD パイプラインの物理健全性テストチェックリスト
"""),

    md("## セットアップ"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.numerics.constants import (
    SAFE_FLOOR, SAFE_FLOOR_STRAIN,
    EPS_PSD, EPS_VARIANCE, EPS_COHERENCE,
    eps_for_dtype,
)
from gwexpy.numerics.scaling import safe_log_scale, safe_epsilon

print("数値床の値:")
print(f"  SAFE_FLOOR        = {SAFE_FLOOR:.2e}  （対数演算の絶対フロア）")
print(f"  EPS_PSD           = {EPS_PSD:.2e}  （Welch 推定量の正則化）")
print(f"  EPS_VARIANCE      = {EPS_VARIANCE:.2e}  （分散・パワーのフロア）")
print(f"  EPS_COHERENCE     = {EPS_COHERENCE:.2e}  （コヒーレンス分母クリッピング）")
print()
print(f"マシンイプシロン (float64): {eps_for_dtype(np.float64):.2e}")
print(f"マシンイプシロン (float32): {eps_for_dtype(np.float32):.2e}")
"""),

    md("""\
## 1. GW 歪みに特別な数値的注意が必要な理由

倍精度浮動小数点のマシンイプシロンは ~2.2×10⁻¹⁶ です。
GW 歪みは ~10⁻²³ — マシンイプシロンの 7 桁上。
これはつまり、**歪みを二乗して PSD を計算し、ログ変換する操作が
適切なフロアなしに大きなビット損失を引き起こす可能性がある**ことを意味します。
"""),

    code("""\
h_typical = 1e-23
h_tiny    = 1e-30

psd_typical = h_typical**2
psd_tiny    = h_tiny**2

print(f"典型的な PSD : {psd_typical:.2e} Hz⁻¹")
print(f"微小な PSD   : {psd_tiny:.2e} Hz⁻¹")
print(f"SAFE_FLOOR   : {SAFE_FLOOR:.2e}")

def safe_log10(x, floor=SAFE_FLOOR):
    return np.log10(np.maximum(x, floor))

print(f"\\nsafe_log10(psd_typical) = {safe_log10(psd_typical):.2f}")
print(f"safe_log10(psd_tiny)    = {safe_log10(psd_tiny):.2f}  "
      f"（{safe_log10(SAFE_FLOOR):.1f} dB にフロア）")
print(f"safe_log10(0.0)         = {safe_log10(0.0):.2f}  （-inf ではない！）")
"""),

    md("## 2. ASD 計算における数値床"),

    code("""\
fs   = 4096.0
T    = 16.0
N    = int(T * fs)
rng  = np.random.default_rng(1)

freqs_n = np.fft.rfftfreq(N, 1.0/fs)[1:]
asd_model = 1e-23 * np.where(freqs_n < 100, (100/freqs_n)**2, 1.0)
fft = asd_model * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(freqs_n)))
noise = np.fft.irfft(np.concatenate([[0.0], fft]), n=N)

ts = TimeSeries(noise, t0=1_300_000_000, sample_rate=fs,
                name="K1:LSC-DARM_STRAIN", unit="strain")

asd = ts.asd(fftlength=4.0, method="median")
freqs = asd.frequencies.value

print(f"ASD 範囲: [{asd.value.min():.2e}, {asd.value.max():.2e}] strain/rtHz")
print(f"ASD に NaN はあるか: {np.any(np.isnan(asd.value))}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.loglog(freqs[1:], asd.value[1:], color="steelblue", lw=1.5, label="ASD")
ax.loglog(freqs[1:], asd_model[:(len(freqs)-1)], "--", color="tomato",
          lw=1.5, label="真の ASD モデル")
ax.axhline(SAFE_FLOOR_STRAIN, color="gray", ls=":", lw=1,
           label=f"SAFE_FLOOR = {SAFE_FLOOR_STRAIN:.0e}")
ax.set_xlabel("周波数 [Hz]")
ax.set_ylabel("ASD [strain/√Hz]")
ax.set_title("数値床付き歪みスケール ASD")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("## 3. astropy.units による単位一貫性の確認"),

    code("""\
asd_u = FrequencySeries(
    asd.value * (u.strain / u.Hz**0.5),
    frequencies=freqs,
    name="DARM ASD",
)

psd_u = asd_u**2
print(f"ASD 単位: {asd_u.unit}")
print(f"PSD 単位: {psd_u.unit}")

df = freqs[1] - freqs[0]
integrated_power = (psd_u.value * df).sum()
rms_from_psd     = np.sqrt(integrated_power)
rms_direct       = ts.std().value

print(f"\\nPSD から RMS  : {rms_from_psd:.4e} strain")
print(f"データから RMS : {rms_direct:.4e} strain")
print(f"相対誤差       : {abs(rms_from_psd - rms_direct)/rms_direct * 100:.2f}%")
assert abs(rms_from_psd - rms_direct) / rms_direct < 0.05, \
    "PSD 積分 RMS はデータ直接 RMS と 5% 以内で一致すべき"
print("PASS: PSD 積分はデータ RMS と一貫しています")
"""),

    md("## 4. safe_log_scale による安全な対数スケールプロット"),

    code("""\
ts2 = TimeSeries(noise * 0.6 + rng.normal(0, 1e-24, N),
                 t0=ts.t0.value, sample_rate=fs, name="witness")
coh = ts.coherence(ts2, fftlength=4.0)

with np.errstate(divide="ignore", invalid="ignore"):
    log_raw = 10 * np.log10(np.maximum(coh.value, 0))

log_safe = safe_log_scale(coh.value, dynamic_range_db=60.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.semilogx(coh.frequencies.value[1:], log_raw[1:],
             color="tomato", lw=1.2, alpha=0.8, label="生の 10·log₁₀")
ax1.set_ylabel("コヒーレンス [dB]")
ax1.set_title("dB 単位のコヒーレンス — 生 vs. 安全")
ax1.set_ylim(-200, 5)
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)

ax2.semilogx(coh.frequencies.value[1:], log_safe[1:],
             color="steelblue", lw=1.2, label="safe_log_scale（60 dB 範囲）")
ax2.set_ylabel("コヒーレンス [dB]")
ax2.set_xlabel("周波数 [Hz]")
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
print(f"safe_log_scale 範囲: [{log_safe.min():.1f}, {log_safe.max():.1f}] dB")
"""),

    md("## 5. 物理健全性チェックリスト"),

    code("""\
def check_asd_pipeline(ts_data: TimeSeries, label: str = "ASD") -> dict:
    results = {}
    asd_check = ts_data.asd(fftlength=4.0, method="median")
    vals = asd_check.value
    freqs_check = asd_check.frequencies.value

    no_nan = not np.any(np.isnan(vals))
    no_inf = not np.any(np.isinf(vals))
    results["NaN なし"]     = ("PASS" if no_nan  else "FAIL", no_nan)
    results["Inf なし"]     = ("PASS" if no_inf  else "FAIL", no_inf)

    above_floor = np.all(vals >= SAFE_FLOOR_STRAIN)
    results["SAFE_FLOOR 以上"] = ("PASS" if above_floor else "FAIL", above_floor)

    df_check = freqs_check[1] - freqs_check[0]
    rms_psd  = np.sqrt((vals**2 * df_check).sum())
    finite_rms = np.isfinite(rms_psd) and rms_psd > 0
    results["有限な RMS"] = ("PASS" if finite_rms else "FAIL", finite_rms)

    rms_data  = ts_data.std().value
    rms_agree = abs(rms_psd - rms_data) / (rms_data + SAFE_FLOOR) < 0.10
    results["RMS 一致（10% 以内）"] = ("PASS" if rms_agree else "FAIL", rms_agree)

    ratio    = vals[1:] / np.maximum(vals[:-1], SAFE_FLOOR_STRAIN)
    no_spike = np.all((ratio < 20.0) & (ratio > 0.05))
    results["急激なスパイクなし"] = ("PASS" if no_spike else "FAIL", no_spike)

    nyq_idx    = len(vals) - 1
    no_nyq     = np.argmax(vals) != nyq_idx
    results["ナイキスト折り返しなし"] = ("PASS" if no_nyq else "FAIL", no_nyq)

    n_pass = sum(1 for v in results.values() if v[1])
    print(f"\\n=== 物理健全性チェックリスト: {label} ===")
    for test, (status, _) in results.items():
        print(f"  [{status}] {test}")
    print(f"\\n  スコア: {n_pass}/{len(results)}")
    return results

res = check_asd_pipeline(ts, label="K1:LSC-DARM_STRAIN")

ts_bad = TimeSeries(np.zeros(N) + 1e-310, t0=ts.t0.value,
                    sample_rate=fs, name="ゼロ近傍データ", unit="strain")
res_bad = check_asd_pipeline(ts_bad, label="ゼロ近傍データ（失敗が期待される）")
"""),

    md("""\
## まとめ

| ツール | 目的 | 使用場面 |
|-------|------|---------|
| `SAFE_FLOOR` | 絶対フロア（1e-50） | `log10(x)` 前 |
| `EPS_PSD` | Welch PSD 分母の正則化 | `ts.asd()` 内部 |
| `EPS_COHERENCE` | コヒーレンス分母のクリッピング | コヒーレンス推定内部 |
| `eps_for_dtype(dtype)` | 作業 dtype のマシンイプシロン | float32/float64 混合時 |
| `safe_log_scale(x, db)` | 動的範囲制限付き dB 変換 | 全 dB/ログプロット |
| `safe_epsilon(x)` | データ相対イプシロン | 適応的閾値処理 |

**ASD パイプラインの物理健全性チェックリスト：**
- [ ] 出力に NaN または Inf がない
- [ ] 全ビンが `SAFE_FLOOR_STRAIN`（1e-50 strain/√Hz）以上
- [ ] PSD 積分が有限・正の RMS を与える
- [ ] PSD 由来の RMS ≈ データ直接の RMS（10% 以内）
- [ ] 隣接ビン間に 20 倍超の振幅ジャンプがない
- [ ] ナイキストビンが支配的でない（折り返しなし）
"""),
]


def write_nb(cells, path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Written: {path}")


if __name__ == "__main__":
    root = Path(__file__).parents[2]
    write_nb(EN_CELLS, root / "docs/web/en/user_guide/tutorials/case_physics_validation.ipynb")
    write_nb(JA_CELLS, root / "docs/web/ja/user_guide/tutorials/case_physics_validation.ipynb")
    print("Done.")
