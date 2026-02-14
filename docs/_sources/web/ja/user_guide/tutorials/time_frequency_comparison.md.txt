# 時間-周波数解析手法: 比較と選択ガイド

:::{admonition} インタラクティブ版が利用可能です
:class: tip

**プロットと実行可能なコード付きのバージョンをお探しですか？**

この理論ガイドに対応する、完全に実行可能な Jupyter Notebook 版があります：
- [時間-周波数解析比較（インタラクティブ）](time_frequency_analysis_comparison.html)
- 8つの完全なプロット、定量評価、ベンチマーク信号

**このページ**: 各手法の理論、使い方ガイド、決定マトリックス
**ノートブック版**: 完全な実装例、出力埋め込み、コピペ可能なコード
:::

このチュートリアルでは、gwexpy のさまざまな時間-周波数解析手法を比較し、解析に適した手法を選択する方法を説明します。

## 概要

重力波信号は**時間変化する周波数内容**を持つことが多い。異なる手法は異なる側面を明らかにします:

| 手法 | 最適な用途 | 時間分解能 | 周波数分解能 |
|--------|----------|-----------------|---------------------|
| **スペクトログラム (STFT)** | 汎用 | 良好 | 良好 |
| **Q変換** | チャープ、過渡現象 | 適応的 | 適応的（定数Q） |
| **ウェーブレット (CWT)** | マルチスケール特徴、チャープ | スケール依存 | スケール依存 |
| **HHT** | 瞬時周波数 | 優秀 | データ適応的 |
| **STLT** | 減衰振動 | 良好 | 良好（+減衰率σ） |
| **ケプストラム** | エコー検出、周期性 | N/A | ケフレンシー領域 |
| **DCT** | 圧縮、平滑特徴 | N/A | N/A（基底係数） |

## セットアップ

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.timeseries import TimeSeries
from gwexpy.noise.wave import chirp, gaussian

# テスト信号の作成: チャープ + ノイズ
sample_rate = 1024  # Hz
duration = 4  # seconds

# 20 Hz から 100 Hz へのチャープ
signal_chirp = chirp(
    duration=duration,
    sample_rate=sample_rate,
    f0=20,  # 開始周波数
    f1=100,  # 終了周波数
    t1=duration
)

# ガウシアンノイズを追加
noise = gaussian(duration=duration, sample_rate=sample_rate, std=0.2)
data = signal_chirp + noise

ts = TimeSeries(data, t0=0, dt=1/sample_rate, unit='strain')
print(f"データ: {len(ts)} サンプル, {duration}秒")
```

## 手法 1: スペクトログラム（STFT ベース）

### 手法の説明

**短時間フーリエ変換（STFT）**: 信号を窓に分割し、各窓で FFT を計算

**公式**: `S(t, f) = |∫ x(τ) w(τ-t) e^(-2πifτ) dτ|²`

### 実装

```python
# 0.5秒窓でスペクトログラムを作成
spec = ts.spectrogram(fftlength=0.5, overlap=0.25)

print(f"スペクトログラムの形状: {spec.shape}")  # (time_bins, freq_bins)
print(f"時間分解能: {spec.dt}")
print(f"周波数分解能: {spec.df}")
```

### 可視化

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# スペクトログラム
im = axes[0].pcolormesh(
    spec.times.value,
    spec.frequencies.value,
    spec.value.T,
    cmap='viridis',
    shading='auto'
)
axes[0].set_ylim(10, 200)
axes[0].set_ylabel('周波数 (Hz)')
axes[0].set_title('スペクトログラム（STFT, 窓=0.5秒）')
fig.colorbar(im, ax=axes[0], label='パワー')

# 真のチャープ周波数を重ねてプロット
t_true = np.linspace(0, duration, 100)
f_true = 20 + (100 - 20) * t_true / duration
axes[0].plot(t_true, f_true, 'r--', linewidth=2, label='真の周波数')
axes[0].legend()

# 参照用の時系列
axes[1].plot(ts.times.value, ts.value, linewidth=0.5)
axes[1].set_xlabel('時間 (s)')
axes[1].set_ylabel('ひずみ')
axes[1].set_title('元の信号')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 長所と短所

✅ **利点:**
- 高速計算
- よく理解された標準手法
- 定常または緩やかに変化する信号に適している

❌ **欠点:**
- 固定された時間-周波数分解能（不確定性原理）
- 急速な周波数変化には不向き
- 窓長が時間と周波数の両方の分解能に影響

### いつ使用するか

✅ **スペクトログラムを使用する場合:**
- 信号が準定常的
- 標準的で確立された手法が必要
- 高速計算が重要
- 周波数変化が窓サイズに対して遅い

## 手法 2: Q変換

### 手法の説明

**定数Q変換**: 定数Q因子を持つ適応的な時間-周波数タイリング

**Q因子**: `Q = f / Δf`（中心周波数と帯域幅の比）

### 実装

```python
# Q=6 のQ変換
q = 6
qgram = ts.q_transform(qrange=(4, 64), frange=(10, 200))

print(f"Q変換の形状: {qgram.shape}")
```

### 可視化

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Q変換
axes[0].imshow(
    qgram.value.T,
    extent=[qgram.times.value[0], qgram.times.value[-1],
            qgram.frequencies.value[0], qgram.frequencies.value[-1]],
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='bilinear'
)
axes[0].set_ylabel('周波数 (Hz)')
axes[0].set_title(f'Q変換（定数Q）')
axes[0].plot(t_true, f_true, 'r--', linewidth=2, label='真の周波数')
axes[0].legend()

# 比較用のスペクトログラム
im = axes[1].pcolormesh(
    spec.times.value,
    spec.frequencies.value,
    spec.value.T,
    cmap='viridis',
    shading='auto'
)
axes[1].set_ylim(10, 200)
axes[1].set_xlabel('時間 (s)')
axes[1].set_ylabel('周波数 (Hz)')
axes[1].set_title('スペクトログラム（比較用）')
axes[1].plot(t_true, f_true, 'r--', linewidth=2, label='真の周波数')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 長所と短所

✅ **利点:**
- 適応的分解能: 高周波でより良い時間分解能
- チャープ（連星合体）に自然
- 定数Qは重力波信号にマッチ

❌ **欠点:**
- STFT より遅い
- より複雑な解釈
- Q因子の調整が必要

### いつ使用するか

✅ **Q変換を使用する場合:**
- チャープを解析（コンパクト連星合体）
- 信号が広い周波数範囲にまたがる
- 高周波過渡現象に良好な時間分解能が必要
- 標準的な重力波過渡現象解析

## 手法 3: ウェーブレット変換 (CWT)

### 手法の説明

**連続ウェーブレット変換 (CWT)**: 拡張・移動されたウェーブレットを用いたマルチスケール解析

**式**: `W(a, b) = ∫ x(t) ψ*((t-b)/a) dt`
- `a`: スケールパラメータ（周波数に反比例）
- `b`: 時間シフトパラメータ
- `ψ`: マザーウェーブレット（例：Morlet）

### 利点と欠点

✅ **利点:**
- 自然なスケールマッチング（ウェーブレットが信号に追従するように「伸縮」）
- STFTより優れた時間-周波数局在化
- チャープとマルチスケール過渡現象に最適
- リッジ抽出により精密な周波数軌跡を提供

❌ **欠点:**
- STFTより高い計算コスト
- ウェーブレット選択が必要（Morlet、Mexican hatなど）
- 冗長表現（過完備）
- 信号境界でのエッジ効果

### いつ使用するか

✅ **ウェーブレットを使用する場合:**
- 複数オクターブにまたがるチャープを解析
- STFTより良好な周波数追跡が必要
- 信号がマルチスケール構造を持つ
- 時間-周波数分解能のトレードオフが重要

❌ **使用しない場合:**
- 信号が純粋に定常（STFTで十分）
- 最速の計算が必要（STFTを使用）
- 周波数範囲が狭い（Q変換の方が良い場合がある）

### 実装

詳細な実装例、可視化、定量評価は英語版ドキュメントまたは `time_frequency_analysis_comparison.ipynb` を参照してください。

## 手法 4: ヒルベルト・ファン変換（HHT）

### 手法の説明

**経験的モード分解（EMD）+ ヒルベルト変換**:
1. 信号を固有モード関数（IMF）に分解
2. ヒルベルト変換により瞬時周波数を計算

### 実装

```python
# EMD を実行
imfs = ts.emd(method='emd', max_imf=5)

print(f"{len(imfs)} 個の IMF を抽出")

# IMF をプロット
fig, axes = plt.subplots(len(imfs), 1, figsize=(12, 10), sharex=True)

for i, (name, imf) in enumerate(imfs.items()):
    axes[i].plot(imf.times.value, imf.value, linewidth=0.5)
    axes[i].set_ylabel(name)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('時間 (s)')
axes[0].set_title('経験的モード分解（EMD）')
plt.tight_layout()
plt.show()
```

### 瞬時周波数

```python
# 支配的な IMF の瞬時周波数を計算
imf_main = imfs['IMF0']
inst_freq = imf_main.instantaneous_frequency()

# プロット
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(imf_main.times.value, imf_main.value, linewidth=0.5)
axes[0].set_ylabel('IMF0 振幅')
axes[0].set_title('支配的固有モード関数')
axes[0].grid(True, alpha=0.3)

axes[1].plot(inst_freq.times.value, inst_freq.value, linewidth=1)
axes[1].plot(t_true, f_true, 'r--', linewidth=2, label='真の周波数')
axes[1].set_ylabel('周波数 (Hz)')
axes[1].set_xlabel('時間 (s)')
axes[1].set_title('瞬時周波数（HHT）')
axes[1].set_ylim(0, 200)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 長所と短所

✅ **利点:**
- データ適応的（窓選択不要）
- 優れた時間分解能
- 非線形、非定常信号を処理
- 直接的な瞬時周波数

❌ **欠点:**
- 計算コストが高い
- EMD にモード混合がある可能性
- STFT/Q変換ほど確立されていない

### いつ使用するか

✅ **HHT を使用する場合:**
- 信号が高度に非定常
- 精密な瞬時周波数が必要
- 標準手法では特徴を分解できない
- グリッチや複雑な過渡現象を解析

❌ **使用しない場合:**
- 信号が定常（過剰）
- 高速計算が必要
- 標準的なスペクトログラムで十分

## 手法 5: 短時間ラプラス変換 (STLT)

### 手法の説明

**STLT**: 信号を**周波数ωと減衰率σの両方**の成分に分解

**式**: `STLT(t, σ, ω) = ∫ x(τ) w(τ-t) e^(-(σ+iω)τ) dτ`

周波数のみを示すSTFTとは異なり、STLTは減衰係数を明らかにします。

### 利点と欠点

✅ **利点:**
- **周波数と減衰率の両方**でモードを分離する独自の能力
- リングダウン品質因子推定に不可欠
- 2D σ-ω表現が減衰構造を明らかにする
- 合体後波形に直接適用可能

❌ **欠点:**
- 高い計算コスト（2D変換）
- 慎重なパラメータ選択が必要
- STFT/Q変換ほど一般的ではない
- 解釈にはσ-ω平面の理解が必要

### いつ使用するか

✅ **STLTを使用する場合:**
- リングダウンモードを解析（ブラックホール準固有モード）
- 品質因子や減衰時間を推定する必要がある
- 複数の減衰振動が周波数で重なる
- 減衰率情報が科学的に重要

❌ **使用しない場合:**
- 信号に減衰がない（代わりにSTFTを使用）
- 周波数情報のみが必要
- 計算リソースが限られている

### 実装

詳細な実装例、可視化、定量評価は英語版ドキュメントまたは `time_frequency_analysis_comparison.ipynb` を参照してください。

## 手法 6: ケプストラム

### 手法の説明

**ケプストラム**: 「ケフレンシー」によるスペクトルの周期性解析

**式**: `C(τ) = IFFT(log|FFT(x)|)`

スペクトルの周期性を時間領域の遅延時間のピークに変換します。

### 利点と欠点

✅ **利点:**
- エコー遅延の直接検出（ケフレンシーピーク）
- スペクトルの周期構造を明らかにする
- ピッチ検出と高調波解析に有用
- 計算効率が高い（二重FFT）

❌ **欠点:**
- 対数が必要（低振幅に敏感）
- 時間局在化されていない（グローバル解析）
- 時間-周波数手法ほど直感的ではない
- 明確なエコーがある信号に最適

### いつ使用するか

✅ **ケプストラムを使用する場合:**
- エコーや反射を検出
- スペクトルの周期構造を解析
- 残響信号の遅延時間を測定
- ピッチ検出または高調波解析

❌ **使用しない場合:**
- 時間局在化解析が必要
- 信号にエコーや周期性がない
- 低SNR（対数演算がノイズを増幅）

### 実装

詳細な実装例、可視化、定量評価は英語版ドキュメントまたは `time_frequency_analysis_comparison.ipynb` を参照してください。

## 手法 7: 離散コサイン変換 (DCT)

### 手法の説明

**DCT**: コサイン基底への変換、平滑信号の圧縮に優れる

**式**: `X(k) = Σ x(n) cos(πk(2n+1)/(2N))`

平滑信号では低周波係数にエネルギーが集中します。

### 利点と欠点

✅ **利点:**
- 平滑信号の優れた圧縮
- スパース表現（少数の係数でエネルギーの大部分をキャプチャ）
- 高速計算（FFT類似）
- 境界アーティファクトなし（Fourierとは異なる）
- 特徴抽出とノイズ除去に理想的

❌ **欠点:**
- 時間局在化されていない
- 不連続な信号には効果が低い
- 時間-周波数手法ほど直感的ではない
- グローバル解析のみ

### いつ使用するか

✅ **DCTを使用する場合:**
- 平滑信号を圧縮
- 特徴抽出（低次係数）
- 平滑背景のノイズ除去
- 機械学習前のデータ削減
- 緩やかなトレンドのモデリング

❌ **使用しない場合:**
- 時間局在化解析が必要
- 信号に急激な過渡現象がある
- 周波数内容が時間的に急速に変化

### 実装

詳細な実装例、可視化、定量評価は英語版ドキュメントまたは `time_frequency_analysis_comparison.ipynb` を参照してください。

## 比較例: 同じ信号に対するすべての手法

```python
fig = plt.figure(figsize=(14, 10))

# 元の信号
ax1 = plt.subplot(4, 1, 1)
ax1.plot(ts.times.value, ts.value, linewidth=0.5, color='black')
ax1.set_ylabel('ひずみ')
ax1.set_title('元の信号: チャープ (20→100 Hz) + ノイズ')
ax1.grid(True, alpha=0.3)

# スペクトログラム
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
im2 = ax2.pcolormesh(spec.times.value, spec.frequencies.value, spec.value.T,
                     cmap='viridis', shading='auto')
ax2.plot(t_true, f_true, 'r--', linewidth=1.5, alpha=0.8)
ax2.set_ylim(10, 150)
ax2.set_ylabel('周波数 (Hz)')
ax2.set_title('スペクトログラム（窓=0.5秒）')

# Q変換
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
ax3.imshow(qgram.value.T,
          extent=[qgram.times.value[0], qgram.times.value[-1],
                  qgram.frequencies.value[0], qgram.frequencies.value[-1]],
          aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear')
ax3.plot(t_true, f_true, 'r--', linewidth=1.5, alpha=0.8)
ax3.set_ylim(10, 150)
ax3.set_ylabel('周波数 (Hz)')
ax3.set_title('Q変換')

# HHT 瞬時周波数
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
ax4.plot(inst_freq.times.value, inst_freq.value, linewidth=1, label='HHT 瞬時周波数')
ax4.plot(t_true, f_true, 'r--', linewidth=2, label='真の周波数')
ax4.set_ylim(0, 150)
ax4.set_ylabel('周波数 (Hz)')
ax4.set_xlabel('時間 (s)')
ax4.set_title('HHT 瞬時周波数')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 決定木

```
どのような情報が必要ですか？

1. 一般的な時間-周波数ビュー？
   → **スペクトログラム (STFT)** を使用

2. 適応分解能による過渡現象検出？
   → **Q変換** を使用

3. チャープ周波数追跡（マルチスケール）？
   → **ウェーブレット (CWT)** を使用

4. 単一値曲線としての瞬時周波数？
   → **HHT** を使用

5. 減衰率と周波数（リングダウン）？
   → **STLT** を使用

6. エコー検出またはスペクトル周期性？
   → **ケプストラム** を使用

7. 信号圧縮または平滑特徴抽出？
   → **DCT** を使用

クイック決定パス:
├─ 定常信号？ → スペクトログラム
├─ 過渡現象/バースト？ → Q変換
├─ オクターブにまたがるチャープ？ → ウェーブレット
├─ 瞬時周波数が必要？ → HHT
├─ 減衰振動？ → STLT
├─ エコーがある？ → ケプストラム
└─ 平滑、圧縮が必要？ → DCT
```

## 性能比較

| 手法 | 計算時間* | メモリ | 最適な用途 |
|--------|------------------|--------|----------|
| スペクトログラム (STFT) | 1×（基準） | 低 | 汎用 |
| DCT | 1-2× | 低 | 圧縮 |
| Q変換 | 5-10× | 中 | 過渡現象 |
| ウェーブレット (CWT) | 8-15× | 中-高 | チャープ |
| ケプストラム | 3-5× | 中 | エコー検出 |
| STLT | 15-25× | 高 | リングダウンモード |
| HHT | 20-50× | 高 | 瞬時周波数 |

*おおよそ、パラメータと信号長に依存

## まとめ表

| 特徴 | STFT | Q変換 | ウェーブレット | HHT | STLT | ケプストラム | DCT |
|---------|------|-------------|---------|-----|------|----------|-----|
| **分解能** | 固定 | 適応Q | スケール適応 | データ適応 | 固定 | ケフレンシー | N/A |
| **最適な信号** | 定常 | チャープ | マルチスケール | AM/FM | 減衰 | エコー | 平滑 |
| **計算コスト** | 低 | 中 | 中-高 | 非常に高 | 高 | 中 | 低 |
| **出力** | 2D (t,f) | 2D (t,f) | 2D (t,f) | f(t) 曲線 | 3D (t,σ,ω) | τ スペクトル | 係数 |
| **時間局在化** | あり | あり | あり | あり | あり | なし | なし |
| **独自情報** | — | 適応分解能 | マルチスケール | 瞬時周波数 | 減衰率σ | 遅延 | 圧縮 |
| **重力波利用** | 汎用 | 過渡現象 | チャープ | グリッチ | リングダウン | 反射 | 特徴 |

## 実践的な推奨事項

### 日常的な解析
**スペクトログラム (STFT)** から始める - 高速でよく理解されており、ほとんどのケースで十分。

### 過渡現象検出
**Q変換** を使用 - 重力波バースト探索の標準。

### チャープ解析
**ウェーブレット (CWT)** を使用 - 複数スケールにわたる精密な周波数軌跡追跡。

### 瞬時周波数
**HHT** を使用 - 単一値瞬時周波数が必要な場合（時間-周波数不確定性なし）。

### リングダウン解析
**STLT** を使用 - 周波数と減衰率（品質因子）を同時に推定。

### エコー検出
**ケプストラム** を使用 - 遅延時間とスペクトルの周期構造を特定。

### データ削減
**DCT** を使用 - 平滑信号の効率的な圧縮と特徴抽出。

### 論文掲載
- **スペクトログラム**（読者に馴染みのある基準線）を含める
- 独自の物理を明らかにする専門的手法を追加（チャープにはウェーブレット、瞬時周波数にはHHT、リングダウンにはSTLTなど）
- ロバスト性を示す：主要な結果が複数の手法で成立することを実証

### 一般的な戦略
1. **シンプルから始める**: 常にスペクトログラムから開始
2. **限界を特定**: STFTが重要な特徴を示せない場所は？
3. **専門的手法を選択**: ニーズに合った「独自情報」を持つ手法を選ぶ（まとめ表参照）
4. **検証**: 可能であれば別の手法でクロスチェック

---

**関連項目:**
- [スペクトログラムチュートリアル](intro_spectrogram.ipynb)
- [HHT チュートリアル](advanced_hht.ipynb)
- [Q変換ドキュメント](../reference/api/qtransform.rst)
