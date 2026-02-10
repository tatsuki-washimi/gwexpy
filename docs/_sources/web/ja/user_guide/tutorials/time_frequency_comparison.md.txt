# 時間-周波数解析手法: 比較と選択ガイド

このチュートリアルでは、gwexpy のさまざまな時間-周波数解析手法を比較し、解析に適した手法を選択する方法を説明します。

## 概要

重力波信号は**時間変化する周波数内容**を持つことが多い。異なる手法は異なる側面を明らかにします:

| 手法 | 最適な用途 | 時間分解能 | 周波数分解能 |
|--------|----------|-----------------|---------------------|
| **スペクトログラム** | 汎用 | 良好 | 良好 |
| **Q変換** | チャープ、過渡現象 | 適応的 | 適応的（定数Q） |
| **HHT** | 非定常、非線形 | 優秀 | データ適応的 |
| **STFT** | 定常セグメント | 固定（窓） | 固定（窓） |
| **ウェーブレット** | マルチスケール特徴 | スケール依存 | スケール依存 |

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

## 手法 3: ヒルベルト・ファン変換（HHT）

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
信号の特性は...

1. 定常または緩やかに変化？
   はい → **スペクトログラム** を使用
   いいえ → 質問2へ

2. チャープまたは急速な過渡現象？
   はい → **Q変換** を使用
   いいえ → 質問3へ

3. 高度に非定常または非線形？
   はい → **HHT** を使用
   いいえ → まず**スペクトログラム**、次にQ変換

4. 瞬時周波数が必要？
   はい → **HHT** を使用
   いいえ → **スペクトログラム**または**Q変換**

5. 高速計算が必要？
   はい → **スペクトログラム** を使用
   いいえ → すべての手法が使用可能
```

## 性能比較

| 手法 | 計算時間* | メモリ | 周波数追跡 |
|--------|------------------|--------|-------------------|
| スペクトログラム | 1×（基準） | 低 | 遅い変化に適している |
| Q変換 | 5-10× | 中 | チャープに優秀 |
| HHT | 20-50× | 高 | すべてに優秀 |

*おおよそ、パラメータに依存

## まとめ表

| 特徴 | スペクトログラム | Q変換 | HHT |
|---------|------------|-------------|-----|
| **時間-周波数分解能** | 固定 | 適応的（定数Q） | データ適応的 |
| **最適な信号タイプ** | 定常 | チャープ | 非定常 |
| **計算コスト** | 低 | 中 | 高 |
| **周波数追跡** | 良好 | 優秀 | 優秀 |
| **解釈の容易さ** | 容易 | 中程度 | 複雑 |
| **標準的な重力波利用** | 汎用 | 過渡現象 | 特殊ケース |

## 実践的な推奨事項

### 日常的な解析
**スペクトログラム**から始める - 高速でよく理解されており、ほとんどのケースで十分。

### 過渡現象検出
**Q変換**を使用 - 重力波バースト探索の標準。

### 詳細な特性評価
**HHT**を使用 - 精密な瞬時周波数が必要な場合、または他の手法が失敗する場合。

### 論文掲載
**スペクトログラム**（読者に馴染みがある）と手法固有の解析（Q変換またはHHT）の両方を含め、ロバスト性を示す。

---

**関連項目:**
- [スペクトログラムチュートリアル](intro_spectrogram.ipynb)
- [HHT チュートリアル](advanced_hht.ipynb)
- [Q変換ドキュメント](../reference/api/qtransform.rst)
