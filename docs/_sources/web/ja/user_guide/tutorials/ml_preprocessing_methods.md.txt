# ML 前処理手法 - 個別テクニック

このチュートリアルでは、gwexpy で利用可能な機械学習前処理手法を、パイプラインに組み合わせる前に**個別に**説明します。

## 概要

重力波解析のための機械学習モデルには、慎重なデータ前処理が必要です:

1. **ホワイトニング**: 色付きノイズを除去 → 平坦なスペクトル
2. **バンドパスフィルタ**: 特定の周波数帯域を抽出
3. **正規化**: チャンネル間で振幅を標準化
4. **分割**: データを訓練/検証セットに分割

このチュートリアルでは、各手法を**個別に**実演することで、以下を理解できます:
- 各手法が何をするか
- いつ使用するべきか
- どのように設定するか
- 出力として何を期待すべきか

## セットアップ

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.timeseries import TimeSeries
from gwexpy.noise.wave import sine, gaussian
from gwexpy.signal.preprocessing import whitening, standardization

# サンプルデータの作成: 60 Hz 信号 + 色付きノイズ
sample_rate = 4096  # Hz
duration = 10  # seconds
t = np.arange(0, duration, 1/sample_rate)

# 信号: 60 Hz 正弦波
signal = 2.0 * np.sin(2 * np.pi * 60 * t)

# 色付きノイズ (1/f ノイズ + ホワイトノイズ)
freqs = np.fft.rfftfreq(len(t), 1/sample_rate)
noise_fft = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
noise_fft[1:] /= np.sqrt(freqs[1:])  # 1/f 着色
noise = np.fft.irfft(noise_fft, len(t)).real * 0.5

# 結合データ
data = signal + noise
ts = TimeSeries(data, t0=0, dt=1/sample_rate, unit='strain', name='H1:GDS-CALIB_STRAIN')

print(f"データ長: {len(ts)} サンプル ({duration}秒)")
```

## 手法 1: ホワイトニング

### ホワイトニングとは？

**目的**: 色付きノイズ（周波数依存PSD）を白色ノイズ（平坦なPSD）に変換

**必要な理由**: ML モデルは定常的な白色ノイズを仮定することが多い。重力波データには強い色付きノイズ（1/f、バイオリンモードなど）がある。

### ホワイトニングの仕組み

1. データの PSD を推定
2. ホワイトニングフィルタを計算: `H(f) = 1 / √PSD(f)`
3. 周波数領域でフィルタを適用
4. 時間領域に戻す

### 実装

```python
from gwexpy.signal.preprocessing.whitening import WhiteningModel

# ホワイトニングモデルの作成
whitening_model = WhiteningModel(
    fftlength=4,  # PSD推定用の4秒セグメント
    overlap=2,    # 50%オーバーラップ
    method='welch'
)

# モデルのフィット（PSDを推定）
whitening_model.fit(ts)

# ホワイトニングを適用
ts_whitened = whitening_model.transform(ts)

print(f"ホワイトニング後のデータ: {ts_whitened.name}")
```

### 可視化: 前後の比較

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 時間領域 - 前
axes[0, 0].plot(ts.times.value[:1000], ts.value[:1000], linewidth=0.5)
axes[0, 0].set_title('元の信号（時間領域）')
axes[0, 0].set_xlabel('時間 (s)')
axes[0, 0].set_ylabel('ひずみ')
axes[0, 0].grid(True, alpha=0.3)

# 時間領域 - 後
axes[0, 1].plot(ts_whitened.times.value[:1000], ts_whitened.value[:1000], linewidth=0.5, color='orange')
axes[0, 1].set_title('ホワイトニング後の信号（時間領域）')
axes[0, 1].set_xlabel('時間 (s)')
axes[0, 1].set_ylabel('ホワイトニング後のひずみ')
axes[0, 1].grid(True, alpha=0.3)

# PSD - 前
psd_original = ts.psd(fftlength=4)
axes[1, 0].loglog(psd_original.frequencies.value, psd_original.value)
axes[1, 0].set_title('元のPSD（色付き）')
axes[1, 0].set_xlabel('周波数 (Hz)')
axes[1, 0].set_ylabel('PSD')
axes[1, 0].grid(True, which='both', alpha=0.3)
axes[1, 0].set_xlim(10, 2000)

# PSD - 後
psd_whitened = ts_whitened.psd(fftlength=4)
axes[1, 1].loglog(psd_whitened.frequencies.value, psd_whitened.value, color='orange')
axes[1, 1].set_title('ホワイトニング後のPSD（平坦）')
axes[1, 1].set_xlabel('周波数 (Hz)')
axes[1, 1].set_ylabel('PSD')
axes[1, 1].grid(True, which='both', alpha=0.3)
axes[1, 1].set_xlim(10, 2000)

plt.tight_layout()
plt.show()
```

**期待される結果:**
- **前**: PSD が高周波で下降（色付きノイズ）
- **後**: PSD がほぼ平坦（白色ノイズ）

### いつ使用するか

✅ **ホワイトニングを使用する場合:**
- 入力データに強い色付きノイズがある
- ML モデルが白色ノイズを仮定
- すべての周波数で特徴を均等に強調したい

❌ **ホワイトニングを使用しない場合:**
- スペクトル形状を明示的に保持したい
- 信号がすでに白色
- 周波数領域解析を行う（代わりに PSD 正規化を使用）

## 手法 2: バンドパスフィルタリング

### バンドパスフィルタリングとは？

**目的**: 特定の周波数範囲の信号を抽出し、その範囲外のノイズを除去

**必要な理由**: 重力波信号は狭い周波数帯域を占めることが多い（CBC の場合 20-500 Hz など）。フィルタリングにより SNR が向上する。

### 実装

```python
# バンドパスフィルタを適用: 50-100 Hz
ts_filtered = ts.bandpass(50, 100, order=8)

print(f"{50}-{100} Hz 帯域にフィルタリング")
```

### 可視化: 周波数応答

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 時間領域の比較
axes[0].plot(ts.times.value[:2048], ts.value[:2048],
            label='元の信号', alpha=0.6, linewidth=0.5)
axes[0].plot(ts_filtered.times.value[:2048], ts_filtered.value[:2048],
            label='バンドパス (50-100 Hz)', linewidth=0.8)
axes[0].set_xlabel('時間 (s)')
axes[0].set_ylabel('ひずみ')
axes[0].set_title('バンドパスフィルタの効果（時間領域）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 周波数領域の比較
psd_filtered = ts_filtered.psd(fftlength=2)

axes[1].loglog(psd_original.frequencies.value, psd_original.value,
              label='元の信号', alpha=0.6)
axes[1].loglog(psd_filtered.frequencies.value, psd_filtered.value,
              label='バンドパス (50-100 Hz)', linewidth=2)
axes[1].axvspan(50, 100, alpha=0.2, color='green', label='通過帯域')
axes[1].set_xlabel('周波数 (Hz)')
axes[1].set_ylabel('PSD')
axes[1].set_title('バンドパスフィルタの効果（周波数領域）')
axes[1].legend()
axes[1].grid(True, which='both', alpha=0.3)
axes[1].set_xlim(10, 500)

plt.tight_layout()
plt.show()
```

**期待される結果:**
- 50-100 Hz 外で PSD が抑制される
- 通過帯域内で信号が保持される
- エッジで滑らかなロールオフ（フィルタ次数で決定）

### 複数帯域

```python
# 複数のバンドパスフィルタを適用して合計
bands = [(30, 40), (50, 70), (100, 150)]
ts_multiband = None

for f_low, f_high in bands:
    ts_band = ts.bandpass(f_low, f_high, order=6)
    if ts_multiband is None:
        ts_multiband = ts_band
    else:
        ts_multiband = ts_multiband + ts_band

print(f"{len(bands)}個のバンドパスフィルタを適用")
```

### いつ使用するか

✅ **バンドパスを使用する場合:**
- 信号の周波数範囲が既知
- 帯域外ノイズを除去したい
- 限定された周波数範囲の ML モデルの前処理

❌ **バンドパスを使用しない場合:**
- 信号周波数が未知または広帯域
- 全スペクトル情報が必要
- フィルタリングのリンギングアーティファクトが問題

## 手法 3: 正規化/標準化

### 正規化とは？

**目的**: データを一貫した範囲にスケーリング、平均を除去

**必要な理由**: ML モデル（特にニューラルネットワーク）は正規化された入力で速く収束する

### 利用可能な手法

1. **Z スコア正規化**: `(x - mean) / std`
2. **ロバスト正規化**: `(x - median) / MAD`（外れ値に頑健）
3. **最小-最大スケーリング**: `(x - min) / (max - min)`

### 実装

```python
from gwexpy.timeseries import TimeSeries

# Z スコア正規化
ts_zscore = ts.standardize(method='zscore')

# ロバスト正規化（中央絶対偏差を使用）
ts_robust = ts.standardize(method='zscore', robust=True)

print(f"元の値:   平均={ts.mean():.3f}, 標準偏差={ts.std():.3f}")
print(f"Z スコア: 平均={ts_zscore.mean():.3e}, 標準偏差={ts_zscore.std():.3f}")
print(f"ロバスト: 中央値={np.median(ts_robust.value):.3e}, MAD*1.4826={np.median(np.abs(ts_robust.value - np.median(ts_robust.value)))*1.4826:.3f}")
```

**期待される出力:**
```
元の値:   平均=0.023, 標準偏差=0.891
Z スコア: 平均≈0.0, 標準偏差≈1.0
ロバスト: 中央値≈0.0, MAD*1.4826≈1.0
```

### 可視化: 分布の比較

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 元の分布
axes[0].hist(ts.value, bins=100, alpha=0.7, edgecolor='black')
axes[0].set_title(f'元の値\nμ={ts.mean():.2f}, σ={ts.std():.2f}')
axes[0].set_xlabel('ひずみ')
axes[0].set_ylabel('カウント')
axes[0].grid(True, alpha=0.3)

# Z スコア分布
axes[1].hist(ts_zscore.value, bins=100, alpha=0.7, color='orange', edgecolor='black')
axes[1].set_title(f'Z スコア正規化\nμ≈0, σ≈1')
axes[1].set_xlabel('正規化されたひずみ')
axes[1].set_ylabel('カウント')
axes[1].grid(True, alpha=0.3)

# ロバスト分布
axes[2].hist(ts_robust.value, bins=100, alpha=0.7, color='green', edgecolor='black')
axes[2].set_title(f'ロバスト正規化\n中央値≈0, MAD≈1')
axes[2].set_xlabel('正規化されたひずみ')
axes[2].set_ylabel('カウント')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 比較: Z スコア vs ロバスト（外れ値あり）

```python
# 人工的な外れ値を追加
ts_with_outliers = ts.copy()
ts_with_outliers.value[1000:1010] = 50  # 大きなスパイク

# 両方の方法で正規化
ts_zscore_out = ts_with_outliers.standardize(method='zscore')
ts_robust_out = ts_with_outliers.standardize(method='zscore', robust=True)

# 比較プロット
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

time_window = slice(900, 1100)  # 外れ値の周辺
times = ts.times.value[time_window]

axes[0].plot(times, ts_with_outliers.value[time_window])
axes[0].set_title('元の値（外れ値あり）')
axes[0].set_ylabel('ひずみ')
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, ts_zscore_out.value[time_window], color='orange')
axes[1].set_title('Z スコア（外れ値に敏感）')
axes[1].set_ylabel('正規化')
axes[1].grid(True, alpha=0.3)

axes[2].plot(times, ts_robust_out.value[time_window], color='green')
axes[2].set_title('ロバスト（外れ値に頑健）')
axes[2].set_ylabel('正規化')
axes[2].set_xlabel('時間 (s)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**期待される結果:**
- Z スコア: 大きな外れ値の分散により信号全体が圧縮される
- ロバスト: 外れ値は可視だが、残りの信号は構造を維持

### いつ使用するか

✅ **Z スコアを使用する場合:**
- データがほぼガウス分布
- 顕著な外れ値がない
- 標準的な ML 前処理

✅ **ロバストを使用する場合:**
- データにグリッチ/外れ値が含まれる
- 非ガウスノイズ
- アーティファクトがあっても信号構造を保持したい

## 手法 4: 訓練/検証分割

### 分割とは？

**目的**: データを重複しない訓練セットと検証セットに分割

**必要な理由**: ML モデルには訓練と性能評価のための別々のデータが必要

### 時系列順序の分割

```python
# 分割: 80% 訓練, 20% 検証
train_fraction = 0.8
split_point = int(len(ts) * train_fraction)

ts_train = ts[:split_point]
ts_valid = ts[split_point:]

print(f"訓練: {len(ts_train)} サンプル ({len(ts_train)/sample_rate:.1f}秒)")
print(f"検証: {len(ts_valid)} サンプル ({len(ts_valid)/sample_rate:.1f}秒)")
```

**期待される出力:**
```
訓練: 32768 サンプル (8.0秒)
検証: 8192 サンプル (2.0秒)
```

### 可視化

```python
plt.figure(figsize=(12, 4))
plt.plot(ts.times.value, ts.value, linewidth=0.5, alpha=0.6, label='全データ')
plt.axvline(ts_train.times.value[-1], color='r', linestyle='--', linewidth=2,
           label='訓練/検証の分割点')
plt.axvspan(ts.times.value[0], ts_train.times.value[-1], alpha=0.1, color='blue',
           label='訓練セット')
plt.axvspan(ts_train.times.value[-1], ts.times.value[-1], alpha=0.1, color='green',
           label='検証セット')
plt.xlabel('時間 (s)')
plt.ylabel('ひずみ')
plt.title('訓練/検証分割（時系列順序）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### いつ使用するか

✅ **時系列順序分割を使用する場合:**
- 時間依存性が存在する（自己相関）
- 将来予測能力をテストする
- 時系列の標準的な実践

❌ **ランダム分割を使用する場合:**
- データが i.i.d.（独立同一分布）
- **警告**: 通常、重力波データには適切ではない！

## まとめ: 決定マトリックス

| 手法 | 入力 | 出力 | いつ使用するか |
|--------|-------|--------|-------------|
| **ホワイトニング** | 色付きノイズ | 白色ノイズ | GW データの ML には常に使用 |
| **バンドパス** | 広帯域信号 | 狭帯域信号 | 既知の周波数範囲 |
| **Z スコア** | 任意のスケール | 平均=0, 標準偏差=1 | ガウスデータ、外れ値なし |
| **ロバスト** | 外れ値あり | 中央値=0, MAD=1 | グリッチのあるデータ |
| **訓練/検証分割** | 完全データセット | 訓練 + 検証セット | すべての教師あり学習 |

## 手法の組み合わせ

これらの手法を最適な順序で組み合わせた完全なワークフローについては、英語版の [ML 前処理パイプライン](../../../en/user_guide/tutorials/case_ml_preprocessing.ipynb) を参照してください。

**推奨順序:**
1. **バンドパス**（該当する場合） - 最初に帯域外ノイズを除去
2. **ホワイトニング** - スペクトルを平坦化
3. **正規化** - スケールを標準化
4. **分割** - 訓練/検証セットを作成

---

**関連項目:**
- [完全な ML パイプライン（英語）](../../../en/user_guide/tutorials/case_ml_preprocessing.ipynb) - エンドツーエンドのワークフロー
- [数値安定性ガイド](../numerical_stability.md) - 精度に関する考慮事項
- [高度な相関解析](advanced_correlation.ipynb) - 特徴エンジニアリング
