# 重力波解析のための線形代数

このチュートリアルでは、gwexpy の Matrix クラスと線形代数手法を使用した重力波データ解析の方法を実演します。

## なぜ重力波解析に線形代数が必要か？

多チャンネル重力波データは自然に行列構造を形成します:
- **相関行列**: チャンネル間のノイズ結合を特定
- **固有モード分解**: 主なノイズ源を発見
- **共分散解析**: チャンネル関係を定量化
- **伝達行列**: システム応答をモデル化

## セットアップ

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.timeseries import TimeSeriesMatrix, TimeSeries
from gwexpy.noise.wave import sine, gaussian

# 多チャンネル合成データの生成
sample_rate = 1024  # Hz
duration = 10  # seconds
n_channels = 6
```

## 1. 相関行列解析

### 多チャンネルデータの作成

```python
# 結合ノイズを持つ6チャンネルをシミュレート
channels = []
base_noise = gaussian(duration=duration, sample_rate=sample_rate, std=1.0)

for i in range(n_channels):
    # 各チャンネルには:
    # 1. 独立ノイズ
    # 2. ベースノイズへの結合（異なる強度）
    coupling = 0.5 ** i  # 指数的減衰
    independent = gaussian(duration=duration, sample_rate=sample_rate, std=0.5)

    channel_data = base_noise * coupling + independent
    channel_data.name = f"Channel {i}"
    channels.append(channel_data)

# TimeSeriesMatrix の作成
tsm = TimeSeriesMatrix.from_list(channels)
print(f"行列の形状: {tsm.shape}")  # (6, 1, 10240)
```

### 相関行列の計算

```python
# 相関行列を計算
corr_matrix = tsm.correlation_matrix()

print(f"相関行列の形状: {corr_matrix.shape}")
print(f"相関行列:\n{corr_matrix}")

# 相関行列を可視化
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='相関係数')
plt.title('チャンネル相関行列')
plt.xlabel('チャンネルインデックス')
plt.ylabel('チャンネルインデックス')
for i in range(n_channels):
    for j in range(n_channels):
        text = plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)
plt.tight_layout()
plt.show()
```

**期待される出力:**
- 隣接チャンネル間の強い相関（>0.8）
- 結合強度の減衰による距離に伴う相関の減少
- 対角線 = 1.0（完全な自己相関）

## 2. 固有モード分解

### 主成分の発見

```python
# 共分散行列と固有値を計算
cov_matrix = tsm.covariance_matrix()
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 固有値の大きさで並べ替え（降順）
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("固有値（各モードで説明される分散）:")
for i, ev in enumerate(eigenvalues):
    percent = 100 * ev / eigenvalues.sum()
    print(f"  モード {i}: {ev:.4f} ({percent:.1f}%)")
```

**期待される出力:**
```
モード 0: 5.2341 (67.8%)  # 支配的モード（共通ノイズ）
モード 1: 1.4562 (18.9%)  # 第2モード
モード 2: 0.6234 (8.1%)   # 第3モード
...
```

### 固有モードの可視化

```python
# 最初の3つの固有モードをプロット
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i in range(3):
    axes[i].bar(range(n_channels), eigenvectors[:, i])
    axes[i].set_ylabel(f'モード {i}\n振幅')
    axes[i].set_title(f'固有モード {i} (λ={eigenvalues[i]:.3f}, '
                     f'{100*eigenvalues[i]/eigenvalues.sum():.1f}% 分散)')
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(0, color='k', linewidth=0.5)

axes[-1].set_xlabel('チャンネルインデックス')
axes[-1].set_xticks(range(n_channels))
plt.tight_layout()
plt.show()
```

**物理的解釈:**
- **モード 0**: すべてのチャンネルが正 → 共通ノイズ源
- **モード 1**: 符号混在 → 差動ノイズ
- **モード 2**: より高周波の空間パターン

## 3. ノイズモード投影

### データを固有モードに投影

```python
# 時系列を固有モードに投影
mode_timeseries = []

for i in range(3):  # 最初の3モード
    # 投影: mode_i(t) = Σ_j eigenvector[j,i] * channel[j](t)
    mode_data = np.zeros(tsm.shape[2])
    for j in range(n_channels):
        mode_data += eigenvectors[j, i] * tsm.value[j, 0, :]

    ts = TimeSeries(
        mode_data,
        t0=tsm.t0,
        dt=tsm.dt,
        unit=tsm.units[0, 0],
        name=f'Mode {i}'
    )
    mode_timeseries.append(ts)

# モード時系列をプロット
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, (ax, ts) in enumerate(zip(axes, mode_timeseries)):
    ax.plot(ts.times.value, ts.value, linewidth=0.5)
    ax.set_ylabel(f'モード {i}')
    ax.set_title(f'固有モード {i} 時系列 '
                f'({100*eigenvalues[i]/eigenvalues.sum():.1f}% 分散)')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('時間 (s)')
plt.tight_layout()
plt.show()
```

## 4. 次元削減

### 縮小モードでデータを再構成

```python
# 最初の k モードのみを使用して再構成
k = 2  # 2つの支配的モードのみを保持

reconstructed = np.zeros_like(tsm.value)
for mode_idx in range(k):
    for ch_idx in range(n_channels):
        reconstructed[ch_idx, 0, :] += (
            eigenvectors[ch_idx, mode_idx] *
            mode_timeseries[mode_idx].value
        )

# 再構成された TimeSeriesMatrix を作成
tsm_reconstructed = TimeSeriesMatrix(
    reconstructed,
    t0=tsm.t0,
    dt=tsm.dt,
    channel_names=[f"Ch{i}_recon" for i in range(n_channels)],
    unit=tsm.units[0, 0]
)

# チャンネル 0 の元データと再構成データを比較
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# 元データ
axes[0].plot(tsm.times.value, tsm.value[0, 0, :], linewidth=0.5, label='元データ')
axes[0].set_ylabel('チャンネル 0\n元データ')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 再構成データ
axes[1].plot(tsm_reconstructed.times.value, tsm_reconstructed.value[0, 0, :],
            linewidth=0.5, color='orange', label=f'再構成 ({k} モード)')
axes[1].set_ylabel('チャンネル 0\n再構成')
axes[1].set_xlabel('時間 (s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 再構成誤差を計算
error = np.linalg.norm(tsm.value - tsm_reconstructed.value)
total = np.linalg.norm(tsm.value)
print(f"\n再構成誤差 (L2 ノルム): {error/total:.1%}")
print(f"捕捉された分散: {100*eigenvalues[:k].sum()/eigenvalues.sum():.1f}%")
```

**期待される出力:**
```
再構成誤差 (L2 ノルム): ~15-20%
捕捉された分散: ~85-90%
```

## 5. 応用

### A. 固有モードフィルタリングによるノイズ除去

```python
# 支配的モード（共通ノイズ）を除去
cleaned = tsm.value.copy()
for ch_idx in range(n_channels):
    cleaned[ch_idx, 0, :] -= (
        eigenvectors[ch_idx, 0] * mode_timeseries[0].value
    )

tsm_cleaned = TimeSeriesMatrix(
    cleaned,
    t0=tsm.t0,
    dt=tsm.dt,
    channel_names=[f"Ch{i}_cleaned" for i in range(n_channels)],
    unit=tsm.units[0, 0]
)

# 相関行列を比較
corr_original = tsm.correlation_matrix()
corr_cleaned = tsm_cleaned.correlation_matrix()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(corr_original, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title('元の相関')
axes[0].set_xlabel('チャンネル')
axes[0].set_ylabel('チャンネル')

im1 = axes[1].imshow(corr_cleaned, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title('モード 0 除去後')
axes[1].set_xlabel('チャンネル')

fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.show()
```

### B. 特定周波数へのモード寄与

```python
# 各モードの PSD を計算
mode_psds = []
for ts in mode_timeseries[:3]:
    psd = ts.psd(fftlength=1)
    mode_psds.append(psd)

# プロット
plt.figure(figsize=(10, 6))
for i, psd in enumerate(mode_psds):
    plt.loglog(psd.frequencies.value, psd.value,
              label=f'モード {i} ({100*eigenvalues[i]/eigenvalues.sum():.1f}%)',
              alpha=0.7)

plt.xlabel('周波数 (Hz)')
plt.ylabel('PSD')
plt.title('主固有モードのパワースペクトル密度')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.xlim(1, sample_rate/2)
plt.tight_layout()
plt.show()
```

## まとめ

gwexpy の線形代数手法により以下が可能になります:

1. **相関解析**: チャンネル結合の特定
2. **固有モード分解**: 主要ノイズ源の発見
3. **次元削減**: 分散を保持しながらデータを圧縮
4. **ノイズ除去**: 共通モードノイズの除去
5. **モードスペクトル解析**: モードの周波数内容の理解

### 主要メソッド

| メソッド | 目的 | 出力 |
|--------|---------|--------|
| `correlation_matrix()` | チャンネル相関 | n×n 行列 |
| `covariance_matrix()` | 分散-共分散 | n×n 行列 |
| `np.linalg.eigh()` | 固有値分解 | 固有値、固有ベクトル |
| モード投影 | ノイズモード抽出 | モードごとの時系列 |

### 次のステップ

- **Field への応用**: ScalarField に適用して空間モードを解析
- **伝達関数行列**: システム同定
- **最適フィルタリング**: 相関構造を使用したウィーナーフィルタリング

---

**関連項目:**
- [TimeSeriesMatrix チュートリアル](matrix_timeseries.ipynb)
- [ノイズバジェット解析例](../examples/case_noise_budget.ipynb)
- [高度な相関手法](advanced_correlation.ipynb)
