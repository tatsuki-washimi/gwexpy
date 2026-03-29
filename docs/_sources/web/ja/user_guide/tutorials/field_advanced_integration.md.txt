# Field API × 高度な解析: 統合ワークフロー

このチュートリアルでは、Field クラス（ScalarField、VectorField、TensorField）と高度な解析手法を**組み合わせて**、エンドツーエンドの科学的ワークフローを構築する方法を実演します。

## なぜ統合が重要か

gwexpy の力は、機能を**組み合わせる**ことから生まれます:
- **Field API**: 4次元時空構造
- **高度な手法**: HHT、ML前処理、相関解析
- **時系列ツール**: TimeSeries、Matrix演算

このチュートリアルでは、孤立した例ではなく**完全なワークフロー**を示します。

## ワークフロー 1: 地震波伝播解析

### 科学的目標

検出器アレイを通じて地震ノイズがどのように伝播するかを以下を使用して解析:
1. ScalarField で時空表現
2. 相互相関で伝播遅延を解析
3. 固有モード解析でノイズ源を特定

### セットアップ

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.fields import ScalarField
from gwexpy.timeseries import TimeSeriesMatrix

# シミュレーションパラメータ
sample_rate = 256 * u.Hz
duration = 10 * u.s
nx, ny, nz = 20, 20, 1  # 2D 空間グリッド (z=0 平面)

# 空間グリッドの作成
x = np.linspace(0, 100, nx) * u.m  # 100m × 100m エリア
y = np.linspace(0, 100, ny) * u.m
z = np.array([0]) * u.m
t = np.arange(int((sample_rate * duration).to_value(u.dimensionless_unscaled))) / sample_rate.value * u.s

nt = len(t)
```

### ステップ 1: 伝播する波を持つ ScalarField の作成

```python
# 点音源からの地震波をシミュレート
source_pos = (30.0, 50.0)  # (x, y) メートル単位
source_freq = 5.0  # Hz
v_propagation = 300.0  # m/s

# フィールドの初期化
data = np.zeros((nt, nx, ny, nz))

# ノイズを追加
np.random.seed(42)
data += np.random.randn(nt, nx, ny, nz) * 0.1

# 伝播する波を追加
for i, xi in enumerate(x.value):
    for j, yj in enumerate(y.value):
        # 音源からの距離
        r = np.sqrt((xi - source_pos[0])**2 + (yj - source_pos[1])**2)

        # 時間遅延
        delay = r / v_propagation

        # 振幅減衰 (1/r)
        amp = 1.0 / (1.0 + r/10)

        # 信号: 遅延した正弦波
        signal = amp * np.sin(2 * np.pi * source_freq * (t.value - delay))
        data[:, i, j, 0] += signal

# ScalarField の作成
field = ScalarField(
    data,
    unit=u.m/u.s,  # 速度
    axis0=t,
    axis1=x,
    axis2=y,
    axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real',
    name='Seismic Velocity'
)

print(f"フィールドの形状: {field.shape}")
print(f"継続時間: {duration}, 空間範囲: {x[-1]} × {y[-1]}")
```

### ステップ 2: 複数の位置で TimeSeries を抽出

```python
# 5つの空間点で時系列を抽出
locations = [
    (10.0*u.m, 50.0*u.m, 0.0*u.m),  # 西端
    (30.0*u.m, 50.0*u.m, 0.0*u.m),  # 音源
    (50.0*u.m, 50.0*u.m, 0.0*u.m),  # 東
    (70.0*u.m, 50.0*u.m, 0.0*u.m),  # 東端
    (30.0*u.m, 80.0*u.m, 0.0*u.m),  # 北
]

timeseries_list = []
for loc in locations:
    # .at() メソッドを使用して特定の空間点で抽出
    ts = field.at(x=loc[0], y=loc[1], z=loc[2])
    ts.name = f"x={loc[0].value:.0f}m, y={loc[1].value:.0f}m"
    timeseries_list.append(ts)

# 相関解析用の TimeSeriesMatrix を作成
from gwexpy.timeseries import TimeSeries
tsm = TimeSeriesMatrix.from_list([ts.squeeze() for ts in timeseries_list])

print(f"{len(timeseries_list)} 個の時系列を抽出")
```

### ステップ 3: 相互相関解析

```python
# 相関行列を計算
corr_matrix = tsm.correlation_matrix()

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 相関行列
im = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title('空間相関行列')
axes[0].set_xlabel('位置インデックス')
axes[0].set_ylabel('位置インデックス')
for i in range(len(locations)):
    for j in range(len(locations)):
        text = axes[0].text(j, i, f'{corr_matrix[i,j]:.2f}',
                           ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=axes[0])

# 時系列のオーバーレイ
for i, ts in enumerate(timeseries_list):
    axes[1].plot(ts.times.value, ts.value.squeeze() + i*2,
                label=ts.name, alpha=0.7)
axes[1].set_xlabel('時間 (s)')
axes[1].set_ylabel('速度（明瞭化のためオフセット）')
axes[1].set_title('異なる位置での時系列')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**期待される観察結果:**
- 波の経路に沿った位置間で最も高い相関
- 時系列オーバーレイで時間遅延が可視化される

### ステップ 4: 空間にわたる PSD 比較

```python
# ScalarField の PSD メソッドを使用して空間的に変化する PSD を計算
psd_west = field.psd(point_or_region=(10.0*u.m, 50.0*u.m, 0.0*u.m), fftlength=2)
psd_source = field.psd(point_or_region=(30.0*u.m, 50.0*u.m, 0.0*u.m), fftlength=2)
psd_east = field.psd(point_or_region=(70.0*u.m, 50.0*u.m, 0.0*u.m), fftlength=2)

# プロット
plt.figure(figsize=(10, 6))
plt.loglog(psd_west.frequencies.value, psd_west.value, label='西 (10m)', alpha=0.7)
plt.loglog(psd_source.frequencies.value, psd_source.value, label='音源 (30m)', linewidth=2)
plt.loglog(psd_east.frequencies.value, psd_east.value, label='東 (70m)', alpha=0.7)
plt.axvline(source_freq, color='r', linestyle='--', label=f'音源周波数 ({source_freq} Hz)')
plt.xlabel('周波数 (Hz)')
plt.ylabel('PSD [(m/s)²/Hz]')
plt.title('位置別のパワースペクトル密度')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.xlim(1, sample_rate.value/2)
plt.tight_layout()
plt.show()
```

### ステップ 5: 空間FFT → K空間解析

```python
# K空間（波数領域）に変換
field_k = field.fft_space(axes=['x', 'y'])

print(f"K空間フィールドの形状: {field_k.shape}")
print(f"空間領域: {field_k.space_domains}")

# 特定時刻の K空間振幅を抽出
t_idx = len(t) // 2  # 時間範囲の中央
kspace_slice = np.abs(field_k[t_idx, :, :, 0].value.squeeze())

# K空間軸を取得
kx = field_k._axis1_index.value
ky = field_k._axis2_index.value

# K空間振幅をプロット
plt.figure(figsize=(10, 8))
plt.pcolormesh(kx, ky, kspace_slice.T, shading='auto', cmap='hot')
plt.colorbar(label='K空間での|振幅|')
plt.xlabel('kx (rad/m)')
plt.ylabel('ky (rad/m)')
plt.title(f't={t[t_idx].value:.2f}秒での K空間振幅')

# 期待される波長と波数
wavelength = v_propagation / source_freq  # λ = v/f
k_expected = 2 * np.pi / wavelength
circle = plt.Circle((0, 0), k_expected, fill=False, color='cyan',
                    linestyle='--', linewidth=2, label=f'期待される|k|={k_expected:.3f} rad/m')
ax = plt.gca()
ax.add_patch(circle)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()
```

**期待される結果:**
- K空間で原点を中心とした円形パターン
- 半径は波長 λ=v/f に対応

## ワークフロー 2: HHT を用いた多モード振動解析

### 科学的目標

以下を使用して多モード振動を解析:
1. VectorField で 3次元変位
2. HHT でモード分解
3. 固有モード解析

### セットアップ

```python
# 3成分変位フィールドの作成
# 2つの振動モードをシミュレート: 5 Hz と 12 Hz

mode1_freq = 5.0  # Hz
mode2_freq = 12.0  # Hz

# モード 1: X方向が支配的
dx_mode1 = np.sin(2 * np.pi * mode1_freq * t.value)[:, None, None, None] * np.ones((1, nx, ny, nz))
dy_mode1 = np.sin(2 * np.pi * mode1_freq * t.value * 0.3)[:, None, None, None] * np.ones((1, nx, ny, nz))

# モード 2: Y方向が支配的
dx_mode2 = np.sin(2 * np.pi * mode2_freq * t.value * 0.2)[:, None, None, None] * np.ones((1, nx, ny, nz))
dy_mode2 = np.sin(2 * np.pi * mode2_freq * t.value)[:, None, None, None] * np.ones((1, nx, ny, nz))

# 結合 + ノイズ
dx_total = (dx_mode1 * 1.0 + dx_mode2 * 0.5 +
            np.random.randn(nt, nx, ny, nz) * 0.1)
dy_total = (dy_mode1 * 0.3 + dy_mode2 * 1.0 +
            np.random.randn(nt, nx, ny, nz) * 0.1)
dz_total = np.random.randn(nt, nx, ny, nz) * 0.05  # 最小限のz方向運動

# ScalarField 成分を作成
from gwexpy.fields import VectorField

field_dx = ScalarField(dx_total, unit=u.mm, axis0=t, axis1=x, axis2=y, axis3=z,
                      axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')
field_dy = ScalarField(dy_total, unit=u.mm, axis0=t, axis1=x, axis2=y, axis3=z,
                      axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')
field_dz = ScalarField(dz_total, unit=u.mm, axis0=t, axis1=x, axis2=y, axis3=z,
                      axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')

# VectorField の作成
displacement = VectorField({'x': field_dx, 'y': field_dy, 'z': field_dz})

print(f"変位ベクトルフィールド成分: {list(displacement.keys())}")
```

### 各成分に HHT を適用

```python
# 中心点で時系列を抽出
center_x = displacement['x'].at(x=50*u.m, y=50*u.m, z=0*u.m).squeeze()
center_y = displacement['y'].at(x=50*u.m, y=50*u.m, z=0*u.m).squeeze()

# 各成分に EMD を適用
try:
    imfs_x = center_x.emd(method='emd', max_imf=3)
    imfs_y = center_y.emd(method='emd', max_imf=3)

    print(f"X成分 IMF: {list(imfs_x.keys())}")
    print(f"Y成分 IMF: {list(imfs_y.keys())}")

    # 瞬時周波数を計算
    inst_freq_x_imf0 = imfs_x['IMF0'].instantaneous_frequency()
    inst_freq_y_imf0 = imfs_y['IMF0'].instantaneous_frequency()

    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # X成分
    axes[0, 0].plot(center_x.times.value, center_x.value, linewidth=0.5)
    axes[0, 0].set_ylabel('X変位 (mm)')
    axes[0, 0].set_title('X成分時系列')
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(inst_freq_x_imf0.times.value, inst_freq_x_imf0.value)
    axes[1, 0].axhline(mode1_freq, color='r', linestyle='--', label=f'モード 1 ({mode1_freq} Hz)')
    axes[1, 0].set_ylabel('周波数 (Hz)')
    axes[1, 0].set_xlabel('時間 (s)')
    axes[1, 0].set_title('X成分瞬時周波数')
    axes[1, 0].set_ylim(0, 20)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Y成分
    axes[0, 1].plot(center_y.times.value, center_y.value, linewidth=0.5, color='orange')
    axes[0, 1].set_ylabel('Y変位 (mm)')
    axes[0, 1].set_title('Y成分時系列')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(inst_freq_y_imf0.times.value, inst_freq_y_imf0.value, color='orange')
    axes[1, 1].axhline(mode2_freq, color='r', linestyle='--', label=f'モード 2 ({mode2_freq} Hz)')
    axes[1, 1].set_ylabel('周波数 (Hz)')
    axes[1, 1].set_xlabel('時間 (s)')
    axes[1, 1].set_title('Y成分瞬時周波数')
    axes[1, 1].set_ylim(0, 20)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

except ImportError:
    print("PyEMD がインストールされていません。HHT 解析をスキップします")
```

**期待される結果:**
- X成分の HHT が ~5 Hz モードを捕捉
- Y成分の HHT が ~12 Hz モードを捕捉
- VectorField への成分ごとの高度な解析を実証

## 主要な統合パターン

### パターン 1: Field → TimeSeries → 高度手法 → Field へ戻る

```python
# 抽出 → 処理 → 再構成 ワークフロー
# 1. Field から TimeSeries を抽出
ts = field.at(x=30*u.m, y=50*u.m, z=0*u.m).squeeze()

# 2. 高度手法を適用（例: ホワイトニング）
from gwexpy.signal.preprocessing.whitening import WhiteningModel
whitener = WhiteningModel(fftlength=2, overlap=1)
whitener.fit(ts)
ts_whitened = whitener.transform(ts)

# 3. 処理されたデータで Field を再構成（概念的）
# 注: 完全な再構成にはすべての空間点の処理が必要
print("パターン 1: Field → TimeSeries → 処理 → 結果")
```

### パターン 2: Field FFT → K空間解析 → 物理的解釈

```python
# 変換 → 解析 → 解釈 ワークフロー
# 1. K空間へ変換
field_k = field.fft_space()

# 2. 支配的な波数を抽出
kx_values = field_k._axis1_index.value
ky_values = field_k._axis2_index.value
# K空間でピークを見つける
kspace_power = np.sum(np.abs(field_k.value)**2, axis=0)  # 時間方向に合計
peak_idx = np.unravel_index(np.argmax(kspace_power), kspace_power.shape[:-1])
peak_kx = kx_values[peak_idx[0]]
peak_ky = kx_values[peak_idx[1]]

# 3. 物理的解釈
wavelength_x = 2 * np.pi / abs(peak_kx) if peak_kx != 0 else np.inf
print(f"パターン 2: 支配的波長 ≈ {wavelength_x:.2f} m")
```

### パターン 3: 多フィールド相関

```python
# 複数のフィールド/成分を比較
# 1. 成分を抽出
comp_x = displacement['x'].at(x=50*u.m, y=50*u.m, z=0*u.m).squeeze()
comp_y = displacement['y'].at(x=50*u.m, y=50*u.m, z=0*u.m).squeeze()

# 2. 相互相関
xcorr = comp_x.xcorr(comp_y, maxlag=0.5)

# 3. 位相関係を発見
peak_idx = np.argmax(np.abs(xcorr.value))
lag_at_peak = xcorr.times.value[peak_idx] - xcorr.t0.value
print(f"パターン 3: X と Y の位相差: {lag_at_peak:.3f}秒")
```

## まとめ: 統合の利点

| 単独使用 | 統合ワークフロー | 利点 |
|------------|---------------------|---------|
| ScalarField 単独 | Field + 相互相関 | 空間伝播解析 |
| VectorField 単独 | Vector + 成分ごとの HHT | モード固有の分解 |
| TimeSeries + FFT | Field + 空間FFT | 波数ベクトル特定 |
| 行列解析 | Field → Matrix → 固有モード | 空間ノイズモード |

## ベストプラクティス

1. **Field 構造で開始** - データが本質的に4次元の場合
2. **TimeSeries を抽出** - 高度な1次元手法（HHT、ML）用
3. **Matrix 演算を使用** - 多チャンネル相関用
4. **K空間に変換** - 波動伝播解析用
5. **結果を結合** - 時間、周波数、波数領域にわたって

---

**関連項目:**
- [ScalarField チュートリアル](field_scalar_intro.ipynb)
- [VectorField チュートリアル](field_vector_intro.md)
- [HHT チュートリアル](advanced_hht.ipynb)
- [線形代数チュートリアル](advanced_linear_algebra.md)
- [TimeSeriesMatrix チュートリアル](matrix_timeseries.ipynb)
