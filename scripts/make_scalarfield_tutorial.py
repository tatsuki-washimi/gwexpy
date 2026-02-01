#!/usr/bin/env python3
"""Generate intro_ScalarField.ipynb tutorial notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

# Title and Introduction
nb.cells.append(
    nbf.v4.new_markdown_cell("""# ScalarField クラス入門

このノートブックでは、`gwexpy` の `ScalarField` クラスの基本的な使い方を学びます。

## ScalarField とは？

`ScalarField` は、時間と3次元空間の4次元データを扱うための特殊なクラスです。物理場の時空間構造を表現し、以下の機能を提供します：

- **軸0（時間軸）**: 時間ドメイン ↔ 周波数ドメインの変換
- **軸1-3（空間軸）**: 実空間 ↔ K空間（波数空間）の変換
- **4D構造の保持**: スライシングしても常に4次元を維持
- **バッチ操作**: `FieldList` と `FieldDict` による複数フィールドの一括処理
""")
)

# Setup
nb.cells.append(
    nbf.v4.new_markdown_cell("""## セットアップ

必要なライブラリをインポートします。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

from gwexpy.fields import ScalarField, FieldList, FieldDict

# 再現性のためのシード設定
np.random.seed(42)""")
)

# Section 1: Initialization
nb.cells.append(
    nbf.v4.new_markdown_cell("""## 1. ScalarField の初期化とメタデータ

`ScalarField` オブジェクトを作成し、そのメタデータを確認します。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 4Dデータの作成（10時点 × 8×8×8 の空間グリッド）
nt, nx, ny, nz = 10, 8, 8, 8
data = np.random.randn(nt, nx, ny, nz)

# 軸座標の定義
t = np.arange(nt) * 0.1 * u.s
x = np.arange(nx) * 0.5 * u.m
y = np.arange(ny) * 0.5 * u.m
z = np.arange(nz) * 0.5 * u.m

# ScalarField オブジェクトの作成
field = ScalarField(
    data,
    unit=u.V,
    axis0=t,
    axis1=x,
    axis2=y,
    axis3=z,
    axis_names=["t", "x", "y", "z"],
    axis0_domain="time",
    space_domain="real"
)

print(f"Shape: {field.shape}")
print(f"Unit: {field.unit}")
print(f"Axis names: {field.axis_names}")
print(f"Axis0 domain: {field.axis0_domain}")
print(f"Space domains: {field.space_domains}")""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell("""### メタデータの確認

- `axis0_domain`: 軸0のドメイン（"time" または "frequency"）
- `space_domains`: 各空間軸のドメイン（"real" または "k"）
- `axis_names`: 各軸の名前

これらのメタデータは、FFT変換時に自動的に更新されます。
""")
)

# Section 2: Slicing
nb.cells.append(
    nbf.v4.new_markdown_cell("""## 2. 4D構造を保持するスライシング

`ScalarField` の重要な特徴は、**スライシングしても常に4次元を維持する**ことです。
整数インデックスを使っても、自動的に長さ1のスライスに変換されます。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 整数インデックスでスライス（通常のndarrayなら3Dになる）
sliced = field[0, :, :, :]

print(f"Original shape: {field.shape}")
print(f"Sliced shape: {sliced.shape}")  # (1, 8, 8, 8) - 4Dを維持！
print(f"Type: {type(sliced)}")  # ScalarField のまま
print(f"Axis names preserved: {sliced.axis_names}")""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 複数の軸で整数インデックスを使用
sliced_multi = field[0, 1, 2, 3]

print(f"Multi-sliced shape: {sliced_multi.shape}")  # (1, 1, 1, 1) - やはり4D
print(f"Value: {sliced_multi.value}")""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell("""この挙動により、ScalarFieldオブジェクトの一貫性が保たれ、メタデータ（軸名やドメイン情報）が失われることがありません。
""")
)

# Section 3: Time-Frequency Transform
nb.cells.append(
    nbf.v4.new_markdown_cell("""## 3. 時間-周波数変換（軸0のFFT）

`fft_time()` と `ifft_time()` メソッドを使って、時間軸を周波数軸に変換できます。
GWpy の `TimeSeries.fft()` と同じ正規化を採用しています。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 時間ドメインのScalarFieldを作成（正弦波）
t_dense = np.arange(128) * 0.01 * u.s
x_small = np.arange(4) * 1.0 * u.m
signal_freq = 10.0  # Hz

# 10 Hz の正弦波を空間的に均一に配置
data_signal = np.sin(2 * np.pi * signal_freq * t_dense.value)[:, None, None, None]
data_signal = np.tile(data_signal, (1, 4, 4, 4))

field_time = ScalarField(
    data_signal,
    unit=u.V,
    axis0=t_dense,
    axis1=x_small,
    axis2=x_small.copy(),
    axis3=x_small.copy(),
    axis_names=["t", "x", "y", "z"],
    axis0_domain="time",
    space_domain="real"
)

# FFT実行
field_freq = field_time.fft_time()

print(f"Time domain shape: {field_time.shape}")
print(f"Frequency domain shape: {field_freq.shape}")
print(f"Axis0 domain changed: {field_time.axis0_domain} → {field_freq.axis0_domain}")""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 周波数スペクトルをプロット（1点のx,y,zを選択）
spectrum = np.abs(field_freq[:, 0, 0, 0].value)
freqs = field_freq._axis0_index.value

plt.figure(figsize=(10, 4))
plt.plot(freqs, spectrum, 'b-', linewidth=2)
plt.axvline(signal_freq, color='r', linestyle='--', label=f'{signal_freq} Hz (入力信号)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [V]')
plt.title('FFT Spectrum (ScalarField)')
plt.xlim(0, 50)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ピーク周波数を確認
peak_idx = np.argmax(spectrum)
peak_freq = freqs[peak_idx]
print(f"Peak frequency: {peak_freq:.2f} Hz (expected: {signal_freq} Hz)")""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell("""### 逆FFT（周波数 → 時間）

`ifft_time()` で元の時間ドメインに戻すことができます。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 逆FFT
field_reconstructed = field_freq.ifft_time()

# 元の信号と比較
original = field_time[:, 0, 0, 0].value
reconstructed = field_reconstructed[:, 0, 0, 0].value

plt.figure(figsize=(10, 4))
plt.plot(t_dense.value, original, 'b-', label='Original', alpha=0.7)
plt.plot(t_dense.value, reconstructed.real, 'r--', label='Reconstructed', alpha=0.7)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.title('IFFT: Frequency → Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 誤差を確認
error = np.abs(original - reconstructed.real)
print(f"Max reconstruction error: {np.max(error):.2e} V")""")
)

# Section 4: Spatial FFT
nb.cells.append(
    nbf.v4.new_markdown_cell("""## 4. 実空間-K空間変換（空間軸のFFT）

`fft_space()` と `ifft_space()` を使って、空間軸を波数空間（K空間）に変換できます。
角波数 k = 2π / λ の符号付きFFTを使用します。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 空間的に周期構造を持つデータを作成
nx, ny, nz = 16, 16, 16
x_grid = np.arange(nx) * 0.5 * u.m
y_grid = np.arange(ny) * 0.5 * u.m
z_grid = np.arange(nz) * 0.5 * u.m

# X方向に波長 4m の正弦波
wavelength = 4.0  # m
k_expected = 2 * np.pi / wavelength  # rad/m

data_spatial = np.sin(2 * np.pi * x_grid.value / wavelength)[None, :, None, None]
data_spatial = np.tile(data_spatial, (4, 1, ny, nz))

field_real = ScalarField(
    data_spatial,
    unit=u.V,
    axis0=np.arange(4) * 0.1 * u.s,
    axis1=x_grid,
    axis2=y_grid,
    axis3=z_grid,
    axis_names=["t", "x", "y", "z"],
    axis0_domain="time",
    space_domain="real"
)

# X軸のみFFT
field_kx = field_real.fft_space(axes=["x"])

print(f"Original space domains: {field_real.space_domains}")
print(f"After fft_space: {field_kx.space_domains}")
print(f"Axis names: {field_kx.axis_names}")""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# K空間スペクトルをプロット
kx_spectrum = np.abs(field_kx[0, :, 0, 0].value)
kx_values = field_kx._axis1_index.value

plt.figure(figsize=(10, 4))
plt.plot(kx_values, kx_spectrum, 'b-', linewidth=2)
plt.axvline(k_expected, color='r', linestyle='--', label=f'k = {k_expected:.2f} rad/m')
plt.axvline(-k_expected, color='r', linestyle='--')
plt.xlabel('Wavenumber kx [rad/m]')
plt.ylabel('Amplitude')
plt.title('Spatial FFT: Real → K space')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ピーク波数を確認
peak_idx = np.argmax(kx_spectrum)
peak_k = kx_values[peak_idx]
print(f"Peak wavenumber: {peak_k:.2f} rad/m (expected: ±{k_expected:.2f} rad/m)")""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell("""### 波長の計算

K空間では、`wavelength()` メソッドで波長を計算できます。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 波長を計算
wavelengths = field_kx.wavelength("kx")

print(f"Wavelength at k={k_expected:.2f}: {2*np.pi/k_expected:.2f} m")
print(f"Calculated wavelengths range: {wavelengths.value[wavelengths.value > 0].min():.2f} - {wavelengths.value[wavelengths.value < np.inf].max():.2f} m")""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell("""### 全空間軸のFFT

`axes` パラメータを省略すると、全空間軸をまとめてFFTできます。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 全空間軸をFFT
field_k_all = field_real.fft_space()
print(f"All spatial axes in K space: {field_k_all.space_domains}")

# 逆FFTで元に戻す
field_real_back = field_k_all.ifft_space()
print(f"Back to real space: {field_real_back.space_domains}")

# 再構成誤差
reconstruction_error = np.max(np.abs(field_real.value - field_real_back.value))
print(f"Max reconstruction error: {reconstruction_error:.2e}")""")
)

# Section 5: Collections
nb.cells.append(
    nbf.v4.new_markdown_cell("""## 5. FieldList と FieldDict によるバッチ操作

複数の `ScalarField` オブジェクトをまとめて処理するには、`FieldList` または `FieldDict` を使用します。
""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell("""### FieldList

リスト形式で複数のフィールドを管理し、一括でFFT操作を適用できます。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 3つの異なる振幅を持つScalarFieldを作成
amplitudes = [1.0, 2.0, 3.0]
fields = []

for amp in amplitudes:
    data_temp = amp * np.random.randn(8, 4, 4, 4)
    field_temp = ScalarField(
        data_temp,
        unit=u.V,
        axis0=np.arange(8) * 0.1 * u.s,
        axis1=np.arange(4) * 0.5 * u.m,
        axis2=np.arange(4) * 0.5 * u.m,
        axis3=np.arange(4) * 0.5 * u.m,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
        space_domain="real"
    )
    fields.append(field_temp)

# FieldList を作成
field_list = FieldList(fields, validate=True)
print(f"Number of fields: {len(field_list)}")""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 一括で時間FFTを実行
field_list_freq = field_list.fft_time_all()

print(f"All fields transformed to frequency domain:")
for i, field in enumerate(field_list_freq):
    print(f"  Field {i}: axis0_domain = {field.axis0_domain}")""")
)

nb.cells.append(
    nbf.v4.new_markdown_cell("""### FieldDict

辞書形式で名前付きフィールドを管理します。
""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 名前付きフィールドの辞書を作成
field_dict = FieldDict({
    "channel_A": fields[0],
    "channel_B": fields[1],
    "channel_C": fields[2]
}, validate=True)

print(f"Field names: {list(field_dict.keys())}")""")
)

nb.cells.append(
    nbf.v4.new_code_cell("""# 一括で空間FFTを実行
field_dict_k = field_dict.fft_space_all(axes=["x", "y"])

for name, field in field_dict_k.items():
    print(f"{name}: {field.space_domains}")""")
)

# Summary
nb.cells.append(
    nbf.v4.new_markdown_cell("""## まとめ

このノートブックでは、`ScalarField` の主要な機能を学びました：

1. **初期化とメタデータ**: 軸情報とドメイン情報の管理
2. **4D保持スライシング**: 整数インデックスでも4D構造を維持
3. **時間-周波数変換**: `fft_time()` / `ifft_time()` による軸0のFFT
4. **実空間-K空間変換**: `fft_space()` / `ifft_space()` による空間軸のFFT
5. **バッチ操作**: `FieldList` と `FieldDict` による複数フィールドの一括処理

これらの機能を組み合わせることで、時空間データの高度な解析が可能になります。

### 次のステップ

- より複雑な物理シミュレーションへの応用
- 実データとの組み合わせ
- カスタム解析パイプラインの構築
""")
)

# Write notebook
with open(
    "/home/washimi/work/gwexpy/examples/tutorials/intro_ScalarField.ipynb", "w"
) as f:
    nbf.write(nb, f)

print("✓ Created: examples/tutorials/intro_ScalarField.ipynb")
