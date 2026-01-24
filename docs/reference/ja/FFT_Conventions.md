# FFTの仕様とコンベンション

このドキュメントでは、`gwexpy` における高速フーリエ変換（FFT）の数学的定義、正規化（Normalization）、および符号の定義について詳述します。

## 1. 時間方向のFFT (`fft_time`)

時間方向のFFTは時間軸（軸0）に対して適用されます。これは `gwpy` および標準的な重力波データ解析の手法との互換性を重視して設計されています。

### 正規化 (Normalization)
順変換において **$1/N$ 正規化** が適用されます。これにより、正弦波 $A \sin(\omega t)$ の振幅は、得られた振幅スペクトル上で $A$ として現れます（片側スペクトルの補正係数考慮後）。
逆変換（`ifft_time`）はこの正規化を解除します。

### スペクトルの定義
- **タイプ**: 片側スペクトル (`rfft`)。非負の周波数成分のみが返されます。
- **係数補正**: 全パワーを維持するため、DC（直流）成分とナイキスト成分を除くすべてのビンが 2 倍されます。
- **周波数軸**: $f = [0, 1/2dt]$。

### 符号の定義 (Sign Convention)
- **順変換 (Forward)**: $X[k] = \frac{1}{N} \sum_{n=0}^{N-1} x[n] e^{-i 2\pi k n / N}$
- **逆変換 (Inverse)**: $x[n] = \sum_{k=0}^{N/2} X[k] e^{i 2\pi k n / N}$ （ビンの重み付けを考慮）

---

## 2. 空間方向のFFT (`fft_space`)

空間方向のFFTは空間軸（軸1, 2, 3）に対して適用されます。多次元格子上の物理フィールド解析向けに設計されています。

### 正規化 (Normalization)
**正規化なし**。順変換では係数の適用を行わず、標準的な `numpy.fft.fftn` の挙動に従います。
逆変換（`ifft_space`）において $1/N$ 正規化が適用されます。

### スペクトルの定義
- **タイプ**: 両側スペクトル。正負両方の波数成分が返されます。
- **波数軸**: デフォルトでは **角波数 (Angular wavenumber)** $k = 2\pi / \lambda$ が生成されます。
- **配列の順序**: ゼロ周波数（DC）成分がインデックス 0 に配置されます（fftshiftは行われません）。可視化の際に中心化が必要な場合は、外部で `numpy.fft.fftshift` を使用してください。

### 符号の定義 (Sign Convention)
- **順変換 (Forward)**: $X[\mathbf{k}] = \sum_{\mathbf{n}} x[\mathbf{n}] e^{-i 2\pi \sum k_j n_j / N_j}$
- **逆変換 (Inverse)**: $x[n] = \frac{1}{\prod N_j} \sum_{\mathbf{k}} X[\mathbf{k}] e^{i 2\pi \sum k_j n_j / N_j}$

---

## 3. スペクトル密度 (`spectral_density`, `compute_psd`)

`spectral_density` メソッドは、Welch法（平均累加ピリオドグラム）を用いたパワースペクトル密度（PSD）推定を提供します。

### スケーリング (Scaling)
- **Density**: 単位周波数/波数分解能あたりのパワーを返します（$V^2 / \text{Hz}$ または $V^2 / [\text{unit}^{-1}]$）。
- **Spectrum**: 各ビンにおけるトータルのパワーを返します（$V^2$）。

### 波数に関する注意
`fft_space` が角波数 ($k = 2\pi/\lambda$) を使用するのに対し、`spectral_density(axis='x')` はデフォルトで **角速度ではない波数** ($f_k = 1/\lambda$) を生成します（`scipy.signal.welch` の仕様に準拠）。
角波数に変換する場合は、得られた軸に $2\pi$ を乗じてください。

---

## 4. 正規化一覧表

| メソッド | 正規化 (順変換) | ドメイン | 符号 ($ikx$) |
| :--- | :--- | :--- | :--- |
| `fft_time` | $1/N$ | 片側 (One-sided) | 負 (Negative) |
| `ifft_time` | $N$ | 片側 (One-sided) | 正 (Positive) |
| `fft_space` | $1$ | 両側 (Two-sided) | 負 (Negative) |
| `ifft_space` | $1/N$ | 両側 (Two-sided) | 正 (Positive) |
| `spectral_density` | Welch / $1/N^2$ | 片側 (One-sided) | N/A (絶対値2乗) |
