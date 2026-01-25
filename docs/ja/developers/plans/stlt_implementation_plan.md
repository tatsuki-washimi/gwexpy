# STLT (Short-Time Laplace Transform) 実装計画

## 1. 概要
現在の `TimeSeries.stlt()` 実装は、STFTの振幅の外積（Magnitude Outer Product）を計算しており、Short-Time Laplace Transform (STLT) の定義（実部軸 $\sigma$ を持つ）を満たしていません。
本計画では、真のSTLTを実装し、出力として $\sigma$ (Laplace real part) と周波数軸を持つ3次元データ構造を提供します。

## 2. 現状分析
- **場所**: `gwexpy/timeseries/_resampling.py` (ライン 634-717)
- **問題点**:
  - `TimePlaneTransform` を返すが、内容は `(Time, Freq, Freq)` のテンソル（振幅積）である。
  - $\sigma$ 軸が存在しない。
  - スペクトル変換メソッドであるにも関わらず、`_resampling.py` (ResamplingMixin) に配置されている（構造的な不整合）。

## 3. デザイン提案

### 3.1 移動と配置
`stlt` メソッドはスペクトル変換の一種であるため、`gwexpy/timeseries/_spectral_special.py` (`TimeSeriesSpectralSpecialMixin`) に移動します。
`gwexpy/timeseries/_resampling.py` からは削除します（または、後方互換性のために deprecation warning を出して新しいメソッドを呼ぶラッパーとして残します。ここでは「Minimal API Breakage」を目指すため、既存のインポート構造を維持しつつ、正しいMixinメソッドが優先されるようにします）。
※ `TimeSeries` クラスは `TimeSeriesResamplingMixin` と `TimeSeriesSpectralMixin` の両方を継承していますが、 `_resampling.py` から削除し `_spectral_special.py` に移動することで、より適切なモジュール構成になります。

### 3.2 メソッドシグネチャ
```python
def stlt(
    self,
    stride: str | Quantity,
    window: str | Quantity,
    *,
    sigma: float | Quantity | Sequence = 0.0,
    frequencies: Optional[Sequence | Quantity] = None,
    **kwargs
) -> TimePlaneTransform:
```
- **sigma**: ラプラス変数の実部 $s = \sigma + j\omega$。スカラーまたは配列を受け付ける。デフォルトは 0 (STFTと等価)。
- **stride, window**: 時間単位での指定（文字列 '1s' 等も可）。

### 3.3 実装ロジック
各時間チャンク（ウィンドウ）に対して以下を計算します：
$$ X(t, \sigma, f) = \sum_{n} x[n] \cdot w[n] \cdot e^{-\sigma t_{rel}[n]} \cdot e^{-j 2\pi f t_{rel}[n]} $$
ここで $t_{rel}$ はウィンドウ内の相対時間です。
1. **チャンク分割**: 指定された `stride`, `window` に基づきデータを切り出し。
2. **重み付け**: 各 $\sigma$ に対して、指数減衰窓 $e^{-\sigma t}$ を適用。
3. **FFT**: 重み付けされたデータに対してFFTを実行。 $\sigma=0$ の場合は通常のSTFTとなります。
4. **出力**: (Time, Sigma, Frequency) の3次元配列。

### 3.4 コンテナクラス
既存の `TimePlaneTransform` を再利用します。
- **データ構造**: `Array3D` を内部で使用し、軸名を `["time", "sigma", "frequency"]` と明示的に設定します。
- **理由**: 新しいクラス (`LaplaceGram` 等) を導入するよりも、既存の汎用コンテナを活用する方が変更が最小限で済むため。

## 4. 作業手順

1. **`gwexpy/timeseries/_spectral_special.py` の修正**:
   - `stlt` メソッドを実装します。内部で `scipy.signal.stft` のロジックを参考にするか、または独自のウィンドウ処理ループを実装します（複数の $\sigma$ に対応するため、カスタムループまたは `np.lib.stride_tricks.sliding_window_view` + `rfft` のベクトル化計算が推奨されます）。

2. **`gwexpy/timeseries/_resampling.py` の修正**:
   - 既存の `stlt` 実装を削除します。

3. **テストの更新 (`gwexpy/timeseries/tests/test_p1_delivery.py`)**:
   - `test_stlt_basic` を更新し、戻り値の形状が `(Time, 1, Freq)` (sigma=0 scalarの場合) または `(Time, N_sigma, Freq)` であることを検証するように変更します。
   - `kind="stlt"` とメタデータを確認します。

## 5. 懸念点と対応
- **パフォーマンス**: 多くの $\sigma$ を指定すると計算量が増大します（FFT回数 = チャンク数 $\times$ sigma数）。
  - -> ドキュメントで注意喚起を行い、デフォルトは `sigma=0` とします。
- **メモリ**: 3次元配列は大きくなりやすいため、`TimePlaneTransform` は生成後に不要なデータを保持しないようにします。

以上
