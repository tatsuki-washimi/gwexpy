# ScalarField

**継承:** `FieldBase`, `Array4D`, `AxisApiMixin`, `StatisticalMethodsMixin`, `GwpyArray`

明示的なドメインメタデータを持つ4次元スカラー場クラスです。軸構造は常に `(axis0, x, y, z)` を保持し、スライスしても軸は削除されず長さ1のまま残ります。時間/周波数と位置/波数のドメインを軸ごとに保持し、FFT操作後もドメインと単位の整合性を検証します。

## 特徴

- **明示的なドメイン保持**: `axis0_domain ∈ {time, frequency}` と空間軸の `space_domain ∈ {real, k}` を明示的に保持。
- **物理整合性を保つスライス**: インデクシング操作を行っても4次元構造が維持され、軸メタデータが欠落しません。
- **単位とドメインの検証**: コンストラクタおよび FFT 操作後にドメインと単位の整合性を自動検証。
- **信号処理機能**: PSD、コヒーレンス、相互相関などの解析機能を内蔵。

## 主要メソッド

### FFT 操作

正規化や符号の定義などの数学的詳細は、[FFTの仕様とコンベンション](FFT_Conventions.md) を参照してください。

#### `fft_time(nfft=None)`

時間軸の FFT を実行し、ドメインを `frequency` に更新。GWpy `TimeSeries.fft()` と同じ正規化（rfft/nfft + 非DC/非Nyquistの倍増）を適用。

#### `ifft_time(nout=None)`

`fft_time` の逆操作。`axis0_domain='time'` に戻し、保存された `_axis0_offset` を用いて時間軸を正確に復元。

#### `fft_space(axes=None, n=None)`

空間軸に対して符号付き両側 FFT を実行。ドメインを `real → k` に、軸名を `x → kx` などに更新。波数は $k = 2\pi/\lambda$ として計算。

#### `ifft_space(axes=None, n=None)`

`k` ドメインの空間軸を実空間に戻し、`real` ドメインを復元。

### 信号処理（インスタンスメソッド）

#### `filter(*args, **kwargs)`

時間軸（axis 0）に対してデジタルフィルタを適用します。`gwpy.signal.filter_design` で設計したフィルタを受け取ります。デフォルトでは位相歪みを防ぐため `filtfilt` を使用します。

#### `resample(rate, **kwargs)`

フィールドを時間軸方向にリサンプリングします。内部的に `scipy.signal.resample` を使用し、すべての軸メタデータと単位を保持します。

### 信号処理解析 (High-level API)

以下の関数が `gwexpy.fields` から利用可能です：

- `compute_psd(field, point_or_region, ...)`: 指定した空間点または領域での Welch PSD 推定。
- `coherence_map(field, ref_point, ...)`: 基準点と断面内の各点とのコヒーレンスマップ。
- `compute_xcorr(field, point_a, point_b, ...)`: 2点間の相互相関関数。
- `time_delay_map(field, ref_point, ...)`: 相関に基づく推定遅延マップの作成。

### デモデータ生成

- `make_demo_scalar_field(...)`: 汎用的な4Dデモフィールドの生成。
- `make_propagating_gaussian(...)`: 伝播するガウス波パケットのシミュレーション。
- `make_sinusoidal_wave(...)`: 平面波のシミュレーション。

## コレクション

複数の ScalarField を扱う場合は `gwexpy.fields.FieldList` / `FieldDict` を利用してください。
