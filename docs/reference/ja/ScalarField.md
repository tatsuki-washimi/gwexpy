# ScalarField

**継承:** FieldBase, Array4D, AxisApiMixin, StatisticalMethodsMixin, Array

明示的なドメインメタデータを持つ4次元スカラー場クラスです。軸構造は常に `(axis0, x, y, z)` を保持し、スライスしても軸は削除されず長さ1のまま残ります。時間/周波数と位置/波数のドメインを軸ごとに保持し、FFT操作後もドメインと単位の整合性を検証します。

**特徴**
- `axis0_domain ∈ {time, frequency}` と空間軸の `space_domain ∈ {real, k}` を明示的に保持。
- コンストラクタおよび FFT 後にドメインと単位の整合性を検証。
- インデクシングは常に4次元を維持し、軸メタデータを保持。

## 主要メソッド

### `fft_time`

時間軸の FFT を実行し、ドメインを `frequency` に更新します。GWpy `TimeSeries.fft()` と同じ正規化（rfft/nfft + 非DC/非Nyquistの倍増）。

### `ifft_time`

`fft_time` の逆操作。`axis0_domain='time'` に戻し、時間軸を復元します。

### `fft_space`

空間軸（任意の部分集合）に対して符号付き両側 FFT を実行。変換した軸のドメインを `real→k` に、軸名を `x→kx` などに更新します。

### `ifft_space`

`k` ドメインの空間軸を実空間に戻し、軸名とドメインを `real` に復元します。

## コレクション

複数の ScalarField を扱う場合は `gwexpy.fields.FieldList` / `FieldDict` を利用してください。
