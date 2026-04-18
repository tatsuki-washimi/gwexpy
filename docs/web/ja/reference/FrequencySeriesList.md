# FrequencySeriesList

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`FrequencySeriesList` は複数の `FrequencySeries` をラベル付きで保持し、一括処理や変換を行うために使います。

## 代表的なシグネチャ

```python
FrequencySeriesList(data: list[FrequencySeries])
FrequencySeriesList.to_matrix()
```

## 最小例

```python
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList
import numpy as np

lst = FrequencySeriesList([FrequencySeries(np.ones(64), df=1.0)])
mat = lst.to_matrix()
```

## 関連理論

- [FFT_Conventions](FFT_Conventions.md)
- [FrequencySeries](FrequencySeries.md)
- [FrequencySeriesMatrix](FrequencySeriesMatrix.md)

## 関連チュートリアル

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- [FrequencySeries チュートリアル](../user_guide/tutorials/intro_frequencyseries.ipynb)
- [FrequencySeries 行列チュートリアル](../user_guide/tutorials/matrix_frequencyseries.ipynb)
- [伝達関数計測](../user_guide/tutorials/case_transfer_function.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** FrequencySeriesBaseList

`FrequencySeries` オブジェクトのリスト。

## 物理コンテキスト

`FrequencySeriesList` は、複数のスペクトルをまとめて処理したいが、各系列の独立した意味は残したい場合に使います。繰り返し測定、複数センサ、同一チャネルの別処理系などが典型例です。

- 各要素は個別のメタデータやチャンネル名を保持できます
- 一括メソッドは便利ですが、単位・`df`・較正状態の一致までは保証しません

## よくある誤読

1. 同じリストに入っているだけで全要素が周波数整列していると思い込む
2. 単位や正規化の違うスペクトルを重ね描きしてそのまま物理解釈する
3. 名前やラベルを付けずに、リスト順だけを provenance だとみなす

## どのページへ進むか

- 各スペクトルの解釈: [FrequencySeries](FrequencySeries.md)
- 整列済み一括解析: [FrequencySeriesMatrix](FrequencySeriesMatrix.md)
- 実務ワークフロー: [伝達関数計測](../user_guide/tutorials/case_transfer_function.ipynb)

## メソッド

### `__init__`

```python
__init__(self, *items: 'Union[_FS, Iterable[_FS]]')
```

self を初期化します。

*(list から継承)*

### `EntryClass`

```python
EntryClass(data, unit=None, f0=None, df=None, frequencies=None, name=None, epoch=None, channel=None, **kwargs)
```

互換性と将来の拡張のための gwpy の FrequencySeries の軽量ラッパー。

### `angle`

```python
angle(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

phase() のエイリアス。新しい FrequencySeriesList を返します。

### `apply_response`

```python
apply_response(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

リスト内の各 FrequencySeries に応答を適用します。

### `crop`

```python
crop(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

リスト内の各 FrequencySeries をクロップします。

### `degree`

```python
degree(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

各 FrequencySeries の位相（度単位）を計算します。

### `differentiate_time` / `integrate_time`

周波数領域での時間微分/積分を各アイテムに適用します。

### `filter`

```python
filter(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

リスト内の各 FrequencySeries にフィルタを適用します。

### `group_delay`

```python
group_delay(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

各アイテムの群遅延を計算します。

### `ifft`

```python
ifft(self, *args, **kwargs)
```

各 FrequencySeries の IFFT を計算します。TimeSeriesList を返します。

### `interpolate`

```python
interpolate(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

リスト内の各 FrequencySeries を補間します。

### `phase`

```python
phase(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

各 FrequencySeries の位相を計算します。

### `plot`

```python
plot(self, **kwargs: 'Any')
```

すべてのシリーズをプロットします。gwexpy.plot.Plot に委譲します。

### `write`

```python
write(self, target: str, *args: Any, **kwargs: Any) -> Any
```

FrequencySeriesList をファイルに書き込みます。

HDF5 出力では `layout` を指定できます（デフォルトは GWpy 互換の dataset-per-entry）。

```python
fsl.write("out.h5", format="hdf5")               # GWpy互換（既定）
fsl.write("out.h5", format="hdf5", layout="group")  # 旧形式（group-per-entry）
```

:::{admonition} warning
:class: warning

信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。
:::

pickle 可搬性メモ: gwexpy の `FrequencySeriesList` は unpickle 時に builtins の `list` を返します
（中身は GWpy の `FrequencySeries`、読み込み側に gwexpy は不要です）。

### `smooth`

```python
smooth(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

各 FrequencySeries を平滑化します。

### `to_db`

```python
to_db(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

各 FrequencySeries を dB に変換します。

### `to_pandas` / `to_xarray`

pandas.DataFrame / xarray.DataArray に変換します。

### `to_cupy` / `to_jax` / `to_tensorflow` / `to_torch` / `to_control_frd`

各アイテムを対応するフレームワークのオブジェクトに変換します。

### `write`

```python
write(self, target: 'str', *args: 'Any', **kwargs: 'Any') -> 'Any'
```

リストをファイル（HDF5, ROOT など）に書き込みます。

### `zpk`

```python
zpk(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

リスト内の各 FrequencySeries に ZPK フィルタを適用します。
