# FrequencySeriesDict

<!-- reference-summary:start -->

**安定性:** Stable

## 主な用途

`FrequencySeriesDict` は複数の `FrequencySeries` をラベル付きで保持し、一括処理や変換を行うために使います。

## 代表的なシグネチャ

```python
FrequencySeriesDict(data: dict[str, FrequencySeries])
FrequencySeriesDict.to_matrix()
```

## 最小例

```python
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
import numpy as np

dct = FrequencySeriesDict({
    "A": FrequencySeries(np.ones(64), df=1.0),
    "B": FrequencySeries(np.ones(64), df=1.0),
})
mat = dct.to_matrix()
```

## 関連理論

- [FFT_Conventions](FFT_Conventions.md)
- [FrequencySeries](FrequencySeries.md)
- [FrequencySeriesMatrix](FrequencySeriesMatrix.md)

## 関連チュートリアル

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- [FrequencySeries チュートリアル](../user_guide/tutorials/intro_frequencyseries.ipynb)
- [伝達関数計測](../user_guide/tutorials/case_transfer_function.ipynb)
- [ノイズバジェット解析](../user_guide/tutorials/case_noise_budget.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** FrequencySeriesBaseDict

ラベルをキーとする `FrequencySeries` オブジェクトの順序付きマッピング。

## 物理コンテキスト

`FrequencySeriesDict` は、ラベルそのものが物理的な識別子である場合に向いています。チャンネル名、センサ位置、構成タグ、処理分岐などをキーとして保持したいときに使います。

- list と違ってキー自体が解析記録の一部になります
- ラベルを失うと比較の意味が薄れる多センサ解析で有用です

## よくある誤読

1. キーが違うだけで較正や整列条件まで違うと勝手に思い込む
2. 出力時にサニタイズされたキーを元の物理チャンネル名だとみなす
3. キーだけで比較し、単位や周波数間隔を確認しない

## どのページへ進むか

- 各スペクトルの解釈: [FrequencySeries](FrequencySeries.md)
- 整列済み解析グリッドへの変換: [FrequencySeriesMatrix](FrequencySeriesMatrix.md)
- 実務ワークフロー: [伝達関数計測](../user_guide/tutorials/case_transfer_function.ipynb), [ノイズバジェット解析](../user_guide/tutorials/case_noise_budget.ipynb)

## メソッド

### `__init__`

```python
__init__(self, *args: 'Any', **kwargs: 'Any')
```

self を初期化します。

*(OrderedDict から継承)*

### `EntryClass`

```python
EntryClass(data, unit=None, f0=None, df=None, frequencies=None, name=None, epoch=None, channel=None, **kwargs)
```

互換性と将来の拡張のための gwpy の FrequencySeries の軽量ラッパー。

### `angle`

```python
angle(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

phase() のエイリアス。新しい FrequencySeriesDict を返します。

### `crop`

```python
crop(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

辞書内の各 FrequencySeries をクロップします。その場で操作（GWpy 互換）。self を返します。

### `degree`

```python
degree(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries の位相（度単位）を計算します。新しい FrequencySeriesDict を返します。

### `differentiate_time` / `integrate_time`

周波数領域での時間微分/積分を各アイテムに適用します。

### `filter`

```python
filter(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries にフィルタを適用します。新しい FrequencySeriesDict を返します。

### `group_delay`

```python
group_delay(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各アイテムの群遅延を計算します。

### `ifft`

```python
ifft(self, *args, **kwargs)
```

各 FrequencySeries の IFFT を計算します。TimeSeriesDict を返します。

### `interpolate`

```python
interpolate(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

辞書内の各 FrequencySeries を補間します。

### `phase`

```python
phase(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries の位相を計算します。

### `plot`

```python
plot(self, label: 'str' = 'key', method: 'str' = 'plot', figsize: 'Optional[Any]' = None, **kwargs: 'Any')
```

データをプロットします。

パラメータ
----------

label : str, optional
    ラベル付け方法: ``'key'``（辞書キーを使用）または ``'name'``（各アイテムの name 属性を使用）
method : str, optional
    `:class:~gwpy.plot.Plot` の呼び出しメソッド。デフォルト: ``'plot'``

### `write`

```python
write(self, target: str, *args: Any, **kwargs: Any) -> Any
```

FrequencySeriesDict をファイルに書き込みます。

HDF5 出力では `layout` を指定できます（デフォルトは GWpy 互換の dataset-per-entry）。

```python
fsd.write("out.h5", format="hdf5")               # GWpy互換（既定）
fsd.write("out.h5", format="hdf5", layout="group")  # 旧形式（group-per-entry）
```

HDF5 のデータセット名（GWpy の `path=` 用）:

- キーは HDF5 で安全な名前にサニタイズされます（例: `H1:ASD` -> `H1_ASD`）。
- サニタイズ後の名前が衝突する場合、`__1` のようなサフィックスが付与されます。
- 元のキーはファイル属性に保存され、gwexpy の `read()` は元キーを復元します。

:::{admonition} warning
:class: warning

信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。
:::

pickle 可搬性メモ: gwexpy の `FrequencySeriesDict` は unpickle 時に builtins の `dict` を返します
（中身は GWpy の `FrequencySeries`、読み込み側に gwexpy は不要です）。

### `smooth`

```python
smooth(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries を平滑化します。

### `to_db`

```python
to_db(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries を dB に変換します。

### `to_matrix`

```python
to_matrix(self)
```

この FrequencySeriesDict を FrequencySeriesMatrix (Nx1) に変換します。

### `to_pandas` / `to_xarray`

pandas.DataFrame / xarray.Dataset に変換します。キーは列/データ変数になります。

### `to_cupy` / `to_jax` / `to_tensorflow` / `to_torch`

各アイテムを対応するフレームワークのテンソル/配列に変換します。

### `write`

```python
write(self, target: 'str', *args: 'Any', **kwargs: 'Any') -> 'Any'
```

辞書をファイル（HDF5, ROOT など）に書き込みます。

### `zpk`

```python
zpk(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

各 FrequencySeries に ZPK フィルタを適用します。
