# TimeSeriesList

**継承元:** PhaseMethodsMixin, TimeSeriesList

TimeSeries オブジェクトのリスト。

## メソッド

### `__init__`

```python
__init__(self, *items)
```

新しいリストを初期化します。

### `EntryClass`

TimeSeries を作成するためのエントリクラス。

### `analytic_signal` / `hilbert`

各アイテムに解析信号変換を適用します。

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

`phase(unwrap=unwrap, deg=deg)` のエイリアス。

### `asd` / `psd`

リスト内の各 TimeSeries の ASD/PSD を計算します。FrequencySeriesList を返します。

### `coalesce`

このリストの連続する要素を単一オブジェクトにマージします。

### `coherence` / `coherence_matrix`

コヒーレンス/コヒーレンス行列を計算します。

### `crop`

```python
crop(self, start=None, end=None, copy=False) -> 'TimeSeriesList'
```

リスト内の各 TimeSeries をクロップします。gwexpy.time.to_gps がサポートするあらゆる時刻形式を受け付けます。

### `csd` / `csd_matrix`

CSD/CSD 行列を計算します。

### `decimate` / `resample`

各 TimeSeries をデシメート/リサンプルします。

### `degree` / `radian`

各アイテムの瞬時位相（度/ラジアン）を計算します。

### `detrend`

各 TimeSeries のトレンドを除去します。

### `envelope`

各アイテムのエンベロープを計算します。

### `fft`

各 TimeSeries に FFT を適用します。FrequencySeriesList を返します。

### `filter` / `notch` / `gate`

各 TimeSeries にフィルタ/ノッチ/ゲートを適用します。

### `heterodyne` / `baseband` / `mix_down` / `lock_in`

各アイテムに信号処理メソッドを適用します。

### `ica` / `pca`

チャンネル間で ICA/PCA 分解を実行します。

### `impute`

```python
impute(self, *, method='interpolate', limit=None, axis='time', max_gap=None, **kwargs)
```

各 TimeSeries の欠損データ（NaN）を補完します。

### `instantaneous_frequency` / `instantaneous_phase`

各アイテムの瞬時周波数/位相を計算します。

### `join`

```python
join(self, pad=None, gap=None)
```

このリストのすべての要素を単一オブジェクトに連結します。

パラメータ
----------
pad : `float`, optional
    ソースデータのギャップを埋める値。デフォルトではギャップは `ValueError` になります。
gap : `str`, optional
    ギャップがある場合の処理: ``'raise'``, ``'ignore'``, ``'pad'``

### `phase`

```python
phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

データの位相を計算します。

### `plot` / `plot_all`

すべてのシリーズをプロットします。

### `write`

```python
write(self, target: str, *args: Any, **kwargs: Any) -> Any
```

TimeSeriesList をファイルに書き込みます。

CSV/TXT 出力は多チャンネル用の「ディレクトリ出力」です（各要素を個別ファイルに保存）。

```python
tsl.write("out_dir", format="txt")  # out_dir/ に series ごとの TXT を保存
```

HDF5 出力では `layout` を指定できます（デフォルトは GWpy 互換の dataset-per-entry）。

```python
tsl.write("out.h5", format="hdf5")               # GWpy互換（既定）
tsl.write("out.h5", format="hdf5", layout="group")  # 旧形式（group-per-entry）
```

> [!WARNING]
> 信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。

pickle 可搬性メモ: gwexpy の `TimeSeriesList` は unpickle 時に **GWpy の `TimeSeriesList`** を返します
（読み込み側に gwexpy は不要です）。

### `q_transform`

各 TimeSeries の Q 変換を計算します。SpectrogramList を返します。

### `rolling_mean` / `rolling_median` / `rolling_std` / `rolling_min` / `rolling_max`

各要素にローリング統計を適用します。

### `spectrogram` / `spectrogram2`

各 TimeSeries のスペクトログラムを計算します。SpectrogramList を返します。

### `stlt`

各アイテムに STLT を適用します。TimePlaneTransform のリストを返します。

### `to_matrix`

```python
to_matrix(self, *, align='intersection', **kwargs)
```

リストをアライメント付きで TimeSeriesMatrix に変換します。

パラメータ
----------
align : str, optional
    アライメント戦略（'intersection', 'union' など）。デフォルトは 'intersection'。

### `to_pandas`

pandas DataFrame に変換します。各要素は列になります。共通の時間軸を前提とします。

### `write`

```python
write(self, target: str, *args: Any, **kwargs: Any) -> Any
```

TimeSeriesList をファイル（HDF5, ROOT など）に書き込みます。
