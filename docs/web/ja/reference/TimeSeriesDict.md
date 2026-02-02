# TimeSeriesDict

**継承元:** PhaseMethodsMixin, TimeSeriesDict

TimeSeries オブジェクトの辞書。

## メソッド

### `EntryClass`

TimeSeries を作成するためのエントリクラス。

### `analytic_signal` / `hilbert`

各アイテムに解析信号/ヒルベルト変換を適用します。

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

`phase(unwrap=unwrap, deg=deg)` のエイリアス。

### `append`

```python
append(self, other, copy=True, **kwargs) -> 'TimeSeriesDict'
```

TimeSeries のマッピングまたは単一の TimeSeries を各アイテムに追加します。

### `asd` / `psd`

```python
asd(self, *args, **kwargs)
psd(self, *args, **kwargs)
```

辞書内の各 TimeSeries の ASD/PSD を計算します。FrequencySeriesDict を返します。

### `coherence` / `coherence_matrix`

コヒーレンスまたはコヒーレンス行列を計算します。

### `crop`

```python
crop(self, start=None, end=None, copy=False) -> 'TimeSeriesDict'
```

辞書内の各 TimeSeries をクロップします。gwexpy.time.to_gps がサポートするあらゆる時刻形式を受け付けます。

### `csd` / `csd_matrix`

クロススペクトル密度/CSD行列を計算します。

### `decimate` / `resample`

各 TimeSeries をデシメート/リサンプルします。

### `degree` / `radian`

各アイテムの瞬時位相（度/ラジアン）を計算します。

### `detrend`

各 TimeSeries のトレンドを除去します。

### `envelope`

各アイテムのエンベロープを計算します。

### `fft`

各 TimeSeries に FFT を適用します。FrequencySeriesDict を返します。

### `filter` / `notch` / `gate`

各 TimeSeries にフィルタ/ノッチ/ゲートを適用します。

### `from_control` / `from_mne` / `from_pandas` / `from_polars`

python-control、MNE、pandas、polars から TimeSeriesDict を作成します。

### `heterodyne` / `baseband` / `mix_down` / `lock_in`

各アイテムに信号処理メソッドを適用します。

### `hht`

各アイテムに Hilbert-Huang 変換を適用します。

### `ica` / `pca`

チャンネル間で ICA/PCA 分解を実行します。

### `impute`

各アイテムの欠損データを補完します。

### `instantaneous_frequency` / `instantaneous_phase`

各アイテムの瞬時周波数/位相を計算します。

### `phase`

```python
phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

データの位相を計算します。

### `plot` / `plot_all`

すべてのシリーズをプロットします。gwexpy.plot.Plot に委譲します。

### `q_transform`

各 TimeSeries の Q 変換を計算します。SpectrogramDict を返します。

### `spectrogram` / `spectrogram2`

各 TimeSeries のスペクトログラムを計算します。

### `to_matrix`

TimeSeriesMatrix に変換します。

### `to_pandas` / `to_polars` / `to_mne`

pandas DataFrame / polars DataFrame / MNE Raw に変換します。

### `write`

```python
write(self, target: str, *args: Any, **kwargs: Any) -> Any
```

TimeSeriesDict をファイル（HDF5, ROOT など）に書き込みます。

CSV/TXT 出力は多チャンネル用の「ディレクトリ出力」です（各要素を個別ファイルに保存）。

```python
tsd.write("out_dir", format="csv")  # out_dir/ に ch ごとの CSV を保存
```

HDF5 出力では `layout` を指定できます（デフォルトは GWpy 互換の dataset-per-entry）。

```python
tsd.write("out.h5", format="hdf5")               # GWpy互換（既定）
tsd.write("out.h5", format="hdf5", layout="group")  # 旧形式（group-per-entry）
```

HDF5 のデータセット名（GWpy の `path=` 用）:
- キーは HDF5 で安全な名前にサニタイズされます（例: `H1:TEST` -> `H1_TEST`）。
- サニタイズ後の名前が衝突する場合、`__1` のようなサフィックスが付与されます。
- 元のキーはファイル属性に保存され、gwexpy の `read()` は元キーを復元します。

> [!WARNING]
> 信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。

pickle 可搬性メモ: gwexpy の `TimeSeriesDict` は unpickle 時に **GWpy の `TimeSeriesDict`** を返します
（読み込み側に gwexpy は不要です）。
