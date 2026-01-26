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
