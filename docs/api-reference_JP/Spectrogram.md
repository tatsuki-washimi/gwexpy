# Spectrogram

**継承元:** Spectrogram

`gwpy.spectrogram.Spectrogram` を拡張し、追加の相互運用（interop）メソッドを提供します。

## メソッド

### `T`

転置された配列のビュー。`self.transpose()` と同じです。

### `abs`

```python
abs(self, axis=None, **kwargs)
```

要素ごとの絶対値を計算します。複素入力の場合、振幅を返します。

### `append`

```python
append(self, other, inplace=True, pad=None, gap=None, resize=True)
```

現在のシリーズの末尾に別のシリーズを接続します。

### `bootstrap_asd`

```python
bootstrap_asd(self, n_boot=1000, average='median', ci=0.68, window='hann', nperseg=None, noverlap=None)
```

ブートストラップ再サンプリングを用いて、このスペクトログラムからロバストな ASD（振幅スペクトル密度）を推定します。
`gwexpy.spectral.bootstrap_spectrogram` のラッパーです。

### `channel`

データに関連付けられた観測チャンネル。

### `copy`

```python
copy(self, order='C')
```

配列のコピーを返します。

### `crop`

```python
crop(self, start=None, end=None, copy=False)
```

指定された x 軸（時間軸）の範囲にシリーズを切り詰めます。

### `crop_frequencies`

```python
crop_frequencies(self, low=None, high=None, copy=False)
```

指定した周波数範囲でスペクトログラムをクロップします。

### `dx`

x 軸（通常は時間軸）のサンプル間隔。

### `dy`

y 軸（通常は周波数軸）のサンプル間隔。

### `filter`

```python
filter(self, *filt, **kwargs)
```

この `Spectrogram` に指定されたフィルタを適用します。

### `from_quantities`

```python
from_quantities(q, times, frequencies)
```

quantities.Quantity から Spectrogram を作成します。

### `from_root`

```python
from_root(obj, return_error=False)
```

ROOT の TH2D から Spectrogram を作成します。

### `from_spectra`

```python
from_spectra(*spectra, **kwargs)
```

スペクトル（FrequencySeries）のリストから新しい `Spectrogram` を構築します。

### `imshow`

```python
imshow(self, **kwargs)
```

imshow を使用してプロットします。*(gwpy から継承)*

### `inject`

```python
inject(self, other)
```

共通する x 軸の値に沿って、互換性のある 2 つのシリーズを加算します。

### `median`

```python
median(self, axis=None, **kwargs)
```

指定された軸方向にメディアン（中央値）を計算します。

### `name`

データセットの名称。

### `pad`

```python
pad(self, pad_width, **kwargs)
```

シリーズを新しいサイズにパディングします。

### `pcolormesh`

```python
pcolormesh(self, **kwargs)
```

pcolormesh を使用してプロットします。*(gwpy から継承)*

### `percentile`

```python
percentile(self, percentile)
```

この `Spectrogram` の指定されたスペクトル・パーセンタイルを計算し、`FrequencySeries` として返します。

### `plot`

```python
plot(self, method='pcolormesh', figsize=(12, 6), xscale='auto-gps', **kwargs)
```

この `Spectrogram` のデータをプロットします。

### `prepend`

```python
prepend(self, other, inplace=True, pad=None, gap=None, resize=True)
```

現在のシリーズの先頭に別のシリーズを接続します。

### `ratio`

```python
ratio(self, operand)
```

リファレンスと比較したスペクトログラムの比率を計算します。
`operand` には 'mean', 'median', または特定の `FrequencySeries` を指定できます。

### `read`

```python
read(source, *args, **kwargs)
```

ファイルから Spectrogram を読み込みます。

### `shift`

```python
shift(self, delta)
```

x 軸方向に `delta` だけシリーズをシフトします。インプレースで動作します。

### `to_quantities`

```python
to_quantities(self, units=None)
```

quantities.Quantity (Elephant/Neo互換) に変換します。

### `to_th2d`

```python
to_th2d(self, error=None)
```

ROOT の TH2D に変換します。

### `to_torch`

スペクトログラム（または SpectrogramList/Dict の各要素）を PyTorch テンソルに変換します。

### `to_cupy`

スペクトログラム（または SpectrogramList/Dict の各要素）を CuPy 配列に変換します。

### `to_matrix`

SpectrogramList または SpectrogramDict を `SpectrogramMatrix` に変換します。

### `unit`

データの物理単位。

### `update`

```python
update(self, other, inplace=True)
```

新しいデータを末尾に追加し、同量のデータを先頭から削除して、シリーズを更新します。

### `value_at`

```python
value_at(self, x, y)
```

指定した (x, y) 座標におけるシリーズの値を返します。

### `variance`

```python
variance(self, bins=None, low=None, high=None, nbins=500, log=False, norm=False, density=False)
```

この `Spectrogram` の `SpectralVariance`（スペクトル分散）を計算します。

### `write`

```python
write(self, target, *args, **kwargs)
```

Spectrogram をファイルに書き出します。

### `x0`, `xindex`, `xspan`, `xunit`

x 軸（時間軸）の開始点、座標配列、範囲、単位。

### `y0`, `yindex`, `yspan`, `yunit`

y 軸（周波数軸）の開始点、座標配列、範囲、単位。

### `zip`

```python
zip(self)
```

`xindex` と `value` 配列をスタックします。

### `zpk`

```python
zpk(self, zeros, poles, gain, analog=True)
```

零点・極・ゲイン（ZPK）フィルタを適用します。
