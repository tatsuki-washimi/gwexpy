# クイックスタート

簡単な時系列データの作成と解析を行います。

:::{note}
より詳しい学習パスは [getting_started](getting_started.md) を参照してください。
:::

## GWpyからの移行

GWpyコードをGWexpyで使用するには、importを置き換えるだけです：

```python
# GWpy (従来)
# from gwpy.timeseries import TimeSeries
# from gwpy.frequencyseries import FrequencySeries

# GWexpy（推奨）
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.frequencyseries import FrequencySeriesMatrix
```

## 複数チャンネル時系列データの生成とプロット

実験的なノイズモデルから複数チャンネルの時系列データを生成します：

```python
import numpy as np
from gwexpy.timeseries import TimeSeriesDict
from gwexpy.signal.noise import PowerLawNoise

# ノイズモデルの設定 (1/f ノイズ: beta=1)
noise_model = PowerLawNoise(beta=1, dt=1/1024)

# 複数チャンネル時系列データの生成
tsd = TimeSeriesDict()
tsd["H1:STRAIN"] = noise_model.generate(duration=64)  # Hanford
tsd["L1:STRAIN"] = noise_model.generate(duration=64)  # Livingston

# プロット
plot = tsd.plot()
plot.show()
```

## TimeSeriesMatrixから周波数行列への一括変換

複数チャンネルの時系列データをFrequencySeriesMatrixに変換し、クロススペクトル密度(CSD)を計算します：

```python
# TimeSeriesDictをmatrixに変換
ts_matrix = tsd.to_matrix()

# CSD計算（Welch法、重なり50%）
csm = ts_matrix.csd(
    fftlength=4,
    overlap=0.5,
    window='hann'
)

# 周波数行列としてプロット
freq_plot = csm.plot()
freq_plot.show()

# 具体的な周波数領域の解析
print(f"周波数範囲: {csm.frequencies[0]:.1f} - {csm.frequencies[-1]:.1f} Hz")
print(f"H1-L1相互スペクトル (10 Hz): {csm['H1:STRAIN', 'L1:STRAIN'].interpolate(10).value:.2e}")
```

