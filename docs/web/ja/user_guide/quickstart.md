# クイックスタート

簡単な時系列データの作成と解析を行います。

:::{note}
より詳しい学習パスは [getting_started](getting_started.md) を参照してください。
:::

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

# サンプルデータの生成 (1024 Hz, 4秒間)
ts = TimeSeries(np.random.randn(4096), t0=0, dt=1/1024, name="demo")

# バンドパスフィルタの適用 (30 - 300 Hz)
bandpassed = ts.bandpass(30, 300)

# 周波数変換 (FFT)
spectrum = bandpassed.fft()

print(spectrum.frequencies[:5])
```

コレクション（複数の時系列）の扱い:

```python
from gwexpy.timeseries import TimeSeriesDict

tsd = TimeSeriesDict()
tsd["H1:TEST"] = ts
tsd["L1:TEST"] = ts * 0.5

# TimeSeriesMatrix に変換
matrix = tsd.to_matrix()
print(matrix.shape)
```

## 時刻ユーティリティと自動Series生成

```python
import numpy as np
from astropy import units as u
import pandas as pd
from gwexpy import as_series
from gwexpy.time import to_gps, from_gps

# GPS時刻への変換
times = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:01"])
gps = to_gps(times)
iso = from_gps(gps)

# 軸データから自動的に Series (TimeSeries / FrequencySeries) を作成
ts_axis = as_series((1419724818 + np.arange(10)) * u.s, unit="h")
fs_axis = as_series(np.arange(5) * u.Hz, unit="mHz")
```
