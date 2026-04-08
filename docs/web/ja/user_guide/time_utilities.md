# :term:`GPS時刻ユーティリティ関数` (`gwexpy.time`)

GWexpy は GWpy の時刻ユーティリティを拡張し、GWpy が対応するスカラーの文字列・datetime に加えて、
pandas・NumPy・Astropy オブジェクトに対するベクトル演算をサポートします。

```python
from gwexpy.time import to_gps, from_gps, tconvert, LIGOTimeGPS
```

---

## `to_gps` — 日時 → GPS 秒

さまざまな時刻表現を GPS 秒に変換します。

### 文字列・datetime オブジェクト

```python
from gwexpy.time import to_gps

# ISO 8601 文字列（タイムゾーン指定なしの場合は UTC として扱われます）
to_gps("2015-09-14 09:50:45 UTC")
# → LIGOTimeGPS(1126259462, 391000000)

# Python datetime（タイムゾーン付き推奨）
from datetime import datetime, timezone
to_gps(datetime(2015, 9, 14, 9, 50, 45, tzinfo=timezone.utc))
```

### Astropy Time

```python
from astropy.time import Time

t = Time("2015-09-14T09:50:45", format="isot", scale="utc")
to_gps(t)
# → 1126259462.391
```

### pandas Series / DatetimeIndex（ベクトル対応）

```python
import pandas as pd

dates = pd.Series(pd.to_datetime(["2015-09-14", "2015-09-15", "2015-09-16"]))
gps_array = to_gps(dates)
# → numpy array([1126224017., 1126310417., 1126396817.])
```

### NumPy datetime64 配列（ベクトル対応）

```python
import numpy as np

dt64 = np.array(["2015-09-14T09:50:45", "2015-09-14T09:51:00"], dtype="datetime64[ns]")
to_gps(dt64)
```

---

## `from_gps` — GPS 秒 → 日時

GPS 秒を人間が読みやすい時刻に変換します。スカラー入力は `datetime`、
配列入力は `astropy.time.Time` 配列を返します。

```python
from gwexpy.time import from_gps

# スカラー → datetime
from_gps(1126259462)
# → datetime.datetime(2015, 9, 14, 9, 50, 45, tzinfo=...)

# リスト → astropy Time 配列
from_gps([1126259462, 1126259472, 1126259482])
```

---

## `tconvert` — 自動判定変換

`tconvert` は入力の型を自動判定し、`to_gps` または `from_gps` に振り分けます。
GWpy の `tconvert` と同様の動作に加え、配列入力に対応しています。

```python
from gwexpy.time import tconvert

# 文字列 / datetime → GPS 秒
tconvert("2015-09-14 09:50:45 UTC")
# → 1126259462

# GPS 秒 → UTC 文字列
tconvert(1126259462)
# → "September 14 2015, 09:50:45 UTC"

# 現在時刻を GPS 秒で取得
tconvert("now")
```

---

## `LIGOTimeGPS` — 高精度 GPS 時刻

`LIGOTimeGPS` は GPS 時刻をナノ秒精度（整数秒 + 整数ナノ秒）で保持します。
LIGO データアクセスライブラリで標準的に使用される表現形式です。

```python
from gwexpy.time import LIGOTimeGPS

t = LIGOTimeGPS(1126259462, 391000000)   # GW150914 合体時刻

print(t.gpsSeconds)      # 1126259462
print(t.gpsNanoSeconds)  # 391000000
print(float(t))          # 1126259462.391

# 算術演算
print(t + 10)    # LIGOTimeGPS(1126259472, 391000000)
print(t - 5)     # LIGOTimeGPS(1126259457, 391000000)
```

---

## TimeSeries との連携

GWexpy の多くのメソッドは、文字列・datetime・GPS 数値などの時刻指定を直接受け付けます。
内部で `gwexpy.time.to_gps` が自動的に呼び出されます。

```python
import gwexpy

ts = gwexpy.TimeSeries.fetch("H1:GDS-CALIB_STRAIN",
                             "2015-09-14 09:50:40",
                             "2015-09-14 09:51:00")

# .crop() は to_gps() が対応するあらゆる形式を受け付けます
segment = ts.crop("2015-09-14 09:50:44", "2015-09-14 09:50:50")
```

---

## 関連ドキュメント

- {doc}`../reference/api/time` — `gwexpy.time` の完全な API リファレンス
- {doc}`gwexpy_for_gwpy_users_ja` — GWexpy の全拡張機能の概要
