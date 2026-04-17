# GPS時刻ユーティリティ関数 (`gwexpy.time`)

GWexpy は GWpy の時刻ユーティリティを拡張し、GWpy が対応するスカラーの文字列・datetime に加えて、pandas・NumPy・Astropy オブジェクトに対するベクトル演算をサポートします。

```python
from gwexpy.time import to_gps, from_gps, tconvert, LIGOTimeGPS
```

## クイックガイド: 重要な注意事項 (FAQ)

変換を行う前に、以下の基本特性を理解しておくことが重要です。

### 閏秒 (Leap Seconds) の扱い
- **GPS 時刻**: 1980年1月6日 00:00:00 UTC を起点（0秒）とし、閏秒を含まない連続した秒数としてカウントされます。
- **変換**: UTC との変換時には、`astropy.time` が保持する最新の IERS 閏秒テーブルが自動的に適用されます。
- **注意**: 未来の時刻や、直近で追加された閏秒を含む変換を行う場合は、`astropy` のデータ更新が必要になる場合があります。

### タイムゾーンのない文字列の扱い
- `"2015-09-14 09:50:45"` のようにタイムゾーンが指定されていない文字列を `to_gps` や `tconvert` に渡すと、**デフォルトで UTC** として解釈されます。
- ローカルタイムとして品質の高い変換を行いたい場合は、必ずタイムゾーン名（例: `"Asia/Tokyo"`）を文字列に含めるか、タイムゾーン付きの `datetime` オブジェクトを渡してください。

### よくある失敗とエラー条件
- `to_gps("not-a-time")` や `tconvert("not-a-time")` のように日時として解釈できない文字列は、`ValueError` で失敗します。
- `from_gps("abc")` のように GPS 秒として数値化できない入力は、`ValueError` で失敗します。
- `to_gps(..., timezone="Asia/Tokyo")` のような未対応キーワード引数は、この実装では受け付けず `TypeError` になります。タイムゾーンは文字列本体または timezone-aware `datetime` で指定してください。
- naive な `datetime` は UTC として扱われます。ローカル時刻を表したい場合は、timezone-aware `datetime` を使ってください。
- 夏時間の切替などで曖昧になるローカル時刻は、このページでは自動解決を保証しません。境界時刻では UTC オフセット付き文字列または明示的な timezone-aware `datetime` を推奨します。

## 関数選択早見表

目的に応じて最適な関数を選択してください。

| 目的 | 推奨関数 | 主な入力の型 | 出力の型 (Scalar / Vector) | 主要引数 |
| :--- | :--- | :--- | :--- | :--- |
| **日時 → GPS秒** | `to_gps` | `str`, `datetime`, `Time`, `Series` | `LIGOTimeGPS` / `f8 ndarray` | — |
| **GPS秒 → 日時** | `from_gps` | `int`, `float`, `LIGOTimeGPS`, `ndarray` | `datetime` / `astropy.time.Time` | — |
| **相互変換 (自動判定)** | `tconvert` | 上記すべて + `"now"` | 文脈に応じた型 (Scalar 優先) | — |
| **高精度オブジェクト** | `LIGOTimeGPS` | `seconds`, `nanoseconds` | `LIGOTimeGPS` (秒+ナノ秒保持) | — |

---

## 基本的な使用例 (Examples)

### 1. 単一時刻の相互変換

```python
from gwexpy.time import to_gps, from_gps, tconvert

# 日本語の日時文字列を GPS 秒へ（デフォルト UTC）
gps = to_gps("2015-09-14 09:50:45")
# → 1126259462.391

# GPS 秒を datetime オブジェクトへ
dt = from_gps(1126259462.391)
# → datetime.datetime(2015, 9, 14, 9, 50, 45, ...)

# tconvert による型自動判別
tconvert("now")     # 現在の GPS 秒
tconvert(1126259462) # 固定文字列（"September 14 2015, ..."）
```

### 2. ベクトル演算 (Vectorized Operations)

リストや NumPy 配列を渡すと、内部的に最適化されたベクトル変換が行われます。

```python
import numpy as np

# リストや配列の一括変換
gps_list = to_gps(["2015-09-14 09:50:00", "2015-09-14 09:51:00"])
# → array([1126259417., 1126259477.])

# 浮動小数点配列から Astropy Time オブジェクトの配列へ返還
times = from_gps(np.arange(1126259400, 1126259410))
# → <Time object: scale='utc' format='gps' value=[1.1262594e+09 ...]>
```

### 3. タイムゾーンと言号（Leap Second）の注意点

```python
# 明示的なタイムゾーン指定（推奨）
to_gps("2024-01-01 09:00:00 JST") 

# タイムゾーンなしは UTC とみなされるため注意
to_gps("2024-01-01 09:00:00") # UTC 09:00:00
```

```python
# naive datetime は UTC として扱われる
from datetime import datetime
to_gps(datetime(2024, 1, 1, 9, 0, 0))

# ローカル時刻なら timezone-aware datetime を使う
from zoneinfo import ZoneInfo
to_gps(datetime(2024, 1, 1, 9, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo")))
```

### 4. 異常入力の扱い

```python
# 日時として解釈できない文字列
to_gps("not-a-time")  # -> ValueError

# 数値化できない GPS 入力
from_gps("abc")       # -> ValueError

# 未対応キーワード引数
to_gps("2024-01-01 09:00:00", timezone="Asia/Tokyo")  # -> TypeError
```

## `to_gps` — 日時 → GPS 秒

**Signature**: `to_gps(t, *args, **kwargs)`

さまざまな時刻表現を GPS 秒に変換します。単一の値（スカラー）だけでなく、リストや配列に対しても効率的なベクトル演算を行います。

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

**Signature**: `from_gps(t, *args, **kwargs)`

GPS 秒を人間が読みやすい時刻に変換します。スカラー入力は `datetime`、
配列入力は `astropy.time.Time` 配列を返します。

- 単一の `int` / `float` / `LIGOTimeGPS` を渡した場合: `datetime.datetime`
- リスト・NumPy 配列・pandas 系のベクトル入力を渡した場合: `astropy.time.Time`

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

**Signature**: `tconvert(t=None, *args, **kwargs)`

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

- [API リファレンス](../reference/api/time.rst) — `gwexpy.time` の完全な API リファレンス
- [GWpy からの移行](gwexpy_for_gwpy_users_ja.md) — GWexpy の全拡張機能の概要
