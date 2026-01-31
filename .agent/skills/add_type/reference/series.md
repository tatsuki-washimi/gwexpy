# Adding Series Types

時系列データ型を追加。

## Overview

Series 型は時間軸を有する1次元・2次元配列。GW信号、センサーデータ等に対応。

## Step 1: Plan & Design

### Base Class

`gwexpy.types.TimeSeries` または `gwexpy.types.Series` を継承：

```python
class MySeries(TimeSeries):
    pass
```

### Metadata Slots

時系列固有のメタデータ：

```python
_metadata_slots = TimeSeries._metadata_slots + (
    "_sampling_rate",       # サンプリングレート（Hz）
    "_epoch",              # エポック時刻（GPS等）
    "_channel_name",       # チャネル名
    "_data_type",          # データ型（raw, processed等）
)
```

### Key Features

- **Time Axis**: 常に軸0は時間
- **Slicing**: 時刻範囲スライス対応
- **Resampling**: サンプリングレート変更対応

## Step 2: Core Class Implementation

### Time-Aware Slicing

```python
def __getitem__(self, key):
    # Handle various slice formats
    if isinstance(key, tuple):
        # Multi-dimensional indexing
        return super().__getitem__(key)
    elif isinstance(key, slice):
        # Time-based slicing
        result = super().__getitem__(key)
        # Time axis metadata は自動保持
        return result
    else:
        # Single index: preserve dimension
        result = super().__getitem__(slice(key, key+1))
        return result

def get_time_range(self, t_start, t_end):
    """時刻範囲でスライス"""
    time_array = self.get_time_axis()
    mask = (time_array >= t_start) & (time_array < t_end)
    return self[mask]
```

### Resampling Support

```python
def resample(self, new_rate):
    """新しいサンプリングレートへリサンプル"""
    # scipy.signal.resample または astropy.time
    from scipy import signal

    factor = new_rate / self._sampling_rate
    new_length = int(self.shape[0] * factor)

    resampled = signal.resample(self, new_length, axis=0)
    result = self.__class__(
        resampled,
        sampling_rate=new_rate,
        epoch=self._epoch,
        channel_name=self._channel_name,
    )
    return result
```

## Step 3: Collections

### SeriesList

```python
class MySeriesList(list):
    """MySeries のコレクション"""

    def concatenate(self, axis=0):
        """全要素を結合"""
        return np.concatenate(self, axis=axis)

    def merge_channels(self):
        """複数チャネルをマージ"""
        arrays = [s.view(np.ndarray) for s in self]
        merged = np.stack(arrays, axis=1)
        return MySeries(
            merged,
            sampling_rate=self[0]._sampling_rate,
        )

    def apply_window(self, window="hann"):
        """全チャネルにウィンドウを適用"""
        from scipy import signal
        window_arr = signal.get_window(window, len(self[0]))
        return [s * window_arr for s in self]
```

### SeriesDict

```python
class MySeriesDict(dict):
    """MySeries の辞書コレクション（チャネル名キー）"""

    def get_channel(self, name):
        """チャネルを取得"""
        return self.get(name)

    def common_time_range(self, t_start, t_end):
        """全チャネルを共通時刻範囲でスライス"""
        return MySeriesDict(
            {k: v.get_time_range(t_start, t_end)
             for k, v in self.items()}
        )
```

## Step 4: Integration

### Export

`gwexpy/types/__init__.py`:

```python
from .series import MySeries, MySeriesList, MySeriesDict

__all__ = [
    # ...
    "MySeries",
    "MySeriesList",
    "MySeriesDict",
]
```

### Documentation Structure

```
docs/reference/en/types/MySeries.md
docs/reference/ja/types/MySeries.md
```

セクション：
- API Reference
- Time Axis Management
- Resampling Guide
- Collections Usage

## Step 5: Testing

### Test Coverage

```python
import pytest
from gwexpy.types import MySeries

class TestMySeries:

    def test_construction(self):
        data = np.random.randn(1000)
        s = MySeries(
            data,
            sampling_rate=100.0,
            epoch=1234567890,
            channel_name="H1:STRAIN"
        )
        assert len(s) == 1000
        assert s._sampling_rate == 100.0

    def test_time_slicing(self):
        s = MySeries(
            np.arange(100),
            sampling_rate=10.0,
            epoch=0
        )
        sliced = s.get_time_range(2.0, 5.0)
        assert len(sliced) > 0

    def test_resampling(self):
        s = MySeries(
            np.sin(np.linspace(0, 2*np.pi, 1000)),
            sampling_rate=100.0
        )
        resampled = s.resample(50.0)
        assert resampled._sampling_rate == 50.0

    def test_collections(self):
        series_list = MySeriesList([
            MySeries(np.random.randn(100), sampling_rate=100),
            MySeries(np.random.randn(100), sampling_rate=100),
        ])
        merged = series_list.merge_channels()
        assert merged.shape == (100, 2)
```

## Time Management Best Practices

- **GPS時刻**: エポックは GPS 秒（TAI 系）を使用
- **精度**: `float64` 以上を推奨（nanosecond 精度）
- **時間軸生成**: `epoch + arange(len(self)) / sampling_rate`
