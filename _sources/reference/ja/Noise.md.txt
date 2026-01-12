# Noise

検出器・環境ノイズのモデル化とASDからの波形生成のユーティリティです。

## `from_asd`

```python
from_asd(asd, duration, sample_rate, t0=0.0, rng=None) -> TimeSeries
```

ASD (`FrequencySeries`) から色付きノイズの `TimeSeries` を生成します。

注意:
- 戻り値は `TimeSeries`（NumPy 配列ではありません）。
- `name` と `channel` は入力ASDから引き継がれます。
- 出力の unit は `asd.unit * sqrt(Hz)` です。
- 開始時刻は `t0` で指定できます。
