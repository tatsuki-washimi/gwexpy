# v0.2.0 実装レーンの移行メモ

このページは、現在の `[Unreleased]` に対する実装証拠を説明します。
v0.2.0 は公開またはリリースされておらず、root が担当する最終検証ゲートは保留中です。

## API ラベル

API 安定性ポリシーのラベルは `stable`、`provisional`、`experimental` の 3 つだけです。
`deferred` はリリース結果であり、4 つ目の API ティアではありません。
分類は保守的に行っており、すべてのレガシー API を暗黙にラベル付けするものではありません。

## 正確なナノ秒起点時刻

入力元が正確な GPS ナノ秒を持つ場合は、キーワード専用の `t0_ns` を使用します。
許可範囲は正確に `0 <= t0_ns <= 2**63 - 1` です。
読み取り専用の `t0_gps_ns` プロパティは、値を保持できた場合に正確な値を返します。

変更前は、浮動小数点の epoch を受け取れても、表現可能な精度に丸められる場合がありました。

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

samples = [0.0, 1.0]
series = TimeSeries(samples, sample_rate=4, epoch=1_400_000_000.000000001)
```

変更後は、整数を明示します。

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

samples = [0.0, 1.0]
series = TimeSeries(samples, sample_rate=4, t0_ns=1_400_000_000_000_000_001)
assert series.t0_gps_ns == 1_400_000_000_000_000_001
```

`t0_ns` はキーワード専用で、その範囲内の正確な整数でなければなりません。
bool、負の値、2**63 - 1 を超える値、`t0` または `epoch` と一致しない値は、構築前に `TypeError` または `ValueError` になります。
浮動小数点秒の既存 API は維持されますが、意味は exact ではなく `quantized` です。

## HDF5 メタデータと provenance の自動復元

正規 HDF5 writer は GWpy ネイティブの payload を変更せず、GWexpy のメタデータ、正確な時刻状態、provenance のために、ファイル root に `_gwexpy_sidecar_json_v1` 属性を 1 つだけ自動で書き込みます。
この sidecar は GWexpy だけが復元し、GWpy 単独の reader はネイティブ payload を読み取れます。
自動遷移は、GWexpy の public read/write 入口に接続されています。
その入口は `StateVector`、`SegmentList`、`DataQualityFlag` を含む 6 種類の native HDF5 sidecar handler も登録します。
GWexpy に入らない GWpy 単独の caller は、GWexpy の bootstrap を起動しません。
root 属性は writer が管理します。メタデータは object と write API を通して更新します。

変更前は、ネイティブの往復で GWpy 単独の reader に見えるのは payload だけでした。

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

source = TimeSeries([0.0, 1.0])
source.write("data.h5", format="hdf5", path="data")
restored = TimeSeries.read("data.h5", format="hdf5", path="data")
```

変更後は、同じ実行入口で正確な時刻、metadata、provenance も自動的に復元されます。

```python
# executable-roundtrip
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gwexpy import register_all
from gwexpy.timeseries import TimeSeries

register_all()
metadata = {"channel": "K1:TEST-STRAIN", "labels": ["synthetic", "tutorial"]}
provenance = {
    "schema": "gwexpy.provenance",
    "algorithm": "migration-roundtrip",
    "parameters": {"sample_rate_hz": 4.0},
}
origin_ns = 1_400_000_000_000_000_001

with TemporaryDirectory() as temporary_directory:
    path = Path(temporary_directory) / "data.h5"
    source = TimeSeries(np.arange(4, dtype=float), sample_rate=4, t0_ns=origin_ns)
    source.metadata = metadata
    source.provenance = provenance
    source.write(path, format="hdf5", path="data")
    restored = TimeSeries.read(path, format="hdf5", path="data")

    assert restored.t0_gps_ns == origin_ns
    assert restored.metadata == metadata
    assert restored.provenance == provenance
```

壊れた JSON、未知の sidecar schema または version、JSON にできない provenance は `ValueError` または `TypeError` になり、黙って無視されません。

## GWF の並列読み込みと `nproc` alias

GWF の読み込みでは `parallel` を推奨します。
`nproc` は互換 alias として残り、このレーンで非推奨または削除にはなりません。
2 つの名前の併用はできません。

変更前は、既存の呼び出しで互換表記を使えました。

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

files = ["example.gwf"]
series = TimeSeries.read(files, format="gwf", nproc=4)
```

変更後は、新しいコードで推奨表記を使います。

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

files = ["example.gwf"]
series = TimeSeries.read(files, format="gwf", parallel=4)
```

次の指定は、どちらかを黙って選ばずに失敗します。

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

files = ["example.gwf"]
TimeSeries.read(files, format="gwf", parallel=4, nproc=4)  # TypeError
```

これらは静的な signature 例であり、外部の example file を読み込んだことを主張しません。
実装経路は spawn、決定的な merge 順序、失敗時のキャンセル、実 lalframe 証拠を使用します。

## NDScope の `dataset_options`

NDScope writer の公開入口は `dataset_options` だけです。
filter や chunk のキーワードを top-level に並べるレガシー surface は使用しません。

変更前は、top-level option は未対応または意味が曖昧な surface でした。

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

series = TimeSeries([0.0, 1.0])
series.write("data.h5", format="hdf.ndscope", compression="gzip")  # TypeError
```

変更後は、承認された HDF5 作成 option を mapping に入れます。

```python
# static-signature-example
from gwexpy.timeseries import TimeSeries

series = TimeSeries([0.0, 1.0])
series.write(
    "data.h5",
    format="hdf.ndscope",
    dataset_options={"compression": "gzip", "shuffle": True},
)
```

これらは静的な signature 例であり、外部の NDScope file を作成したことを主張しません。
未知の key、無効な filter、適合しない chunk、利用できない codec は、ファイル作成前の preflight で拒否されます。

## provenance、median bias、coupling segment

provenance は厳格な JSON mapping です。
必要に応じて RNG method、bit generator、seed、software version、すべての algorithm parameter を記録します。
runtime の provenance 伝播は、provenance を持つ `Spectrogram` の解析 output に限って実装されています。
対応する copy、slice、ufunc、二項演算は、検証済みで独立した provenance snapshot と operation tree を保持します。
対応する HDF5 sidecar round-trip は、JSON-safe な provenance を対応オブジェクト上で保持します。
`GauChResult.metadata` は provenance mapping の alias のままです。

```python
# static-signature-example
from gwexpy.spectrogram import Spectrogram

result = Spectrogram([[0.0]], dt=1, f0=0, df=1)
result.provenance = {
    "schema": "gwexpy.provenance",
    "algorithm": "example",
    "parameters": {"window": "hann"},
    "rng": {"method": "numpy", "bit_generator": "PCG64", "seed": 7},
}
```

array、有限でない数、対応しない object は、曖昧な JSON に変換せず失敗します。
`median_bias(N)` は、独立な chi-square-2 または指数分布標本に対するレビュー済みの式と限界を使用します。

```python
# static-signature-example
from gwexpy.signal.spectral import median_bias

alpha_3 = median_bias(3)
assert alpha_3 == 5 / 6
```

重複標本や相関標本への補正を主張するものではありません。

カップリングの coupling segment v1 の長形式では、`start_gps_ns`、`duration_ns`、`source_channel`、`response_channel`、`frequency_hz`、`coupling_factor`、`coupling_factor_unit` を必須列とします。
upper limit の列は `upper_limit` 行でだけ許可し、周波数は Hz に正規化します。
この schema の安定性ラベルは `Experimental` であり、科学的に広い一般性を主張しません。

```python
# static-signature-example
import pandas as pd

from gwexpy.coupling import validate

segments = pd.DataFrame(
    {
        "start_gps_ns": [1_400_000_000_000_000_000],
        "duration_ns": [1_000_000_000],
        "source_channel": ["K1:PEM-MIC"],
        "response_channel": ["K1:STRAIN"],
        "frequency_hz": [10.0],
        "coupling_factor": [0.25],
        "coupling_factor_unit": ["1"],
    }
)
validate(segments)
```

## SeriesMatrix direct-ufunc 制限と #637 fallback

**安定性:** provisional。

#637 の composition prototype と証拠は隔離環境で完了しましたが、candidate runtime は
integration にコピーされておらず、v0.2.0 では SeriesMatrix の composition/B1 を採用しません。

B0 の公開可能な source of truth は contract ledger で、直接 ufunc の可否を
明示します。

`np.sqrt(matrix)` は v0.2.0 B0 の direct NumPy ufunc としては非対応です。

代替として次を利用してください。

```python
result = matrix ** 0.5
```

この代替経路は `TimeSeriesMatrix`、`FrequencySeriesMatrix`、`SpectrogramMatrix`
の3 familyすべてで contract-tested されています。B0 の
クラス、数値、セル単位、軸、ラベル、metadata の意味は維持されます。

`np.log(matrix)`、`np.exp(matrix)`、`np.isfinite(matrix)`、`np.isnan(matrix)` に対する
metadata-preserving な B0 workaround は現在未定義です。

`np.isreal(matrix)` は `SpectrogramMatrix` でのみ direct support で、`TimeSeriesMatrix`
と `FrequencySeriesMatrix` では B0 で未対応（`UnitConversionError`）です。

`(2 * u.s) * matrix` と `matrix * (2 * u.s)` は両方とも既に B0 で対応済みです。

`np.asarray(matrix)` は metadata model を離脱する raw-array の境界で、意図的に bare
ndarray を返します。metadata-preserving workaround ではありません。

未対応の direct 呼び出しは silent downgrade ではなく **明示的失敗** になり、
`ndarray` や `Quantity` への自動劣化を許容しません。

#637 の将来 redesign は v0.2.0 の確定版ではないため、採用バージョン/日付は
このページでは定めていません。

## リリース状態

このページは `[Unreleased]` の実装文書と証拠準備です。
公開済み version、tag、commit、pull request、issue state、または完了した最終 integration gate を主張しません。
