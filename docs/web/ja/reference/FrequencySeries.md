# FrequencySeries

**継承元:** FrequencySeriesAnalysisMixin, SignalAnalysisMixin, FrequencySeriesSpectralMixin, StatisticsMixin, FittingMixin, PhaseMethodsMixin, RegularityMixin, BaseFrequencySeries (gwpy.frequencyseries.FrequencySeries)

互換性と将来の拡張のための gwpy の FrequencySeries の軽量ラッパー。

## Pickle / shelve の可搬性

> [!WARNING]
> 信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。

gwexpy の pickle は可搬性を優先しており、unpickle 時に **GWpy 型**を返す設計です
（読み込み側に gwexpy が無くても、gwpy があれば復元できます）。

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `df` | 周波数間隔 |
| `f0` | 開始周波数 |
| `frequencies` | 周波数配列 |
| `channel` | データチャンネル |
| `name` | データセット名 |
| `unit` | 物理単位 |
| `epoch` | GPS エポック |

## スペクトル変換

| メソッド | 説明 |
|---------|------|
| `ifft()` | 逆高速フーリエ変換 |
| `idct()` | 逆離散コサイン変換 |

## 信号処理

| メソッド | 説明 |
|---------|------|
| `filter()` | フィルタリング（振幅応答） |
| `apply_response()` | 複素周波数応答の適用 |
| `zpk()` | ZPK フィルタ |
| `smooth()` | 平滑化 |

## 周波数領域解析

| メソッド | 説明 |
|---------|------|
| `differentiate_time()` | 周波数領域での時間微分 |
| `integrate_time()` | 周波数領域での時間積分 |
| `group_delay()` | 群遅延 |

## 位相解析

| メソッド | 説明 |
|---------|------|
| `phase()` | 位相計算 |
| `angle()` | phase() のエイリアス |
| `degree()` | 位相（度） |
| `radian()` | 位相（ラジアン） |
| `unwrap_phase()` | 位相アンラップ |

## 統計

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `rms()` | 統計量 |
| `abs()` | 絶対値 |
| `real()` / `imag()` | 実部/虚部 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | 周波数範囲でクロップ |
| `interpolate()` | 補間 |
| `pad()` | パディング |
| `to_db()` | dB に変換 |

## 相互運用性

| メソッド | 説明 |
|---------|------|
| `to_pandas()` | pandas 変換 |
| `to_torch()` / `to_tensorflow()` / `to_jax()` | ML フレームワーク変換 |
| `to_control_frd()` | python-control FRD 変換 |
