# Spectrogram

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`Spectrogram` は単一の時間周波数マップを表し、GWexpy の解析・描画・変換ヘルパを利用できます。

## 代表的なシグネチャ

```python
Spectrogram(data, t0=None, dt=None, f0=None, df=None, ...)
Spectrogram.percentile(q, axis="time")
```

## 最小例

```python
from gwexpy.spectrogram import Spectrogram
import numpy as np

sgm = Spectrogram(np.random.randn(16, 32), dt=1.0, df=1.0)
med = sgm.percentile(50, axis="time")
```

## 関連理論

- [FFT_Conventions](FFT_Conventions.md)
- [Spectrogram チュートリアル](../user_guide/tutorials/intro_spectrogram.ipynb)
- [時間-周波数解析: 手法比較ガイド](../user_guide/tutorials/time_frequency_comparison.md)

## 関連チュートリアル

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- [Spectrogram チュートリアル](../user_guide/tutorials/intro_spectrogram.ipynb)
- [グリッチ詳細解析](../user_guide/tutorials/case_glitch_analysis.ipynb)
- [HHT: 解析](../user_guide/tutorials/advanced_hht.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** SpectrogramAnalysisMixin, SpectrogramInteropMixin, SignalAnalysisMixin, PhaseMethodsMixin, RegularityMixin, BaseSpectrogram ([`gwpy.spectrogram.Spectrogram`](https://gwpy.readthedocs.io/en/stable/reference/gwpy.spectrogram.Spectrogram/))

追加の相互運用メソッドを備えた gwpy.spectrogram.Spectrogram の拡張。

## 物理コンテキスト

`Spectrogram` は「単一チャネルの時間周波数マップ」を表します。非定常雑音、短時間バースト、線スペクトルのドリフト、制御線の立ち上がり、glitch の周波数進化など、**時間と周波数の両方を同時に見たい現象**を扱うときの基本クラスです。

- **二つの軸の意味**: `t0` / `dt` は時間ビン、`f0` / `df` は周波数ビンを決めます。1 ピクセルは「ある時間窓での、ある周波数帯の強度」を表します。
- **上流処理への依存**: `Spectrogram` は通常 `TimeSeries.spectrogram()`、`spectrogram2()`、`q_transform()`、`hht(..., output="spectrogram")` などの結果です。したがって、窓長・オーバーラップ・変換手法により分解能と見え方が変わります。
- **単位の意味**: ピクセル値はパワー、ASD 相当、正規化強度、位相量などになり得ます。まず「色が何を表しているか」を固定してから読む必要があります。

## 解析上の注意点

### 時間分解能と周波数分解能のトレードオフ

スペクトログラムは便利ですが、すべてを同時に高分解能で見られるわけではありません。

- 短い窓では時間局在は良くなるが、周波数分解能は粗くなる
- 長い窓では細い線を見やすくなるが、短いトランジェントはにじむ
- Q 変換や HHT では、このトレードオフの現れ方が STFT と異なる

### 色の強さをそのまま物理振幅だと思わない

可視化では対数圧縮、正規化、percentile clipping が入ることがあります。見た目の明るさだけで「信号が強い」「エネルギーが保存されている」と判断してはいけません。

- `plot(norm="log")` や dB 表示の有無を確認する
- 比較図では colormap と color scale を揃える
- percentile ベースの背景推定をした場合は、元の絶対量と区別する

### よくある誤読

1. ピクセル幅を物理イベント継続時間そのものとみなす
2. 別設定の spectrogram を同じ色スケールだと思って比較する
3. 周波数ドリフトを手法由来の smear と区別せずに解釈する
4. percentile や bootstrap の統計出力を生データの瞬時強度と混同する

## どのページへ進むか

- 上流の時間領域入口: [TimeSeries](TimeSeries.md)
- 時間周波数手法の比較: [時間-周波数解析: 手法比較ガイド](../user_guide/tutorials/time_frequency_comparison.md)
- インタラクティブ比較: [時間-周波数解析: インタラクティブ比較](../user_guide/tutorials/time_frequency_analysis_comparison.md)
- glitch の実例: [グリッチ詳細解析](../user_guide/tutorials/case_glitch_analysis.ipynb)
- HHT との比較: [HHT: 解析](../user_guide/tutorials/advanced_hht.ipynb)

## Pickle / shelve の可搬性

:::{admonition} warning
:class: warning

信頼できないデータを `pickle` / `shelve` で読み込まないでください。ロード時に任意コード実行が起こり得ます。
:::

gwexpy の pickle は可搬性を優先しており、unpickle 時に **GWpy 型**を返す設計です
（読み込み側に gwexpy が無くても、gwpy があれば復元できます）。

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `dt` | 時間間隔 |
| `t0` | 開始時刻 |
| `times` | 時間配列 |
| `df` | 周波数間隔 |
| `f0` | 開始周波数 |
| `frequencies` | 周波数配列 |
| `channel` | データチャンネル |
| `name` | データセット名 |
| `unit` | 物理単位 |

## スペクトル解析

| メソッド | 説明 |
|---------|------|
| `percentile()` | パーセンタイル計算 |
| `bootstrap_asd()` | ブートストラップ ASD 推定 |
| `rebin()` | 時間/周波数リビン |
| `interpolate()` | 補間 |

## 位相解析

| メソッド | 説明 |
|---------|------|
| `phase()` | 位相計算 |
| `angle()` | phase() のエイリアス |
| `degree()` | 位相（度）、単位 'deg' を設定 |
| `radian()` | 位相（ラジアン）、単位 'rad' を設定 |
| `unwrap_phase()` | 位相アンラップ |

## 統計

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `median()` | 統計量 |
| `abs()` | 絶対値 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | 時間範囲でクロップ |
| `crop_frequencies()` | 周波数範囲でクロップ |

## 相互運用性

| メソッド | 説明 |
|---------|------|
| `to_pandas()` | pandas 変換 |
| `to_timeseries_list()` | 周波数ビンごとに TimeSeriesList へ変換 |
| `to_frequencyseries_list()` | 時刻ビンごとに FrequencySeriesList へ変換 |
| `to_torch()` / `to_tensorflow()` / `to_jax()` / `to_cupy()` | ML フレームワーク変換 |
| `to_xarray()` | xarray 変換 |

## 入出力

| メソッド | 説明 |
|---------|------|
| `read()` / `write()` | ファイル入出力 |

## 可視化

| メソッド | 説明 |
|---------|------|
| `plot()` | スペクトログラムプロット |
