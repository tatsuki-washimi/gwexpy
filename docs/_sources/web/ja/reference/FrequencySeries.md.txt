# FrequencySeries

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`FrequencySeries` は単一スペクトルを表す基本クラスで、GWexpy のフィッティング・統計・フィルタ・描画拡張を含みます。

## 代表的なシグネチャ

```python
FrequencySeries(data, unit=None, f0=None, df=None, frequencies=None, ...)
FrequencySeries.ifft(...)
```

## 最小例

```python
from gwexpy.frequencyseries import FrequencySeries
import numpy as np

fs = FrequencySeries(np.ones(128), df=1.0, unit="V / Hz")
phase = fs.phase()
```

## 関連理論

- [FFT_Conventions](FFT_Conventions.md)
- [FrequencySeries チュートリアル](../user_guide/tutorials/intro_frequencyseries.ipynb)
- [伝達関数計測](../user_guide/tutorials/case_transfer_function.ipynb)

## 関連チュートリアル

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- [FrequencySeries チュートリアル](../user_guide/tutorials/intro_frequencyseries.ipynb)
- [ノイズバジェット解析](../user_guide/tutorials/case_noise_budget.ipynb)
- [フィッティング上級編](../user_guide/tutorials/advanced_fitting.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** FrequencySeriesAnalysisMixin, SignalAnalysisMixin, FrequencySeriesSpectralMixin, StatisticsMixin, FittingMixin, PhaseMethodsMixin, RegularityMixin, BaseFrequencySeries ([`gwpy.frequencyseries.FrequencySeries`](https://gwpy.readthedocs.io/en/latest/api/gwpy.frequencyseries.FrequencySeries/))

互換性と将来の拡張のための gwpy の FrequencySeries の軽量ラッパー。

## 物理コンテキスト

`FrequencySeries` は「単一チャネルの周波数領域表現」を表します。`TimeSeries` を FFT した複素スペクトル、PSD/ASD、伝達関数、応答関数、位相付き周波数応答など、**各周波数ビンが一つの周波数帯の情報を表す量**を保持するときの基本クラスです。

- **時間領域との対応**: 多くの `FrequencySeries` は `TimeSeries.fft()`、`psd()`、`asd()`、`csd()` の結果として得られます。したがって、`df`・窓長・平均化方法は上流の時間領域処理に依存します。
- **単位の意味**: `unit` は振幅スペクトル・PSD・応答関数で意味が変わります。たとえば ASD はしばしば `strain / sqrt(Hz)`、PSD は `strain2 / Hz`、応答関数は `m / V` や `count / N` のような複合単位になります。
- **複素数の意味**: `real` / `imag` / `phase()` / `unwrap_phase()` は、単なる数値操作ではなく、遅延・共振・制御系の位相余裕・因果的な応答の解釈につながります。

## 解析上の注意点

### 何のスペクトルなのかを先に固定する

同じ `FrequencySeries` でも、振幅スペクトル・PSD・ASD・複素伝達関数では読み方が異なります。

- ノイズ床を比較したいなら ASD / PSD として解釈する
- 入出力関係を見たいなら伝達関数・応答関数として解釈する
- `ifft()` で時間波形へ戻したいなら、上流の FFT 規約と Hermitian 条件を意識する

### FFT 規約と正規化

`FrequencySeries` は「もう周波数領域にある」ため、見た目だけでは正規化規約が分かりません。別コードや論文図と比較するときは、以下を先に確認します。

1. 片側スペクトルか両側スペクトルか
2. 窓関数補正が入っているか
3. 振幅量なのかパワー量なのか
4. dB 表示前の基準量が何か

規約は [FFT_Conventions](FFT_Conventions.md) を基準にしてください。

### よくある誤読

1. ASD と PSD を同じ量として比較する
2. `to_db()` 後の値を線形振幅のまま足し引きする
3. 位相の wrap をそのまま物理的ジャンプだとみなす
4. `df` や平均化窓を無視してピーク幅・共振 Q を解釈する

## どのページへ進むか

- 周波数領域規約の確認: [FFT_Conventions](FFT_Conventions.md)
- 時間領域からの入口: [TimeSeries](TimeSeries.md)
- フィッティングや応答推定: [フィッティング上級編](../user_guide/tutorials/advanced_fitting.ipynb)
- 実務の例: [伝達関数計測](../user_guide/tutorials/case_transfer_function.ipynb)
- ノイズ床比較: [ノイズバジェット解析](../user_guide/tutorials/case_noise_budget.ipynb)

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
