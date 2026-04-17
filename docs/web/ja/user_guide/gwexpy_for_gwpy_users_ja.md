# GWpy ユーザー向け移行ガイド

このページは、**GWpy から GWexpy へ移るときの入口**です。  
全 API を網羅するページではありません。まずは「何がそのまま動くか」と「どこから差分を使うと効果が大きいか」を短時間で掴むことを目的にしています。

完全な API 一覧ではなく、**GWpy との差分だけを引きたい**場合は [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md) を参照してください。  
全 API の仕様を追いたい場合は [API リファレンス](../reference/index.rst) を参照してください。

## まず最初に押さえること

- **単一チャネルの基本操作はかなりそのままです。**
  `TimeSeries` / `FrequencySeries` / `Spectrogram` を使った読み込み、プロット、基本スペクトル解析は、まず import を `gwexpy` 側へ置き換えるだけで試せます。
- **差分が大きいのは多チャネル処理です。**
  `TimeSeriesDict` を手作業でループする代わりに、`to_matrix()` で `TimeSeriesMatrix` へ変換して一括処理できます。
- **外部ライブラリ呼び出しをデータオブジェクト側へ寄せられます。**
  代表例は `.find_peaks()`, `.fit()`, `.hht()`, `.arima()` です。
- **I/O と interop は別ガイドを見る前提です。**
  直接 I/O は [ファイル I/O 対応フォーマットガイド](io_formats.md)、外部ライブラリ変換は [Interop / 変換ガイド](interop.md) を正本とします。

## どこから書き換えると効果が大きいか

| 目的 | まず見る差分 | 深掘り先 |
| --- | --- | --- |
| 複数チャンネルをまとめて解析したい | `TimeSeriesDict.to_matrix()` → `TimeSeriesMatrix` | [Matrix チュートリアル](tutorials/matrix_timeseries.ipynb) |
| SciPy / Statsmodels 呼び出しを整理したい | オブジェクトメソッド化された追加 API | [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md) |
| 既存の単一チャネルコードを早く移したい | import だけ差し替えてから必要箇所だけ差分 API を追加 | [クイックスタート](quickstart.md) |
| 結果共有の互換性を知りたい | Transparent Pickle の挙動 | [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md) |

## 差分レシピ 1: `TimeSeriesDict` の手ループを `TimeSeriesMatrix` に寄せる

GWpy では、複数チャンネルの比較やペアごとのスペクトル計算を、自分でループして組み立てることが多くなります。  
GWexpy では `to_matrix()` を入口にして、チャンネル集合をそのまま一括解析の対象にできます。

### GWpy style

```python
from gwpy.timeseries import TimeSeriesDict

tsd = TimeSeriesDict.read(cache, channels)
reference = tsd["H1:STRAIN"]

csd = {}
for name, ts in tsd.items():
    if name == "H1:STRAIN":
        continue
    csd[name] = ts.csd(reference, fftlength=4)
```

### GWexpy style

```python
from gwexpy.timeseries import TimeSeriesDict

tsd = TimeSeriesDict.read(cache, channels)
matrix = tsd.to_matrix()

csm = matrix.csd(fftlength=4)
csm.plot().show()
```

この差分が効く場面:

- ループを減らして、チャンネル集合をそのまま計算対象にしたいとき
- 多チャネル解析を `TimeSeriesMatrix` / `FrequencySeriesMatrix` へ揃えたいとき

関連ページ:

- [Matrix チュートリアル](tutorials/matrix_timeseries.ipynb)
- [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md)
- [TimeSeriesDict リファレンス](../reference/TimeSeriesDict.md)
- [TimeSeriesMatrix リファレンス](../reference/TimeSeriesMatrix.md)

## 差分レシピ 2: 外部関数呼び出しをオブジェクトメソッドへ寄せる

GWpy ベースのコードでは、NumPy 配列へ取り出してから SciPy / Statsmodels を直接呼ぶ流れが自然です。  
GWexpy では、その一部がデータオブジェクトのメソッドとしてまとまっています。

### GWpy style

```python
import numpy as np
from scipy.signal import find_peaks
from gwpy.frequencyseries import FrequencySeries

spec = FrequencySeries(...)
peaks, props = find_peaks(np.asarray(spec.value), height=0.2)
```

### GWexpy style

```python
from gwexpy.frequencyseries import FrequencySeries

spec = FrequencySeries(...)
peaks, props = spec.find_peaks(threshold=0.2)
```

同じ方向の差分として、`gwexpy` では次のような API も追加されています。

- `.fit()` : データオブジェクトに対するフィッティング
- `.hht()` : Hilbert-Huang Transform
- `.arima()` : 時系列予測・モデル化

関連ページ:

- [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md)
- [周波数系列チュートリアル](tutorials/intro_frequencyseries.ipynb)
- [フィッティング](tutorials/advanced_fitting.ipynb)
- [HHT](tutorials/advanced_hht.ipynb)
- [ARIMA](tutorials/advanced_arima.ipynb)

## 差分レシピ 3: 単一チャネルコードは大きく変えなくてよい

GWpy の基本クラスに慣れている場合、最初から全面的に書き換える必要はありません。  
まずは import を `gwexpy` 側へ置き換え、必要になった箇所だけ差分 API を足す進め方が現実的です。

### GWpy style

```python
from gwpy.timeseries import TimeSeries

ts = TimeSeries.read("data.gwf", "H1:STRAIN")
asd = ts.asd(fftlength=4)
asd.plot().show()
```

### GWexpy style

```python
from gwexpy.timeseries import TimeSeries

ts = TimeSeries.read("data.gwf", "H1:STRAIN")
asd = ts.asd(fftlength=4)
asd.plot().show()
```

実務上の見方:

- まずは既存の単一チャネル処理をそのまま持ってくる
- 多チャネル化や追加メソッドが必要になった地点で `gwexpy` 固有 API を使う

関連ページ:

- [クイックスタート](quickstart.md)
- [チュートリアル一覧](tutorials/index.rst)
- [API リファレンス](../reference/index.rst)

## 差分レシピ 4: Pickle 共有時の互換性を見る

GWexpy は、解析結果の共有時に「受け取り側が GWexpy を入れていない」ケースも意識しています。  
このページでは安全性の一般論ではなく、**GWpy ユーザーが共有運用で何を期待できるか**だけを押さえます。

### GWpy style

```python
import pickle
from gwpy.timeseries import TimeSeries

ts = TimeSeries(...)

with open("result.pkl", "wb") as f:
    pickle.dump(ts, f)

# 受け取り側も、同じ型を読める環境を前提に共有する
```

### GWexpy style

```python
import pickle
from gwexpy.timeseries import TimeSeries

ts = TimeSeries(...)

with open("result.pkl", "wb") as f:
    pickle.dump(ts, f)

# 受け取り側に GWexpy がなくても、GWpy があれば基本クラスとして復元できる
```

:::{important}
Pickle は信頼できるデータだけをロードしてください。
:::

関連ページ:

- [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md)
- [インストールガイド](installation.md)

## I/O と外部ライブラリ連携は別ページを見る

このページでは、I/O 形式や外部ライブラリ連携の一覧を再掲しません。  
それぞれ既に専用ページがあるため、移行時は次を正本として参照してください。

- 直接 I/O: [ファイル I/O 対応フォーマットガイド](io_formats.md)
- 外部ライブラリ変換: [Interop / 変換ガイド](interop.md)

## 次のステップ

- [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md) - 追加 API を差分観点で引く
- [チュートリアル一覧](tutorials/index.rst) - 差分レシピから実例へ進む
- [ファイル I/O 対応フォーマットガイド](io_formats.md) - 読み書き形式を確認する
- [Interop / 変換ガイド](interop.md) - 外部ライブラリとの橋渡しを見る
- [API リファレンス](../reference/index.rst) - 全 API の仕様を確認する
