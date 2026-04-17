# GWpy 差分 API 一覧

このページは、**GWpy と比べて何が増えるか**を差分観点で引くための補助インデックスです。  
全 API を網羅するページではありません。完全な仕様一覧は [API リファレンス](../reference/index.rst) を参照してください。

移行の入口から読みたい場合は [GWpy ユーザー向け移行ガイド](gwexpy_for_gwpy_users_ja.md) へ戻ってください。

## まずカテゴリで見る

| カテゴリ | こういうときに見る | 代表項目 | 入口 |
| --- | --- | --- | --- |
| マルチチャネル | 複数チャンネルを一括処理したい | `to_matrix()`, `TimeSeriesMatrix`, `FrequencySeriesMatrix` | [Matrix チュートリアル](tutorials/matrix_timeseries.ipynb) |
| 追加メソッド群 | 外部関数呼び出しをデータオブジェクト側へ寄せたい | `.find_peaks()`, `.fit()`, `.hht()`, `.arima()` | [追加メソッド群](#追加メソッド群) |
| Field API | 時空間データや 4 次元構造を扱いたい | `ScalarField`, `FieldList`, `FieldDict`, `fft_space()` | [Field API](#field-api) |
| 共有 / 互換性 | 結果共有や復元互換性を確認したい | Transparent Pickle | [共有-互換性](#共有--互換性) |

## 詳細一覧

### マルチチャネル

| 代表 API | 安定性 | GWpy との差分 | 詳細リンク |
| --- | --- | --- | --- |
| `TimeSeriesDict.to_matrix()` -> `TimeSeriesMatrix` | Stable | チャンネル集合を行列コンテナへ変換し、一括スペクトル解析や統計処理へ繋げられる | [Matrix チュートリアル](tutorials/matrix_timeseries.ipynb), [TimeSeriesDict](../reference/TimeSeriesDict.md), [TimeSeriesMatrix](../reference/TimeSeriesMatrix.md) |
| `FrequencySeriesDict.to_matrix()` -> `FrequencySeriesMatrix` | Stable | 周波数系列の集合を、ペア比較や一括解析向けのコンテナへ揃えられる | [FrequencySeriesDict](../reference/FrequencySeriesDict.md), [FrequencySeriesMatrix](../reference/FrequencySeriesMatrix.md) |
| `SpectrogramDict.to_matrix()` / `SpectrogramList.to_matrix()` -> `SpectrogramMatrix` | Stable | 時間周波数データの集合を行列として扱い、まとめて後段へ渡せる | [SpectrogramDict](../reference/SpectrogramDict.md), [SpectrogramList](../reference/SpectrogramList.md), [SpectrogramMatrix](../reference/SpectrogramMatrix.md) |

### 追加メソッド群

| 代表 API | 安定性 | GWpy との差分 | 詳細リンク |
| --- | --- | --- | --- |
| `.find_peaks()` | Stable | NumPy 配列へ降ろして SciPy を直接呼ぶ代わりに、データオブジェクト上でピーク検出できる | [周波数系列チュートリアル](tutorials/intro_frequencyseries.ipynb), [TimeSeries](../reference/TimeSeries.md), [FrequencySeries](../reference/FrequencySeries.md) |
| `.fit()` | Stable | フィッティング処理を、データオブジェクトからそのまま開始できる | [フィッティング](tutorials/advanced_fitting.ipynb), [Fitting Reference](../reference/fitting.md) |
| `.hht()` | Experimental | Hilbert-Huang Transform をオブジェクトメソッドとして呼べる | [HHT](tutorials/advanced_hht.ipynb), [TimeSeries](../reference/TimeSeries.md) |
| `.arima()` | Experimental | 時系列モデル化と予測を、時系列オブジェクトのメソッドとして呼べる | [ARIMA](tutorials/advanced_arima.ipynb), [TimeSeries](../reference/TimeSeries.md) |

### Field API

| 代表 API | 安定性 | GWpy との差分 | 詳細リンク |
| --- | --- | --- | --- |
| `ScalarField` | Experimental | 時間 + 空間軸を持つ 4 次元フィールドを、メタデータ付きで保持できる | [Field API 入門](tutorials/field_scalar_intro.ipynb), [ScalarField](../reference/ScalarField.md) |
| `ScalarField.fft_space()` | Experimental | 空間方向の変換を、Field オブジェクトの文脈のまま実行できる | [Field API 入門](tutorials/field_scalar_intro.ipynb), [ScalarField](../reference/ScalarField.md) |
| `FieldList` / `FieldDict` | Experimental | 複数の `ScalarField` をまとめて扱い、バッチ処理や整合性確認を揃えられる | [Field API 入門](tutorials/field_scalar_intro.ipynb), [FieldList](../reference/FieldList.md), [FieldDict](../reference/FieldDict.md) |

### 共有 / 互換性

| 代表 API / 挙動 | 安定性 | GWpy との差分 | 詳細リンク |
| --- | --- | --- | --- |
| Transparent Pickle | Stable | 送信側が GWexpy でも、受信側に GWexpy がなく、GWpy があれば基本クラスとして復元できる | [GWpy ユーザー向け移行ガイド](gwexpy_for_gwpy_users_ja.md), [インストールガイド](installation.md) |

## このページに載せていないもの

- **直接 I/O の形式一覧**は [ファイル I/O 対応フォーマットガイド](io_formats.md) を参照してください。
- **外部ライブラリ変換の一覧**は [Interop / 変換ガイド](interop.md) を参照してください。
- **全 API の仕様**は [API リファレンス](../reference/index.rst) を参照してください。
