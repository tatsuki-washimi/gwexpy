# WP3: 環境制約（`environment_blocked` 19 API）検証・境界画定台帳

## 1. 概要
初期公的台帳において `environment_blocked` として記録された19 APIについて、
ローカルで利用可能な依存関係（LAL 7.7.0, PyCBC 2.8.2, NDS2ライブラリ）、公開ネットワークサービス（GWOSC）、
および外部認証・C ABIバイナリ互換性の制約境界をサブプロセス隔離実行により厳密に検証・画定しました。

| 検証群 | 件数 | 実行・検証結果 | 最終判定 |
| :--- | :---: | :--- | :--- |
| **Group 1: LAL・PyCBC変換** | **8** | LAL 7.7.0 / PyCBC 2.8.2 を用いたサブプロセス隔離実行により、GWpyと完全な数値等価性（`numeric_value_parity`）を確認 | **`verified` へ昇格** (8件) |
| **Group 2: Arrakis・NDS2バッファ変換** | **4** | 合成Arrakis / NDS2バッファを用いた変換契約をGWpy並行実行で確認（Arrakis実ライブラリ欠落・NDS2接続要件を明文化） | **`verified` へ昇格** (4件) |
| **Group 3: 外部データ取得** | **6** | ・`fetch_open_data` (2件): 公開GWOSCサービスから非認証HTTPSで正常取得成功<br>・`fetch` (2件): Kerberos認証・NDS2サーバー接続が必要な境界を確認<br>・`find` (2件): LIGO DataFindサーバーまたはローカルフレームキャッシュが必要な境界を確認 | **`verified` へ昇格** (2件)<br>**`environment_blocked` を維持** (4件) |
| **Group 4: ARIMA依存環境** | **1** | `pmdarima` のC拡張とNumPy 2.x (`numpy 2.5.4`) 間のC ABI不整合（`ValueError: dtype size 96 vs 88`）をサブプロセスで再現・記録 | **`environment_blocked` を維持** (1件) |
| **合計** | **19** | **実行可能14件を実質検証完了 / 外部認証・ABI制約5件の境界を確定** | **verified: 14件 / blocked維持: 5件** |

## 2. 個別検証結果一覧（19件）

| # | 検証群 | 識別子 | 元Disposition | 新Disposition | 新Reachability | GWpy比較 | 検証内容・根拠 |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | lal_pycbc_conversion | `gwexpy.timeseries.timeseries.TimeSeries/to_lal` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Elevated from environment_blocked: lal 7.7.0 and pycbc 2.8.2 installed; side-by-side execution matches GWpy exactly |
| 2 | lal_pycbc_conversion | `gwexpy.timeseries.timeseries.TimeSeries/from_lal` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Elevated from environment_blocked: lal 7.7.0 and pycbc 2.8.2 installed; side-by-side execution matches GWpy exactly |
| 3 | lal_pycbc_conversion | `gwexpy.timeseries.timeseries.TimeSeries/to_pycbc` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Elevated from environment_blocked: lal 7.7.0 and pycbc 2.8.2 installed; side-by-side execution matches GWpy exactly |
| 4 | lal_pycbc_conversion | `gwexpy.timeseries.timeseries.TimeSeries/from_pycbc` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Elevated from environment_blocked: lal 7.7.0 and pycbc 2.8.2 installed; side-by-side execution matches GWpy exactly |
| 5 | lal_pycbc_conversion | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/to_lal` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Elevated from environment_blocked: lal 7.7.0 and pycbc 2.8.2 installed; side-by-side execution matches GWpy exactly |
| 6 | lal_pycbc_conversion | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/from_lal` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Elevated from environment_blocked: lal 7.7.0 and pycbc 2.8.2 installed; side-by-side execution matches GWpy exactly |
| 7 | lal_pycbc_conversion | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/to_pycbc` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Elevated from environment_blocked: lal 7.7.0 and pycbc 2.8.2 installed; side-by-side execution matches GWpy exactly |
| 8 | lal_pycbc_conversion | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/from_pycbc` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Elevated from environment_blocked: lal 7.7.0 and pycbc 2.8.2 installed; side-by-side execution matches GWpy exactly |
| 9 | arrakis_nds2_buffer | `gwexpy.timeseries.timeseries.TimeSeries/from_arrakis` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Synthetic buffer contract verified with GWpy parity; noted that genuine Arrakis library is third-party/missing from PyPI |
| 10 | arrakis_nds2_buffer | `gwexpy.timeseries.collections.TimeSeriesDict/from_arrakis` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Synthetic block contract verified with GWpy parity; noted that genuine Arrakis library is third-party/missing from PyPI |
| 11 | arrakis_nds2_buffer | `gwexpy.timeseries.timeseries.TimeSeries/from_nds2_buffer` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Synthetic NDS2 buffer contract verified with GWpy parity; live buffer generation requires NDS2 server connection |
| 12 | arrakis_nds2_buffer | `gwexpy.timeseries.collections.TimeSeriesDict/from_nds2_buffers` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Synthetic NDS2 buffer collection contract verified with GWpy parity; live buffer generation requires NDS2 server connection |
| 13 | external_data_service | `gwexpy.timeseries.timeseries.TimeSeries/fetch_open_data` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Public GWOSC open data fetch operates over unauthenticated HTTPS; successfully received open data |
| 14 | external_data_service | `gwexpy.timeseries.collections.TimeSeriesDict/fetch_open_data` | `environment_blocked` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | Public GWOSC open data fetch operates over unauthenticated HTTPS; successfully received open data dict |
| 15 | external_data_service | `gwexpy.timeseries.timeseries.TimeSeries/fetch` | `environment_blocked` | **`environment_blocked`** | `environment_blocked` | `incomparable` | Maintained as environment_blocked: requires Kerberos authentication ticket and live NDS2 server access |
| 16 | external_data_service | `gwexpy.timeseries.collections.TimeSeriesDict/fetch` | `environment_blocked` | **`environment_blocked`** | `environment_blocked` | `incomparable` | Maintained as environment_blocked: requires Kerberos authentication ticket and live NDS2 server access |
| 17 | external_data_service | `gwexpy.timeseries.timeseries.TimeSeries/find` | `environment_blocked` | **`environment_blocked`** | `environment_blocked` | `incomparable` | Maintained as environment_blocked: requires LIGO DataFind server connection or local frame cache |
| 18 | external_data_service | `gwexpy.timeseries.collections.TimeSeriesDict/find` | `environment_blocked` | **`environment_blocked`** | `environment_blocked` | `incomparable` | Maintained as environment_blocked: requires LIGO DataFind server connection or local frame cache |
| 19 | arima_environment | `gwexpy.timeseries.arima.fit_arima` | `environment_blocked` | **`environment_blocked`** | `environment_blocked` | `incomparable` | Maintained as environment_blocked: pmdarima requires NumPy 1.x C ABI; current audit environment runs numpy 2.5.4 |
