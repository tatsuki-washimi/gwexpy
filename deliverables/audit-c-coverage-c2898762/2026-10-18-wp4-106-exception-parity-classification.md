# 106件の Exception Parity 原因別分類・再評価台帳

**監査対象コミット**: `c2898762ca68e20a1a241fffb3c3262b0a27290b`  
**作成日**: 2026-10-18  
**対象母集団**: C監査ベースライン台帳において `gwpy_comparison == "exception_parity"` と記録された 106 API  

## 1. エグゼクティブサマリー

旧台帳で「GWpy と例外が一致した」として `exception_parity` に分類されていた 106 件について、実行コードと例外内容を精査した結果、**94件（88.7%）がハーネスの引数不足によるゼロ引数呼び出し（TypeError）起因**であることが判明しました。

| 分類カテゴリ | 件数 | 構成比 | 発生原因・挙動の真相 | 実質的影響と評価 |
| :--- | :---: | :---: | :--- | :--- |
| **カテゴリA: 単なる引数不足の一致** | 94 | 88.7% | ハーネスが引数を要するメソッドを無引数 `gw_attr()` で呼び出し、双方が `TypeError: missing required argument` を発火。 | 製品の真のパリティを証明しておらず、数値比較が未実施のまま例外一致として処理された「見かけのパリティ」。 |
| **カテゴリB: 異常・ドメイン境界入力の一致** | 10 | 9.4% | 空配列・無効フォーマット・存在しないファイル指定等により、双方が `ValueError`, `IndexError`, `IORegistryError` を発火。 | 異常系・境界条件における例外契約の正当な一致。 |
| **カテゴリC: 上流 Matplotlib 描画挙動** | 2 | 1.9% | Matplotlib のバージョン差異や描画スタイル指定（drawstyle step）により双方が `AttributeError` を発火。 | 上流ライブラリ仕様に起因する挙動一致（WP1/WP2是正済み）。 |
| **合計** | **106** | **100.0%** | **見かけの一致（94件）と正当な境界例外（12件）を厳格に分離** |

## 2. 旧 Disposition とのクロス集計

| 旧 Disposition | カテゴリA（引数不足） | カテゴリB（異常系・境界） | カテゴリC（上流Matplotlib） | 合計 |
| :--- | :---: | :---: | :---: | :---: |
| `verified` | 48 | 2 | 0 | **50** |
| `contract_pending` | 29 | 2 | 1 | **32** |
| `environment_blocked` | 14 | 0 | 0 | **14** |
| `differ` | 3 | 6 | 1 | **10** |

### 重大な監査知見
1. **Verified 50件の性質**: `verified` に属する 50 件は、GWexpy 側の positive control では有効なフィクスチャによる契約検証に成功しているものの、GWpy 実機比較においては無引数呼び出しによる `TypeError` が記録されていました。したがって、これらの「GWpy 比較」は数値パリティの実証ではなく「比較保留（`comparison_pending`）」として扱うのが誠実な記録です。
2. **Differ 10件の性質**: `differ` に属する 10 件のうち、6 件は `whiten`, `gate` などの信号処理契約であり、デフォルトのサンプル長不足（10サンプル）で両者または GWexpy が `ValueError` を発火したものです（WP1で既に原因特定済み）。

## 3. カテゴリB（異常・境界入力）およびカテゴリC（上流挙動）の12件全件一覧

| 識別子 | 旧Disposition | 観測例外 | 分類 | 評価・備考 |
| :--- | :---: | :---: | :--- | :--- |
| `gwexpy.plot.plot.Plot/colorbar` | `contract_pending` | `ValueError` | abnormal_or_domain_boundary_input | Both raised ValueError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.spectrogram.spectrogram.Spectrogram/from_spectra` | `contract_pending` | `ValueError` | abnormal_or_domain_boundary_input | Both raised ValueError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.spectrogram.spectrogram.Spectrogram/step` | `contract_pending` | `AttributeError` | upstream_matplotlib_behavior | Both raised AttributeError due to upstream Matplotlib drawing/backend method differences (e.g. drawstyle step) |
| `gwexpy.spectrogram.spectrogram.Spectrogram/write` | `verified` | `IndexError` | abnormal_or_domain_boundary_input | Both raised IndexError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.timeseries.timeseries.TimeSeries/find_gates` | `differ` | `ValueError` | abnormal_or_domain_boundary_input | Both raised ValueError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.timeseries.timeseries.TimeSeries/gate` | `differ` | `ValueError` | abnormal_or_domain_boundary_input | Both raised ValueError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.timeseries.timeseries.TimeSeries/mask` | `differ` | `ValueError` | abnormal_or_domain_boundary_input | Both raised ValueError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.timeseries.timeseries.TimeSeries/q_gram` | `differ` | `IndexError` | abnormal_or_domain_boundary_input | Both raised IndexError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.timeseries.timeseries.TimeSeries/q_transform` | `differ` | `ValueError` | abnormal_or_domain_boundary_input | Both raised ValueError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.timeseries.timeseries.TimeSeries/whiten` | `differ` | `ValueError` | abnormal_or_domain_boundary_input | Both raised ValueError on domain boundary, invalid format, or missing target file conditions |
| `gwexpy.types.array2d.Array2D/step` | `differ` | `AttributeError` | upstream_matplotlib_behavior | Both raised AttributeError due to upstream Matplotlib drawing/backend method differences (e.g. drawstyle step) |
| `gwexpy.types.array2d.Array2D/write` | `verified` | `IORegistryError` | abnormal_or_domain_boundary_input | Both raised IORegistryError on domain boundary, invalid format, or missing target file conditions |
