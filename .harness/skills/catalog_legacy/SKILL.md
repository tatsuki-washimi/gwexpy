---
name: catalog_legacy
description: |
  docs_internal/references/SampleCodes_GWpy/ の13,711件レガシーコードカタログを
  検索・参照し、gwexpy の新機能開発・サンプルノートブック作成・ギャップ分析に活用する。
  API_MAPPING.md v2（5カテゴリ）との連携で GWpy / scipy / numpy / pandas 等あらゆる
  レガシーパターンの移行パスを提示できる。
---

# Catalog Legacy GWpy Codes

`make_notebook` / `analyze_external` / `extend_gwpy` スキルと連携し、
13,711件のレガシーコードを gwexpy 開発の資産として活用するスキル。

## 依存ファイル

| ファイル | 役割 |
|---------|------|
| `docs_internal/references/SampleCodes_GWpy/CATALOG.json` | 機械可読カタログ（gitignore対象、要事前生成）|
| `docs_internal/references/SampleCodes_GWpy/CATALOG_SUMMARY.md` | 人間可読サマリ（gitignore対象）|
| `docs_internal/references/SampleCodes_GWpy/STRATEGY.md` | 活用戦略・Phase別実行結果 |
| `docs_internal/references/SampleCodes_GWpy/API_MAPPING.md` | GWpy/低レベルAPI→gwexpy 包括マッピング v2（5カテゴリ）|
| `scripts/dev_tools/catalog_legacy_codes.py` | カタログ生成スクリプト |

## カタログ統計（最終スキャン: 2026-03-29）

| 項目 | 値 |
|------|-----|
| 総ファイル数 | 13,711 (.py: 12,751, .ipynb: 960) |
| ユニークパターン | ~7,224（重複除去後）|
| gwexpy 使用ファイル | 69（最優先参照対象）|
| GWpy 使用ファイル | 4,228 |
| GWpy 未使用・外部依存あり | 4,139（scipy/numpy/pandas 等）|
| 総サイズ | 245.6 MB |
| lib_vendor | 0（削除済み）|

## Quick Usage

```
/catalog_legacy                          # カタログ検索（対話モード）
/catalog_legacy --topic PEM              # トピック指定で検索
/catalog_legacy --api TimeSeries.read    # GWpy API 使用例を検索
/catalog_legacy --dep scipy.signal       # 外部依存で検索（低レベル移行対象）
/catalog_legacy --gwexpy                 # gwexpy 使用ファイルを一覧（69件）
/catalog_legacy --gap                    # gwexpy チュートリアルのギャップ分析
/catalog_legacy --new-feature            # 新機能候補の優先度リストを表示
/catalog_legacy --migration scipy        # scipy → gwexpy 移行パターン検索
/catalog_legacy --update                 # カタログを再生成
```

---

## ステップ 1: カタログ確認・生成

CATALOG.json が存在するか確認する:

```bash
ls docs_internal/references/SampleCodes_GWpy/CATALOG.json
```

存在しない場合は生成する（所要時間: 約5〜10分）:

```bash
python scripts/dev_tools/catalog_legacy_codes.py --verbose
```

生成済みサマリで素早く概要を確認:

```bash
cat docs_internal/references/SampleCodes_GWpy/CATALOG_SUMMARY.md
```

---

## ステップ 2: カタログ読み込みと検索

Python を使ってカタログを読み込む（エージェントが直接実行する場合）:

```python
import json
from pathlib import Path

catalog = json.loads(
    Path("docs_internal/references/SampleCodes_GWpy/CATALOG.json").read_text()
)

# --- トピック検索 ---
def search_by_topic(topic: str) -> list[dict]:
    return [e for e in catalog if topic.lower() in e["topic"].lower()]

# --- パス部分一致検索（サブディレクトリ指定）---
def search_by_path(keyword: str) -> list[dict]:
    return [e for e in catalog if keyword.lower() in e["path"].lower()]

# --- GWpy API 検索 ---
def search_by_gwpy_api(api: str) -> list[dict]:
    return [e for e in catalog if any(api in a for a in e["gwpy_apis"])]

# --- 外部依存検索（scipy / pandas / sklearn 等）---
def search_by_dep(dep: str) -> list[dict]:
    return [e for e in catalog if any(dep in d for d in e["external_deps"])]

# --- gwexpy API 使用ファイル検索 ---
def search_by_gwexpy_api(api: str) -> list[dict]:
    return [e for e in catalog if any(api in a for a in e["gwexpy_apis"])]

# --- 重複除去: 代表ファイルのみ取得 ---
def get_unique_entries(entries: list[dict]) -> list[dict]:
    seen_groups: set[str] = set()
    unique: list[dict] = []
    for e in entries:
        g = e["duplicate_group"]
        if g is None:
            unique.append(e)
        elif g not in seen_groups:
            seen_groups.add(g)
            unique.append(e)
    return unique

# --- gwexpy 使用ファイルを優先取得（最優先参照対象）---
gwexpy_files = [e for e in catalog if e["gwexpy_apis"]]

# --- GWpy 未使用で外部依存あり（Category 3 移行対象）---
low_level_files = [
    e for e in catalog
    if not e["gwpy_apis"] and e["external_deps"] and e["lines_of_code"] > 30
]

# --- scipy 使用ファイル（移行優先度高）---
scipy_files = search_by_dep("scipy")

# --- コード量の多い実装を取得（軽量ファイル除外）---
substantial = [e for e in catalog if e["lines_of_code"] > 100 and not e["is_untitled"]]
```

### トピック分類の注意点

`topic` フィールドはディレクトリ名ベースのヒューリスティック分類:

```python
# injection 解析ファイルの多くは topic='PEM' に分類されている
# （PEM/PEMinjection/ 配下のため）
# → topic ではなく path で絞り込む
injection_files = search_by_path("injection")
unique_injection = get_unique_entries(injection_files)

# calibration の大半は GWpy 未使用（scipy.optimize 直接使用）
# → Category 3（移行対象）の主要ターゲット
cal_files = search_by_topic("calibration")
cal_low_level = [e for e in cal_files if not e["gwpy_apis"]]
```

---

## ステップ 3: 重複グループを尊重する

13,711件の大半は重複ファイルである。読む前に必ず重複除去を行う。

| 重複グループ | ファイル数 | 代表ファイル |
|------------|---------|------------|
| residual | 196 | `residual.py` |
| plot_each_meas | 115 | `plot_each_meas.py` |
| __init__ (886e95) | 112 | `__init__.py` |
| WSK_calibration_meas | 102 | `WSK_calibration_meas.py` |
| NInjA 系 | 42+ | `GoogleDrive/PEM/PEMinjection/NInjA.py` |
| read_adx3 系 | 38 | `GoogleDrive/PEM/Acoustic/infrasound/Analysis/Dev_Prod/read_adx3.py` |
| InjectionAnalysis 系 | 36 | `GoogleDrive/PEM/PEMinjection/tmp/InjectionAnalysis.py` |
| gbd2gwf 系 | 29 | `GoogleDrive/PEM/Acoustic/infrasound/Analysis/gbd2gwf.py` |

```python
# 代表ファイルを1件だけ取得してから読む
injection_files = search_by_path("PEMinjection")
unique_injection = get_unique_entries(injection_files)
representative_path = unique_injection[0]["path"]
# → Read tool でファイルを読む
```

---

## ステップ 4: API_MAPPING.md v2 で移行ガイドを確認

レガシーコードを gwexpy へ移行する際は `API_MAPPING.md` を参照する。

```bash
cat docs_internal/references/SampleCodes_GWpy/API_MAPPING.md
```

### API_MAPPING.md v2 の5カテゴリ

| カテゴリ | 内容 | 対象ファイル数 |
|---------|------|-------------|
| **Cat.1 直接互換** | GWpy API → gwexpy（サブクラスで完全互換）| 4,228 |
| **Cat.2 拡張互換** | gwexpy 独自強化（Matrix, 分解, Bruco, 結合解析等）| — |
| **Cat.3 移行対象** | scipy/numpy/pandas/sklearn → gwexpy | ~4,139 |
| **Cat.4 連携** | 外部ツール ↔ gwexpy（50+ interop）| — |
| **Cat.5 新機能候補** | 未実装 P0-P3 | — |

### 主要移行パターン早見表

| レガシーパターン | 件数 | gwexpy 対応 | カテゴリ |
|----------------|------|------------|---------|
| `TimeSeries.read()` | 1,050 | そのまま使える | Cat.1 |
| `ts.asd()` / `.psd()` | 526 / 122 | そのまま使える | Cat.1 |
| `scipy.signal.welch()` | ~150 | `ts.psd()` | Cat.3 |
| `scipy.signal.find_peaks()` | 49 | `ts.find_peaks()` | Cat.3 |
| `scipy.optimize.curve_fit()` | ~180 | `fit_series()` | Cat.3 |
| `sklearn.FastICA` | 5 | `matrix.ica_fit()` | Cat.3 |
| `pandas.DataFrame.rolling()` | 17 | `ts.rolling_mean()` 等 | Cat.3 |
| `geopandas.GeoDataFrame` 地図描画 | 46 | `GeoMap` | Cat.3 |
| `plot_mmm()` 自作 | 181 | `gwexpy.plot.plot_mmm()` | Cat.3 |
| `obspy.read()` | 412 | `TimeSeries.read(format='miniseed')` | Cat.4 |
| python-control FRD | 1 | `to_control_frd()` / `from_control_frd()` | Cat.4 |
| dttxml | 少数 | `read(..., format='dttxml', products=...)` | Cat.4 |
| ADX3 reader | 38 | **P1 未実装** | Cat.5 |
| Lorentzian モデル | 29 | **P1 未実装** | Cat.5 |

---

## ステップ 5: ギャップ分析

### P0: 実装済みだがチュートリアル未作成

| 機能 | gwexpy モジュール | レガシー出現数 |
|------|-----------------|-------------|
| ICA/PCA 分解 | `TimeSeriesMatrix.ica_fit()` / `.pca_fit()` | 18 |
| GBD フォーマット読み込み | `TimeSeries.read(format='gbd')` | 17 |
| ObsPy 地震データ読み込み | `TimeSeries.read(format='miniseed'/'sac')` | 24 |
| カップリング関数解析 | `gwexpy.analysis.coupling` | 36 |
| ホワイトニング前処理 | `gwexpy.signal.preprocessing.whiten()` | ~30 |

### P1: 高頻度・未実装

| 新機能 | 出現数 | 提案 API |
|--------|--------|---------|
| ADX3 CSV リーダー | 38 | `format='adx3'` |
| Lorentzian フィットモデル | 29 | `fitting.models.lorentzian` |
| インジェクション解析パイプライン (NInjA 互換) | 36 | `InjectionAnalysisPipeline` |
| スペクトログラムクリーニング | 30+ | `sg.clean()` |
| SNR スペクトログラム | 37 | `sg.normalize(method='snr')` |

### コードによるギャップ確認

```python
# gwexpy 使用中の API 種類
gwexpy_apis_used = set()
for e in gwexpy_files:
    gwexpy_apis_used.update(e["gwexpy_apis"])
print("gwexpy API 使用状況:", gwexpy_apis_used)

# scipy 使用ファイルで GWpy 未使用（移行優先候補）
cat3_candidates = [
    e for e in catalog
    if not e["gwpy_apis"]
    and any("scipy" in d for d in e["external_deps"])
    and e["lines_of_code"] > 50
]
print(f"Category 3 移行候補: {len(cat3_candidates)} ファイル")
```

---

## ステップ 6: 代表ファイルの詳細解析

catalog_legacy でファイルを特定した後は、`analyze_external` スキルで詳細解析する。

```
/catalog_legacy --api TimeSeriesDict.coherence
→ 使用ファイルのパスリスト取得
→ /analyze_external --code で詳細解析（手動実装パターンの把握）
→ /make_notebook で gwexpy 版チュートリアル生成
```

### gwexpy 使用ファイル（最優先）の参照パス

```python
# BifrequencyMap / Schumann 解析
schumann_files = search_by_gwexpy_api("BifrequencyMap")
# → GoogleDrive/PEM/MAG/MFS/analysis_LongTerm/schumann_analysis_gwexpy.py

# fit_series 使用
fitting_files = search_by_gwexpy_api("fit_series")
# → GoogleDrive/PEM/MAG/MFS/analysis_LongTerm/Schumann_pipeline.py

# NInjA 使用（injection 解析）
ninja_files = search_by_gwexpy_api("NInjA")
# → GoogleDrive/PEM/PEMinjection/NInjA.py

# ADX3 実装パターン（P1 実装前の参照用）
adx3_files = get_unique_entries(search_by_path("adx3"))
# → GoogleDrive/PEM/Acoustic/infrasound/Analysis/Dev_Prod/read_adx3.py
```

---

## 他スキルとの連携フロー

### フロー A: サンプルノートブック作成

```
1. /catalog_legacy --topic <topic>  OR  --dep scipy.signal
   → 代表ファイルを特定（重複除去済み）
2. /analyze_external --code <path>
   → 手動処理パターン・scipy/numpy の使い方を把握
3. API_MAPPING.md (Cat.3) で gwexpy 対応 API を確認
4. /make_notebook
   → gwexpy 版チュートリアルを生成（Before/After コード例付き）
```

**例: scipy.signal.welch → gwexpy チュートリアル**

```
1. /catalog_legacy --dep scipy.signal
   → welch 使用ファイルを検索（~150件）→ 代表を特定
2. /analyze_external --code <path>
   → welch のパラメータ・使用コンテキストを把握
3. API_MAPPING.md 3.1 で ts.psd() との対応確認
4. /make_notebook case_scipy_migration.ipynb
```

### フロー B: 新機能実装

```
1. /catalog_legacy --new-feature
   → 優先度マトリクスを確認（ADX3 リーダーが P1 最優先）
2. /catalog_legacy --api "adx3"
   → 全実装例を収集（38件 → 代表 1件）
3. /analyze_external --code <代表ファイル>
   → 仕様を精読（バリアント検出・タイムスタンプ修復ロジック等）
4. /extend_gwpy
   → gwexpy/timeseries/io/adx3.py を実装
5. /make_notebook case_adx3_reader.ipynb
```

**例: Lorentzian モデル追加**

```
1. /catalog_legacy --dep iminuit
   → Minuit 使用 29件の代表ファイルを確認
2. /analyze_external --code <代表ファイル>
   → どの関数形式（Lorentzian / multi-peak）が使われているか把握
3. /extend_gwpy
   → gwexpy/fitting/models.py に lorentzian(), multi_lorentzian() を追加
4. /make_notebook case_lorentzian_fit.ipynb
```

### フロー C: チュートリアルギャップ解消（P0）

```
1. /catalog_legacy --gap
   → P0リスト確認（ICA/PCA チュートリアルが最優先）
2. /catalog_legacy --dep sklearn（sklearn 使用 69件を検索）
   → get_unique_entries() で代表を絞る
3. /analyze_external --code <代表ファイル>
   → 実際の FastICA 使用パターンを把握
4. /make_notebook case_ica_pca_decomposition.ipynb
   → gwexpy TimeSeriesMatrix.ica_fit() を使用した ICA チュートリアル
```

### フロー D: 外部ツール連携ワークフロー

```
1. /catalog_legacy --dep obspy
   → 412件のうち代表を特定
2. API_MAPPING.md (Cat.4) で obspy ↔ gwexpy 変換 API を確認
3. /analyze_external --code <代表ファイル>
   → obspy の使い方（Trace/Stream 操作, FDSN クライアント等）を把握
4. /make_notebook case_seismic_analysis.ipynb
   → TimeSeries.read(format='miniseed') + from_obspy_trace() の使い方
```

### フロー E: python-control / finesse 連携確認

```
1. /catalog_legacy --dep control
   → VIS / MIF での使用例を確認（1件）
2. API_MAPPING.md (Cat.4) で to_control_frd() / from_finesse_frequency_response() を確認
3. /make_notebook case_control_interop.ipynb
   → 懸架系 TF を FrequencySeriesMatrix → python-control FRD 変換
```

---

## ユースケース別クイックレファレンス

### 「このレガシーパターンを gwexpy で書き直したい」

```python
# 1. 該当ファイルを検索
files = search_by_dep("scipy.signal")  # または search_by_gwpy_api("TimeSeries.read")

# 2. 重複除去
unique = get_unique_entries(files)
print(f"{len(files)} 件 → {len(unique)} ユニークパターン")

# 3. 代表ファイルのパスを取得
for e in unique[:5]:
    print(e["path"], e["lines_of_code"], "lines")

# 4. API_MAPPING.md の該当セクションを確認
# → Cat.3 なら Before/After コード例あり
```

### 「新機能追加に使えるレガシーの実装例を探したい」

```python
# ADX3 パーサの全バリアントを収集
adx3_variants = search_by_path("adx3")
unique_adx3 = get_unique_entries(adx3_variants)

# ファイルサイズ・LOC 順でソート（最も充実した実装を優先）
unique_adx3.sort(key=lambda e: e["lines_of_code"], reverse=True)

# 上位3件を参照
for e in unique_adx3[:3]:
    print(e["path"], e["lines_of_code"], "lines")
```

### 「gwexpy をすでに使っているコードを参考にしたい」

```python
# gwexpy 使用ファイル全69件
for e in sorted(gwexpy_files, key=lambda e: e["lines_of_code"], reverse=True):
    print(e["path"], e["gwexpy_apis"])
```

---

## 注意事項

- CATALOG.json / CATALOG_SUMMARY.md は `.gitignore` 対象（自動生成）
- SampleCodes_GWpy/ の実体ファイルも gitignore 対象（245.6 MB）
- カタログはスキャン時点のスナップショット。ファイル追加・更新後は再生成:
  `python scripts/dev_tools/catalog_legacy_codes.py --verbose`
- `is_untitled: true` のファイルは内容確認まで参照価値不明
- injection 解析ファイルは `topic == "PEM"` に分類されているため `search_by_path("injection")` を使う
- DGS / camera_DAQ は guardian/EPICS 制御・カメラサーバーが主体で gwexpy 対象外が多い
- VIS は python-control 連携が有効（Cat.4）
