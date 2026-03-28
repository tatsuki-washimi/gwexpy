# 実施概要

GWexpyでは現行 `gwpy>=3.0,<4.0` を仮定しており、GWpy 4.0.0 で多数の破壊的変更が導入されています【12†L146-L148】【124†L51-L59】。本調査では、GWpy 3.9→4.0 の変更点のうち gwexpy に影響する点を抽出し、影響範囲、修正方針、移行手順を提示します。以下、大見出しごとに要約・詳細を述べます。

## 1. GWpy 4.0 の主要な破壊的変更と新機能

GWpy 4.0.0 では以下の主要変更があり、gwexpy への影響が予想されます【124†L51-L59】【124†L95-L97】【124†L134-L137】。影響度の高いものを列挙します：

- **Python要件の更新**：Python 3.11 以上が必須に（3.9/3.10廃止、3.14 追加）【124†L51-L59】。  
- **I/Oシステム刷新**：`.read()/write()` 系処理が Astropy のクラスベース I/O レジストリに移行【124†L51-L59】。既存の `register_reader` 等は新しい `gwpy.io.registry.default_registry` への登録に切り替える必要があります【124†L62-L64】。並列処理はプロセス (`nproc`) からスレッド (`parallel`) へ変更されています【124†L58-L64】【124†L134-L136】。  
- **gwpy.io.mp の削除**：標準ライブラリの `multiprocessing/concurrent.futures` を推奨【124†L95-L97】。gwexpy 独自の `gwexpy/io/mp.py` は不要となるか置き換えが必要です。  
- **TimeSeries.get系の変更**：`TimeSeries.get()` メソッドに `source` 引数が追加され、`fetch`/`find` は内部的に `get` に統合【124†L66-L74】【124†L112-L119】。既存コードで明示的にデータソースを指定している場合は引数名や挙動を確認します。  
- **その他重要変更**：`gwpy.signal.filter_design` の機能更新（`bilinear_zpk` 削除等）【124†L85-L93】、`igwn_ligolw` への切替【124†L78-L84】、`DataQualityDict.query_dqsegdb` の `url→host` 名変更【124†L121-L124】、`verbose` キーワードの廃止【124†L130-L133】、`gopen` 関数廃止【124†L138-L140】など。

これらのうち、**gwexpy で直接使われている可能性が高い**のは「I/Oレジストリの変更」「`nproc→parallel`」「`gwpy.io.mp` の削除」「一部関数引数名変更」です。例えば、独自のファイル形式読込機能（WAV, TDMS, GBD, WIN, ATS 等）はすべて I/O レジストリを用いて実装されているため、4.0 への対応が必要です。

## 2. gwexpy 内の影響箇所マッピング

gwexpy のコード中で影響が大きいのは主に *I/O 関連モジュール* です。該当ファイルと主な依存先 API、影響度を表にまとめました。

| ファイルパス                                     | 依存関数/クラス           | 参照 GWpy API              | 想定影響                 | 対応工数 |
|:---------------------------------------------|:-----------------------|:-----------------------|:---------------------|:------:|
| `gwexpy/io/mp.py`                              | `gwpy.io.mp` 全部       | `gwpy.io.mp` モジュール        | **破壊的**: モジュール削除【124†L95-L97】 | 高     |
| `gwexpy/timeseries/io/wav.py`                  | 独自 WAV リーダ           | `gwpy.io.registry.register_reader`  | レジストリ方式変更【124†L51-L59】   | 中     |
| `gwexpy/timeseries/io/tdms.py`                 | 独自 TDMS リーダ          | `gwpy.io.registry.register_reader`  | レジストリ方式変更           | 中     |
| `gwexpy/timeseries/io/win.py` (WIN/WIN32)     | 独自 WIN リーダ          | `gwpy.io.registry.register_reader`  | レジストリ方式変更           | 中     |
| `gwexpy/timeseries/io/gbd.py` (GRAPHTEC GBD)  | 独自 GBD リーダ          | `gwpy.io.registry.register_reader`  | レジストリ方式変更           | 中     |
| `gwexpy/timeseries/io/ats.py` (Metronix ATS)  | 独自 ATS/ATS.MTH5 リーダ   | `gwpy.io.registry.register_reader`  | レジストリ方式変更           | 中     |
| `gwexpy/timeseries/matrix.py`                 | 継承: `gwpy.timeseries.TimeSeries` | `gwpy.timeseries.TimeSeries`     | 対応不要* (API互換維持)       | 低     |
| `gwexpy/plot/plot.py`                         | 継承: `gwpy.plot.Plot`   | `gwpy.plot.Plot`            | 対応不要 (互換)             | 低     |
| その他独自処理 (DBアクセス等)                    | `parallel`/`nproc` 引数   | `.read()`系引数              | 引数名変更【124†L134-L136】   | 低     |

※「影響度」は大まかに **高/中/低** で評価。  
以上から、**優先度が最も高いのは I/O モジュール全般**で、次いで `gwexpy/io/mp.py` の処理です。TimeSeriesBaseやPlot継承部分は直接の破壊的変更は少なく、影響は限定的です。

## 3. 具体的な移行対応例

### 3.1 I/O レジストリの更新

*Before (旧API)*:
```python
from gwpy.io import registry as io_registry
# ...
io_registry.register_reader("tdms", TimeSeries, read_timeseries_tdms)
```

*After (GWpy 4.0)*:
```python
from gwpy.io.registry import default_registry as io_registry
# あるいは import gwpy.io.registry as registry; io_registry = registry.default_registry
# ...
io_registry.register_reader("tdms", TimeSeries, read_timeseries_tdms)
```

GWpy 4.0 では `gwpy.io.registry.default_registry` を用いて登録します【124†L62-L64】【125†L12-L16】。たとえば `wav.py` での修正例：

```diff
-from gwpy.io import registry as io_registry
+from gwpy.io.registry import default_registry as io_registry
```

さらに、並列処理引数の変更にも注意します。例えば古い `.read(..., nproc=4)` なら、4.0 では `.read(..., parallel=4)` に置き換えます【124†L58-L64】【124†L134-L136】。このほか `verbose` キーワードを利用している場合は削除し、代わりにログ設定を行います。

### 3.2 gwpy.io.mp の削除

`gwexpy/io/mp.py` では単純に GWpy の `io.mp` を再エクスポートしていました:

```python
from gwpy.io.mp import *
```

GWpy 4.0 では `gwpy.io.mp` が削除されたので、**このファイルごと削除**するか、必要な機能のみ標準ライブラリ（`multiprocessing`, `concurrent.futures`）で置き換えます【124†L95-L97】。例えば、何らかの並列 I/O サポートが必要なら Python 標準のパラレル処理を用意し直します。

### 3.3 `nproc`→`parallel` 対応

GWpy 4.0 では `.read(..., nproc=)` は非推奨となり、`.read(..., parallel=)` が新仕様です【124†L58-L64】【124†L134-L136】。gwexpy 側で独自の `.read()` クラスメソッドを持つ場合（現状のコード上は特殊実装なし）、呼び出し側が `nproc` を使っていれば `parallel` に書き換えます。加えて、もし `verbose` パラメータを渡している場合も削除し、ログ出力は `gwpy.log` 経由に移行すべきです【124†L130-L133】。

### 3.4 パッチ例 (wav.py 抜粋)

```diff
-from gwpy.io import registry as io_registry
+from gwpy.io.registry import default_registry as io_registry

 # --- Reader registration ---
-io_registry.register_reader("wav", TimeSeriesDict, read_timeseriesdict_wav)
+io_registry.register_reader("wav", TimeSeriesDict, read_timeseriesdict_wav)
```

このようにインポート先を変更し、以降の `io_registry.register_*` 呼び出しは同様に行います。また、`registry.register_identifier` も同様に `default_registry` で実行できます。

## 4. 実装タスクの優先順位と見積

移行作業は以下のようなタスクに分割できます。

| タスク                             | 優先度 | 見積時間 | リスク | テスト追加例 |
|:---------------------------------|:----:|:------:|:----:|:---------|
| **I/O モジュールの修正**<br>―― WAV/TDMS/WIN/GBD/ATS 等 (I/O 登録部分の更新) | 高    | 1–2日各 | 中   | 各フォーマットの読み込みテスト（既存サンプルファイルで読めるか） |
| **gwexpy/io/mp.py の削除・代替** | 高    | 1日   | 中   | `mp` を使ったAPIの動作確認（必要なら並列タスクのテスト） |
| `nproc→parallel` 修正           | 中    | 0.5日  | 低   | `.read(parallel=...)` が動くか確認 |
| `verbose` 等の不要引数削除       | 中    | 0.5日  | 低   | ログ出力テスト |
| **パッケージ設定更新**<br>―― `python_requires >=3.11` など | 高    | 0.5日  | 低   | CI でPython 3.11+ 環境でのテスト |
| CI/GHA設定更新                     | 高    | 0.5日  | 低   | CI上でGWpy4をインストールしたテスト実行 |
| ドキュメント更新（リリースノート）     | 低    | 0.5日  | 低   | – |

- **優先度**: I/O モジュール修正と mp 削除は必須。Pythonバージョン要件更新も早期に実施。
- **見積時間**は経験則で、小中規模の修正であるためおおむね数日～数人日程度（並行作業含む）。
- **リスク**: I/O 変更は想定通りに動かなくなる可能性中程度。CI上でGWpy4を使った自動テストを十分増やします。
- **テスト**: 各ファイルフォーマットの例データを用い、4.0 環境下での読み込み検証テストを追加します。また、`gpw.io.registry` や新引数の働きを確認する単体テストを用意します。

## 5. 互換性戦略

GWpy 3.x と 4.x の両方をどう扱うかの方針です：

- **同時サポート方針**: 可能であれば一時的に両対応することで移行期間を短縮します。具体的には、条件分岐でバージョンを判定し、GWpy 3.x と 4.x それぞれで適切な処理を呼び出します。ただし Python 3.9/3.10 は GWpy4 では動作しないため、サポートバージョンの調整が必要です。 
- **シム/条件付きインポート**: 例として次のように書けます。
  ```python
  try:
      from gwpy import __version__ as gwpy_version
  except ImportError:
      gwpy_version = None

  if gwpy_version and gwpy_version.startswith("4."):
      from gwpy.io.registry import default_registry as io_registry
      NPROC_ARG = "parallel"
  else:
      from gwpy.io import registry as io_registry
      NPROC_ARG = "nproc"
  ```
  ただし、よりシンプルには「GWpy4対応版を次期メジャー（v0.2 など）でリリースし、旧版で3.x専用と区別」する選択肢もあります。
- **最低バージョン指定**: PyPI や `pyproject.toml` で `gwpy>=3.0,<5.0` とし、`python_requires=">=3.11"` に変更します。過渡的に 3.x/4.x 両対応する場合は例えば `">=3.0,<4.1"` とピン留めしない方法も考えられますが、最終的には 3.11+ を前提とする必要があります。

## 6. CI およびパッケージ変更

- **CI (GitHub Actions/Tox)**: テスト環境に Python 3.11/3.12 を追加し、依存に `gwpy>=4.0.0` のテストケースを設けます。これにより 4.0 環境下で全テストがパスすることを保証します。既存の 3.x テストも維持し、必要なら両方通すマトリクスを組みます。  
- **パッケージ設定**: `setup.py` または `pyproject.toml` の `python_requires` を `>=3.11` に変更します。依存リストの `gwpy` を `"gwpy>=3.0,<5.0"` 等に緩めるか、 4.0対応リリースでは `>=4.0` を推奨します。  
- **テストの追加**: 新しい I/O や `parallel` 引数などを扱うテストスクリプトを用意し、またドキュメントビルド（Sphinxなど）で `gwpy>=4` に対応できるように確認します。

## 7. ユーザー向けリリースノート案

移行ユーザーに向け、以下の要点をリリースノートで案内します：

- **GWpy 4.0.0 対応**：GWpy 4.x をサポートするための更新を行いました。Python 3.11 以上が必要です（3.9/3.10 は非対応）【124†L51-L59】。
- **ファイル読み込み (I/O) 方式の変更**：内部で Astropy I/O レジストリを使うようになり、独自フォーマット読込コードの登録方法が変わりました。GWexpyユーザーが書いたカスタムフォーマット reader がある場合は、`from gwpy.io import registry` を使っていれば新しいコードでは自動変換済みのはずですが、何か問題があれば `gwpy.io.registry.default_registry` を用いた登録方法を参照してください【124†L62-L64】。例：`from gwpy.io.registry import default_registry as registry` に変更します。  
- **`nproc→parallel`**：読み込み系クラスメソッドの `nproc` 引数は廃止され `parallel` になりました【124†L134-L136】。旧引数を使うと警告またはエラーになるため、新版では `parallel` キーワードを指定してください。  
- **マルチプロセスサポート**：`gwexpy/io/mp.py` は不要になりました。大規模並列処理が必要な場合は `multiprocessing` など標準ライブラリを利用してください【124†L95-L97】。  
- **Python 要件変更**：GWexpy v0.1 系では Python 3.9+ と表明していましたが、GWpy4 対応版では 3.11+ が最小要件となります【124†L51-L59】。  
- **その他変更**：（必要に応じて）`verbose` 等の廃止キーワードや外部依存切替（`igwn-ligolw` など）への対応方針を説明します。

## 8. 参考資料・ソース

- GWexpy リポジトリ README【12†L146-L148】（現行は `gwpy<4.0` が必要である旨記載）。  
- GWpy 公式 CHANGELOG【124†L51-L59】【124†L58-L64】【124†L95-L97】【124†L134-L137】（4.0 の破壊的変更点）。  
- GWpy 4.0.1 API リファレンス（I/O レジストリの仕様）【125†L12-L16】。  

上記に加え、必要に応じて GWpy の日本語ドキュメントや Astropy の I/O registry ドキュメントを参照しつつ実装を行います。  

**引用例：**「GWpy 4.0.0 では I/O システムが Astropy のクラスベースレジストリに刷新され、`nproc`→`parallel` への変更や `gwpy.io.mp` の削除など複数の破壊的変更が導入されています【124†L51-L59】【124†L95-L97】」。  

