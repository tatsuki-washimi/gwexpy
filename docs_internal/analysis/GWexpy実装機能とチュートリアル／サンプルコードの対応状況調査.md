# GWexpy 実装機能とチュートリアル／サンプルコードの対応状況調査

## エグゼクティブサマリー

本調査では、GWexpy の「実装された主要機能（公開API・ユーザー向け機能）」に対して、対応するチュートリアル／サンプルコードが不足なく用意されているかを、リポジトリ内の公開API（`gwexpy/__init__.py`）と、ドキュメントのチュートリアル索引（英語・日本語）および主要ガイド（Quickstart、機能一覧、I/Oガイド）を突き合わせて検証しました。fileciteturn185file0L1-L1 fileciteturn196file0L1-L1 fileciteturn197file0L1-L1 fileciteturn201file0L1-L1 fileciteturn207file0L1-L1 fileciteturn211file0L1-L1

結論として、英語（`en`）側は「主要機能（TimeSeries / FrequencySeries / Spectrogram / ノイズ生成 / フィッティング / 行列コンテナ / Field API / セグメント解析 / 多数の高度解析）」に対して網羅的にチュートリアルが用意されている一方、日本語（`ja`）側は少なくとも次の重要カテゴリが**欠落**しています。

- **ノイズ生成の基礎チュートリアル**（英語には `Noise Generation Basics` があるが、日本語索引に無い）fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  
- **スペクトルフィッティングの基礎チュートリアル**（英語には `Spectral Fitting Basics` があるが、日本語索引に無い）fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  
- **セグメント解析の「高度パイプライン」チュートリアル**（英語には `Segment Analysis Pipeline (Advanced)` があるが、日本語索引に無い）fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  

加えて、「実装は存在するが、チュートリアル／サンプルの導線が弱い（または未整備）」項目として、**`gwexpy.time` のGPS時刻ユーティリティ**、**CLI（現状プレースホルダ）**、**GUI（pyaggui）**が確認されました。fileciteturn200file0L1-L1 fileciteturn203file0L1-L1 fileciteturn205file0L1-L1 fileciteturn206file0L1-L1 fileciteturn207file0L1-L1

本レポートの後半に、**不足分として追加すべきチュートリアル／サンプル項目（必須・推奨）**を具体化してまとめます。

## 調査範囲と方法

調査の基準を「実装＝ユーザーが直接使う可能性が高い公開機能」とし、次の一次情報を中心に突合しました。

- 公開API（トップレベルで再公開されるクラス・サブパッケージ）を `gwexpy/__init__.py` から抽出fileciteturn185file0L1-L1  
- ドキュメントのチュートリアル索引（英語・日本語）から、提供されているチュートリアルページ（Notebook由来）を列挙fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  
- Quickstart、GWpyユーザー向け機能一覧、I/O対応フォーマットガイドも「サンプルコードが含まれる公式説明」として補助的に評価fileciteturn201file0L1-L1 fileciteturn207file0L1-L1 fileciteturn211file0L1-L1  

注意点として、`gwexpy.astro`・`gwexpy.detector` はGWpyの該当モジュールを動的再公開している設計（PEP 562）であり、GWexpy独自機能というより「利便性のための再export」に近いものです。fileciteturn198file0L1-L1 fileciteturn199file0L1-L1  
このため、これらは「GWexpyで新たに必須となるチュートリアル」の優先度は相対的に下げつつ、必要ならGWpyドキュメントへの誘導を整備する、という扱いが合理的です（後述）。

## 実装機能インベントリの要約

`gwexpy/__init__.py` では、GWpy拡張の主要データ構造（TimeSeries / FrequencySeries / Spectrogram と各種 Dict/List）、追加データ構造（Histogram、各種 Matrix、Field API）、および多くのサブパッケージがトップレベルから到達可能になっています。fileciteturn185file0L1-L1

また `gwexpy.time` は、GWpyの `gwpy.time` を再公開しつつ、`to_gps`・`from_gps`・`tconvert` を独自に明示エクスポートする構造です。fileciteturn200file0L1-L1  
この「時刻指定の柔軟性」は `TimeSeries.crop(...)` 等の振る舞いにも影響する、と機能一覧ページで明記されています。fileciteturn207file0L1-L1

さらに、パッケージとして CLI と GUI のエントリポイントが定義されており（`pyproject.toml` の scripts）、CLIは現状プレースホルダ、GUIはPyQt5ベースの `pyaggui` が実装されています。fileciteturn180file0L1-L1 fileciteturn203file0L1-L1 fileciteturn205file0L1-L1 fileciteturn206file0L1-L1

## チュートリアル／サンプルの現状カバレッジ

### 英語ドキュメントのチュートリアル網羅性

英語版チュートリアル索引（`docs/web/en/user_guide/tutorials/index.html`）には、以下の大項目が明確に用意されています。

- Core Data Structures（TimeSeries / FrequencySeries / Spectrogram / Noise / Plotting / Map plotting / Interop / Histogram）  
- Matrix Containers（TimeSeriesMatrix / FrequencySeriesMatrix / SpectrogramMatrix）  
- High-dimensional Fields（Scalar/Vector/Tensor + 応用）  
- Advanced Signal Processing（フィッティング、スペクトログラム処理、ピーク検出、HHT、時周波解析、ARIMA、相関、ML前処理、線形代数、結合解析、分解、地震波解析、GBD I/O など）  
- Specialized Tools（Bruco、ICA、Bilinear coupling、Violin mode、Schumann resonance）  
- Segment Analysis（SegmentTable基礎〜高度パイプライン、ASDパイプライン、可視化、イベント同期）  

これらの一覧が索引ページに揃っており、公開APIの主要領域（データ構造・行列・Field・ノイズ・フィッティング・解析パイプライン）に対して「最低1つ以上のチュートリアル」が配置されている状態です。fileciteturn196file0L1-L1

加えて、Quickstart でも「マルチチャンネル生成」「行列変換」「CSD」など、初期導入のサンプルが提示されています。fileciteturn201file0L1-L1  
Interopについては、`intro_interop` のNotebook（生成HTML）が存在し、多数の外部ツール連携をまとめて示す構成であることが確認できます。fileciteturn188file0L1-L1

### 日本語ドキュメントのカバレッジと英語との差分

日本語版チュートリアル索引（`docs/web/ja/user_guide/tutorials/index.html`）は、TimeSeries / FrequencySeries / Spectrogram / Plotting / Map plotting / Interop / Histogram、行列コンテナ、Field API、多くの高度解析、特殊ツール、セグメント解析（基礎・ASD・可視化・イベント同期）を含みます。fileciteturn197file0L1-L1

しかし、英語索引に存在する次のページが日本語索引から欠落しています。

- `Noise Generation Basics`（英語にあり、日本語に無し）fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  
- `Spectral Fitting Basics`（英語にあり、日本語に無し）fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  
- `Segment Analysis Pipeline (Advanced)`（英語にあり、日本語に無し）fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  

一方で、日本語には「フィッティング & スペクトル線解析（advanced_fitting）」自体は存在します。fileciteturn192file0L1-L1  
つまり日本語側は「いきなり応用編はあるが、入門編が欠ける」構図になっており、初学者の導入における段差が大きい状態です（後述の追加項目に含めます）。

## 不足している対応項目

ここでは「実装機能 ⟷ チュートリアル／サンプル」の観点で、欠落または不足が確認できた項目を、**必須（リリース前に埋めたい）**と**推奨（品質向上）**に分けて整理します。  
（位置情報は、該当する索引HTMLや実装ファイルパスで提示します。）

### 必須と判断できる不足

| 不足項目 | 実装側の根拠（位置） | 現状の不足状況（位置） | 追加すべき内容（最小セット） |
|---|---|---|---|
| 日本語：ノイズ生成 入門チュートリアル | 英語チュートリアルに `intro_noise` が存在（ノイズ生成を「基礎」として独立扱い）fileciteturn196file0L1-L1 | 日本語チュートリアル索引に `intro_noise` が存在しないfileciteturn197file0L1-L1 | `docs/web/ja/user_guide/tutorials/intro_noise.ipynb` を追加（英語の翻訳＋日本語注釈）。日本語索引に追記。 |
| 日本語：スペクトルフィッティング 入門チュートリアル | 英語チュートリアルに `intro_fitting` が存在fileciteturn196file0L1-L1 | 日本語索引に `intro_fitting` が存在しない（応用 `advanced_fitting` のみ）fileciteturn197file0L1-L1 | `docs/web/ja/user_guide/tutorials/intro_fitting.ipynb` を追加（「最小のfit」「モデル指定」「誤差・推定値の読み方」）。索引に追記。 |
| 日本語：セグメント解析の高度パイプライン（intro_table） | 英語チュートリアルに `intro_table` が存在fileciteturn196file0L1-L1 | 日本語索引に `intro_table` が存在しないfileciteturn197file0L1-L1 | `docs/web/ja/user_guide/tutorials/intro_table.ipynb`（翻訳）を追加し、セグメント解析の「基礎→応用→ASD→可視化→ケーススタディ」の導線を再構成。 |

上記3件は、英語で「入門として独立ページがある」と明確に位置付けられているにもかかわらず、日本語で欠けているため、**日本語ユーザーの学習経路が実装に追従していない**状態です。fileciteturn196file0L1-L1 fileciteturn197file0L1-L1

### 推奨（ただしユーザー体験上の改善余地が大きい不足）

| 不足／弱い導線 | 実装側の根拠（位置） | 現状の不足状況（位置） | 追加すべき内容（提案） |
|---|---|---|---|
| `gwexpy.time`（to_gps / from_gps / tconvert）の「使い方チュートリアル」 | `gwexpy.time` が独自に `to_gps` 等を公開し、GWpyを超えた利便性の入口になっているfileciteturn200file0L1-L1 | 機能一覧では「cropが to_gps による柔軟な時刻指定に対応」と書かれるが、手を動かす最小チュートリアルは索引からは見当たらないfileciteturn207file0L1-L1 fileciteturn196file0L1-L1 | `intro_time.ipynb`（EN/JA）を新設し、(1) 文字列・datetime・配列のGPS変換、(2) `TimeSeries.crop`/`SegmentTable`での実例、(3) timezone注意点、(4) ベクトル化のメリット、までを短くまとめる。 |
| CLI（`gwexpy` コマンド）に対する「実用サンプル」の欠如 | CLIモジュールは存在するが「現状プレースホルダ」と明記されているfileciteturn203file0L1-L1 | ドキュメント側にCLIの現状・使い方・非対応範囲の説明ページが見当たらない（チュートリアル索引にも無し）fileciteturn196file0L1-L1 fileciteturn197file0L1-L1 | リリース前に方針が必要：A) CLIを「非推奨/実験」と明記するページ（`cli.md`）を追加し、`--version`など現状動作を記述。B) ある程度使えるなら、最低限「spectrum」「spectrogram」を実装して、短いCLIレシピを載せる。 |
| GUI（`gwexpy.gui` / `pyaggui`）の実行例・ユーザーガイド不足 | GUIモジュールが実装され、PyQt5アプリとして `main()` があり、ファイルを開く引数も受けるfileciteturn205file0L1-L1 fileciteturn206file0L1-L1 | 機能一覧では「GUI Applications」として存在のみ言及されるが、実行手順・対応フォーマット・スクリーンショット等が無いfileciteturn207file0L1-L1 | `gui.md`（EN/JA）を追加し、(1) インストール（extras `gui` が必要な旨）、(2) 起動方法（`gwexpy.gui` or `python -m ...` の推奨）、(3) 対応入力例、(4) 既知の制限、(5) 最低1枚の画面キャプチャ、を整備する。 |

## 必要な追加項目まとめ

最後に、「不足を埋めるために追加すべき項目」を、作業単位として落とし込みます。ここでは、実装変更が不要な「翻訳追加」と、実装・仕様整理が必要な「導線整備」を分けます。

### 追加が必須のチュートリアル（日本語側の欠落補完）

- `docs/web/ja/user_guide/tutorials/intro_noise.ipynb` を追加し、日本語チュートリアル索引（`docs/web/ja/user_guide/tutorials/index.rst` → 生成 `index.html`）に組み込む。英語側に同名の入門があるため、翻訳が最短ルートです。fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  
- `docs/web/ja/user_guide/tutorials/intro_fitting.ipynb` を追加し、`advanced_fitting` の前に「基礎」を置く。日本語側は応用編が既にあるため、学習導線として“基礎→応用”が自然です。fileciteturn197file0L1-L1 fileciteturn192file0L1-L1  
- `docs/web/ja/user_guide/tutorials/intro_table.ipynb` を追加し、セグメント解析の「高度パイプライン」を欠落なくする。fileciteturn196file0L1-L1 fileciteturn197file0L1-L1  

### 追加が強く推奨のチュートリアル／サンプル（機能はあるが学習単位が不足）

- `docs/web/en/user_guide/tutorials/intro_time.ipynb` と `docs/web/ja/user_guide/tutorials/intro_time.ipynb` を新設し、`gwexpy.time`（to_gps / from_gps / tconvert）の使い方を“短い成功体験”として提供する。`gwexpy.time` は独自に明示エクスポートされているため、ユーザーが存在を知っている前提にしない方が安全です。fileciteturn200file0L1-L1  
- `docs/web/*/user_guide/cli.md`（英日）を追加し、CLIが現状プレースホルダであること、現状できること（例：`--version`）、将来予定、代替手段（Python API）を明確化する。CLI実装自体が未成熟なので、**「チュートリアル不足」というより、プロダクト仕様の誤解を防ぐための必須ドキュメント」**に近いです。fileciteturn203file0L1-L1  
- `docs/web/*/user_guide/gui.md`（英日）を追加し、`pyaggui` の起動・依存・対応入力・制限をガイド化する。GUIは実装がある以上、入口が無いと「存在するのに使えない」と受け取られやすいです。fileciteturn205file0L1-L1 fileciteturn206file0L1-L1 fileciteturn207file0L1-L1  

以上が、「実装された機能に対して対応するチュートリアル／サンプルコードが不足なく用意されているか」の調査結果と、必要な追加項目のまとめです。