# ドキュメントのノートブック実行問題の修正レポート

ドキュメントのWebページでサンプルコードの出力が表示されていなかった問題を調査し、解決しました。

## 問題の概要

GitHub Actionsによるドキュメントビルドプロセスにおいて、`NBS_EXECUTE: never` が設定されており、デプロイ時のノートブックの自動実行が無効化されていました。同時に、リポジトリ内のJupyter Notebookファイル自体の出力がクリアされていたため、Webサイト上で出力セルが空になっていました。

## 実施した解決策（オプションB）

採用された計画に基づき、専用の環境ですべてのノートブックを手動実行し、その結果（プロット画像を含む）を直接 `.ipynb` ファイルに保存しました。これにより、安全のために自動実行を無効化したままでも、ドキュメントサイトに正しい出力が表示されるようになります。

### 実施ステップ:

1.  **環境構築**: 仮想環境 `.venv-docs-exec` を作成し、`gwexpy` とチュートリアルの実行に必要なすべてのオプション依存関係（`control`, `statsmodels`, `ligo.skymap` など）をインストールしました。
2.  **実行スクリプトの作成**: `docs/web/en/` および `docs/web/ja/` にある全50個のノートブックの実行を自動化する `scripts/run_all_notebooks.py` を作成しました。
3.  **ノートブックの処理**: `jupyter nbconvert` を使用してすべてのノートブックを実行しました。日本語版の `intro_timeseries.ipynb` などの主要なチュートリアルに画像データと実行結果が含まれていることを確認しました。
4.  **検証**: ノートブックのファイルサイズが大幅に増加（例：スタブ状態から 1.9MB 〜 5.4MB へ）したことを確認し、プロットが保存されていることを裏付けました。

## 変更内容

- [MODIFY] `docs/web/en/user_guide/tutorials/` および `docs/web/ja/user_guide/tutorials/` 内のすべての `.ipynb` ファイル（実行結果を保持するようになりました）。
- [NEW] [run_all_notebooks.py](file:///home/washimi/work/gwexpy/scripts/run_all_notebooks.py)（今後のメンテナンス用ユーティリティスクリプト）。

## 検証結果

- `docs/web/ja/user_guide/tutorials/intro_timeseries.ipynb`: 1.9MB（出力およびPNGプロットを確認済み）。
- `docs/web/ja/user_guide/tutorials/matrix_timeseries.ipynb`: 5.4MB（大規模な出力を確認済み）。
- 全50個のノートブックがエラーで停止することなく実行完了しました。

これらの更新されたノートブックをリポジトリにコミットすることで、GitHub Actionsによる次回のビルド時に保存された出力がWebページに反映されます。
