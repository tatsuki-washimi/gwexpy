# Notebook 運用ポリシー (Notebook Policy)

GWexpy のドキュメントおよびサンプルで使用される Jupyter Notebook (.ipynb) の運用、CI での検証、および出力管理に関するガイドラインです。

## 1. ノートブックの分類 (Classification)

計算負荷、依存関係、および用途に基づいて 3 つのカテゴリに分類します。

| カテゴリ | 特徴 | CI での扱い | セル出力の管理 |
| :--- | :--- | :--- | :--- |
| **Light** | 実行時間が短く（数分以内）、外部データ不要 | `papermill` で全実行・検証 | `nbstripout` で自動除去（差分を最小化） |
| **Heavy** | 実行が長い or LIGO 内部データ/GPU 必須 | 構文確認 (`nbval --nbval-lax`) のみ | `nbstripout` で自動除去 |
| **Display-only** | 手動で入念に調整された出力結果を重視 | 検証対象外（または読み込み確認のみ） | 出力を保持（コミットに含む） |

## 2. CI での判定方法

ノートブックの最初のセルにメタデータタグを付与することで、CI 実行を制御します。

- **Light**: `metadata: { "tags": ["ci-light"] }`
- **Heavy**: `metadata: { "tags": ["ci-heavy"] }`
- **Display-only**: `metadata: { "tags": ["display-only"] }`

## 3. 出力管理 (`nbstripout`)

リポジトリの肥大化を防ぐため、原則としてノートブックの出力はコミットしません。

- `pre-commit` フックにより、「Display-only」以外のノートブックは自動的に `nbstripout` が適用されます。
- 特定のノートブックを `nbstripout` から除外する場合は、`.gitattributes` に以下を記述します：

  ```text
  path/to/display_only_notebook.ipynb filter=nbstripout-ignore
  ```

## 4. 新規追加時のチェックリスト

- [ ] 適切なカテゴリ分類を行い、メタデータタグを付与したか？
- [ ] タイトルと概要が記述され、JA/EN の両方が用意されているか？
- [ ] 他のドキュメント（Quickstart, Tutorial 一覧）から正しくリンクされているか？
- [ ] 依存ライブラリが `docs/requirements.txt` または `pyproject.toml` に含まれているか？
- [ ] `display-only` の場合、`.gitattributes` に除外設定を追記したか？
