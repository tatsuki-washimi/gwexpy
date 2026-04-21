# Notebook 運用ポリシー (Notebook Policy)

GWexpy のドキュメントおよびサンプルで使用される Jupyter Notebook (.ipynb) の運用、CI での検証、および出力管理に関するガイドラインです。

## 0. 正本管理 (Source of Truth)

- 公開ドキュメントとして配信する notebook は `docs/web/{en,ja}/user_guide/tutorials/` を正本とします。
- `examples/` は legacy notebook、公開候補、ローカル実行向け補助資料の置き場です。
- 同名または同等内容の notebook が両方に存在する場合、公開内容については `docs/web/.../tutorials/` 側を authoritative とみなします。
- `examples/` から notebook を公開昇格する場合は、`docs/web/{en,ja}/user_guide/tutorials/` へ取り込み、対応する index へ追加してから保守対象にします。
- `docs/web/` と `examples/` の source notebook は tracked のまま保ちますが、原則として clean な状態を維持します。docs は実行済み notebook から生成してよく、tracked source notebook は commit 前に出力を strip します。
- セクション 14-10 に関連するコードコメント修正は `docs/developers/guides/notebook_physics_comment_rubric.md` に従って行います。

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

GitHub Actions で notebook を実行するジョブは fail-closed を既定とします。
`NBS_ALLOW_ERRORS` は明示的に無効化し、実行された notebook の例外を成功扱いにしません。
PR では `scripts/notebook_gen/check_changed_notebooks.py` による changed-notebook 検証を使い、
docs build 自体は `nbsphinx_execute=never` の非実行パスを維持しつつ、
notebook を実行する検証ジョブ側では同じ fail-closed 方針を維持します。

## 2.5. 日本語プロット用フォント設定

docs / notebook build では共有の Matplotlib 設定を `MPLCONFIGDIR/matplotlibrc` に生成し、
notebook kernel からも継承されるようにします。

- `backend: Agg`
- `font.family: sans-serif`
- `font.sans-serif: Noto Sans CJK JP, IPAexGothic, IPAGothic, DejaVu Sans`
- `axes.unicode_minus: False`

GitHub Actions では workflow レベルで CJK フォントをインストールし、
notebook ごとの個別フォント回避策は追加しない方針とします。

## 3. 出力管理 (`nbstripout`)

リポジトリの肥大化を防ぐため、原則としてノートブックの出力はコミットしません。

- `pre-commit` フックにより、`docs/web/` と `examples/` 配下の「Display-only」以外のノートブックには自動的に `nbstripout` が適用されます。
- 通常の source notebook は tracked のまま保ちますが、commit 前に clean にしておく運用です。
- docs は実行済み notebook を元に生成してかまいませんが、tracked source notebook には出力を残さない方針です。
- 特定のノートブックを `nbstripout` から除外する場合は、`.gitattributes` に以下を記述します：

  ```text
  path/to/display_only_notebook.ipynb filter=nbstripout-ignore
  ```

- repo-size 保護と notebook hygiene のチェックは `scripts/check_forbidden_artifacts.py` と `scripts/check_repo_hygiene.py` が担います。

## 4. 新規追加時のチェックリスト

- [ ] 適切なカテゴリ分類を行い、メタデータタグを付与したか？
- [ ] タイトルと概要が記述され、JA/EN の両方が用意されているか？
- [ ] 他のドキュメント（Quickstart, Tutorial 一覧）から正しくリンクされているか？
- [ ] 依存ライブラリが `docs/requirements.txt` または `pyproject.toml` に含まれているか？
- [ ] `display-only` の場合、`.gitattributes` に除外設定を追記したか？
- [ ] 解析手順に物理的判断が含まれる場合、コードコメントが `notebook_physics_comment_rubric.md` を満たしているか？
