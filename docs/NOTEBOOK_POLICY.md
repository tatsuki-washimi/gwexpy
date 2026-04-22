# Notebook 運用ポリシー (Notebook Policy)

GWexpy のドキュメントおよびサンプルで使用される Jupyter Notebook (.ipynb) の運用、CI での検証、および出力管理に関するガイドラインです。

## 0. 正本管理 (Source of Truth)

- 公開ドキュメントとして配信する notebook は `docs/web/{en,ja}/user_guide/tutorials/` を正本とします。
- `examples/` は legacy notebook、公開候補、ローカル実行向け補助資料の置き場です。
- 同名または同等内容の notebook が両方に存在する場合、公開内容については `docs/web/.../tutorials/` 側を authoritative とみなします。
- `examples/` から notebook を公開昇格する場合は、`docs/web/{en,ja}/user_guide/tutorials/` へ取り込み、対応する index へ追加してから保守対象にします。
- `docs/web/` と `examples/` の tracked notebook は、原則として **clean な source** として保ちます。Git に残す正本は notebook の内容であり、公開用の実行結果は publish 時に別途生成します。
- セクション 14-10 に関連するコードコメント修正は `docs/developers/guides/notebook_physics_comment_rubric.md` に従って行います。

## 1. ノートブックの分類 (Classification)

計算負荷、依存関係、および用途に基づいて 3 つのカテゴリに分類します。

| カテゴリ | 特徴 | CI での扱い | セル出力の管理 |
| :--- | :--- | :--- | :--- |
| **Light** | 実行時間が短く（数分以内）、外部データ不要 | `papermill` で全実行・検証 | 出力は strip して commit |
| **Heavy** | 実行が長い or LIGO 内部データ/GPU 必須 | 構文確認 (`nbval --nbval-lax`) のみ | 出力は strip して commit |
| **Display-only** | 手動で入念に調整された出力結果を重視 | 検証対象外（または読み込み確認のみ） | 例外的に出力保持を許可 |

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

## 3. 出力管理

最終方針は次のとおりです。

- `docs/web/` と `examples/` の tracked notebook は、`display-only` を除き **出力と `execution_count` を strip して commit** します。
- 公開用の plot / table / image は source notebook に保持せず、publish 時に実行して得られた一時成果物から HTML を生成します。
- `display-only` は「CI の changed-notebook 実行をスキップしつつ、review 済み出力を意図的に保持する notebook」のための例外タグです。
- `scripts/check_repo_hygiene.py` はこの方針をそのまま強制し、非 `display-only` notebook の出力残しを reject します。
- `pre-commit` では `scripts/strip_example_notebook_outputs.py` が `docs/web/` と `examples/` の非 `display-only` notebook を自動で clean に戻します。
- repo-size 保護と notebook hygiene のチェックは `scripts/check_forbidden_artifacts.py` と `scripts/check_repo_hygiene.py` が担います。

## 3.5. Documentation Site Build Workflow (GitHub Pages)

Tracked notebook source は clean に保ち、Sphinx build は notebook source そのものを再実行しない構成を維持します。
その代わり、公開時には一時的な executed notebook tree または同等の generated artifact を事前に作成し、
HTML build はその生成物に対して `nbsphinx_execute = "never"` で行います。

Notebook execution is handled separately:

- changed-notebook CI checks use `scripts/notebook_gen/check_changed_notebooks.py`
- `light` notebooks are executed with `papermill`
- `heavy` notebooks are syntax-checked with `pytest --nbval-lax`
- publish / release path では `docs/web/` notebook を実行し、clean source とは別の temp tree を生成してから docs build に渡します
- `display-only` notebooks are skipped by the changed-notebook execution job and may retain intentional outputs only when explicitly reviewed

### Rationale

Separating notebook execution from tracked source keeps Git diffs small, avoids accidental
site regressions caused by a casual rerun, and prevents notebook output JSON from dominating
repository size. Notebook-specific CI still verifies that changed notebooks remain runnable
under the project policy.

### Recommended workflow when notebooks change

1. Edit tracked notebooks under `docs/web/` or `examples/` as clean source.
2. Run the affected notebooks locally, or use the CI-style runner, to validate behavior and inspect plots.
3. Clear outputs from non-`display-only` notebooks before committing.
4. For publish validation, create a temp executed notebook tree or equivalent generated artifacts, then verify the docs build with `python -m sphinx -b html -W --keep-going -D nbsphinx_execute=never docs /tmp/gwexpy-docs-html`
   or an equivalent `sphinx-build` command against that prepared tree.
5. Commit standalone generated artifacts only when the page depends on tracked files outside the notebook JSON itself.

> **Rule: tracked notebooks are committed clean; rendered outputs are publish artifacts, not source-of-truth.**

## 4. 新規追加時のチェックリスト

- [ ] 適切なカテゴリ分類を行い、メタデータタグを付与したか？
- [ ] タイトルと概要が記述され、JA/EN の両方が用意されているか？
- [ ] 他のドキュメント（Quickstart, Tutorial 一覧）から正しくリンクされているか？
- [ ] 依存ライブラリが `docs/requirements.txt` または `pyproject.toml` に含まれているか？
- [ ] `display-only` の場合、changed-notebook CI から除外してよい理由と retained output の必要性を確認したか？
- [ ] 解析手順に物理的判断が含まれる場合、コードコメントが `notebook_physics_comment_rubric.md` を満たしているか？
