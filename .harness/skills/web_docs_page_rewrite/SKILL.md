---
name: web_docs_page_rewrite
description: docs/web の GitHub Pages で、installation、io_formats、time_utilities、numerical_stability など個別ページを読みやすく再設計する時に使う。冒頭要約、早見表、FAQ、危険事項の強調、API への戻り導線、ja/en 同期を揃えたい時に使う
---

# Web Docs Page Rewrite

`docs/web` の個別ページを、利用者が判断しやすい形へ再設計するための skill。  
対象は `*.md`、一部の `*.rst`、Notebook 由来ページの入口文面。

## Responsibilities

1. 冒頭要約と対象読者を明示する
2. 早見表、選択表、FAQ を必要ページに追加する
3. 危険事項や制約を目立つ位置に移す
4. チュートリアル、ケーススタディ、API への戻り導線を付ける
5. `ja/en` ペアで内容差分を管理する

## Standard Page Shape

ページ再設計時は、必要に応じて次の順に並べる。

1. 何のページか
2. 誰向けか
3. まず何を判断できるか
4. 早見表または最短例
5. 詳細解説
6. FAQ / 注意事項
7. 次に読むページ

## Page-Specific Patterns

### installation

- 最小導入 / 推奨導入 / 開発用 / 特殊依存 / トラブルシューティングに分ける
- 未リリース手順は主導線に置かない
- バージョン要件を他ページと揃える

### io_formats

- 冒頭に判断表を置く
- 必須引数、自動判別可否、外部依存、重要制約を表で示す
- 内部実装パスや stub 情報を一般向け本文に出しすぎない

### time_utilities

- `to_gps / from_gps / tconvert` の使い分け表を置く
- 入出力型、戻り値、タイムゾーン、配列時の挙動を明示する

### numerical_stability / validated_algorithms / architecture

- 「通常は何もしなくてよいか」を最初に書く
- 数式には変数定義を付ける
- 対応 API へ戻るリンクを入れる

## Checklist

1. 対象ページの目的と読者を一文で言えるか確認する
2. 冒頭に要約、対象読者、前提、次アクションを置く
3. 表、FAQ、警告のどれが必要か決める
4. 「危険」「制約」「例外条件」を本文下部から前方へ移す
5. `ja/en` 対応ページを同時に編集する
6. `cd docs && make html` で確認する

## Verification

```bash
cd docs && make html
```

確認項目:

- 冒頭を読むだけでページ用途が分かる
- 重要制約が埋もれていない
- 関連する Tutorial / Example / Reference へ移動できる

## When Not to Use

- サイト全体の構成や入口設計を直す時
- 公開前の build / linkcheck / Notebook gate が主目的の時

その場合は `web_docs_ia_overhaul` または `web_docs_release_gates` を使う。
