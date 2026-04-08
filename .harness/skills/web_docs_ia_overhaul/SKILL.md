---
name: web_docs_ia_overhaul
description: docs/web の GitHub Pages で、トップページ、索引、チュートリアル一覧、ケーススタディ一覧、リファレンス入口の情報設計や導線を整理する時に使う。入口重複、分類の曖昧さ、学習順の不明瞭さ、ja/en の非対称を直したい時に使う
---

# Web Docs IA Overhaul

`docs/web` の GitHub Pages 入口設計を整理するための skill。  
対象は `index.rst`、`tutorials/index.rst`、`examples/index.rst`、`reference/index.rst` などの案内ページ。

## Responsibilities

1. 入口ページの役割を一つに絞る
2. Tutorial / Case Study / Reference / Advanced Guide の境界を明確にする
3. 学習導線と辞書的参照導線を分ける
4. `ja/en` の構造差分を同時に解消する
5. 変更後に Sphinx build が通ることを確認する

## Checklist

1. 対象ページの現状構成を読む
2. 重複導線、孤立ページ、曖昧な分類を列挙する
3. 各ページの役割を次のいずれかに固定する
   - 入口
   - 学習導線
   - 実例導線
   - API 索引
   - 深掘り解説
4. `ja/en` の対応ページをペアで編集対象にする
5. 次の観点で導線を整理する
   - 誰向けか
   - 何を学べるか
   - 次にどこへ行くか
6. `cd docs && make html` で確認する

## Editing Rules

- `index.rst` では Hero、対象読者別導線、主要入口の3ブロックを優先する
- 「次のステップ」「学習パス」「ガイド選択」の重複は残さない
- `tutorials/index.rst` には難易度、前提、得られる成果を加える
- `examples/index.rst` では各ケースの問題設定、利用 API、期待出力を短く示す
- `reference/index.rst` は単なる toctree にせず、用途別入口を持たせる
- 片言語のみ直さない

## Verification

```bash
cd docs && make html
```

確認項目:

- トップページで重複導線が消えている
- Tutorials / Examples / Reference が役割別に分かれている
- `ja/en` の入口構成が大きくずれていない

## When Not to Use

- 個別ページ本文の書き換えが主目的の時
- Notebook 実行や linkcheck など公開前検証が主目的の時

その場合は `web_docs_page_rewrite` または `web_docs_release_gates` を使う。
