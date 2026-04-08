---
name: web_docs_release_gates
description: docs/web の GitHub Pages 変更を PR 前に検証する時に使う。make html、linkcheck、Notebook の軽重分類、ja/en 構造差分、公開ブロッカーの切り分けをまとめて確認したい時に使う
---

# Web Docs Release Gates

`docs/web` の GitHub Pages 変更を、公開前または PR 前に検証するための skill。  
`manage_docs` が一般的な doc build と同期を扱うのに対し、この skill は GitHub Pages 品質ゲートに特化する。

## Responsibilities

1. `docs/web` の build 成功を確認する
2. linkcheck を実行し、重大リンク切れを洗い出す
3. Notebook を lightweight / heavy / display-only に分類して扱う
4. `ja/en` の構造差分を確認する
5. docs だけで閉じない release blocker を分離する

## Verification Flow

### 1. HTML build

```bash
cd docs && make html
```

### 2. Link check

```bash
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

### 3. Notebook gate

- Lightweight: CI またはローカルで実行確認
- Heavy: 専用環境または別ジョブへ送る
- Display-only: 構文、リンク、表示崩れのみ確認

### 4. Bilingual sync

最低限、次を確認する。

- `ja/en` の対応ページが両方存在する
- 主要見出しと toctree の構造が大きくずれていない
- 片言語だけで新設したページが残っていない

### 5. Release blocker split

次は docs PR から分離して記録する。

- PyPI / Conda 公開状況
- Notebook の重依存問題
- 品質バッジや coverage 公開
- docs 以外の CI 整備

## PR Checklist

- [ ] `cd docs && make html`
- [ ] `sphinx-build -b linkcheck docs docs/_build/linkcheck`
- [ ] 主要 one-liner または Quickstart サンプルの確認
- [ ] `ja/en` 対応ページの同時更新
- [ ] Notebook の分類と未検証項目の明記
- [ ] release blocker の切り出し

## Reporting Format

検証結果は次の形式で残す。

- Passed:
  - build
  - linkcheck
  - lightweight notebooks
- Needs follow-up:
  - heavy notebooks
  - external release blockers
  - translation drift

## When Not to Use

- 入口設計や分類そのものを考える時
- 個別ページの本文を作り直す時

その場合は `web_docs_ia_overhaul` または `web_docs_page_rewrite` を使う。
