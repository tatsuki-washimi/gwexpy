---
name: manage_docs
description: ドキュメントのビルド、同期、検証を一括管理する（Sphinx対応）。build後の生成物差分やautosummaryの扱いもここで管理する
---

# Manage Documentation

ドキュメントの管理を行うスキル。ビルドと同期の両方に対応。

## Quick Start

```bash
# ドキュメントビルド
cd docs && make html

# または直接
sphinx-build -b html docs/web/en/ docs/_build/html/en/
sphinx-build -b html docs/web/ja/ docs/_build/html/ja/
```

## Tasks

### 1. Docstring の更新

コード変更後、docstring を更新:

1. 関数シグネチャを確認
2. `Parameters`, `Returns`, `Raises` セクションを更新
3. 例があれば動作確認

**フォーマット**: NumPy スタイル

```python
def func(x: float, y: int = 1) -> float:
    """Short description.

    Parameters
    ----------
    x : float
        Description of x.
    y : int, optional
        Description of y. Default is 1.

    Returns
    -------
    float
        Description of return value.

    Examples
    --------
    >>> func(1.0, y=2)
    0.5
    """
```

### 2. ドキュメントファイルの更新

- `docs/web/en/` - 英語版
- `docs/web/ja/` - 日本語版

**両方を更新すること**。

### 3. ドキュメントビルド

```bash
# 英語版
sphinx-build -b html docs/web/en/ docs/_build/html/en/

# 日本語版
sphinx-build -b html docs/web/ja/ docs/_build/html/ja/

# 警告をエラーとして扱う（CI と同じ厳格モード）
sphinx-build -W -b html docs/web/en/ docs/_build/html/en/
```

- **重要**: `main` ブランチへプッシュする前に、`-W` フラグ付きでビルドがパスすることを確認してください。

### 4. 検証

- `docs/_build/html/en/index.html` を確認
- リンク切れがないか確認
- nitpick 警告は `conf.py` の `nitpick_ignore` で管理

### 5. 生成物の扱い

docs build 後は必ず `git status --short` を確認し、生成物を分類する。

- `docs/_build/`: 原則コミットしない
- `docs/web/**/_autosummary/*.rst`: API 追加や autosummary 再生成が意図通りならコミット候補
- `docs/web/**/*.ipynb`: source notebook を明示的に更新した時だけコミット候補
- 英語版・日本語版で対応する差分が揃っているか確認する

大量差分が出た時は、build の副作用なのか source 変更の反映なのかを分けて確認する。

## Bilingual Support

gwexpy は英語・日本語の両方でドキュメントを提供:

| 言語   | ディレクトリ   |
| ------ | -------------- |
| 英語   | `docs/web/en/` |
| 日本語 | `docs/web/ja/` |

新機能追加時は**両方**を更新。

## API リファレンスの追加

新しいモジュールを追加した場合:

1. `docs/web/en/reference/` に `.rst` ファイルを作成
2. `index.rst` の toctree に追加
3. 日本語版も同様
