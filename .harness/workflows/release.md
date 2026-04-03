---
name: release
description: GWexpy リリース準備ワークフロー。バージョン更新・CHANGELOG・TestPyPI検証・タグ付けの手順。
trigger: manual
---

# GWexpy リリースワークフロー

## 前提確認

```bash
conda run -n gwexpy pytest tests/ -m "not gui and not nds and not cvmfs"
conda run -n gwexpy ruff check gwexpy/ tests/
conda run -n gwexpy mypy gwexpy/
```

全て PASS であること。

## ステップ 1: prep_release スキル

```
/prep_release
```

内部で以下を実施:
- バージョン文字列の同期確認（`pyproject.toml`, `__version__`, `CITATION.cff`, `codemeta.json`）
- CHANGELOG 更新
- TestPyPI への試験ビルド

## ステップ 2: バージョン同期チェック（自動）

確認対象ファイル:
- `pyproject.toml` — `version = "X.Y.Z"`
- `gwexpy/__init__.py` — `__version__`
- `CITATION.cff` — `version:` と `date-released:`
- `codemeta.json` — `"version":`

## ステップ 3: タグ付けと GitHub Release

```bash
git tag -s vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

GitHub Actions の `release.yml` が PyPI 公開を自動実行。

## ロールバック手順

PyPI 公開後の問題発覚時:
1. PyPI では削除不可（`yank` のみ）
2. `X.Y.Z+1` でホットフィックスリリースを行う
3. `git tag -d vX.Y.Z && git push origin :vX.Y.Z` でタグのみ削除可能
