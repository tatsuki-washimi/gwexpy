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

## Pre-Release Gates

リリース前に以下の各項目を再チェックする。

- [ ] **Tests & Coverage**: `pytest` が全テストを通っていること。カバレッジが低下していないか。
- [ ] **Static Analysis**: `ruff` と `strict mypy` が変更ファイルについて clean であること。
- [ ] **Docs Build**: `cd docs && make html` で警告なくビルドでき、新機能が反映されていること。
- [ ] **Notebook Examples**: 主要なチュートリアルノートブックが現在のブランチで正常に動作すること。

## ステップ 1: prep_release スキル

```
/prep_release
```

内部で以下を実施:
- バージョン文字列の同期確認（`pyproject.toml`, `__version__`, `CITATION.cff`, `codemeta.json`）
- CHANGELOG 更新（リリース日とタグの整合）
- TestPyPI への試験ビルド

## Packaging Verification

実際に `pip` でインストール可能か、および配布物が正しいかを確認する。

1. **Build Distribution**: `python -m build`
2. **Twine Check**: `twine check dist/*`
3. **Clean Install Test**: 新しい仮想環境で `pip install dist/*.whl` を行い、`import gwexpy` が成功すること。
4. **Metadata Sync**: `CITATION.cff` の日付が最終リリース日と一致しているか。

## ステップ 2: バージョン同期チェック（自動）

確認対象ファイル:
- `pyproject.toml` — `version = "X.Y.Z"`
- `gwexpy/_version.py` — `__version__`
- `CITATION.cff` — `version:` と `date-released:`
- `codemeta.json` — `"version":`

差異がある場合はタグ作成前に必ず同期し、CHANGELOG の対象バージョン見出しとも整合させる。
`metadata-checker` エージェントを実行し、不整合がないか確認してください。

```
/metadata-checker
```

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
