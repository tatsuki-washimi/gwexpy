---
name: docs-sync
description: 公開 API・install 手順・主要挙動の変更時に、README・docs・notebook の追従漏れを検知・修正する。
trigger: manual
---

# Docs & Notebook Drift Detector

コードの変更がドキュメントやチュートリアルに反映されているかを確認します。

## When to Use
- **Public API 変更**: クラス名、関数名、引数の変更
- **インストーラ/依存関係**: `pyproject.toml` の変更
- **主要な挙動の変更**: 物理的な計算ロジックやデフォルト値の変更

## Drift Detection Checklist

1. **README / docs/ (README.md, docs/*.rst)**
   - インストール手順が最新か (`pip install`, `conda install`)
   - クイックスタートのコード例が動作するか
   - 物理的な単位や前提条件の変更が反映されているか

2. **Tutorial / Notebook Assets**
   - notebook やチュートリアル資産が存在する場合、API 変更によりセルやコード例がエラーにならないか
   - プロットの結果が意図せず大きく変わっていないか
   - ノートブックが未整備の領域では README / docs のコード例を優先して確認する

3. **Metadata (pyproject.toml, CITATION.cff, optional codemeta.json)**
   - バージョン番号、著者、引用情報が最新か
   - `codemeta.json` が存在しない場合はチェック対象から外す

## Integration with Feature Development
実装完了前に以下のコマンドでドキュメントの差分を確認してください。

```bash
git diff --name-only main | grep -E "README|docs/|pyproject.toml"
```

ドキュメントの更新が不要な場合は、`evidence-pack` の「Known Gaps」または PR コメントにその理由を簡潔に記述してください（例: "Internal refactoring only, no API changes"）。

## Docs Build Verification
```bash
cd docs
make html
```
*ビルド中に警告が発生していないか、特に `autosummary` や `nitpick` のエラーがないか確認してください。*
