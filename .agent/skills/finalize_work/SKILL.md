---
name: finalize_work
description: 作業完了時の検証・整理・コミットを一括実行する（quick/full モード対応）
---

# Finalize Work

作業完了時の一連のフローを実行するスキル。モードに応じて実行範囲を調整可能。

## Quick Start

```bash
# クイックモード（コミットのみ）
finalize_work --quick

# フルモード（検証・テスト・リント・ドキュメント更新・コミット）
finalize_work --full
```

## Modes

### Quick Mode (デフォルト)

最小限のクリーンアップとコミットのみ実行:

1. 一時ファイルの削除
2. Git ステータス確認
3. コミット

### Full Mode

完全な検証フローを実行:

1. 物理検証（該当する場合）
2. テスト実行
3. リント・型チェック
4. ドキュメント同期
5. ディレクトリ整理
6. コミット
7. 結果報告

## Instructions

### Step 1: クリーンアップ

```bash
# キャッシュの削除
find . -maxdepth 4 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -maxdepth 4 -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null
rm -rf .ruff_cache .pytest-ipython build/ *.egg-info/
```

### Step 2: Git ステータス確認

```bash
git status --short
```

- 大量の変更がある場合は `git add .` を使わず、個別にファイルを追加
- `.gitignore` にないアーティファクトがあれば追加を検討

### Step 3: コミット

```bash
git add <files>
git commit -m "<conventional commit message>"
```

**Conventional Commits 形式**:

- `feat:` 新機能
- `fix:` バグ修正
- `docs:` ドキュメント
- `refactor:` リファクタリング
- `test:` テスト追加
- `chore:` その他

## Full Mode Details

詳細な手順は以下を参照:

- [物理検証](reference/physics.md)
- [テスト実行](reference/tests.md)
- [ドキュメント更新](reference/docs.md)

## Best Practices

- コミット前に必ず `git status` で変更内容を確認
- 削除するファイルは確認を求める（標準的な一時ファイルを除く）
- `__pycache__`, `.vscode`, `.idea` などはコミットしない
