---
name: organize_project
description: プロジェクトのディレクトリ構造整理、配置ミスのクリーンアップ、および .gitignore の管理を行う
---

# Organize Project

プロジェクト構造の整理と .gitignore 管理を統合的に行います。

## Quick Usage

```bash
/organize_project              # Audit & organize files
/organize_project --ignore     # Manage .gitignore
```

## Modes

### 1. File Organization (デフォルト)

プロジェクトのファイル配置を監査・整理：

- ルートディレクトリの不要ファイルチェック
- テストファイルの配置確認（`tests/` 内か）
- ソースファイルの配置確認（パッケージディレクトリ内か）
- 空ディレクトリ・一時ファイルのクリーンアップ提案

**注意**: ファイル移動時は import パスも更新すること。

### 2. Gitignore Management

`.gitignore` の追加・管理：

1. **Check**: ファイルが既に `.gitignore` にあるか、git 追跡中か確認
2. **Update**: パターンを `.gitignore` に追加（コメント付き）
3. **Untrack**: 追跡済みファイルは `git rm --cached <path>` で解除
