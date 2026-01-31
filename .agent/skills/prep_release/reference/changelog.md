# Step 2: CHANGELOG Management

CHANGELOG.md の整備。

## Instructions

### 1. Read Current CHANGELOG

`CHANGELOG.md` ファイルを読込：

```bash
cat CHANGELOG.md | head -50
```

### 2. Verify Unreleased Section

"Unreleased" セクションが存在することを確認：

```markdown
## Unreleased

### Added
- Feature 1
- Feature 2

### Fixed
- Bug fix 1
- Bug fix 2

## [0.1.0] - 2026-01-20
```

### 3. Create Release Header

新しいリリースヘッダーを作成（日付は本日）：

```markdown
## [0.4.1] - 2026-01-31

### Added
- [Content from Unreleased/Added]

### Fixed
- [Content from Unreleased/Fixed]

### Changed
- [Content from Unreleased/Changed]

### Deprecated
- [Content from Unreleased/Deprecated]

### Removed
- [Content from Unreleased/Removed]
```

### 4. Update Unreleased Section

残された "Unreleased" セクション（新しい内容）：

```markdown
## Unreleased

### Added
- [Empty or placeholder for next release]
```

### 5. Verify Format

- セクション名が標準的か確認：Added, Fixed, Changed, Deprecated, Removed
- リンク参照が正しいか確認 `[0.4.1]: https://github.com/...`

## CHANGELOG Format (Keep a Changelog)

推奨フォーマット：[Keep a Changelog](https://keepachangelog.com/)

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-01-20

### Added
- Initial release

[Unreleased]: https://github.com/...
[0.1.0]: https://github.com/...
```

## Categories

- **Added**: 新機能
- **Fixed**: バグ修正
- **Changed**: 既存機能の変更
- **Deprecated**: 非推奨化
- **Removed**: 削除
- **Security**: セキュリティ問題の修正

## Tips

- 各バージョンを日付付きで記録
- User-facing changes のみを記載
- 開発者向けの内部変更は記載しない
- Git commit ハッシュを含める場合は参考程度に
