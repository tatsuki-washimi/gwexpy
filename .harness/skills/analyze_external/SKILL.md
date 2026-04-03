---
name: analyze_external
description: 外部コード、ドキュメント（PDF/Word/Excel）、メディアファイル、およびWeb情報を分析する
---

# Analyze External Sources

プロジェクト外部のあらゆるソースを分析・知見化します。

## Quick Usage

```bash
/analyze_external              # General analysis
/analyze_external --code       # External code analysis
/analyze_external --document   # Office document analysis
/analyze_external --media      # Multimedia analysis
/analyze_external --web        # Web research
```

## Modes

### 1. External Code Analysis

外部ライブラリや他言語のコードを分析：

- インストール済みライブラリのソース調査
- 非Pythonコード（C++、MEDM等）の解析
- データ構造・アルゴリズム・ロジックフローの説明

詳細：[reference/code.md](reference/code.md)

### 2. Office Document Analysis

PDF、Word、Excel、PowerPoint からデータ抽出：

詳細：[reference/documents.md](reference/documents.md)

### 3. Multimedia Analysis

動画・音声ファイルの分析：

詳細：[reference/media.md](reference/media.md)

### 4. Web Research

Web 情報収集・技術トレンド調査：

詳細：[reference/web.md](reference/web.md)
