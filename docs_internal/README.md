# docs_internal/

内部向けドキュメントを管理するディレクトリです。

## 目的

git で管理しますが、**公開 Web サイト（Sphinx ビルド）には含まれません**。
プロジェクトの計画・報告・論文原稿など、外部公開が不要または望ましくない
文書を収容します。

## ディレクトリ構成

```
docs_internal/
├── plans/           # プロジェクト計画・実装方針
├── reports/         # 完了報告・進捗記録
├── archive/         # 過去の計画・報告（参照用）
├── reviews/         # レビュー会話記録
├── prompts/         # AI エージェント向けプロンプト
├── references/      # 外部文献・資料の抽出
├── analysis/        # 技術分析・設計検討
└── publications/    # 論文・学会発表資料
    └── paper_softwarex/   # SoftwareX 論文原稿
```

## 公開ドキュメントとの違い

| 項目 | docs/ | docs_internal/ |
|------|-------|----------------|
| Sphinx でビルド | Yes | No |
| 公開 Web サイトに掲載 | Yes | No |
| git 管理 | Yes | Yes |
| 対象読者 | ユーザー・貢献者 | 開発者（内部） |
