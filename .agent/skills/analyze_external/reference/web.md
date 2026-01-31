# Web Research

Web からの情報収集・技術トレンド調査。

## 1. Web Search

- **クエリ最適化**: 具体的かつ英語検索を組み合わせ
- **ソース評価**: 公式ドキュメント（GitHub, PyPI, 公式 wiki）を優先

## 2. Content Examination

- **静的分析**: `read_url_content` で Markdown 変換テキストを読み込み
- **動的分析**: CSR サイトやインタラクティブ操作が必要な場合は `browser_subagent` を利用

## 3. Summarization

- 収集情報をそのまま出力するのではなく、現プロジェクトに「どう活用できるか」を軸にサマリー
- ユーザーがフォロー検証できるよう引用元を明記

## Precautions

- 認証が必要なプライベート情報は収集しない
- 情報の鮮度（Date）を確認し、廃止された API やライブラリを推奨しない
