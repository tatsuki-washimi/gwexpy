# External Code Analysis

外部ライブラリや他言語のコード分析。

## Instructions

### 1. Locate Source

- **Python ライブラリ**: `.venv/lib/pythonX.X/site-packages/` から検索
- **Non-Python**: ディレクトリツリーからファイルを特定

### 2. Read and Analyze

- ソースコードを読み込み、実装を解析
- バイナリファイルや未知フォーマットはテキストとして読み取り試行
  - 例：MEDM `.adl` ファイルはテキスト形式
- 外部ロジックと現プロジェクトの関連性をマッピング

### 3. Report

- データ構造、アルゴリズム、ロジックフローの説明
- 現在の実装との差異をハイライト
- 適用可能なパターンの提案
