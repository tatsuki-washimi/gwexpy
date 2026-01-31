# [Task for GPT 5.2] Phase 2: 依存関係の確認

## 目的
gwexpy v0.1.0b1 のリリース準備として、`pyproject.toml` の依存関係定義を確認し、過不足やバージョン制約の問題がないか検証する。

---

## 背景

Phase 1 でパッケージメタデータの基本検証は完了していますが、以下を最終確認する必要があります：

1. **必須依存関係（dependencies）**: 不足や過剰がないか
2. **オプション依存関係（optional-dependencies）**: 各エクストラの整合性
3. **バージョン制約**: 適切な制約設定か
4. **実際の使用との整合性**: ソースコード内のimportと一致しているか

---

## 実施タスク

### Task 1: pyproject.toml の依存関係確認

#### 1-1. 依存関係定義の読み取り

以下のセクションを Read ツールで確認してください：

```toml
[project]
dependencies = [...]

[project.optional-dependencies]
gw = [...]
stats = [...]
fitting = [...]
...
```

#### 1-2. 依存関係の分類整理

以下のカテゴリに分類してください：

**必須依存関係（dependencies）:**
- 基盤ライブラリ（gwpy, numpy, scipy, astropy, pandas）
- ユーティリティ（matplotlib, h5py）

**オプション依存関係（optional-dependencies）:**
- `[gw]`: 重力波解析専用
- `[stats]`: 統計解析
- `[fitting]`: フィッティング
- `[astro]`: 天体物理
- `[geophysics]`: 地球物理
- `[audio]`: 音響解析
- `[bio]`: 生体信号
- `[interop]`: 相互運用性
- `[control]`: 制御工学
- `[plot]`: プロット・マッピング
- `[analysis]`: 高度な信号処理
- `[gui]`: GUI
- `[all]`: 全依存関係

---

### Task 2: 実際の使用との整合性確認

#### 2-1. ソースコード内のimport調査

以下のコマンドでimport文を抽出してください：

```bash
# gwexpyパッケージ内のimport文を抽出
grep -rh "^import\|^from.*import" gwexpy/ --include="*.py" | \
  grep -v "^from gwexpy" | \
  sort -u | head -50
```

#### 2-2. 未宣言の依存関係の検出

抽出されたimportと pyproject.toml の依存関係を比較し、以下を確認：

**確認項目:**
1. import されているが依存関係に含まれていないライブラリ
2. 依存関係に含まれているが実際には使用されていないライブラリ
3. 標準ライブラリとの区別（標準ライブラリは依存関係不要）

**標準ライブラリ例:**
- sys, os, pathlib, datetime, json, re, collections, typing, logging, warnings

---

### Task 3: バージョン制約の妥当性確認

#### 3-1. 現在のバージョン制約の確認

各依存関係のバージョン制約を確認してください：

**確認項目:**
- 最小バージョン指定の妥当性（`>=X.Y.Z`）
- 上限制約の必要性（`<X.0.0` は避ける）
- ピン留めの妥当性（`==X.Y.Z` は避ける）

**推奨パターン:**
```toml
# 良い例
"numpy>=1.21.0"         # 最小バージョンのみ
"gwpy>=3.0.0"           # メジャーバージョン固定

# 避けるべき例
"numpy>=1.21.0,<2.0"    # 上限制約（非推奨）
"pandas==1.3.0"         # ピン留め（非推奨）
```

#### 3-2. Python バージョンとの整合性

`requires-python = ">=3.9"` との整合性を確認：

- Python 3.9 でサポートされていないライブラリはないか
- Python 3.9 で動作する最小バージョンになっているか

---

### Task 4: オプション依存関係の整合性確認

#### 4-1. エクストラの重複確認

複数のエクストラに同じライブラリが含まれていないか確認：

```bash
# pyproject.toml から optional-dependencies を抽出して分析
```

**期待される動作:**
- 基盤ライブラリは `dependencies` にのみ存在
- 特定用途のライブラリは適切なエクストラにのみ存在

#### 4-2. `[all]` エクストラの完全性確認

`[all]` が他のすべてのエクストラを包含しているか確認：

**確認方法:**
1. 各エクストラの依存関係をリストアップ
2. `[all]` が全てを含んでいるか検証
3. PyPI で提供されないライブラリ（nds2-client等）の扱いを確認

---

### Task 5: 問題の修正（必要な場合）

#### 5-1. 不足依存関係の追加

ソースコードで使用されているが宣言されていないライブラリがある場合：

```toml
dependencies = [
    # 既存の依存関係...
    "新規ライブラリ>=X.Y.Z",
]
```

#### 5-2. 過剰依存関係の削除

宣言されているが実際には使用されていないライブラリがある場合：

**注意:**
- 将来的に使用予定の場合は残す
- ドキュメントで言及されている場合は残す
- テストでのみ使用される場合は `[dev]` に移動

#### 5-3. バージョン制約の調整

Python 3.9 互換性の問題がある場合、最小バージョンを引き上げ：

```toml
# 修正前
"numpy>=1.19.0"

# 修正後（Python 3.9対応）
"numpy>=1.21.0"
```

---

## 制約事項

### 変更してはいけないもの

1. **動作中の依存関係**: 問題がない限り変更しない
2. **エクストラの構造**: 既存の分類を大きく変更しない

### 慎重に扱うべきもの

1. **バージョン制約の追加**: 必要最小限にとどめる
2. **依存関係の削除**: 本当に不要か十分確認

---

## 成果物

以下の情報を報告してください：

### 1. 依存関係の現状

```
必須依存関係数: 〇〇個
オプション依存関係（エクストラ数）: 〇〇個

主要な依存関係:
- gwpy: >=X.Y.Z
- numpy: >=X.Y.Z
- scipy: >=X.Y.Z
- ...
```

### 2. 整合性チェック結果

```
ソースコード内のimport調査:
- 総import数: 〇〇個
- 標準ライブラリ: 〇〇個
- サードパーティ: 〇〇個

未宣言の依存関係:
- [ライブラリ名]: 使用箇所
または: なし

未使用の依存関係:
- [ライブラリ名]: 依存関係に含まれているが未使用
または: なし
```

### 3. バージョン制約の評価

```
適切なバージョン制約: 〇〇個
上限制約あり（要確認）: 〇〇個
ピン留めあり（要確認）: 〇〇個

Python 3.9 互換性:
- 問題あり: [ライブラリ名]
または: 問題なし
```

### 4. オプション依存関係の評価

```
エクストラ重複:
- [ライブラリ名]: [エクストラ1], [エクストラ2] に重複
または: なし

[all] エクストラの完全性:
- 包含されていないエクストラ: [名前]
または: すべて包含済み
```

### 5. 修正内容（実施した場合）

```
追加した依存関係:
- [ライブラリ名]>=X.Y.Z: 理由

削除した依存関係:
- [ライブラリ名]: 理由

バージョン制約の変更:
- [ライブラリ名]: 変更前 → 変更後: 理由

または: 修正不要
```

---

## 完了の定義

以下の条件をすべて満たした時点で、このタスクは完了とします：

- [ ] pyproject.toml の依存関係を確認している
- [ ] ソースコード内のimportを調査している
- [ ] 未宣言/未使用の依存関係を特定している
- [ ] バージョン制約の妥当性を評価している
- [ ] Python 3.9 互換性を確認している
- [ ] オプション依存関係の整合性を確認している
- [ ] 必要な修正を実施している（または不要と判断）
- [ ] 依存関係の現状と推奨事項が報告されている

---

## タイムライン

このタスクは **Phase 2: 品質向上** の一部です。

**並行作業:**
- CI設定確認（GPT 5.2 - 同一セッション）

**次のタスク:**
- TestPyPI アップロード準備

完了後、結果を Claude Sonnet 4.5 に報告してください。

---

## 参考情報

### 依存関係の確認スクリプト例

```python
import ast
import sys
from pathlib import Path

def extract_imports(file_path):
    """Extract all import statements from a Python file"""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

    return imports

# gwexpy/ 内の全Pythonファイルをスキャン
all_imports = set()
for py_file in Path('gwexpy').rglob('*.py'):
    all_imports.update(extract_imports(py_file))

print("\\n".join(sorted(all_imports)))
```

### check_deps スキルの活用

gwexpy には `check_deps` スキルが用意されています：

```bash
# ソースコード内のimportとpyproject.tomlの依存関係をチェック
# （スキルが利用可能な場合）
```
