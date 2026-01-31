# [Task for GPT 5.2] Phase 2: CI設定の確認

## 目的
gwexpy v0.1.0b1 のリリース準備として、CI設定（pytest マーカー、テスト環境、ノートブック検証）の動作を確認し、問題があれば修正する。

---

## 背景

現在のCI設定（GitHub Actions）が以下の要件を満たしているか確認する必要があります：

1. **pytest マーカーの動作**: 特定のテスト（NDS接続、GUI等）を除外可能か
2. **ノートブック検証**: タイムアウト対策が適切か
3. **GUI環境整備**: GUI テスト環境が適切に設定されているか

---

## 実施タスク

### Task 1: CI設定ファイルの確認

#### 1-1. GitHub Actions ワークフローの読み取り

以下のファイルを Read ツールで読み取ってください：

```
.github/workflows/test.yml
.github/workflows/docs.yml
```

#### 1-2. pytest設定の確認

以下のファイルで pytest 設定を確認してください：

```
pyproject.toml (tool.pytest.ini_options)
pytest.ini (存在する場合)
```

**確認項目:**
- マーカーの定義（`markers = ...`）
- テストパスの設定
- 警告フィルタの設定
- タイムアウト設定

---

### Task 2: pytest マーカーの動作確認

#### 2-1. 定義済みマーカーの一覧取得

```bash
pytest --markers
```

**期待されるマーカー（例）:**
- `nds`: NDS接続が必要なテスト
- `gui`: GUIテストが必要なテスト
- `slow`: 実行時間が長いテスト

#### 2-2. マーカーの動作テスト

以下のコマンドでマーカーが正しく機能するか確認してください：

```bash
# NDSテストを除外
pytest tests/ -v -m "not nds" --collect-only | grep -c "test"

# GUIテストを除外
pytest tests/ -v --ignore=tests/gui/ --collect-only | grep -c "test"

# slowテストのみ実行（定義されている場合）
# pytest tests/ -v -m "slow" --collect-only
```

**期待される動作:**
- マーカーで適切にテストがフィルタされる
- 警告メッセージがない（未登録マーカーの警告など）

---

### Task 3: ノートブック検証の確認

#### 3-1. nbmake設定の確認

```bash
# nbmake がインストールされているか確認
pytest --version | grep nbmake

# または
pip list | grep nbmake
```

#### 3-2. ノートブックテストの実行

```bash
# タイムアウト設定を確認しながら実行
pytest docs/web/en/guide/tutorials/ --nbmake --nbmake-timeout=300 -v
```

**確認項目:**
- タイムアウト値が適切か（300秒 = 5分）
- 高負荷なノートブック（advanced_arima.ipynb等）で問題ないか
- スキップされるべきノートブックがスキップされているか

#### 3-3. 問題のあるノートブックの特定

タイムアウトが発生する場合、以下を記録してください：

```
ノートブック名: advanced_arima.ipynb
タイムアウト時間: 300秒
推奨対応: タイムアウト延長 or CI除外
```

---

### Task 4: CI設定ファイルの修正（必要な場合）

#### 4-1. pytest マーカーの追加

pyproject.toml に未定義マーカーがある場合、追加してください：

```toml
[tool.pytest.ini_options]
markers = [
    "nds: tests requiring NDS connection",
    "gui: tests requiring GUI environment",
    "slow: slow running tests (> 1 minute)",
]
```

#### 4-2. ノートブックタイムアウトの調整

`.github/workflows/test.yml` または `pyproject.toml` で設定を調整：

```yaml
# GitHub Actions例
- name: Test notebooks
  run: pytest docs/ --nbmake --nbmake-timeout=600  # 10分に延長
```

または

```toml
# pyproject.toml例
[tool.pytest.ini_options]
addopts = "--nbmake-timeout=600"
```

#### 4-3. 高負荷ノートブックの除外

CI で実行が困難なノートブックを除外：

```yaml
# .github/workflows/test.yml
- name: Test notebooks
  run: |
    pytest docs/ --nbmake \
      --ignore=docs/web/en/guide/tutorials/advanced_arima.ipynb \
      --ignore=docs/web/ja/guide/tutorials/advanced_arima.ipynb
```

---

### Task 5: 検証

#### 5-1. マーカー動作の再確認

```bash
pytest --markers | grep -E "nds|gui|slow"
```

#### 5-2. テスト実行の検証

```bash
# 全テスト（GUI除外）
pytest tests/ -v --ignore=tests/gui/

# NDSテスト除外
pytest tests/ -v -m "not nds"
```

**期待される結果:**
- マーカー警告なし
- 適切なテスト除外
- 全テスト合格

---

## 制約事項

### 変更してはいけないもの

1. **既存のテストロジック**: テスト自体の内容は変更しない
2. **動作しているCI設定**: 問題がない設定は変更しない

### 慎重に扱うべきもの

1. **タイムアウト値**: 大きすぎるとCI時間が増加
2. **テスト除外**: 必要なテストまで除外しない

---

## 成果物

以下の情報を報告してください：

### 1. CI設定の現状

```
GitHub Actions ワークフロー:
- test.yml: [主要設定の概要]
- docs.yml: [主要設定の概要]

pytest設定:
- マーカー定義: [nds, gui, slow など]
- タイムアウト: [現在の値]秒
- 警告フィルタ: [設定内容]
```

### 2. pytest マーカー動作確認結果

```
定義済みマーカー:
- nds: [動作OK/NG]
- gui: [動作OK/NG]
- slow: [定義なし/動作OK/NG]

マーカーテスト結果:
- 全テスト数: 〇〇件
- nds除外後: 〇〇件
- gui除外後: 〇〇件
```

### 3. ノートブック検証結果

```
nbmake インストール: Yes/No
タイムアウト設定: 〇〇秒

問題のあるノートブック:
- [ノートブック名]: タイムアウト発生 → 推奨対応
- [ノートブック名]: エラー発生 → 推奨対応

または: 問題なし
```

### 4. 修正内容（実施した場合）

```
修正ファイル:
1. [ファイル名]: 修正内容の概要
2. [ファイル名]: 修正内容の概要

修正前後の比較:
- [具体的な変更内容]
```

### 5. 検証結果

```
pytest --markers: [マーカー警告の有無]
pytest tests/ (GUI除外): [結果]
pytest tests/ (NDS除外): [結果]
```

---

## 完了の定義

以下の条件をすべて満たした時点で、このタスクは完了とします：

- [ ] CI設定ファイル（test.yml, pyproject.toml）を確認している
- [ ] pytest マーカーが定義され、動作している
- [ ] マーカー警告が発生していない
- [ ] ノートブック検証の動作を確認している（または問題点を特定）
- [ ] 必要な修正を実施している（または不要と判断）
- [ ] 修正後の検証を完了している
- [ ] CI設定の現状と推奨事項が報告されている

---

## タイムライン

このタスクは **Phase 2: 品質向上** の一部です。

**並行作業:**
- 依存関係確認（GPT 5.2 - 同一セッション）

**次のタスク:**
- TestPyPI アップロード準備

完了後、結果を Claude Sonnet 4.5 に報告してください。

---

## 参考情報

### pytest マーカーの使用例

```python
# テストファイル内
import pytest

@pytest.mark.nds
def test_nds_connection():
    """Requires NDS server connection"""
    pass

@pytest.mark.slow
def test_heavy_computation():
    """Takes > 1 minute"""
    pass
```

### GitHub Actions での pytest実行例

```yaml
- name: Run tests
  run: |
    pytest tests/ -v \
      --ignore=tests/gui/ \
      -m "not nds" \
      --cov=gwexpy \
      --cov-report=xml
```

### ノートブックテストのスキップ

```python
# ノートブック内のセルで
# pytest: skip
```
