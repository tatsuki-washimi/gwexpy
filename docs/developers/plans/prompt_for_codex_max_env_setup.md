# [Task for GPT 5.1-Codex-Max] Phase 1: 環境準備とパッケージ検証

## 目的
gwexpy v0.1.0b1 のリリース準備として、PyPIパッケージング用のツール（twine）をインストールし、パッケージメタデータとビルド成果物の検証を実施する。

---

## 背景

現在、gwexpy は初公開ベータ版（v0.1.0b1）のリリース準備中です。TestPyPI経由での公開前に、以下を確認する必要があります：

1. パッケージビルドツール（twine）の利用可能性
2. パッケージメタデータの妥当性（PyPI表示用の長文説明、classifiers等）
3. ビルド成果物の内容（必須ファイルの同梱確認）

---

## 実施タスク

### Task 1: twine のインストール

#### 1-1. 現在の状態確認

まず、twine が既にインストールされているか確認してください。

```bash
twine --version
```

#### 1-2. インストール（必要な場合）

twine がインストールされていない場合、以下のコマンドでインストールしてください。

```bash
pip install twine
```

インストール後、再度バージョンを確認してください。

```bash
twine --version
```

---

### Task 2: パッケージのビルド

#### 2-1. ビルドツールの確認

`build` パッケージがインストールされていることを確認してください。

```bash
python -m build --version
```

インストールされていない場合：

```bash
pip install build
```

#### 2-2. パッケージのビルド

プロジェクトルート（`/home/washimi/work/gwexpy`）で以下を実行してください。

```bash
python -m build
```

これにより、`dist/` ディレクトリに以下が生成されます：
- `gwexpy-0.1.0b1.tar.gz`（ソース配布物）
- `gwexpy-0.1.0b1-py3-none-any.whl`（ビルド済み配布物）

---

### Task 3: twine check によるメタデータ検証

#### 3-1. メタデータの検証

ビルドされたパッケージのメタデータを検証してください。

```bash
twine check dist/*
```

#### 3-2. 結果の記録

以下の情報を記録してください：

- **検証結果**: PASSED / FAILED
- **警告メッセージ**: 警告がある場合はその内容
- **エラーメッセージ**: エラーがある場合はその内容と該当箇所

---

### Task 4: パッケージ内容の確認

#### 4-1. tarball の内容確認

ソース配布物に必須ファイルが含まれているか確認してください。

```bash
tar -tzf dist/gwexpy-0.1.0b1.tar.gz | grep -E "(LICENSE|README|py.typed|pyproject.toml)" | head -10
```

以下のファイルが含まれていることを確認：
- `LICENSE`
- `README.md`
- `gwexpy/py.typed`（型ヒント対応マーカー）
- `pyproject.toml`

#### 4-2. wheel の内容確認

```bash
unzip -l dist/gwexpy-0.1.0b1-py3-none-any.whl | grep -E "(LICENSE|README|py.typed|METADATA)" | head -10
```

#### 4-3. バージョン整合性の確認

パッケージメタデータのバージョンが `0.1.0b1` であることを確認してください。

```bash
unzip -p dist/gwexpy-0.1.0b1-py3-none-any.whl gwexpy-0.1.0b1.dist-info/METADATA | grep "^Version:"
```

期待される出力: `Version: 0.1.0b1`

---

### Task 5: pyproject.toml のメタデータ確認

#### 5-1. PyPI表示用メタデータの読み取り

以下の項目が適切に設定されているか確認してください。

```bash
grep -A 5 "^\[project\]" pyproject.toml
```

確認項目：
- `name = "gwexpy"`
- `version = "0.1.0b1"`
- `description` が簡潔で明確か
- `readme = "README.md"`
- `license` が設定されているか
- `classifiers` が適切か（Python 3.9+, MIT License等）

---

## 成果物

以下の情報を報告してください：

### 1. 環境情報
```
- twine バージョン
- build バージョン
- Python バージョン
```

### 2. ビルド結果
```
- ビルド成功/失敗
- 生成されたファイル（dist/ 内のファイル一覧）
```

### 3. twine check 結果
```
- 検証ステータス（PASSED/FAILED）
- 警告・エラーメッセージ（あれば全文）
```

### 4. パッケージ内容チェック
```
- LICENSE 含まれている: Yes/No
- README.md 含まれている: Yes/No
- py.typed 含まれている: Yes/No
- バージョン整合性: 0.1.0b1 で一致 Yes/No
```

### 5. 問題点・推奨事項
```
- 発見された問題点
- 修正が必要な項目
- 次のステップへの推奨事項
```

---

## 制約事項

- **実行のみ**: コードの修正は行わないでください。検証と情報収集のみを実施してください。
- **結果の記録**: すべてのコマンド出力を記録してください。
- **エラーハンドリング**: エラーが発生した場合は、その内容を詳細に報告してください。

---

## 完了の定義

以下の条件をすべて満たした時点で、このタスクは完了とします：

- [ ] twine がインストールされている
- [ ] パッケージが正常にビルドされている（dist/ に .tar.gz と .whl が存在）
- [ ] `twine check` が実行され、結果が報告されている
- [ ] パッケージ内容（LICENSE, README, py.typed）が確認されている
- [ ] バージョンが 0.1.0b1 で一致している
- [ ] すべての検証結果が報告されている

---

## タイムライン

このタスクは **Phase 1: 基盤修正** の一部であり、以下と並列実行されます：

- GPT 5.2-Codex: Python 3.9 型ヒント修正
- Claude Sonnet 4.5: ドキュメント更新（完了）

完了後、結果を Claude Sonnet 4.5 および他のチームメンバーに報告してください。
