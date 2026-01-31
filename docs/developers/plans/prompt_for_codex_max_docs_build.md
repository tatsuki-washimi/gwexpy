# [Task for GPT 5.1-Codex-Max] Phase 2: ドキュメントビルド検証

## 目的
gwexpy v0.1.0b1 のリリース準備として、Sphinx ドキュメントのビルドを実行し、警告・エラーがないことを確認する。

---

## 背景

Phase 1, 2 でドキュメント内容の整備は完了していますが、実際のビルドとリンクチェックを実施し、以下を確認する必要があります：

1. **ビルド成功**: sphinx-build が警告なしで完了するか
2. **リンクチェック**: 外部リンク・内部リンクが有効か
3. **HTML表示**: 生成されたHTMLが適切に表示されるか

---

## 実施タスク

### Task 1: ドキュメントビルド環境の確認

#### 1-1. Sphinx のインストール確認

```bash
sphinx-build --version
```

必要に応じてインストール：

```bash
pip install sphinx sphinx-rtd-theme nbsphinx myst-parser
```

#### 1-2. ドキュメント構成の確認

```bash
ls -la docs/web/
ls -la docs/web/en/
ls -la docs/web/ja/
```

**確認項目:**
- conf.py の存在
- index.rst の存在
- 各種 .rst/.md ファイル

---

### Task 2: ドキュメントビルドの実行

#### 2-1. 英語版ドキュメントのビルド

```bash
cd docs/web/en
sphinx-build -b html . _build/html -nW
```

**オプション説明:**
- `-b html`: HTML出力
- `-nW`: 警告をエラーとして扱う（厳密モード）
- `-n`: nit-picky モード（より厳密な警告）

**期待される出力:**
```
build succeeded, 0 warnings.
```

#### 2-2. 日本語版ドキュメントのビルド

```bash
cd docs/web/ja
sphinx-build -b html . _build/html -nW
```

#### 2-3. 警告・エラーの記録

ビルド中に警告やエラーが発生した場合、以下を記録してください：

```
警告/エラー内容:
ファイル名: [ファイルパス]
行番号: [行番号]
メッセージ: [エラーメッセージ]
```

---

### Task 3: リンクチェックの実行

#### 3-1. 英語版リンクチェック

```bash
cd docs/web/en
sphinx-build -b linkcheck . _build/linkcheck
```

**確認項目:**
- 外部リンクの有効性
- 内部リンクの有効性
- アンカーリンクの有効性

#### 3-2. リンクチェック結果の確認

```bash
cat _build/linkcheck/output.txt | grep -E "broken|redirect"
```

**許容される警告:**
- 一時的なネットワークエラー
- リダイレクト（301/302）
- localhost へのリンク（開発用）

**対応が必要な警告:**
- 404 エラー（リンク切れ）
- 500 エラー（サーバーエラー）
- タイムアウト（恒久的）

---

### Task 4: HTML表示の確認

#### 4-1. 生成されたHTMLの確認

```bash
# 英語版
ls -lh docs/web/en/_build/html/index.html

# 日本語版
ls -lh docs/web/ja/_build/html/index.html
```

#### 4-2. 主要ページの存在確認

以下のページが生成されているか確認：

```bash
# 英語版
ls docs/web/en/_build/html/guide/installation.html
ls docs/web/en/_build/html/guide/quickstart.html
ls docs/web/en/_build/html/guide/tutorials/index.html

# 日本語版
ls docs/web/ja/_build/html/guide/installation.html
ls docs/web/ja/_build/html/guide/quickstart.html
ls docs/web/ja/_build/html/guide/tutorials/index.html
```

---

### Task 5: 問題の修正（必要な場合）

#### 5-1. 一般的な警告の対応

**警告例1: 参照エラー**
```
WARNING: undefined label: some-label
```

**対応:** ラベルの定義を追加または参照を修正

**警告例2: 画像ファイルなし**
```
WARNING: image file not readable: path/to/image.png
```

**対応:** 画像ファイルのパスを修正または画像を追加

**警告例3: toctree 警告**
```
WARNING: toctree contains reference to nonexisting document
```

**対応:** toctree のパスを修正またはファイルを追加

#### 5-2. リンク切れの対応

**404エラーの場合:**
- リンクURLを修正
- リンクを削除
- 代替URLに変更

---

### Task 6: 検証

#### 6-1. 修正後の再ビルド

修正を実施した場合、再度ビルドを実行：

```bash
cd docs/web/en
sphinx-build -b html . _build/html -nW

cd docs/web/ja
sphinx-build -b html . _build/html -nW
```

#### 6-2. 最終確認

```bash
# 英語版の警告数
cd docs/web/en
sphinx-build -b html . _build/html 2>&1 | grep -c "WARNING"

# 日本語版の警告数
cd docs/web/ja
sphinx-build -b html . _build/html 2>&1 | grep -c "WARNING"
```

**期待される結果:** 0 warnings

---

## 制約事項

### 変更してはいけないもの

1. **既存のドキュメント内容**: 内容の大幅な変更は行わない
2. **Sphinx設定（conf.py）**: 問題がない限り変更しない

### 慎重に扱うべきもの

1. **外部リンク**: リンク先が一時的にダウンしている可能性
2. **画像パス**: 相対パスと絶対パスの扱い

---

## 成果物

以下の情報を報告してください：

### 1. ビルド結果

```
英語版ドキュメント:
- ビルド: 成功/失敗
- 警告数: 〇〇件
- エラー数: 〇〇件
- ビルド時間: 〇〇秒

日本語版ドキュメント:
- ビルド: 成功/失敗
- 警告数: 〇〇件
- エラー数: 〇〇件
- ビルド時間: 〇〇秒
```

### 2. リンクチェック結果

```
英語版:
- チェックしたリンク数: 〇〇個
- 正常: 〇〇個
- リダイレクト: 〇〇個
- エラー: 〇〇個

日本語版:
- チェックしたリンク数: 〇〇個
- 正常: 〇〇個
- リダイレクト: 〇〇個
- エラー: 〇〇個

問題のあるリンク:
- [URL]: エラー内容
または: なし
```

### 3. 生成されたHTML

```
英語版HTMLサイズ: 〇〇MB
日本語版HTMLサイズ: 〇〇MB

主要ページの存在:
- installation.html: Yes/No
- quickstart.html: Yes/No
- tutorials/index.html: Yes/No
```

### 4. 修正内容（実施した場合）

```
修正ファイル:
1. [ファイル名]: 修正内容の概要
2. [ファイル名]: 修正内容の概要

または: 修正不要
```

### 5. 残存する問題（ある場合）

```
警告1: [内容] - 影響度: 低/中/高
警告2: [内容] - 影響度: 低/中/高

推奨対応: [対応方針]

または: 問題なし
```

---

## 完了の定義

以下の条件をすべて満たした時点で、このタスクは完了とします：

- [ ] 英語版ドキュメントのビルドが成功している
- [ ] 日本語版ドキュメントのビルドが成功している
- [ ] ビルド警告が0件（または許容範囲内）
- [ ] リンクチェックを実行している
- [ ] 重大なリンク切れがない（404エラー等）
- [ ] 主要ページのHTMLが生成されている
- [ ] ビルド結果が報告されている

---

## タイムライン

このタスクは **Phase 2: 品質向上** の最終タスクです。

**前提条件:**
- Phase 2 の他のタスクが完了済み

**次のタスク:**
- パッケージ再ビルド
- TestPyPI アップロード

完了後、結果を Claude Sonnet 4.5 に報告してください。

---

## 参考情報

### Sphinx ビルドの一般的なコマンド

```bash
# クリーンビルド
sphinx-build -E -a -b html . _build/html

# 厳密モードでビルド
sphinx-build -W --keep-going -b html . _build/html

# 特定のファイルのみ再ビルド
sphinx-build -b html . _build/html specific_file.rst
```

### conf.py の主要設定

```python
# 警告を無視する設定（使用は慎重に）
suppress_warnings = ['image.nonlocal_uri']

# nbsphinx の実行制御
nbsphinx_execute = 'never'  # ノートブックを実行しない
```

### リンクチェックのオプション

```bash
# タイムアウトを延長
sphinx-build -b linkcheck -D linkcheck_timeout=30 . _build/linkcheck

# 特定のパターンを無視
# conf.py で linkcheck_ignore = [r'http://localhost.*'] を設定
```
