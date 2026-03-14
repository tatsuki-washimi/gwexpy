# Phase 3 Step 11: 統合テスト・検証 ✅

## 目的
Steps 6-10で実装したメタデータ抽出機能が正常に動作し、既存テストに回帰がないことを確認します。

## 実装内容の確認（参考）

### Step 6: tinytag依存追加
- [pyproject.toml:94-97](../../../pyproject.toml#L94-L97): `[project.optional-dependencies]` に `audio` グループを追加
- 依存パッケージ: pydub, tinytag>=1.10

### Step 7: メタデータ抽出ヘルパー
- [gwexpy/io/utils.py:231-299](../../../gwexpy/io/utils.py#L231-L299): `extract_audio_metadata()` 関数実装
- tinytag未インストール時や抽出失敗時は警告を出して空の辞書を返す

### Step 8: audio.py メタデータ統合
- [gwexpy/timeseries/io/audio.py:20,48,67-71,136-140](../../../gwexpy/timeseries/io/audio.py): `extract_metadata: bool = False` パラメータ追加
- メタデータ抽出ロジックをプロベナンスに統合

### Step 9: wav.py メタデータ統合
- [gwexpy/timeseries/io/wav.py:6,33,50-54,115-129](../../../gwexpy/timeseries/io/wav.py): audio.py と同一パターン実装
- WAVファイルの読み込みで `extract_metadata` オプションをサポート

### Step 10: メタデータテスト作成
- [tests/io/test_audio_metadata.py](../../../tests/io/test_audio_metadata.py): 新規テストファイル (230行)
  - 11個のテスト関数
  - tinytag可用性に応じた条件付き実行
  - tinytag未インストール時の警告テスト
  - メタデータ抽出失敗時のエラーハンドリングテスト

---

## タスク

### 1️⃣ 全I/Oテスト実行
```bash
pytest tests/io/ -v
```

**検証内容:**
- [ ] 全テストが PASS
- [ ] Step 10 で新規追加した `test_audio_metadata.py` が含まれている
- [ ] 既存テスト（magic number, GBD, ATS, WAV）に回帰なし
- [ ] 実行時間を記録（パフォーマンス基準値）

**期待される結果:**
```
tests/io/test_magic_number_identifiers.py::test_identify_gbd_with_valid_header PASSED
tests/io/test_audio_metadata.py::test_extract_audio_metadata_from_wav PASSED (skipped if tinytag missing)
tests/io/test_audio_metadata.py::test_extract_audio_metadata_no_tinytag PASSED
...

======= XXX passed, YYY skipped in Z.XXs =======
```

**失敗時の対応:**
- [ ] FAIL した場合は、テスト名と出力を詳細に報告
- [ ] エラーメッセージから根本原因を特定
- [ ] 必要に応じて `pytest tests/io/test_audio_metadata.py -vv` で詳細ログを確認

---

### 2️⃣ カバレッジ確認
```bash
pytest tests/io/ --cov=gwexpy.timeseries.io --cov-report=term-missing
```

**検証内容:**
- [ ] カバレッジ結果を記録（目標: > 80%）
- [ ] 新規関数のカバレッジ確認:
  - `extract_audio_metadata()` (gwexpy/io/utils.py)
  - `identify_gbd()` (gwexpy/timeseries/io/gbd.py)
  - `identify_ats()` (gwexpy/timeseries/io/ats.py)
  - `read_timeseriesdict_audio(extract_metadata=True)` (gwexpy/timeseries/io/audio.py)
  - `read_timeseriesdict_wav(extract_metadata=True)` (gwexpy/timeseries/io/wav.py)
- [ ] カバレッジが低い行をリスト化（90%以上が目標）

**期待される結果:**
```
Name                                          Stmts   Miss  Cover   Missing
─────────────────────────────────────────────────────────────────────────────
gwexpy/timeseries/io/__init__.py                XX      X    XX%
gwexpy/timeseries/io/_registration.py          XX      X    XX%
gwexpy/timeseries/io/ats.py                    XX      X    XX%
gwexpy/timeseries/io/gbd.py                    XX      X    XX%
gwexpy/timeseries/io/audio.py                  XX      X    XX%
gwexpy/timeseries/io/wav.py                    XX      X    XX%
gwexpy/io/utils.py                             XX      X    XX%
─────────────────────────────────────────────────────────────────────────────
TOTAL                                          XXX     X    XX%
```

---

### 3️⃣ 型チェック
```bash
mypy gwexpy/timeseries/io
```

**検証内容:**
- [ ] 型チェックエラーが 0 個
- [ ] 新規関数の型アノテーションが正しい
- [ ] Optional型や Union型の使用が適切か

**期待される結果:**
```
Success: no issues found in 7 source files
```

**警告がある場合:**
- [ ] `error:` で始まる行がないことを確認（warnings は許容）
- [ ] 新規コードに関するエラーがないことを確認

---

## 回帰テスト: Phase 1-2実装との互換性確認

以下のコマンドで、Phase 1-2の実装に影響がないことを確認:

```bash
# Phase 1-2関連テスト（ATS, GBD, WAV, Audio読み込み）
pytest tests/io/test_io_improvements_extended.py -v

# 既存のmagic number識別テスト
pytest tests/io/test_magic_number_identifiers.py -v
```

**確認項目:**
- [ ] `test_io_improvements_extended.py`: 全テスト PASS
- [ ] `test_magic_number_identifiers.py`: 全テスト PASS
- [ ] 他のI/Oモジュール（GWF, HDF5など）に影響なし

---

## 出力フォーマット

実行結果を以下のフォーマットで報告してください:

```
## ✅ Step 11: 統合テスト・検証 - 完了報告

### 1. 全I/Oテスト実行
- **コマンド**: pytest tests/io/ -v
- **結果**: XXX passed, YYY skipped in Z.XXs ✅
- **テスト項目**:
  - GBD magic number: 6 tests PASS ✅
  - ATS magic number: 12 tests PASS ✅
  - Audio metadata: 6 tests (N skipped if tinytag missing) ✅
  - WAV metadata: 2 tests (N skipped if tinytag missing) ✅
  - その他既存テスト: NNN tests PASS ✅

### 2. カバレッジ確認
- **全体カバレッジ**: XX.X% ✅
- **gwexpy.timeseries.io**: XX.X%
  - _registration.py: XX.X%
  - gbd.py: XX.X%
  - ats.py: XX.X%
  - audio.py: XX.X%
  - wav.py: XX.X%
- **gwexpy.io.utils**: XX.X%

### 3. 型チェック
- **mypy gwexpy/timeseries/io**: Success (0 errors) ✅

### 回帰テスト
- **Phase 1-2実装**: 全テスト PASS ✅
- **既存I/Oモジュール**: 影響なし ✅

---

### 🎯 最終判定
- [ ] ✅ **全検査 PASS** → Step 12へ進む（Sonnet: ドキュメント・コミット）
- [ ] ⚠️ **一部失敗あり** → 詳細を報告（修正後の再検証）
```

---

## トラブルシューティング

**Q: tinytag関連テストがスキップされる**
- これは正常です（tinytag はオプショナル依存）
- tinytag がインストールされていない場合、テストはスキップされます
- 実装は動作していますが、メタデータ抽出では警告を出します

**Q: test_audio_metadata.py で ImportError が出る**
- `from unittest.mock import patch` が使用可能な Python 3.11+ を確認
- `pytest --version` が最新であることを確認

**Q: カバレッジが低い**
- エラーハンドリング分岐（Exception 処理）は tinytag 未インストール環境ではカバーされません
- スキップされたテストをカウントするかどうかで判断してください

**Q: mypy エラーが出た場合**
- 新規関数の型アノテーションを確認
- `dict[str, Any]` や `str | Path` の型が正しいか確認
- 必要に応じて `# type: ignore` コメントを追加（最小限）

---

## 関連ファイル
- **新規テスト**: tests/io/test_audio_metadata.py
- **新規関数**: gwexpy/io/utils.py (extract_audio_metadata)
- **修正ファイル**: gwexpy/timeseries/io/{audio,wav}.py
- **参考**: .claude/plans/snug-honking-pelican.md (Step 11)
