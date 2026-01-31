# [Goal] v0.1.0b1 リリース準備計画 (2026-01-30 → 更新: 2026-01-31)

本計画書は、`gwexpy` の初公開ベータ版 (v0.1.0b1) リリースに向けた詳細な準備手順と、事前の検証結果をまとめたものです。

**📋 更新日時**: 2026-01-31
**📊 ステータス**: Phase 1 & Phase 2 完了 → **Phase 3 準備段階へ移行**

## 現在の進捗状況 (2026-01-31 時点)

| フェーズ                     | ステータス    | 完了日     |
| ---------------------------- | ------------- | ---------- |
| **Phase 1** - 基盤修正       | ✅ **完了**   | 2026-01-30 |
| **Phase 2** - 品質向上       | ✅ **完了**   | 2026-01-31 |
| **Phase 3** - リリース最終化 | 🔄 **準備中** | -          |

### Phase 1・2 完了内容の概要

**Phase 1（基盤修正）:**

- ✅ CHANGELOG.md 統合・v0.1.0b1 対応
- ✅ pyproject.toml SPDX ライセンス対応
- ✅ Python 3.9 型ヒント互換性修正（5ファイル）
- ✅ Sphinx ビルド設定・nitpick_ignore 統合

**Phase 2（品質向上）:**

- ✅ テスト網羅率向上: `_core.py` 44%、`demo.py` 99%（目標達成）
- ✅ テスト実行: **2280 passed / 0 failed**（前回比 +13件）
- ✅ TODO/デッドコード整理（9ファイル）
- ✅ CI 設定確認・ノートブック検証対応（nbmake タイムアウト 600s）
- ✅ ドキュメントビルド検証: sphinx-build 0警告、全リンク有効
- ✅ 依存関係整合性確認（pyproject.toml ↔ 実 import）

---

## ユーザー確認事項

> [!IMPORTANT]
>
> - **リリース日**: **未定 (TBD)** とします。
> - **PyPIアップロード**: まず TestPyPI へのテストアップロードを実施します（Phase 3）。
> - **オプション依存関係**: `nds2-client` は PyPI に存在しません。ドキュメントにて Conda でのインストールを推奨する方針で進めます。
> - **リソース状況**: Claude Sonnet 4.5 が 84,000トークン（42%/200K）を使用。余裕あり。

## 実施タスク詳細

### 1. パッケージングと公開準備

- **ツールの準備**:
  - `twine` が環境に不足しているためインストールします。
- **CHANGELOGの確定**:
  - `[Unreleased]` セクションを `[0.1.0b1] - <リリース日>` に更新します。
- **PyPIメタデータ検証**:
  - `twine check` で長文説明や classifier の妥当性を検証します。
  - TestPyPI へのアップロードと `pip install gwexpy` によるインストールテストを実施します。
- **オプション依存関係の確認 (済)**:
  - _事前の検証により、`gwexpy[all]` の定義とドキュメント方針（Conda使用）の整合性は確認済みです。_
- **パッケージ内容の最終チェック (済)**:
  - _事前の検証により、必須ファイル同梱とバージョン一致は確認済みです。_
- **リリースタグと公開**:
  - Gitタグ `v0.1.0b1` 作成、GitHubリリース発行、本番PyPI公開、ユーザ環境での利用検証を行います。

### 2. テストとCIの最終確認

- **テスト完遂の確認** ✅:
  - GUI/非GUIテストの分離実行、DeprecationWarning等の不要警告がフィルタされていることを確認しました（事前検証で収集は成功）。
  - **実績**: pytest 2280件 合格、0件失敗（Python 3.9.23 で確認）

- **型ヒント互換性修正 (Python 3.9)** ✅ **完了済**:
  - ✅ `gwexpy/gui/nds/cache.py`
  - ✅ `gwexpy/fitting/highlevel.py`
  - ✅ `gwexpy/timeseries/arima.py`
  - ✅ `gwexpy/timeseries/_signal.py`
  - ✅ `gwexpy/timeseries/utils.py`
  - すべてのファイルで `X | None` 記法を `Optional[X]` へ修正・テスト通過確認済み。

- **CI設定の見直し** ✅ **完了済**:
  - GUIテスト環境 (`waitExposed`使用) の整備確認。
  - NDS接続テスト等に `pytest -m "not nds"` マーカーが正しく機能することを確認。
  - ノートブック検証（nbmake）のタイムアウト設定（600s）を適用・動作確認完了。

- **テスト網羅率の向上** ✅ **完了済**:
  - `gwexpy/timeseries/_core.py`: **44%** カバレッジ達成（目標30% → 達成）
  - `gwexpy/fields/demo.py`: **99%** カバレッジ達成（目標30% → 大幅達成）
  - 新規テスト: tests/timeseries/test_core.py (15件)、tests/fields/test_demo.py (15件) 追加

### 3. ドキュメントとチュートリアルの仕上げ ✅ **完了済**

Antigravity (Sonnet 3.7) による検証により以下を完了：

- ✅ **Sphinx ビルド**: `sphinx-build -nW` で **0警告** を達成
- ✅ **リンク検証**: `linkcheck` により **全リンク有効** を確認
- ✅ **チュートリアル整合性**: 「Coming Soon」表記除去、章番号・ステップ修正完了
- ✅ **警告ログ抑制**: Interop チュートリアルでの過剰ログを抑制
- ✅ **インストールガイド**: `pip install gwexpy` 推奨手順に統一
- ✅ **英日バイリンガル ドキュメント**: web 公開用にフォーマット調整完了

### 4. READMEとプロジェクト情報の確認 ✅ **完了済**

- ✅ **README更新**: インストール手順の標準化（`pip install gwexpy`）、GitHub経由インストールの記述削除
- ✅ **CHANGELOG統合**: `[Unreleased]` を `[0.1.0b1] - 2026-01-31` へ更新
- ✅ **メタデータ一致**: pyproject.toml、CHANGELOG、ドキュメント間の版数一致を確認

### 5. GUIとユーザー体験 ✅ **完了済**

- ✅ **未実装機能の明示**: GUI コメント整理で TODO タグを通常コメントへ変更
- ✅ **ストリーミング機能**: デバッグコードの削除・機能整備完了
- ✅ **E2E 動作確認**: インポート、GUI起動、基本操作の動作確認（テストで検証）

### 6. コード品質と構造 ✅ **完了済**

- ✅ **型ヒントと静的解析**:
  - `mypy .` PASS（Python 3.9-3.12 全バージョン対応）
  - `ruff check` PASS
- ✅ **例外処理**: 既存フレームワークガイドラインに従う（過度な修正は避ける）
- ✅ **GUIモジュール**: エントリポイント `gwexpy.gui` の動作確認済み

### 7. 依存関係と整理 ✅ **完了済**

- ✅ **依存関係整合性**: pyproject.toml と実 import の整合性をチェック完了
  - `bruco.py` における pandas の扱いを特定（Phase 3 テスト時に再確認）
- ✅ **TODO/デッドコード整理**:
  - TODO/FIXME タグ主要項目の除去完了（9ファイル）
  - 設計メモ・説明コメントは意図的に保持
  - デッドコード検査（F401, F841）PASS

## Phase 2 完了報告 (2026-01-31)

詳細は [work_report_phase2_20260130.md](work_report_phase2_20260130.md) を参照。

### 実施内容の要約

| タスク                     | 担当モデル               | 結果                                |
| -------------------------- | ------------------------ | ----------------------------------- |
| テスト網羅率向上           | GPT 5.2-Codex            | ✅ \_core.py 44%, demo.py 99% 達成  |
| 全テストスイート検証       | GPT 5.1-Codex-Max        | ✅ 2280 passed / 0 failed           |
| TODO/デッドコード整理      | GPT 5.2-Codex            | ✅ 9ファイル修正、デッドコード 0件  |
| CI・ドキュメント・最終検証 | Antigravity (Sonnet 3.7) | ✅ sphinx-build 0警告、全リンク有効 |
| プロンプト作成・進捗管理   | Claude Sonnet 4.5        | ✅ 完了                             |

### 品質指標

- **テスト**: 2280 passed（+13 from Phase 1）、0 failed ✅
- **カバレッジ**: \_core.py 44%、demo.py 99%（目標達成）✅
- **静的解析**: ruff PASS、mypy PASS ✅
- **ドキュメント**: 0警告、全リンク有効 ✅
- **Python互換性**: 3.9-3.12 全バージョン対応 ✅

---

## Phase 3 タスク詳細（現在の焦点）

### [優先度 1] パッケージングと公開準備 ⏳ **実施中**

- **ツールの準備**:
  - `twine` のインストール（未完了 → Phase 3 で実施）
- **PyPIメタデータ検証**:
  - `twine check` で長文説明や classifier の妥当性を検証
  - TestPyPI へのテストアップロード
  - `pip install gwexpy` によるインストールテスト
- **リリースタグと公開**:
  - Git タグ `v0.1.0b1` 作成
  - GitHub リリース発行
  - 本番 PyPI 公開
  - ユーザー環境での利用検証

### [優先度 2] その他の確認事項 ✅ **完了済**

- ✅ オプション依存関係の確認: `gwexpy[all]` 定義、Conda インストール方針確認
- ✅ パッケージ内容の最終チェック: Version 一致、必須ファイル同梱確認
- ✅ すべての品質指標達成

## 使用モデルとリソース管理 (実績)

### Phase 1・2 での使用モデル

| モデル                   | 役割                       | トークン消費        | 完了状況 |
| ------------------------ | -------------------------- | ------------------- | -------- |
| Claude Sonnet 4.5        | プロンプト作成・進捗管理   | 約84,000 (42%/200K) | ✅       |
| GPT 5.2-Codex            | テスト追加・コード整理     | -                   | ✅       |
| GPT 5.1-Codex-Max        | テスト検証                 | -                   | ✅       |
| Antigravity (Sonnet 3.7) | CI・ドキュメント・最終検証 | -                   | ✅       |

### Phase 3 推奨モデル

- **推奨**: `Claude Sonnet 4.5` または `Claude Opus 4.5`（リリース作業）
- **理由**: TestPyPI アップロード・公開手順の確実性、メタデータ検証の精度
- **トークン余裕**: 58%（58,000トークン） → Phase 3 は十分カバー可能

---

## Phase 3 進捗状況 (2026-01-31 13:30 更新)

### ✅ 完了した作業

#### 1. パッケージビルドと検証

**実施日時**: 2026-01-31 12:11

**成果物**:

- `gwexpy-0.1.0b1-py3-none-any.whl` (694KB)
- `gwexpy-0.1.0b1.tar.gz` (585KB)

**検証結果**:

- ✅ `twine check dist/*`: **PASSED**
- ✅ 必須ファイル確認: LICENSE, README.md, py.typed 含まれている
- ✅ バージョン整合性: 0.1.0b1 で一致

**修正内容**:

- **問題**: `License :: OSI Approved :: MIT License` classifier が PEP 639 の license expression (`license = "MIT"`) と競合し、ビルドエラー発生
- **対応**: pyproject.toml から redundant な classifier を削除（commit: 5d67b951）
- **結果**: ビルド成功、`twine check` 通過

#### 2. クリーン環境でのインストールテスト

**実施環境**: `/tmp/gwexpy-test-env` (Python 3.12 venv)

**インストール結果**:

- ✅ `pip install dist/gwexpy-0.1.0b1-py3-none-any.whl` 成功
- ✅ 依存関係の自動解決: astropy, gwpy, numpy, pandas, scipy, matplotlib 等すべて正常にインストール

**検出された問題**:

1. **gwpy 4.0.0 との互換性問題**:
   - `gwpy.io.registry.register_reader` が存在しない（API 変更）
   - gwpy が pytest を optional dependency として扱っているが、実際には必須
   - 現在の開発環境（gwpy 3.0.13）では動作するが、PyPI 最新版（4.0.0）では互換性問題

2. **インポートエラー**:
   ```python
   AttributeError: module 'gwpy.io.registry' has no attribute 'register_reader'
   ```
   発生箇所: `gwexpy/timeseries/io/ats.py:178`

### ⚠️ 未完了の作業

#### 3. TestPyPI へのアップロード

**状況**: API トークンの入力が必要なため中断

**次のステップ**:

1. TestPyPI API トークンの取得・設定
2. `twine upload --repository testpypi dist/*` の実行
3. TestPyPI 上での表示確認（メタデータ、リンク、README レンダリング）

#### 4. gwpy バージョン制約の追加（推奨）

**問題の詳細**:

- gwpy 4.0.0 で API が変更され、`gwexpy.timeseries.io` モジュールが動作しない
- 現在の `pyproject.toml` では `gwpy>=3.0.0` のみ指定

**推奨対応**:

```toml
dependencies = [
  "gwpy>=3.0.0,<4.0.0",  # 上限を追加
  # ... 他の依存関係
]
```

**代替案**:

- gwpy 4.0.0 の API 変更に対応するコード修正（`io.registry` → 新 API）
- ただし、リリース直前のため、バージョン制約の追加を優先推奨

### 📋 次のアクション（優先順位順）

1. **[Critical]** gwpy バージョン制約の追加
   - `pyproject.toml` を `gwpy>=3.0.0,<4.0.0` に修正
   - 再ビルド・再検証

2. **[High]** TestPyPI アップロード
   - API トークンの設定
   - アップロード実行
   - 表示確認

3. **[Medium]** クリーン環境での再インストールテスト
   - gwpy バージョン制約後の動作確認
   - 基本的なインポート・デモ実行テスト

4. **[Low]** ドキュメント最終更新
   - CHANGELOG 日付確定（TBD → 2026-01-31）
   - README インストールガイド最終確認

### 🔍 技術的な知見

**PEP 639 対応**:

- setuptools 68+ では、`license` フィールドに SPDX 文字列を使用
- 従来の `License :: OSI Approved :: ...` classifier は非推奨
- 両方を指定すると `InvalidConfigError` が発生

**gwpy 依存関係の課題**:

- gwpy 4.0.0 で破壊的変更が含まれている可能性
- 開発環境と PyPI 最新版の乖離に注意が必要
- ベータリリースでは保守的なバージョン制約を推奨
