# gwexpy プロジェクト監査・分析統合レポート

最終更新日: 2026-04-03
ステータス: **P3・P4段階へ移行（再現性・ドキュメント精緻化）**

## 1. エグゼクティブサマリー

`gwexpy` は、研究用ソフトウェアとしては非常に高い完成度（ドキュメント・CI・再現性の骨格）を備えていますが、外部コントリビュータや第三者ユーザーが安心して関与するための「OSS運用の標準インフラ」が依然として薄い状態にあります。

2026-04-03 時点の追跡監査により、P0（セキュリティ）・P1（外部貢献・マルチ環境）・P2（透明性と同期の自動化）の施策が全て完了しました。引き続き **「再現性・ドキュメント精緻化（P3・P4）」** に関して実装が必要な項目が残っています。

本レポートは、監査から得られた確証をもとにした、ライブなロードマップです。

---

## 2. 解決済み事項の整理

以前の監査（2026-04-01〜2026-04-02）で指摘された重大な課題のうち、以下の項目はすでに解消されています。

- **[DONE] バージョン情報の単一ソース化**: `pyproject.toml` に `dynamic = ["version"]` を導入し、`_version.py` を唯一の正解としました。
- **[DONE] ワイルドカードインポートの排除**: `gwexpy/io/` 内での `from gwpy.io.* import *` を、明示的な import に置き換えました。
- **[DONE] 広範な例外捕捉の具体化**: I/O 層を中心に、`except Exception:` を具体的な例外（`OSError`, `ValueError` 等）へ修正しました。
- **[DONE] 欠落していたチュートリアル作成**: `Histogram`, `Table`, `Noise`, `Fitting` 等の主要チュートリアルを `docs/` に追加しました。
- **[DONE] セキュリティポリシーの設定 (P0)**: `SECURITY.md` を作成し、脆弱性報告手順と pickle に関する警告を明記しました。
- **[DONE] セキュリティスキャンの自動化 (P0)**: `.github/workflows/security.yml` を導入し、`pip-audit`, `bandit`, `CodeQL` による自動スキャン（PR毎/週次）を構築しました。
- **[DONE] コミュニティ基盤の整備 (P1)**: `CODE_OF_CONDUCT.md` (English), Issue/PR テンプレート, `.pre-commit-config.yaml` を追加しました。
- **[DONE] マルチOS Smoke Test (P1)**: Windows/macOS 上での `TimeSeries`, `FrequencySeries`, `Spectrogram` の動作確認を CI に追加しました。
- **[DONE] mypy の CI 統合と fail-on-error 化 (P1)**: `mypy` ジョブを CI に統合し、fail-on-error（`continue-on-error` なし）で型チェックを強制。
- **[DONE] テスト再現性の完全確保 (P3)**: 75種類以上のフォーマットを網羅する自動フィクスチャ生成基盤 (`generate_fixtures.py`) を導入し、バイナリ依存のない 100% 網羅テストを構築しました。
- **[DONE] リリースメタデータ同期 (P2)**: `_version.py` (0.1.1)、`CITATION.cff` (0.1.1)、`.zenodo.json` (0.1.1) がバージョン統一済み。
- **[DONE] GOVERNANCE 文書整備 (P2)**: `MAINTAINERS.md` を作成し、保守責任と引き継ぎ手順を明記しました。
- **[DONE] Dependabot 設定と運用フロー (P2)**: `.github/dependabot.yml` を追加し、脆弱性 PR の triage ルールを `SECURITY.md` に明文化しました。
- **[DONE] 依存 lock の導入 (P3)**: `pip-compile --extra=dev` による `requirements-dev.txt`（296行、全バージョン固定）を整備。`ci.yml` も lock ファイル参照に統一し、CI の再現性を担保しました。

---

## 3. OSS運用インフラ監査テーブル（現状スナップショット）

学術ソフトウェアとしての第三者検証・引用・継続保守のコストを下げるための「運用上の信頼性」に関する課題です。

| 観点 | 現状とギャップ | リスク/影響 | 推奨アクション |
| :--- | :--- | :--- | :--- |
| **セキュリティ** | **[DONE]** `SECURITY.md` および脆弱性報告導線、`security.yml` による自動スキャンを導入済み。 | 脆弱性報告の漏れやサプライチェーンリスクの検知遅れ。 | 導入済み（継続的な監視とアラート対応）。 |
| **Dependabot** | **[DONE]** `.github/dependabot.yml` を追加済み。脆弱性 PR の triage ルールを `SECURITY.md` に明文化。 | 自動 PR が来ても誰もレビューしない状態になる。 | 導入済み（運用フロー確立）。 |
| **品質ゲート** | **[DONE]** `mypy` ジョブを CI に統合し fail-on-error 化。型チェック強制中。 | 型の退行が検知されず、メンテナンスコストが増大する。 | 導入済み（引き続きエラー数削減と型安全性向上）。 |
| **マルチOS保証** | **[DONE]** Windows/macOS での `smoke-test` ジョブを CI に追加。 | ユーザー層が限定され、環境依存バグの発見が遅れる。 | 導入済み（主要なクラスのインポート・動作を確認）。 |
| **コミュニティ** | **[DONE]** `CODE_OF_CONDUCT.md`, Issue/PR テンプレート, `pre-commit` を整備。 | 外部協力の"入口摩擦"が増え、貢献の品質が安定しない。 | 導入済み（外部貢献への準備完了）。 |
| **GOVERNANCE** | **[DONE]** `MAINTAINERS.md` を作成済み。保守責任と期待値を明記。 | 外部貢献者が責務や期待値を把握しにくい。長期保守への信用低下。 | 導入済み（保守体制確立）。 |
| **メタデータ同期** | **[DONE]** `_version.py` (0.1.1)、`CITATION.cff` (0.1.1)、`.zenodo.json` (0.1.1) がバージョン統一済み。 | DOI/CITATION と PyPI リリースが一致せず、引用者が混乱する。 | 解決済み（CI での自動チェック検討中）。 |
| **再現性/配布** | **[DONE]** 75+ フォーマットを網羅する動的生成基盤 (`generate_fixtures.py`) を導入。 | 外部コントリビュータがデバッグ・PR 作成を断念する。 | 導入済み（100% 網羅達成）。 |
| **依存 lock** | **[DONE]** `requirements-dev.txt` を `pip-compile --extra=dev` で生成済み（296行、全バージョン固定）。`ci.yml` も lock ファイル参照に統一済み。 | 将来の依存更新で CI やユーザー環境が壊れる。 | 導入済み（`requirements-dev.txt` で CI 再現性を担保）。 |
| **ドキュメントビルド** | `autodoc_mock_imports` (37件)、`nitpick_ignore` (~130件) が肥大。`nbsphinx` 実行の非決定性あり。 | API 自動抽出の欠落、リンク切れの見逃し。 | mock 精査・削減、ノートブック実行を CI 限定化（**P4**）。 |
| **副作用設計** | `import gwexpy` 時に副作用（`enable_series_fit()`, `register_all()` 等）が発生する設計の opt-out が弱い。 | ユーザーが予期しない挙動に遭遇する。 | opt-in/out の API 規約を `CONTRIBUTING.md` に明示し、テストでカバー（**P4**）。 |

---

## 4. 優先度付き改善ロードマップ (2026-Q2)

### ✅ P0: 完了（信頼の基盤）

1. **[DONE]** セキュリティポリシーの策定 (`SECURITY.md`)。
2. **[DONE]** 脆弱性スキャンの自動化 (`pip-audit`, `bandit`, `CodeQL`)。

### ✅ P1: 完了（外部貢献とマルチ環境対応）

1. **[DONE]** 行動規範 (CoC) とテンプレートの導入。
2. **[DONE]** マルチOS Smoke Test の CI 追加。
3. **[DONE]** mypy CI 統合と fail-on-error 化。
4. **[DONE]** `pre-commit` 設定の導入。

### ✅ P2: 完了（透明性と同期の自動化）

1. **[DONE]** リリースメタデータ同期: `_version.py` (0.1.1)、`CITATION.cff` (0.1.1)、`.zenodo.json` (0.1.1) 統一済み。
2. **[DONE]** GOVERNANCE 文書: `MAINTAINERS.md` 作成済み。
3. **[DONE]** Dependabot 設定と運用フロー: `.github/dependabot.yml` 追加済み、`SECURITY.md` に triage ルール明文化。

### ✅ P3: 再現性と開発環境の堅牢化

1. **[DONE]** 依存 lock の導入: `requirements-dev.txt`（pip-compile 生成、296行）で CI 再現性を担保。`ci.yml` も lock ファイル参照に統一済み。
2. **[DONE]** テスト fixture の同期と自動生成基盤の導入。
3. **[DONE]** mypy の fail-on-error 化: CI に `continue-on-error` なしで統合済み。

### 📌 P4: ドキュメントと設計の精緻化

1. **Sphinx の mock/nitpick 整理**: `autodoc_mock_imports` と `nitpick_ignore` を精査・削減。
2. **`nbsphinx` 実行ポリシー見直し**: CIでの限定実行化（非決定性の排除）。
3. **import 副作用の opt-out 整備**: opt-in/out API 規約のドキュメント化とテストカバレッジの追加。

---

## 5. リリース前チェックリスト

| 項目 | 状態 |
| :--- | :--- |
| バージョン情報が統一されているか（`_version.py`/`CITATION.cff`/`.zenodo.json`） | ✅ 対応済み（0.1.1 統一） |
| `MAINTAINERS.md` がリポジトリルートにあるか | ✅ 対応済み |
| `requirements-dev.lock` または `constraints/` を使って CI の再現性を担保しているか | ✅ 対応済み（`requirements-dev.txt` で全バージョン固定、`ci.yml` も参照済み） |
| テストが 75+ フォーマットの自動生成によりローカル再現可能か | ✅ 対応済み |
| `mypy` が fail-on-error で強制されているか | ✅ 対応済み（`continue-on-error` なし） |
| `dependabot.yml` と脆弱性対応フローが `SECURITY.md`/運用ドキュメントに記載されているか | ✅ 対応済み |
| PyPI 上の公開版バージョン（0.1.1）が実際に公開済みか | ❓ 要確認 |
| `SECURITY.md` の連絡先メールアドレスに実際の宛先が設定されているか | ⬜ 要確認・更新 |

---

## 6. 検証計画

各フェーズ完了時に、機能的な回帰テストに加え、新設した「運用ゲート」の確認を行います。

1. **運用ゲート確認**:
   - `pip-audit` がクリーンであること。
   - PR テンプレートが GitHub 上で正しく表示されること。
   - `mypy` ジョブがパスすること（将来：fail-on-error）。
2. **マルチOS検証**:
   - GitHub Actions 上で Windows/macOS ジョブが `PASS` すること。
3. **ドキュメント整合性**:
   - `docs/` 内の最新の機能（Histogram, Table等）が API Reference に正しく反映されていること。

---

## 6. 次フェーズ（P4）の優先度目安

| 優先度 | 項目 | 影響度 |
| :--- | :--- | :--- |
| **P4** | Sphinx の mock/nitpick 削減 | 低（ドキュメント品質向上） |
| **P4** | import 副作用の opt-out 整備 | 低（ユーザー体験向上） |

---

*本レポートは、学術ソフトウェアとしての信頼性を高め、長期的な保守を可能にするための「生きた文書」として、タスクの完了に合わせて更新される。最終更新: 2026-04-03（P0/P1/P2/P3 完了、P4 進行中）。*
