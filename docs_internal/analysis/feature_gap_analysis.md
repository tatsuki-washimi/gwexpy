# gwexpy プロジェクト監査・分析統合レポート

最終更新日: 2026-04-02
ステータス: **継続的改善フェーズ（OSS標準化・信頼性向上）**

## 1. エグゼクティブサマリー

`gwexpy` は、研究用ソフトウェアとしては非常に高い完成度（ドキュメント・CI・再現性の骨格）を備えていますが、外部コントリビュータや第三者ユーザーが安心して関与するための「OSS運用の標準インフラ」が依然として薄い状態にあります。

2026-04-02 時点の深掘り監査により、P0（セキュリティ）と P1（外部貢献・マルチ環境）の主要施策の実装が完了しましたが、引き続き **「運用品質（リリースメタデータ同期・再現性・GOVERNANCE 等）」** に関して未対応または追加確認が必要な重要項目が残っています。

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
- **[DONE] mypy の CI 可視化 (P1)**: `mypy` ジョブを CI に統合。現在は `continue-on-error` による可視化フェーズです。

---

## 3. OSS運用インフラ監査テーブル（現状スナップショット）

学術ソフトウェアとしての第三者検証・引用・継続保守のコストを下げるための「運用上の信頼性」に関する課題です。

| 観点 | 現状とギャップ | リスク/影響 | 推奨アクション |
| :--- | :--- | :--- | :--- |
| **セキュリティ** | **[DONE]** `SECURITY.md` および脆弱性報告導線、`security.yml` による自動スキャンを導入済み。 | 脆弱性報告の漏れやサプライチェーンリスクの検知遅れ。 | 導入済み（継続的な監視とアラート対応）。 |
| **Dependabot** | `.github/dependabot.yml` の有無を確認・整備中。脆弱性 PR の triage ルールが未明確。 | 自動 PR が来ても誰もレビューしない状態になる。 | `dependabot.yml` を追加し、`SECURITY.md` に対応フローを明記。 |
| **品質ゲート** | **[DONE]** `mypy` ジョブを CI に統合。現在は `continue-on-error` で可視化フェーズ（23件のエラー）。 | 型の退行が検知されず、メンテナンスコストが増大する。 | 導入済み（今後、エラー修正を進め強制化へ）。 |
| **マルチOS保証** | **[DONE]** Windows/macOS での `smoke-test` ジョブを CI に追加。 | ユーザー層が限定され、環境依存バグの発見が遅れる。 | 導入済み（主要なクラスのインポート・動作を確認）。 |
| **コミュニティ** | **[DONE]** `CODE_OF_CONDUCT.md`, Issue/PR テンプレート, `pre-commit` を整備。 | 外部協力の"入口摩擦"が増え、貢献の品質が安定しない。 | 導入済み（外部貢献への準備完了）。 |
| **GOVERNANCE** | `MAINTAINERS.md`, `Support policy`, 公開 `ROADMAP.md` が未整備。 | 外部貢献者が責務や期待値を把握しにくい。長期保守への信用低下。 | 各ドキュメントを作成し、Issue→Milestone を紐付ける。 |
| **メタデータ同期** | `_version.py` は 0.1.1 だが `CITATION.cff`/`.zenodo.json` が 0.1.0 のまま。CI での自動チェックがない。 | DOI/CITATION と PyPI リリースが一致せず、引用者が混乱する。 | `release.yml` にバージョン一致確認ジョブを追加。 |
| **再現性/配布** | テスト用 fixture が git 外にあり、新規参加者がローカルでテストを再現しづらい。 | 外部コントリビュータがデバッグ・PR 作成を断念する。 | 小さな fixture を `tests/fixtures/` に同梱（checksum 付き）。 |
| **依存 lock** | `pyproject.toml` は整備済みだが、lockfile による CI 再現性の担保が薄い。 | 将来の依存更新で CI やユーザー環境が壊れる。 | `pip-compile` 等による `requirements-dev.lock` を導入。 |
| **ドキュメントビルド** | `autodoc_mock_imports` と `nitpick_ignore` が肥大。`nbsphinx` 実行の非決定性あり。 | API 自動抽出の欠落、リンク切れの見逃し。 | mock 精査、ノートブック実行を CI 限定化。 |
| **副作用設計** | `import gwexpy` 時に副作用（`enable_series_fit()` 等）が発生する設計の opt-out が弱い。 | ユーザーが予期しない挙動に遭遇する。 | opt-in/out の API 規約を `CONTRIBUTING.md` に明示し、テストでカバー。 |

---

## 4. 優先度付き改善ロードマップ (2026-Q2)

### ✅ P0: 完了（信頼の基盤）

1. **[DONE]** セキュリティポリシーの策定 (`SECURITY.md`)。
2. **[DONE]** 脆弱性スキャンの自動化 (`pip-audit`, `bandit`, `CodeQL`)。

### ✅ P1: 完了（外部貢献とマルチ環境対応）

1. **[DONE]** 行動規範 (CoC) とテンプレートの導入。
2. **[DONE]** マルチOS Smoke Test の CI 追加。
3. **[DONE]** mypy CI 可視化ジョブの統合。
4. **[DONE]** `pre-commit` 設定の導入。

### 📌 P2: 透明性と同期の自動化（次期対応）

1. **リリースメタデータ同期チェック**: `_version.py` と `CITATION.cff`/`.zenodo.json`/`CHANGELOG.md` の一致を `release.yml` で検証・ブロック。
2. **MAINTAINERS.md + Support policy + 公開 ROADMAP**: 外部貢献の信頼性向上のため作成。
3. **Dependabot 設定と運用フロー**: `.github/dependabot.yml` を追加し、脆弱性 PR の triage ルールを明文化。

### 📌 P3: 再現性と開発環境の堅牢化

1. **依存 lock の導入**: `pip-compile` 等で `requirements-dev.lock` を整備し、CI の再現性を担保。
2. **テスト fixture の同梱**: 小さなサンプルデータを `tests/fixtures/` に同梱（checksum 付き）。
3. **mypy の fail-on-error 化**: 現在の 23 件のエラーを解消し、`continue-on-error` を除去して強制化。

### 📌 P4: ドキュメントと設計の精緻化

1. **Sphinx の mock/nitpick 整理**: `autodoc_mock_imports` と `nitpick_ignore` を精査・削減。
2. **`nbsphinx` 実行ポリシー見直し**: CIでの限定実行化（非決定性の排除）。
3. **import 副作用の opt-out 整備**: opt-in/out API 規約のドキュメント化とテストカバレッジの追加。

---

## 5. リリース前チェックリスト

| 項目 | 状態 |
| :--- | :--- |
| `release.yml` に `_version.py` と `CITATION.cff`/`.zenodo.json`/`CHANGELOG.md` の一致確認ジョブがあるか | ⬜ 未対応 |
| `MAINTAINERS.md` と簡易 `Support policy` がリポジトリルートにあるか | ⬜ 未対応 |
| `requirements-dev.lock` または `constraints/` を使って CI の再現性を担保しているか | ⬜ 未対応 |
| テストが小さな fixture でローカル再現可能か（CI 上でノートブック実行が決定性を持つか） | ⬜ 未対応 |
| `mypy` の fail-on-error 化計画がスケジュールされているか（段階的適用） | ⬜ 計画中（現在 23 件のエラーを許容中） |
| `dependabot.yml` と脆弱性対応フローが `SECURITY.md`/運用ドキュメントに記載されているか | ⬜ 未対応 |
| PyPI 上の公開版バージョン（0.1.1）が実際に公開済みか（リモート確認要） | ❓ 要確認 |
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

> [!WARNING]
> **メタデータ同期のリスク（高優先）**: `_version.py` は 0.1.1 を示していますが、`CITATION.cff` や `.zenodo.json` が 0.1.0 のまま残っています。次のリリース前に必ず手動で同期し、P2 で CI 自動化を実施してください。

---

*本レポートは、学術ソフトウェアとしての信頼性を高め、長期的な保守を可能にするための「生きた文書」として、タスクの完了に合わせて更新される。*
