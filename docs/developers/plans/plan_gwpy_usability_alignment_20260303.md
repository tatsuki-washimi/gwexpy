# 実施計画: gwpy互換の使い勝手整合（API UX Alignment）

**日付**: 2026年3月3日  
**担当**: Codex (AI Assistant)  
**ステータス**: 実施中 (Phase 1-3 完了、Phase 4 一部)

## Objectives & Goals

- `gwpy==4.0.x` の呼び出し習慣と矛盾しない API 体験を `gwexpy` で実現する。
- 「同名 API は同じ呼び方で動く」を原則にし、破壊的差分は明示的にドキュメント化する。
- 互換対象と非互換対象（gwexpy独自 API）を明確に分離し、利用者の混乱を防ぐ。

### 成功条件

- `gwpy` 互換対象 API の主要呼び出しパターンで回帰テストが全通過する。
- ドキュメントのシグネチャ・引数説明・チュートリアルが実装と一致する。
- 既知の齟齬（`transfer_function` 位置引数、`nproc/parallel` 記述揺れ等）が解消される。

## スコープ

### In Scope

- `TimeSeries` の `gwpy` 同名 API 互換性（特に位置引数・キーワード引数）。
- `TimeSeriesDict/List` での `gwpy` と誤認しやすい API の呼び方整理。
- `fftlength/overlap` と `nfft/noverlap` の受理ルール統一（実装・ドキュメント）。
- 互換方針を保証するテスト追加。

### Out of Scope

- `gwpy` に存在しない API を完全に同一化すること（例: `lock_in`, `TimeSeriesDict.csd`）。
- アルゴリズム性能改善のみを目的とした改修。
- 大規模リファクタ（モジュール分割や設計刷新）。

## 事前確認済みの主要差分（優先修正候補）

1. `TimeSeries.transfer_function`  
   `gwpy` の位置引数呼び出し（`other, fftlength, overlap`）に対し、`gwexpy` は `mode` が第2引数に来るため非互換。
2. `TimeSeriesDict/List.csd` / `coherence`  
   `fftlength` / `overlap` が keyword-only で、位置引数呼び出しが `TypeError` になる。
3. `nproc` と `parallel` の説明・実装不整合  
   API により受理引数が異なるため、ドキュメントと実装の整合が不足。
4. `nperseg` / `noverlap` / `nfft` / `fftlength` の案内揺れ  
   実装上受理するパターンと、ドキュメントの表現が一致していない箇所がある。

## Detailed Roadmap (by Phase)

### Phase 1: 互換ポリシーの固定（0.5日）

- 互換基準を `gwpy==4.0.1` に固定し、対象 API 一覧を確定する。
- 「同名 API は互換」「独自 API は明示」のルールを `docs/developers/compatibility/` に記録する。
- 判定表を作成する:
  - 完全互換
  - 互換ラッパで吸収
  - 非互換維持（理由を明記）

### Phase 2: API互換レイヤ実装（1.5日）

- `TimeSeries.transfer_function`:
  - `gwpy` 形式の位置引数を受理する互換パースを追加。
  - 既存の `mode=` 指定は維持し、曖昧ケースは明示エラーにする。
- `TimeSeriesDict/List.csd` / `coherence`:
  - `*args` 互換レイヤを追加し、`(other, fftlength, overlap)` を受理。
  - keyword と positional の混在は `TypeError` で明確化。
- `nproc/parallel`:
  - APIごとに受理方針を固定（互換 alias か、明示非互換かを統一）。
  - 非推奨扱いにする場合は警告メッセージを統一。

### Phase 3: ドキュメント整合（1.0日）

- `docs/web/en|ja/reference/` のシグネチャ記述を実装と一致させる。
- `gwpyユーザー向け` ガイドに互換差分と移行例を追記。
- `nproc/parallel`, `fftlength/overlap`, `nfft/noverlap` の受理ルールを一覧化。

### Phase 4: 回帰テストとリリース前確認（1.0日）

- 互換回帰テストを追加:
  - `transfer_function` の gwpy流位置引数
  - `TimeSeriesDict/List.csd/coherence` の位置引数・キーワード引数両系統
  - `nproc/parallel` の受理/拒否仕様
- 既存テスト群と合わせて実行し、失敗時は互換ポリシーに基づき修正。
- 変更点を `CHANGELOG` に整理（Breaking/Behavioral change を明記）。

## Testing & Verification Plan

### 追加テスト（必須）

- `tests/timeseries/test_transfer_function_compat.py`
  - `gwpy` 互換呼び出し（positional）と `gwexpy` 拡張呼び出し（mode 指定）の両方を検証。
- `tests/timeseries/test_collections_spectral_compat.py`
  - `TimeSeriesDict/List.csd/coherence` の引数受理パターンとエラーメッセージを検証。
- `tests/timeseries/test_fft_param_compat.py`
  - `fftlength/overlap` と `nfft/noverlap` の受理ルール、混在時エラーを検証。

### 実行コマンド（想定）

```bash
pytest tests/timeseries/test_transfer_function_compat.py -q
pytest tests/timeseries/test_collections_spectral_compat.py -q
pytest tests/timeseries/test_fft_param_compat.py -q
pytest tests/timeseries -q
```

## Models, Recommended Skills, and Effort Estimates

### 推奨モデル

- 実装・テスト作成: `GPT-5 Codex`
- 互換仕様レビュー（独立観点）: `Claude Opus` または `code-reviewer-pro` 相当のレビュー手順

### 推奨スキル

- `setup_plan`（計画更新）
- `run_tests`（回帰検証）
- `lint_check`（静的チェック）
- `code-reviewer-pro`（互換性観点レビュー）

### 工数見積

- Phase 1: 0.5日
- Phase 2: 1.5日
- Phase 3: 1.0日
- Phase 4: 1.0日
- 合計: **4.0日**（レビュー往復を含めると 4.5日）

## リスクと対策

- リスク: 互換レイヤ追加で曖昧な引数解釈が増える。  
  対策: 混在ケースをすべて `TypeError` で明示し、テストで固定する。
- リスク: ドキュメント更新漏れ。  
  対策: API差分一覧を単一表に集約し、参照元を一本化する。
- リスク: `gwpy` 側仕様変更への追従遅れ。  
  対策: 互換基準バージョン（4.0.1）を明記し、将来更新時に差分比較を再実施する。

## 完了定義 (Definition of Done)

- 互換対象 API のテストがすべて通る。
- 参照ドキュメントと実装シグネチャの差分がゼロ。
- 互換方針と例外（非互換維持 API）が開発者向け文書に明示されている。

## 実施ログ (2026-03-03)

### 完了

- `TimeSeries.transfer_function` を gwpy 互換順（`other, fftlength, overlap, window, average`）で受理するよう更新。
- `TimeSeriesDict/List` の `csd` / `coherence` / `csd_matrix` / `coherence_matrix` に位置引数互換レイヤを追加。
- 互換ポリシー文書を追加:
  - `docs/developers/compatibility/gwpy/API_UX_POLICY_20260303.md`
- 互換テストを追加:
  - `tests/timeseries/test_transfer_function_compat.py`
  - `tests/timeseries/test_collections_spectral_compat.py`
  - `tests/timeseries/test_fft_param_compat.py`
- 参照ドキュメント更新:
  - `docs/web/en/reference/TimeSeriesDict.md`
  - `docs/web/en/reference/TimeSeriesList.md`

### 未完了 / 制約

- `pytest` 実行は環境依存で失敗（`ws-base` 環境の `astropy`/`numpy` 不整合、`main` 環境は sandbox 下でパッケージ追加不可）。
- 代替として、`python` 実行スクリプトで以下を確認済み:
  - `transfer_function` の gwpy 位置引数互換
  - collections の位置引数互換と混在時 `TypeError`
  - `asd` の `nfft/noverlap` 受理と `nperseg` エラー

---

_本計画書は、2026年3月3日時点の実装照合結果に基づいて作成。_
