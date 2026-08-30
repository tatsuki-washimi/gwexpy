# HDF5 P0 Code-Quality Review Prompt

### Handover Instructions for the Next Model

## Role

HDF5 exact-epoch P0 実装の independent code-quality reviewer として作業する。
Specification reviewer とは別の fresh session を使用する。
この session では実装、テスト、計画を変更せず、レビュー結果だけを作成する。

## Start Gate

最新の specification review report を先に読む。
未解決の Critical または Important finding が一件でもある場合は code-quality review を開始せず、その条件を報告して停止する。
Specification finding の修正 commit がある場合は、その commit を含む最新 `HEAD` を reviewed head として記録する。

## Reference Materials

- Specification: `docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md`
- Canonical plan: `docs/superpowers/plans/2026-08-29-hdf5-exact-epoch-identity.md`
- Latest specification review: `docs/developers/reviews/review_HDF5P0Specification_<TIMESTAMP>.md`
- Focused implementation files:
  - `gwexpy/timeseries/io/hdf5.py`
  - `gwexpy/timeseries/io/_hdf5_exact_epoch.py`
  - `tests/timeseries/test_hdf5_exact_epoch_codec.py`
  - `tests/timeseries/test_hdf5_exact_t0.py`
  - `tests/timeseries/test_hdf5_exact_t0_transactions.py`
- Local logs:
  - `.agent/tmp/gwexpy_conda_jobs/gwexpy-pytest-focused-task10-20260830.log`
  - `.agent/tmp/gwexpy_conda_jobs/gwexpy-pytest-task10-20260830.log`
  - `.agent/tmp/gwexpy_conda_jobs/gwexpy-ruff-task10-20260830.log`
  - `.agent/tmp/gwexpy_conda_jobs/gwexpy-ruff-source-task10-20260830.log`
  - `.agent/tmp/gwexpy_conda_jobs/gwexpy-mypy-task10-20260830.log`

Use `rtk` for every shell invocation.
Use conda environment `gwexpy` for Python tooling.

## Review Scope

次の項目を specification compliance とは別に確認する。

- exception propagation と `_RollbackError` state classification の一貫性。
- setup、commit、rollback、cleanup の各 failure boundary と error aggregation。
- unlinked HDF5 ID、file-like temporary、pathname stage の resource lifecycle。
- private helper の責務、命名、型安全性、重複、到達不能コード。
- adversarial test double が production failure を正しく模擬していること。
- RSS、growth、call-count test の閾値と偽陽性、偽陰性の可能性。
- object reference、region reference、dimension scale、alias test の identity assertion。
- `requires-python >=3.11` と MyPy assumptions の整合性。
- 変更範囲が private API に限定され、公開 API drift がないこと。

`docs_redesign/conf.py:242` D103 は既知の未変更 finding であり、このレビューの対象外とする。

## Procedure

1. `rtk git status --short --branch` と reviewed head を記録する。
2. `origin/main...HEAD` の focused implementation diff を読む。
3. production helper と対応 test を対で確認する。
4. failure injection が intended branch を通ることを確認する。
5. finding を Critical、Important、Minor に分類する。

## Output Contract

各 finding には severity、`file:line`、failure scenario、impact、RED-first test、最小修正案を記載する。
Critical または Important finding がない場合は、その事実を明記する。
確認できない事項を pass と推定しない。
レビュー結果以外のファイルを変更せず、commit、push、tag、release、upload、workflow dispatch を行わない。
レビュー結果を `docs/developers/reviews/review_HDF5P0CodeQuality_<TIMESTAMP>.md` に保存して停止する。
