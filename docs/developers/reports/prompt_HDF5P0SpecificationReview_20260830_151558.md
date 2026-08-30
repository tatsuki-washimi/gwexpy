# HDF5 P0 Specification Review Prompt

### Handover Instructions for the Next Model

## Role

HDF5 exact-epoch P0 実装の independent specification reviewer として作業する。
この session では実装、テスト、計画を変更せず、レビュー結果だけを作成する。

## Preconditions

- Worktree: `/home/washimi/.config/superpowers/worktrees/gwexpy/v020-post-release-qualification`
- Branch: `test/v020-post-release-qualification`
- Reviewed implementation head: `98028bcc0`
- Implementation comparison: `origin/main...98028bcc0`
- Tasks 1–9 と Task 10 local gates は完了している。
- P0 は未承認である。
- P1 bootstrap と公開操作は対象外である。

## Reference Materials

次の資料と implementation diff だけをレビュー根拠として使用する。

- Specification: `docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md`
- Canonical plan: `docs/superpowers/plans/2026-08-29-hdf5-exact-epoch-identity.md`
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

次の specification requirements と実装の対応を確認する。

- dataset-local marker と v2 sidecar の authority policy。
- aliases、moves、copies、external links、path validation の意味論。
- native writer が pathname、file-like、File、Group の成功と post-write failure で最大一回であること。
- pathname stage の同一ディレクトリ atomic replacement と拒否条件。
- file-like backup、working image、fixed-chunk copy、byte/position state、cleanup warning の境界。
- caller-owned handle の verified recovery artifact、delete-before/after-raise、durable recreation、sidecar restoration、object identity preservation。
- `_RollbackError` の `state`、`recovery_path`、`byte_state`、`position_state` が specification の observable state と一致すること。
- success と incomplete rollback で private recovery object 数の上限が守られること。
- acceptance tests が specification の主張を実際に検証していること。

`docs_redesign/conf.py:242` D103 は既知の未変更 finding であり、このレビューの対象外とする。

## Procedure

1. `rtk git status --short --branch` で対象 branch と clean state を確認する。
2. `rtk git diff origin/main...98028bcc0 -- <focused files>` で implementation diff を読む。
3. Specification の各 requirement を implementation と test node に対応付ける。
4. ログに記録された pass、skip、xfail を区別して確認する。
5. finding を Critical、Important、Minor に分類する。

## Output Contract

各 finding には次を記載する。

- Severity: Critical、Important、Minor のいずれか。
- 根拠となる specification の節。
- 実装またはテストの `file:line`。
- observable impact。
- RED-first で追加または変更すべき test node。
- 最小修正案。

Critical または Important finding がない場合は、その事実を明記する。
確認できない事項を pass と推定しない。
レビュー結果以外のファイルを変更せず、commit、push、tag、release、upload、workflow dispatch を行わない。
レビュー結果を `docs/developers/reviews/review_HDF5P0Specification_<TIMESTAMP>.md` に保存して停止する。
