# HDF5 exact-epoch P0 仕様レビュー

## 判定

**判定：未承認。**

Critical は0件、Important は3件、Minor は0件である。

Critical finding はない。

Important finding が残るため、現時点の実装を P0 仕様適合として承認できない。

レビュー対象は、指定された設計仕様、canonical plan、`origin/main...98028bcc0` の focused implementation diff、Task 10 の指定ログに限定した。

## Findings

### I-1：pathname の非 exact external overwrite が marked target を拒否しない

- **Severity**：Important
- **仕様節**：`Write metadata policy` の「External storage retains the existing conservative rule: it cannot replace a sidecar-managed or canonically marked dataset」および `Test-driven implementation order / 1. Dataset marker and authority` の external storage 拒否要件。
- **file:line**：`gwexpy/timeseries/io/hdf5.py:1270`。pathname の `append=False, overwrite=True` 分岐は既存 v2 文書を構文検証するだけで、`_reject_stale_external_sidecar` を呼ばない。既存テストも `tests/timeseries/test_hdf5_exact_t0.py:1353` で `append=True` の場合しか検証していない。
- **observable impact**：exact marker と v2 sidecar を持つ既存 pathname に対し、非 exact 配列を `external=..., overwrite=True, append=False` で書くと、拒否されずに HDF5 pathname と external raw file の両方が更新される。隔離一時ディレクトリで再現したところ、例外は発生せず、`target_changed=True` かつ `raw_changed=True` になった。
- **RED-first test node**：`tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_path_external_overwrite_without_append_rejects_marked_target` を追加する。marked target と raw file のバイト列、native writer 呼び出し数を保存し、`ValueError`、両ファイル不変、呼び出し0回を要求する。
- **最小修正案**：`_preflight_native_external_write` の既存 pathname が HDF5 である分岐で、`append` の値にかかわらず requested object に対して `_reject_stale_external_sidecar` を実行する。`overwrite=True, append=False` で既存ファイル全体を置換する場合は、置換で失われる marked または sidecar-managed dataset がないことも native writer の前に確認する。

### I-2：disposable stage が ExternalLink／SoftLink の leaf を通常 dataset に置換する

- **Severity**：Important
- **仕様節**：`Native path` の「Write resolution rejects an ExternalLink at the leaf」「It also rejects a SoftLink at the leaf」および `Transaction architecture / Disposable stage writer`。
- **file:line**：`gwexpy/timeseries/io/hdf5.py:760`。`_write_disposable_stage` は private parent と external ancestor を検査した直後に native writer を呼ぶが、leaf の link type を検査しない。leaf を拒否できる `_existing_dataset` は `gwexpy/timeseries/io/hdf5.py:1047` の caller-owned open-container 経路でしか使われない。既存の `test_hdf5_link_write_policy` も `tests/timeseries/test_hdf5_exact_t0.py:4378` で `File` と `Group` だけを対象にしている。
- **observable impact**：既存 ExternalLink leaf または SoftLink leaf を指定して pathname／file-like に `append=True, overwrite=True` で書くと、要求された `ValueError` が発生せず、referring file の leaf が HardLink dataset に変わる。pathname と file-like の ExternalLink leaf を隔離環境で再現し、どちらも例外なし、target bytes 更新、leaf type が `ExternalLink` から `HardLink` へ変更されることを確認した。external file 自体は変更されなかったが、リンクの型と参照先を保存するという仕様には違反する。
- **RED-first test node**：`tests/timeseries/test_hdf5_exact_t0.py::test_hdf5_disposable_target_rejects_leaf_links_before_native_write` を pathname／file-like と ExternalLink／SoftLink の直積で追加する。例外、native writer 0回、target bytes 不変、link type／target 不変、external file 不変を検証する。
- **最小修正案**：`_write_disposable_stage` で stage を開いた後、native writer より前に transaction coordinate の leaf を `getlink=True` で検査する。既存の `_existing_dataset` を再利用する場合は、relative／absolute path と Group 基準の座標を保ったまま SoftLink／ExternalLink leaf の拒否だけを共通化する。

### I-3：recovery group creation が作成後に失敗すると partial private group を追跡できない

- **Severity**：Important
- **仕様節**：`Caller-owned open-container transaction` の「Creation of the group ... belongs to the transaction envelope」「A failure before the recovery artifact is complete ... removes the partial private group」「If that cleanup also fails, the structured error reports state=\"old\" and the partial recovery path」。
- **file:line**：`gwexpy/timeseries/io/hdf5.py:878`。`_create_handle_recovery_group` の呼び出しが `try` より前にあり、group path と handle は呼び出しが正常に返った後でしか `_HandleRecovery` に保存されない。既存の group-create failure row は `tests/timeseries/test_hdf5_exact_t0_transactions.py:136` で作成前にただ例外を送出するため、create-after-mutation failure を検証していない。
- **observable impact**：recovery group を実際に作成してから creation seam が例外を送出するよう注入すると、公開 dataset は old のままだが、`/__gwexpy_t0_rollback_*` が残り、通常の `RuntimeError` に `state` と `recovery_path` が付かない。隔離 HDF5 file で `state=None`、`recovery_path=None`、private group 1個残存を再現した。これは partial setup cleanup と structured observable state の両契約を破る。
- **RED-first test node**：既存の `tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_open_recovery_setup_failure_preserves_public_state[group-create]` を、実際の creator で group を作成した後に例外を送出する注入へ変更する。cleanup 成功時は private group 0個と original error の再送出を、cleanup も失敗する行では `_RollbackError(state="old", recovery_path=<残存group>)` を要求する。
- **最小修正案**：rollback 名を group creation 前に確定して transaction envelope に保持し、その path を使って creation 自体を `try` 内で実行する。create call が例外を送出しても path の存在を検査して削除できるようにし、削除失敗時はその path を持つ `_RollbackError(state="old")` を構築する。

## 確認した仕様対応

focused diff では、dataset-local v2 marker を唯一の exact authority とし、v2 sidecar の path を診断専用にする読取方針を確認した。

hard-link alias、copy lineage、resolved external file authority、whole-document v2 validation、v1 非 authority、native GWpy reader 経由の view 変換に対応する実装とテストを確認した。

pathname の same-directory replacement、file-like の disk-backed backup／working image と fixed-chunk copy、open handle の identity-preserving recovery link、`_RollbackError` の state fields、native writer 最大1回の既存テストを確認した。

ただし、上記3件の未検証分岐で仕様違反を再現したため、これらの確認を P0 全体の pass とは扱わない。

## ログと read-only 検証

- focused pytest log：769 passed。skip と xfail は記録されていない。
- surrounding pytest log：2784 passed、72 skipped、3 xfailed。skip と xfail は pass に算入していない。
- repository Ruff log：既知で対象外の `docs_redesign/conf.py:242` D103 の1件だけを記録している。
- source/test Ruff log：All checks passed。
- MyPy log：396 source files で issue なし。
- `rtk git diff origin/main...98028bcc0 --check`：問題なし。
- conda `gwexpy` の隔離一時ディレクトリで、I-1、pathname／file-like の I-2、I-3 の failure injection を再現した。リポジトリ内の実装、テスト、計画は変更していない。

## 変更ファイル

- `docs/developers/reviews/review_HDF5P0Specification_20260830_171632.md`
