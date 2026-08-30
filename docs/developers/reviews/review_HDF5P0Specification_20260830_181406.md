# HDF5 exact-epoch P0 specification review

## Review metadata

- Review type: independent specification review, final round 3
- Branch: `test/v020-post-release-qualification`
- Reviewed implementation head: `f72ee74a9338`
- Comparison: `origin/main...f72ee74a9`
- Specification: `docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md`
- Canonical plan: `docs/superpowers/plans/2026-08-29-hdf5-exact-epoch-identity.md`
- Review start state: clean (`## test/v020-post-release-qualification...origin/main [ahead 57]`)
- Scope restriction: specification、canonical plan、focused implementation diff、指定された fresh local logs のみを根拠にした。
- Excluded evidence: `docs/developers/reviews/` の既存レビュー、`docs/developers/reports/`、会話ログは参照していない。

## Verdict

**P0 specification verdict: changes required.**

Critical は 0 件、Important は 2 件、Minor は 0 件である。

2 件とも、正常系の exact-epoch authority や旧データの保持そのものではなく、transaction setup で作成した一時ファイルの cleanup failure を構造化して報告する契約に反する。

したがって、現 head を P0 specification compliant と判定できない。

本レビューは人間による最終 P0 承認を行わない。

## Findings

### Important 1: file-like backup の setup cleanup failure が元例外と保持パスを失う

**仕様根拠**

- Specification「File-like transaction」では、完全な byte rollback と position rollback の後に temporary cleanup が失敗した場合、`_RollbackError` に cleanup failure を追加し、`state="old"` と、存在する場合の保持パスを報告することを要求している。
- Specification「Error policy」では、複数のエラーを報告するときに `_RollbackError` を使うことを要求している。
- Specification「Acceptance criteria」では、cleanup failure が旧状態、保持された recovery location、または artifact persistence failure を含む structured indeterminate report のいずれかとして観測可能であることを要求している。
- Canonical plan Task 7 Step 3 は backup create、write、flush、fsync、close と tempfile cleanup の failure injection を要求している。

**該当箇所**

- `gwexpy/timeseries/io/hdf5.py:1214`
- `gwexpy/timeseries/io/hdf5.py:1223`
- `gwexpy/timeseries/io/hdf5.py:1386`
- `gwexpy/timeseries/io/hdf5.py:1407`
- `tests/timeseries/test_hdf5_exact_t0_transactions.py:1321`
- `tests/timeseries/test_hdf5_exact_t0_transactions.py:1495`

**根拠**

`_create_filelike_backup()` は `mkstemp()` 後の `fchmod()` または `fdopen()` が失敗すると、同じ `except` 内で descriptor close と path unlink を実行する。

この unlink も失敗すると、unlink 例外が setup の元例外を置き換え、作成済み path は呼出側へ返らない。

外側の `_write_filelike_transaction()` は `backup_path is None` のまま例外を処理するため、保持ファイルを検出できず、rollback error がないものとして plain `OSError` を再送出する。

隔離再現では、`fchmod` failure と backup unlink failure を同時に注入した結果、次を観測した。

```text
FILELIKE OSError 'injected backup unlink failure' state=None recovery_path=None
retained_backup=True caller_position=7
```

元の `injected backup setup failure` は observable error metadata から消失した。

既存テストは precommit stage failure、commit write failure、rollback failure、success/rollback 後の close/unlink failure を検証しているが、backup setup 自体とその cleanup の複合失敗を検証していない。

**Observable impact**

caller bytes と position が旧状態であっても、mode 0600 の named temporary が未報告で残る。

呼出側は `_RollbackError.state`、`rollback_errors`、`recovery_path` から複数故障と残存ファイルを発見できず、仕様が要求する cleanup failure の可観測性を失う。

**RED-first test node**

`tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_filelike_backup_setup_cleanup_failure_reports_old_and_retains_path`

この node では `fchmod` または `fdopen` の setup failure と unlink failure を同時注入し、次を要求する。

- caller bytes と元 position が不変である。
- `_RollbackError.state == "old"` である。
- `byte_state == "old"` かつ `position_state == "old"` である。
- `operation_error` が setup failure である。
- `rollback_errors` が unlink failure を含む。
- `recovery_path` が実在する retained path を指す。
- 各注入 seam が 1 回だけ呼ばれる。

**最小修正案**

`_create_filelike_backup()` が path 確保後の setup と cleanup を別々に捕捉し、元の setup exception、cleanup exceptions、実在する retained path を呼出側へ失わず伝えるようにする。

外側では caller position の restoration 結果と合わせ、複数故障を `_RollbackError(state="old", byte_state="old", position_state="old", recovery_path=...)` に正規化する。

cleanup が成功した単独 setup failure は、旧状態を確認して元例外を再送出してよい。

### Important 2: pathname stage の descriptor close failure が stage を cleanup せず未報告で残す

**仕様根拠**

- Specification「Filesystem-path transaction」では secure sibling stage を transaction 内で管理し、failed stage では original path を不変にし、stage cleanup も失敗した場合は両例外、`state="old"`、保持 stage path を structured error で報告することを要求している。
- Specification「Error policy」では mutation 開始後の final rollback cleanup failure まで捕捉することを要求している。
- Specification「Acceptance criteria」では cleanup failure の recovery location または artifact-persistence failure が observable であることを要求している。
- Canonical plan Task 6 Step 5 は sibling stage を `O_CREAT | O_EXCL` で作成し、失敗時の cleanup と状態分類を要求している。

**該当箇所**

- `gwexpy/timeseries/io/hdf5.py:1313`
- `gwexpy/timeseries/io/hdf5.py:1320`
- `gwexpy/timeseries/io/hdf5.py:1344`
- `tests/timeseries/test_hdf5_exact_t0_transactions.py:960`
- `tests/timeseries/test_hdf5_exact_t0_transactions.py:994`

**根拠**

`_create_sibling_transaction_file()` は `os.open()` で stage を作成した後、`os.close()` を cleanup guard なしで呼ぶ。

`os.close()` が raise すると関数は stage path を返さないため、`_write_path_transaction()` の `temporary_path` 代入と `try` block に到達しない。

そのため、作成済み stage の unlink は試行されず、structured state と保持 path のない plain exception が返る。

隔離再現では、descriptor を実際に close した直後に close failure を注入した結果、次を観測した。

```text
PATH OSError 'injected stage descriptor close failure' state=None recovery_path=None
target_exists=False retained_stage=True
```

既存の pathname tests は `os.replace` failure と、その後の stage unlink failure を検証しているが、stage creation が path を返す前の cleanup failure を検証していない。

**Observable impact**

original target は旧状態（この再現では不存在）のままだが、同一 directory に stage が未報告で残る。

呼出側は exception metadata から stage の場所を取得できず、P0 の cleanup 可観測性と temporary lifecycle の契約を満たせない。

**RED-first test node**

`tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_path_stage_setup_failure_cleans_or_reports_retained_stage`

この node は descriptor close failure を注入し、unlink 成功時と unlink failure 時を parameterize する。

- unlink 成功時は target が旧状態で、stage が残らず、元の close failure が再送出される。
- unlink failure 時は `_RollbackError.state == "old"`、元の close failure、unlink failure、実在する `recovery_path` が観測できる。
- native writer は 0 回である。

**最小修正案**

stage path の確保直後から cleanup envelope に入れる。

`os.close()` failure 時にも stage unlink を試行し、unlink 成功なら元の close exception を再送出する。

unlink も失敗した場合は `_RollbackError` に close exception と unlink exception を分離して格納し、`state="old"` と実在する stage path を設定する。

## Requirement-to-evidence mapping

以下は source と acceptance test の対応を確認した範囲である。

- Dataset-local marker と v2 sidecar authority: codec の payload、digest、float bits、unit binding、whole-document validation と、authority truth tableを確認した。
- Aliases、moves、copies、external links: hard/soft aliases、move、same/cross-file H5Ocopy、without-attrs copy、resolved external file authority の nodes を確認した。
- Path validation: relative/absolute `str`、UTF-8 `bytes`、NUL、raw dot components、external/soft link write policy の matrix を確認した。
- Native writer 最大 1 回: pathname、file-like、File、Group の success と post-write failure の parameterized node を確認した。
- Pathname atomic stage: nonregular target、multiply-linked regular file、replace failure、replace plus unlink failure、append、fresh overwrite、growth の nodes を確認した。
- File-like transaction: chunk-bounded copy、short writes、commit rollback、byte/position classification、post-commit warning、normal cleanup、growth、RSS の nodes を確認した。
- Caller-owned handle recovery: setup、delete-before/after-raise、durable recreation、public/sidecar restoration、object identity、references、dimension scale、private-object bound の nodes を確認した。
- `_RollbackError`: stable metadata fields `operation_error`、`rollback_errors`、`state`、`recovery_path`、`byte_state`、`position_state` を source と tests で確認した。

上記の coverage は Finding 1 と Finding 2 の setup-before-return paths を含まない。

## Fresh log assessment

- Focused pytest: `782 passed in 59.50s`。skip と xfail の表示はない。
- Qualification claim: `1 passed in 0.80s`。skip と xfail の表示はない。
- Surrounding selectors: `54 passed, 1 skipped in 21.62s`。1 件は pass と数えていない。`-q` log だけでは skip 理由を確認できない。
- Focused Ruff: pass。
- Source/test Ruff: pass。
- Format check: 5 files formatted。
- MyPy: 2 source files、0 issues。
- `git diff --check`: output なし、exit code 0。

Fresh logs の成功は、未収集の setup cleanup failure paths を pass と推定する根拠にはしていない。

## Read-only reproduction

Conda environment `gwexpy` と隔離 `TemporaryDirectory` を使い、repository file を変更せずに次を再現した。

1. file-like backup setup failure と backup unlink failure の複合注入。
2. pathname sibling stage の descriptor close-after-close failure 注入。

いずれも original target state は保持されたが、retained temporary が存在し、例外には `state` と `recovery_path` がなかった。

## Review boundary

実装、テスト、spec、plan は変更していない。

Commit、push、tag、release、upload、workflow dispatch は行っていない。

P1、candidate qualification、人間による最終 P0 承認は本レビューの対象外である。
