# HDF5 exact-epoch P0 code-quality review

## Review basis

このレビューはproduction readiness、例外分類、リソース所有権、private APIの凝集性、型安全性、テストの妥当性を対象とした。

既存の`docs/developers/reviews/`、`docs/developers/reports/`、会話履歴、他agentの出力、git log内のreview本文は参照していない。

依頼に記載されたHead SHA `f154b5ae4`は、このworktreeではcommitとして解決できなかった。

branchの実HEADは`f154b5a346b8c1002323bfe18112290c1aa2e349`だったため、Base SHA `209d061f11576f4014422fe8e68fbdbf092575a9`から実HEADまでをレビューした。

対象は次の実装、テスト、仕様、計画、round 5ログに限定した。

- `gwexpy/timeseries/io/hdf5.py`
- `gwexpy/timeseries/io/_hdf5_exact_epoch.py`
- `tests/timeseries/test_hdf5_exact_epoch_codec.py`
- `tests/timeseries/test_hdf5_exact_t0.py`
- `tests/timeseries/test_hdf5_exact_t0_transactions.py`
- `docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md`
- `docs/superpowers/plans/2026-08-29-hdf5-exact-epoch-identity.md`
- 依頼で指定された`.agent/tmp/gwexpy_conda_jobs/hdf5_p0_round5_*_20260830.log`

## Strengths

`_hdf5_exact_epoch.py`はcodec、sidecar schema、immutable carrierに責務を限定しており、registry処理やHDF5 mutationを持ち込んでいない。

markerは長さ、文字種、canonical prefix、payload field、digest、unit conversion bits、binary64 projection、byte-for-byte再構成を段階的に検証している。

`SidecarDocument`は`MappingProxyType`を用い、record carrierもfrozen dataclassであるため、validation後の値が誤って書き換わりにくい。

pathname、file-like、caller-owned handleの三つのtransaction戦略は別helperへ分離されており、native writerを一度だけ呼ぶ共通経路も明確である。

caller-owned handleのrollbackは古いdataset IDを保持したままhard linkをrecovery objectへ移し、object reference、region reference、dimension scale、hard-link aliasをpublic assertionで検証している。

`_RollbackError`は`operation_error`、順序付き`rollback_errors`、`state`、`byte_state`、`position_state`、`recovery_path`を保持している。

主要なrollback testは例外の型だけでなく、元の例外object、rollback errorの順序、call count、public bytes、exact epoch、sidecar、recovery path、close/reopen後のobject identityを確認している。

file-like copyは各readを`FILELIKE_COPY_CHUNK`以下に制限し、short positive writeを再試行し、不正なreadまたはwrite countを拒否する。

production codeに`BytesIO`、`getvalue()`、引数なし`read()`によるfull snapshot経路は見つからなかった。

resource testsはrepeated pathname overwrite、repeated file-like overwrite、16 MiB stage image、8 MiB対32 MiB subprocess RSS、native writer call countを別々に検証している。

公開APIの追加または変更、HDF5以外への実装拡張、release操作は見つからなかった。

## Critical findings

なし。

## Important findings

### 1. Pathname cleanupの存在確認失敗がprimary errorとrecovery metadataを失わせる

**分類**：例外分類、resource ownership、production readiness。

**場所**：`gwexpy/timeseries/io/hdf5.py:1426`、特に`gwexpy/timeseries/io/hdf5.py:1427`。

`_write_path_transaction`はprimary operationの失敗後、cleanupを始める前に`temporary_path.exists()`をcleanup用の`try`の外で評価する。

この存在確認が`PermissionError`などを送出すると、`operation_error`は上書きされ、`unlink()`は試行されず、stageが残る。

送出される例外は`_RollbackError`ではないため、callerは元のoperation error、ordered cleanup errors、`state="old"`、`recovery_path`を取得できない。

隔離再現では、`os.replace` seamでstage directoryの権限を外してprimary failureを起こしたところ、target bytesはoldのまま、stageは残存した一方、結果はraw `PermissionError`で`isinstance(error, _RollbackError) == False`だった。

現行test `tests/timeseries/test_hdf5_exact_t0_transactions.py:1120`は`unlink()`失敗後の`os.lstat()`失敗を検証するが、`temporary_path.exists()`自体の失敗を通らないため、この経路を検出しない。

**修正案**：先行する`exists()`を削除し、`unlink()`を無条件に試行する。

`FileNotFoundError`だけをcleanup済みとして扱い、それ以外は`cleanup_errors`へ追加したうえで`_retained_temporary_path()`を呼び、元の`operation_error`を保持した`_RollbackError(state="old")`を送出する。

cleanup errorとinspection errorは発生順に`rollback_errors`へ格納する。

**必要test**：`os.replace`のprimary failure後にcleanup前のpath inspectionまたはdirectory accessを失敗させ、targetがoldであること、stageの残存状態、元のoperation error identity、ordered rollback errors、`state`、`recovery_path`をassertするpublic write testを追加する。

### 2. Commit後cleanup warningがwarnings-as-errors設定で成功返却を破る

**分類**：例外分類、public transaction contract、テストgap。

**場所**：`gwexpy/timeseries/io/hdf5.py:1309`から`gwexpy/timeseries/io/hdf5.py:1319`、呼出し元は`gwexpy/timeseries/io/hdf5.py:1549`から`gwexpy/timeseries/io/hdf5.py:1557`。

file-like commit後のcleanup failureは`warnings.warn(..., ResourceWarning)`を直接呼び、その送出を保護していない。

callerまたはtest runnerが`warnings.simplefilter("error", ResourceWarning)`を設定していると、`ResourceWarning`が例外として送出される。

この時点ではtarget bytesとexact epochがnew stateへcommit済みなので、callerは失敗として処理または再試行し得る一方、実データは更新済みである。

隔離した実書込再現では、backup unlink failureと`ResourceWarning`のerror filterを同時に設定すると、呼出しは`ResourceWarning`で終了したが、再読込したexact epochは新値`456`で、mode-0600 backupも残存した。

現行test `tests/timeseries/test_hdf5_exact_t0_transactions.py:1991`および`tests/timeseries/test_hdf5_exact_t0_transactions.py:2035`は通常の`pytest.warns`設定だけを使うため、warning filterが例外化する経路を検出しない。

**修正案**：commit後の通知を専用helperへ閉じ込め、warning filterやcustom warning hookによる例外がpublic writeから伝播しないようにする。

通常設定では分類済み`ResourceWarning`を維持し、通知処理が失敗してもnew stateを返すというpost-commit不変条件を優先する。

**必要test**：backup unlink failureを注入し、`warnings.simplefilter("error", ResourceWarning)`の下でもwriteが例外を送出せず、新しいbytes、exact epoch、caller position、残存backup pathが期待どおりであることをassertするintegration testを追加する。

## Minor findings

なし。

## Recommendations

Important finding 1と2をRED-first testで修正し、変更したfailure-injection nodes、focused HDF5 suite、changed-file Ruff、format check、production MyPy、repository scoped gatesを再実行する。

`_RollbackError.state`、`byte_state`、`position_state`はprivate APIでも`Literal`またはenum相当の型に限定すると、将来の分岐追加で不正な組合せを作りにくい。

subprocess RSS testは現行のfull-buffer退行を検出する設計だが、`ru_maxrss`はprocess high-water markなので、初回writeのpeakに後続allocationが隠れる可能性がある。

将来のnon-blocking hardeningとして、同一sizeのnative writer基準processとの差分も取得すると、「native behavior relative」というresource claimをより直接に検証できる。

## Verification results

### Round 5 logs

- Resource selectors：12 passed in 6.64s。
- Focused HDF5 suite：797 passed in 60.68s。
- Exact qualification claim：1 passed in 0.79s。
- Surrounding selectors：54 passed in 22.02s。
- Changed-file Ruff：passed。
- Changed-file format check：5 files already formatted。
- Production MyPy：2 source files、no issues。
- Repository scoped Ruff：passed。
- Full-repository Ruff：failed only at unchanged `docs_redesign/conf.py:242:5` D103。
- Repository MyPy：396 source files、no issues。
- Repository pytest scope：2812 passed、72 skipped、3 xfailed、47 warnings in 136.02s。
- Saved round 5 diff check：outputなしでpass。

skip、xfail、warningはpassへ合算していない。

full-repository RuffのD103は計画に記録された既知のunchanged findingであり、このHDF5 P0差分のfindingには分類しない。

### Additional isolated verification

`rtk conda run -n gwexpy pytest tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_filelike_success_cleanup_failure_warns_new_and_returns tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_path_replace_failure_cleans_or_reports_retained_stage -q`は5 passed in 0.93sだった。

warning helper単体を`ResourceWarning`のerror filter下で呼ぶと、`ResourceWarning TimeSeries HDF5 write committed; state=new; ...`が送出された。

実際のfile-like writeへbackup unlink failureを注入した再現結果は`ResourceWarning 456 1 True`だった。

これは順に、送出例外型、新しいexact epoch、作成backup数、backup残存を表す。

pathname writeへreplace failureとstage directory access failureを注入した再現結果は、raw `PermissionError`、target bytes unchanged、stage count 1、stage retained、`_RollbackError`ではない、だった。

`rtk rg`によるproduction scanでは`BytesIO`、`getvalue()`、引数なし`read()`を検出しなかった。

`rtk git diff --check 209d061f1..HEAD`はoutputなしでpassした。

## Assessment

**Changes required**。

Critical findingはないが、二つのImportant findingはいずれもfailure pathで実際のstateとcallerが受け取る結果を食い違わせる。

修正と再検証が終わるまで、production-readyとは判定しない。

この判定は独立したcode-quality reviewであり、人間によるHDF5 P0最終承認ではない。
