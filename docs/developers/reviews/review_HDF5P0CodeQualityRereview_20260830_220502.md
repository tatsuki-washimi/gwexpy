# HDF5 exact-epoch P0 code-quality fresh re-review

## Review scope

Review HEAD `38c552f039a56b7d07b99d239f061274ba7c9de4` を、remediation base `f154b5a346b8c1002323bfe18112290c1aa2e349` と比較した。

指定された設計仕様、実装計画、production 2ファイル、focused test 3ファイル、remediation diff、remediation後のfresh logsだけを根拠にした。

`docs/developers/reviews/`の既存レビュー、`docs/developers/reports/`、会話履歴、他agent出力、git logのレビュー本文は参照していない。

このレビューでは実装、test、spec、planを変更していない。

## Strengths

pathnameのprimary failure後の処理は、`gwexpy/timeseries/io/hdf5.py:1430`から`gwexpy/timeseries/io/hdf5.py:1447`でstageの先行存在確認を行わず、`Path.unlink()`を1回だけ試している。

`FileNotFoundError`だけをcleanup済みとして扱い、それ以外ではprimary errorを`operation_error`に保持し、unlink errorと後続inspection errorを順序どおり`rollback_errors`へ格納する。

retained stageを確認できた場合だけ`recovery_path`を返し、targetの`state="old"`を維持する分類も一貫している。

public regressionは`tests/timeseries/test_hdf5_exact_t0_transactions.py:1225`から`tests/timeseries/test_hdf5_exact_t0_transactions.py:1296`で、probe 0回、unlink 1回、old target、native writer 1回、retained stage、primary error identityを同時に確認している。

delete-after-raiseとinspection failureの順序は、同ファイル`1121`から`1222`の既存public parameterizationが補完しており、新しいseamだけで成功する偽陽性にはなっていない。

file-like commit後の通知は、`gwexpy/timeseries/io/hdf5.py:1309`から`gwexpy/timeseries/io/hdf5.py:1323`でmessage構築と`warnings.warn()`だけを狭い`try`に置いている。

通常filterでは`ResourceWarning`を維持し、warnings-as-errorsや通常のcustom notification exceptionはcommit済みwriteの返却を破らない。

`Exception`だけを捕捉しているため、transaction cleanupで必要な`BaseException`捕捉と、通知層でのprocess-control exception非抑止が区別されている。

public regressionは`tests/timeseries/test_hdf5_exact_t0_transactions.py:2110`から`tests/timeseries/test_hdf5_exact_t0_transactions.py:2176`で、new bytes、exact epoch、committed position、native writer 1回、cleanup 1回、retained backupを検証している。

通常の`ResourceWarning`は同ファイル`2066`から`2107`で別に検証されている。

private helperの責務は、stage作成、copy、flush、cleanup、notification、handle recoveryに分離されている。

transaction envelopeの`BaseException`捕捉は、mutation後のclose、unlink、sidecar restore、recovery persistenceを完遂して状態を分類する目的に限定されている。

post-commit notificationの`Exception`捕捉は上記のとおり狭く、型検査もproduction 2ファイルで通過している。

resource testsはnative writer call count、16 MiB stage image比較、20回の1 MiB置換、file-like RSSの8 MiB対32 MiB比較を独立に持つ。

RSS testはsubprocess内でinput arraysと既存caller bufferをbaseline前に確保し、replacement中に増えるwrapper peakを測るため、full Python snapshotの再導入を検出できる構成である。

public APIの追加やconstructor変更はなく、remediation production diffは`hdf5.py`の2 failure pathに限定されている。

## Critical

0件。

## Important

0件。

## Minor

### 1. custom notification exceptionのpublic regressionがtracked testにない

**分類**：test coverage、保守性。

**場所**：`tests/timeseries/test_hdf5_exact_t0_transactions.py:2110`。

このpublic testは`ResourceWarning`をerrorへ変換するfilterを検証するが、`warnings.warn`自体が`RuntimeError`などを送出するcustom notification failureは検証しない。

productionの`gwexpy/timeseries/io/hdf5.py:1313`から`gwexpy/timeseries/io/hdf5.py:1323`はmessage構築と通知の通常exceptionを正しく抑止し、今回の隔離public writeでも成功したため、現時点のproduction behaviorに不具合はない。

ただし、将来`try`の範囲や捕捉型を変更した場合、warnings-as-errorsのrowだけではcustom notifierの回帰を特定できない。

**修正案**：`test_hdf5_filelike_committed_cleanup_warning_cannot_become_failure`をnotification modeでparameterizeし、既存のwarnings-as-errorsに加えて、`exact_hdf5.warnings.warn`が`RuntimeError("injected notification failure")`を送出するrowを追加する。

**必要test**：新しいrowでもnew bytes、`t0_gps_ns == 456`、committed position、native writer 1回、cleanup 1回、retained backupを既存rowと同じpublic write経路で確認する。

## Verification results

指定されたfresh logsの結果は次のとおりだった。

- resource：12 passed in 6.63s。
- focused HDF5：799 passed in 60.29s。
- exact claim：1 passed in 0.87s。
- surrounding：54 passed in 22.41s。skip、xfailはない。
- changed Ruff：pass。
- changed format check：5 files already formatted。
- production MyPy：2 source files、no issues。
- repository scoped Ruff：pass。
- repository full Ruff：`docs_redesign/conf.py:242:5`の既知かつ未変更のD103だけでfail。
- repository MyPy：396 source files、no issues。
- repository pytest：2814 passed、72 skipped、3 xfailed、47 warnings in 200.19s。
- fresh diff check：pass。

追加の隔離検証では、2 remediation public nodesを直接実行し、`2 passed in 0.93s`だった。

さらに`warnings.warn`を`RuntimeError`送出へ差し替えたpublic file-like writeを実行し、new bytes、`t0_gps_ns == 456`、committed position、native writer 1回、cleanup 1回、retained backup 1件を確認した。

## Recommendations

Minor 1のcustom notification rowを次の通常保守でtracked regressionへ追加する。

既知D103、repository pytestの72 skipsと3 xfailsは今回のHDF5 remediationと分離して管理する。

## Residual risks

pathname commitはatomic replacementであり、inode identity、owner、ACL、拡張属性を保持する契約ではない。

caller-owned open HDF5 handleでは旧object identityを守るため、HDF5 free-spaceによる物理file size増加は上限保証の対象外である。

post-commit cleanup通知はfilterやcustom notifier failure時に観測されない場合があるが、commit済み成功をfailureへ変えないための意図したtrade-offである。

repository全体のskip、xfail、既知D103は残っており、本レビューはそれらを解消したとは評価しない。

## Assessment

**Ready with minor follow-up**。

Critical 0件、Important 0件、Minor 1件である。

2つのremediation経路はproduction readinessを満たし、Minor 1は現行behaviorの欠陥ではなくtracked regressionの不足である。

本レビューはHDF5 exact-epoch P0実装のcode-quality判定であり、人間によるP0承認、candidate qualification、tag、push、release、upload、workflow dispatchを承認するものではない。
