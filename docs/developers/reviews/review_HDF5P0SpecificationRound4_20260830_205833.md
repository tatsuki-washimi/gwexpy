# HDF5 P0 第四巡 fresh specification review

## 対象と独立性

2026-08-30 20:58:33 JST に、`test/v020-post-release-qualification` の HEAD `a60f85c7125529b62d1aafe8525ae9e1c6ca1e30` を、第四巡 base `8369778c5` と比較した。

レビューには、指定された設計仕様、実装計画、focused production 2ファイル、focused tests 3ファイル、両 revision 間の限定差分、指定された fresh logs だけを使用した。

`docs/developers/reviews/` の既存文書、`docs/developers/reports/`、会話履歴、他 agent 出力、Git log の review 本文は参照していない。

public API、P1 bootstrap、公開操作は評価対象外とした。

この文書は人間による P0 最終承認ではない。

## Verdict

**Changes required**

Critical は0件、Important は1件、Minor は0件である。

実装計画が第五巡を許可していないため、このレビューでは修正を開始せず、人間へ判断を戻す。

## Findings

### Critical

なし。

### Important

#### I-1：pathname の replace 失敗後に stage unlink が delete-after-raise になると、存在しない recovery path を報告する

**分類**：仕様、エラー状態、テスト。

**根拠**：`gwexpy/timeseries/io/hdf5.py:1426` から `gwexpy/timeseries/io/hdf5.py:1436` は、`os.replace` などの primary operation が失敗した後、stage が存在するときに `Path.unlink()` を呼ぶ。

しかし `Path.unlink()` が entry を削除してから例外を送出した場合、例外処理は削除後の実在性を再確認せず、常に `str(temporary_path)` を `_RollbackError.recovery_path` に格納する。

第四巡で追加された `_retained_temporary_path` は `gwexpy/timeseries/io/hdf5.py:1229` から `gwexpy/timeseries/io/hdf5.py:1240` にあり、descriptor-close setup failure では削除後の実在性を確認するが、この replace-failure cleanup 経路では使われていない。

既存の acceptance test `tests/timeseries/test_hdf5_exact_t0_transactions.py:1145` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:1190` は unlink が削除前に失敗する場合だけを注入しており、delete-after-raise を区別しない。

設計仕様は、structured error が retained recovery object path を報告することを `docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md:588` から `docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md:594` で定める。

pathname についても、failed stage と cleanup failure の両方を報告し、retained stage path を示すことを同仕様 `:631` から `:633` で定める。

さらに、path replacement とその後の stage unlink failure は明示的な failure-injection 対象である（同仕様 `:785` から `:799`）。

**隔離再現**：tracked file を変更せず、conda `gwexpy` 環境の一時ディレクトリで公開 `TimeSeries.write(...)` を実行した。

既存 pathname を `append=True, overwrite=True` で置換し、`os.replace` に primary `OSError`、stage の `Path.unlink` に「実際に削除してから `OSError`」を注入した。

結果は `_RollbackError`、`state="old"`、`operation_error` は replace error だったが、`recovery_path='/tmp/.../.x.hdf5.gwexpy-....hdf5'` に対して `Path.exists()` は `False` だった。

したがって public target の旧 bytes は atomicity により保持される一方、例外の recovery location は事実と一致しない。

**影響**：caller は structured error が示す stage を回収できず、診断または手動復旧で存在しない path を authoritative artifact と誤認する。

これはデータ本体の atomicity を破らないが、P0 acceptance criterion の cleanup classification を満たさないため Important と判定する。

**必要な修正**：replace-failure cleanup でも unlink 例外後に `os.lstat` 相当で entry の実在性を確認し、残っている場合だけ `recovery_path` を設定する。

delete-after-raise では `recovery_path=None`、fail-before-delete では実在する stage path、実在性確認自体が失敗した場合はその inspection error も順序を保って `rollback_errors` に含める必要がある。

descriptor close と同様に、unlink を再試行してはならない。

**必要なテスト**：`TimeSeries.write(...)` 経由で `os.replace` failure と stage unlink の `success`、`fail-before-delete`、`delete-after-raise` を分ける parameterized acceptance test を追加する。

各 row で primary operation error、順序付き cleanup errors、`state="old"`、stage の実在性と `recovery_path` の一致、元 pathname bytes、native writer の呼出回数、unlink の呼出回数1回を確認する。

### Minor

なし。

## 第四巡の限定要件

| 要件 | 実装とテスト | 評価 |
|---|---|---|
| file-like backup setup の `fchmod` または `fdopen` failure で primary error を保持する | `_FilelikeBackupSetupError` が primary error と setup cleanup errors を運び、`_write_filelike_transaction` が public `_RollbackError` へ展開する（`gwexpy/timeseries/io/hdf5.py:110`、`:1243`、`:1469`）。carrier 自体は public write から漏れない | 適合 |
| close、unlink、lstat、position restore の複合 failure を発生順に公開する | setup cleanup を close、unlink、lstat の順で収集し、その後に position restore error と外側 temporary cleanup error を追加する（同 `:1253` から `:1268`、`:1470` から `:1502`） | 適合 |
| retained backup path と byte、position、overall state を分類する | surviving entry のみ setup carrier に保持し、bytes は `old`、position は復元結果により `old` または `indeterminate`、overall state は両者から決める（同 `:1229` から `:1240`、`:1479` から `:1501`） | 適合 |
| setup descriptor close を再試行しない | `_create_filelike_backup` の raw descriptor close は1回だけで、tuple assignment 未完了のため外側 generic cleanup は同 descriptor を再度受け取らない（同 `:1243` から `:1270`） | 適合 |
| pathname sibling 作成後の descriptor close failure で unlink を1回試す | `_create_sibling_transaction_file` は close error 後に unlink を1回だけ呼ぶ（同 `:1356` から `:1386`） | 適合 |
| pathname setup cleanup の success、fail-before-delete、delete-after-raise を分ける | unlink error 後に `_retained_temporary_path` で実在性を確認し、entry が残る場合だけ path を報告する。公開 write の新規 tests は `tests/timeseries/test_hdf5_exact_t0_transactions.py:961` から `:1108` | 適合 |
| pathname setup failure では native writer を呼ばず、元 pathname を保持する | 新規 tests が old bytes、native writer 0回、close 1回、unlink 1回を確認する | 適合 |

## 主要要件の対応表

| 主要要件 | production の根拠 | focused acceptance の根拠 | 評価 |
|---|---|---|---|
| digits-only marker、float-bit、unit、digest、canonical re-encoding | `gwexpy/timeseries/io/_hdf5_exact_epoch.py:498` から `:705` | `tests/timeseries/test_hdf5_exact_epoch_codec.py` の marker envelope、canonicality、corruption、limit nodes | 適合 |
| strict sidecar v2 と marker-only authority | `gwexpy/timeseries/io/_hdf5_exact_epoch.py:91` から `:376`、`gwexpy/timeseries/io/hdf5.py:593` から `:694` | codec sidecar nodes と `tests/timeseries/test_hdf5_exact_t0.py:479` から `:830` | 適合 |
| alias、move、copy、stale replacement、GWpy-only read | resolved dataset marker と resolved file sidecar を使う `gwexpy/timeseries/io/hdf5.py:1713` から `:1753` | `tests/timeseries/test_hdf5_exact_t0.py:620` から `:830`、`:3537` 以降 | 適合 |
| native path と link safety、reload idempotence | `gwexpy/timeseries/io/hdf5.py:304` から `:437`、`:1755` 以降 | `tests/timeseries/test_hdf5_exact_t0.py:3869` 以降と `:5251` 以降 | 適合 |
| one native write と disposable stage | `gwexpy/timeseries/io/hdf5.py:753` から `:785` | `tests/timeseries/test_hdf5_exact_t0_transactions.py:819` から `:876` | 適合 |
| pathname atomicity | `gwexpy/timeseries/io/hdf5.py:1158` から `:1438` | path target、replace、growth tests | 一部不適合。I-1 |
| bounded file-like copying、backup durability、byte/position state | `gwexpy/timeseries/io/hdf5.py:1186` から `:1306`、`:1441` から `:1552` | copy、backup、commit、rollback、cleanup、RSS tests | 適合 |
| caller-owned handle identity と sidecar rollback | `gwexpy/timeseries/io/hdf5.py:807` から `:1155` | `tests/timeseries/test_hdf5_exact_t0_transactions.py:109` から `:817` | 適合 |
| resource invariants | one-write、growth、no duplicate stage、bounded RSS、private recovery object tests | fresh resource log は12 passed | 適合 |

## Fresh logs

- resource qualification：12 passed in 6.51s。
- focused HDF5 suite：795 passed in 60.50s。
- exact qualification claim：1 passed in 0.81s。
- surrounding compatibility selectors：54 passed in 21.99s。skip は記録されていない。
- changed-file Ruff：All checks passed。
- changed-file format check：5 files already formatted。
- focused production MyPy：2 source files、no issues。
- repository scoped Ruff：All checks passed。
- full-repository Ruff：失敗。既知かつ scope 外の `docs_redesign/conf.py:242:5` D103 1件だけ。
- repository MyPy：396 source files、no issues。
- repository pytest：2810 passed、72 skipped、3 xfailed、47 warnings、183.77s。skip と xfail を pass に含めていない。
- fresh diff-check log：出力なし。

full-repository Ruff の D103 は approved specification が明示した既知の unchanged finding と一致し、今回の Important finding とは別である。

## 追加検証

conda `gwexpy` 環境で第四巡の5 test functionsを直接実行し、parameterized rowsを含む13件が `13 passed in 0.90s` だった。

対象は pathname setup の cleanup success、fail-before-delete、delete-after-raise、および file-like backup setup の `fchmod`、`fdopen`、close、unlink、lstat、position restore の組合せである。

実行後の `git status --short` は空だった。

別の一時ディレクトリで、I-1 の public `TimeSeries.write(...)` 再現を実行した。

正確な観測は `_RollbackError`、`state="old"`、operation message `replace`、cleanup errors `['unlink-after-delete']`、non-`None` recovery path、`Path.exists() == False`、元 target bytes 不変、native writer 1回、unlink 1回だった。

## 残余リスクと境界

I-1 が解消されるまでは pathname cleanup の recovery metadata を仕様適合として承認できない。

raw descriptor の `close()` が例外を返した場合、OS 上で descriptor が閉じたかどうかは platform semantics に依存する。

現実装と tests は危険な close retry を行わないことを保証しているが、close-before-failure を注入した場合の descriptor 生存そのものは保証できない。

full-repository pytest の72 skipsと3 xfailsは環境または既知期待としてログに残っており、pass として評価していない。

別 fresh reviewer による code-quality review、人間の P0 承認、P1 bootstrap、candidate qualification、tag、push、release、upload、workflow dispatch は未実施であり、この verdict の範囲外である。
