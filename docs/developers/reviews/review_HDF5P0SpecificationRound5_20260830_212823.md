# HDF5 P0 仕様適合レビュー 第五巡

## 結論

**Verdict: Approved for specification compliance**

Review HEAD `0a7ea1bbe` は、approved specification と implementation plan に適合している。

Critical 0件、Important 0件、Minor 0件である。

この判定は仕様適合性だけを対象とし、人間によるP0最終承認、code-quality review、P1 bootstrap、公開操作を承認しない。

## レビュー条件

- 対象worktree：`/home/washimi/.config/superpowers/worktrees/gwexpy/v020-post-release-qualification`
- 対象branch：`test/v020-post-release-qualification`
- Review HEAD：`0a7ea1bbe5e976c47b3a0c6e3a0af4a2dfbbe6b9`
- 第五巡base：`0d4dfe327`
- 差分範囲：指定されたplan、focused production、focused testsだけの `0d4dfe327..0a7ea1bbe`
- 証跡範囲：指定されたround 5 fresh logsだけ

過去または既存のreview文書、`docs/developers/reports/`、会話履歴、他agent出力、git logのreview本文は参照していない。

## Findings

### Critical

0件。

### Important

0件。

### Minor

0件。

CriticalまたはImportantが残っていないため、「第六巡を開始せず人間判断へ戻す」という停止条件には該当しない。

ただし、planが定める別担当のcode-quality reviewと人間によるP0判断は未実施であり、本レビューからP0承認へ進めることはできない。

## 第五巡findingの確認

`gwexpy/timeseries/io/hdf5.py:1426` は pathname replacement を含む主操作の例外objectを `operation_error` として保持する。

stage entryが存在する場合、`gwexpy/timeseries/io/hdf5.py:1427` から `gwexpy/timeseries/io/hdf5.py:1429` は unlink を一度だけ試みる。

unlinkが例外を返した場合、`gwexpy/timeseries/io/hdf5.py:1431` から `gwexpy/timeseries/io/hdf5.py:1435` は cleanup errorを先に格納し、`_retained_temporary_path()` でentryを検査する。

`gwexpy/timeseries/io/hdf5.py:1229` から `gwexpy/timeseries/io/hdf5.py:1240` は `os.lstat()` 成功時だけpathを返し、`FileNotFoundError` では `None`、その他のinspection errorではその例外をcleanup errorの後へ追加して `None` を返す。

したがって、4経路は次のように区別される。

| cleanup outcome | 例外 | `rollback_errors` | `recovery_path` | stage entry |
|---|---|---|---|---|
| success | 元のreplace errorを再送出 | 該当なし | 該当なし | 消去済み |
| fail-before-delete | `_RollbackError` | unlink error | 実在するstage path | 残存 |
| delete-after-raise | `_RollbackError` | unlink error | `None` | 消去済み |
| inspection-failure | `_RollbackError` | unlink error、inspection errorの順 | `None` | 実在を主張しない |

`tests/timeseries/test_hdf5_exact_t0_transactions.py:1111` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:1218` は4経路を一つのparameterized testで観測する。

テストは private helperを直接呼ばず、`tests/timeseries/test_hdf5_exact_t0_transactions.py:1176` および `tests/timeseries/test_hdf5_exact_t0_transactions.py:1186` の `TimeSeries.write(..., format="hdf5")` を通る。

次の観測は要求どおりである。

- 元pathname bytes：`tests/timeseries/test_hdf5_exact_t0_transactions.py:1202`
- native writer 1回：`tests/timeseries/test_hdf5_exact_t0_transactions.py:1203`
- replace、unlink、lstat各seam回数：`tests/timeseries/test_hdf5_exact_t0_transactions.py:1204` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:1206`
- 元replace例外object：`tests/timeseries/test_hdf5_exact_t0_transactions.py:1183` および `tests/timeseries/test_hdf5_exact_t0_transactions.py:1193`
- `rollback_errors` のobject identityと順序：`tests/timeseries/test_hdf5_exact_t0_transactions.py:1194` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:1197`
- `state`、`byte_state`、`position_state`：`tests/timeseries/test_hdf5_exact_t0_transactions.py:1198` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:1200`
- stage実在性と `recovery_path` の一致：`tests/timeseries/test_hdf5_exact_t0_transactions.py:1208` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:1218`

unlink seamの呼び出し回数を全経路で1回と固定しているため、unlink再試行を見逃さない。

## 主要要件の対応表

| 仕様要件 | 実装根拠 | テスト根拠 | 判定 |
|---|---|---|---|
| digits-only canonical marker、single boundary decode、float-bit preservation、unit binding | `gwexpy/timeseries/io/_hdf5_exact_epoch.py:378` から `gwexpy/timeseries/io/_hdf5_exact_epoch.py:709` | `tests/timeseries/test_hdf5_exact_epoch_codec.py:691` から `tests/timeseries/test_hdf5_exact_epoch_codec.py:1235` | 適合 |
| strict v2 sidecar、whole-document validation、bounded schema、diagnostic-only paths | `gwexpy/timeseries/io/_hdf5_exact_epoch.py:91` から `gwexpy/timeseries/io/_hdf5_exact_epoch.py:376` | `tests/timeseries/test_hdf5_exact_epoch_codec.py:77` から `tests/timeseries/test_hdf5_exact_epoch_codec.py:667` | 適合 |
| dataset marker authority、resolved-file sidecar、native GWpy read後のview、crop順序 | `gwexpy/timeseries/io/hdf5.py:1560` から `gwexpy/timeseries/io/hdf5.py:1758` | `tests/timeseries/test_hdf5_exact_t0.py:366` から `tests/timeseries/test_hdf5_exact_t0.py:865`、`tests/timeseries/test_hdf5_exact_t0.py:3496` | 適合 |
| caller metadata policy、marker reset、live-object compaction、v1除去 | `gwexpy/timeseries/io/hdf5.py:217` から `gwexpy/timeseries/io/hdf5.py:302`、`gwexpy/timeseries/io/hdf5.py:439` から `gwexpy/timeseries/io/hdf5.py:688` | `tests/timeseries/test_hdf5_exact_t0.py:1295` から `tests/timeseries/test_hdf5_exact_t0.py:1999`、`tests/timeseries/test_hdf5_exact_t0.py:2305` から `tests/timeseries/test_hdf5_exact_t0.py:3180` | 適合 |
| native path forms、link safety、reload-safe registration | `gwexpy/timeseries/io/hdf5.py:304` から `gwexpy/timeseries/io/hdf5.py:438`、`gwexpy/timeseries/io/hdf5.py:1760` から `gwexpy/timeseries/io/hdf5.py:1909` | `tests/timeseries/test_hdf5_exact_t0.py:3892` から `tests/timeseries/test_hdf5_exact_t0.py:4845`、`tests/timeseries/test_hdf5_exact_t0.py:5256` から `tests/timeseries/test_hdf5_exact_t0.py:5458` | 適合 |
| caller-owned handle identityとsidecar rollback | `gwexpy/timeseries/io/hdf5.py:788` から `gwexpy/timeseries/io/hdf5.py:1156` | `tests/timeseries/test_hdf5_exact_t0_transactions.py:121` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:819` | 適合 |
| pathname one-write、atomic replace、stage cleanup分類、bounded storage | `gwexpy/timeseries/io/hdf5.py:1356` から `gwexpy/timeseries/io/hdf5.py:1443` | `tests/timeseries/test_hdf5_exact_t0_transactions.py:821` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:1319` | 適合 |
| file-like disk-backed backup、bounded copy、byte/position state、cleanup warning | `gwexpy/timeseries/io/hdf5.py:1179` から `gwexpy/timeseries/io/hdf5.py:1307`、`gwexpy/timeseries/io/hdf5.py:1446` から `gwexpy/timeseries/io/hdf5.py:1558` | `tests/timeseries/test_hdf5_exact_t0_transactions.py:1431` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:2357` | 適合 |
| targetごとのnative writer最大1回、disposable stageにrecovery hard linkなし | `gwexpy/timeseries/io/hdf5.py:744` から `gwexpy/timeseries/io/hdf5.py:786` | `tests/timeseries/test_hdf5_exact_t0_transactions.py:821` から `tests/timeseries/test_hdf5_exact_t0_transactions.py:880` | 適合 |

第四巡までに修正対象となったresource invariants、one-write、atomic pathname、bounded file-like、handle identityとsidecar、exact timeの受入条件について、第五巡差分による後退は認められない。

第五巡production差分は pathname cleanup classificationの8行追加と3行置換に限定され、codec、file-like、handle、read authority、registrationの実装は変更していない。

## fresh logの確認結果

指定されたround 5 fresh logsを個別に読み、次のように分類した。

- resource：12 passed in 6.64s
- focused HDF5：797 passed in 60.68s
- official exact claim：1 passed in 0.79s
- surrounding selectors：54 passed in 22.02s。skipは記録されていない
- changed-file Ruff：pass
- changed-file format check：5 files already formatted
- production MyPy：2 source files、issue 0
- repository scoped Ruff：pass
- full-repository Ruff：fail。`docs_redesign/conf.py:242` の既知かつ変更外のD103が1件だけ
- repository MyPy：396 source files、issue 0
- repository pytest：2812 passed、72 skipped、3 xfailed、47 warnings in 136.02s
- diff check：出力なし

full-repository Ruffの既知findingをpassとして扱っていない。

pytestのskipとxfailもpass数へ合算していない。

## 追加検証

現HEADで次を追加実行した。

```bash
rtk conda run -n gwexpy env PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_path_replace_failure_cleans_or_reports_retained_stage -q
```

結果は `4 passed in 0.87s`、exit code 0である。

実行後の `rtk git status --short` は空であり、追加検証によるtracked/untracked fileは生じていない。

## 残余リスク

- 本レビューは指定されたfocused production、focused tests、第五巡diff、fresh logsだけを根拠とする。対象外コードとの未記録の相互作用は評価していない。
- pathname transactionの動作はfilesystemとOSの `os.replace`、unlink、`lstat` の意味に依存する。指定された4つの決定的failure seamは検証済みだが、並行する外部processがstage entryを操作する競合は仕様の受入条件に含まれず、検証していない。
- full-repository Ruffは既知の変更外D103によりcleanではない。P0対象source/test scopeのRuff結果とは分けて扱う必要がある。
- P0最終承認には、planに残る独立code-quality reviewと人間判断が必要である。

## 最終判定

**Approved for specification compliance**

CriticalまたはImportant findingは残っていない。

この結果は人間によるP0最終承認ではない。
