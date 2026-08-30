# 作業報告: HDF5 P0 Checkpoint A 引継ぎ

**日時**: 2026-08-30 13:30:56 JST
**モデル**: Codex（GPT-5）
**状態**: ⏳ 継続中。Task 1–5 は完了し、Task 6–8 のスコープ判断を待っている。
**ブランチ**: `test/v020-post-release-qualification`
**worktree**: `/home/washimi/.config/superpowers/worktrees/gwexpy/v020-post-release-qualification`
**本文書作成前の HEAD**: `209d061f1 fix: preserve native HDF5 paths across reloads`
**本文書作成前のリポジトリ状態**: clean、`origin/main` より 41 commits ahead
**リソース記録**: 複数セッションをまたいだ総経過時間とトークン使用量は未計測。
**公開状態**: push、tag、release、package upload、conda-forge更新、Zenodo操作、remote workflow dispatchは実施していない。

## 現在の結論

v0.2.0 で確認された HDF5 の**厳密時刻（exact epoch）**の直接的な不具合は、ローカルで修正済みである。
公式の opt-in claim は元の整数時刻を 0 ns 差で復元し、Task 1–5 は marker authority、strict v2 sidecar、path-independent lineage、GWpy compatibility、native path forms、reload behavior を検証している。

P0 は未承認である。
Checkpoint A では、現在の transaction wrapper が native writer を二度呼ぶこと、最終 cleanup が success envelope の外にあること、file-like image 全体を Python memory に複製することを確認した。
承認済み計画は、Task 6–8 を v0.2.1 に含める前に明示的なスコープ判断を要求している。

本文書は `report_Phase1QualificationHandover_20260829_171704.md` の作業状態を更新する。
旧レポートは Phase 1 全体の背景と、別作業である P1 bootstrap の参照資料として残す。

## 完了したローカル作業

### Task 1: canonical epoch marker

private marker codec は native binary64 値を保ったまま、exact integer epoch と lineage information を保持する。
境界値、単位、canonical form、checksum、長さ、seed固定randomの各テストはコミット済みである。

### Task 2: strict sidecar v2

v2 JSON document は、文書全体、cross-field consistency、record/path limits、canonical ordering、bounded serialization を検証する。
version 1 metadata は exact-time authority を与えない。

### Task 3: GWpy HDF5 integration

HDF5 read は保存済みの native GWpy reader を先に呼び、その結果を GWexpy type として view した後、marker と sidecar の検証が通った場合だけ exact authority を付与する。
write は dataset-local marker を維持し、live v2 document を再構築する。
metadata handling は scalar text normalization、fixed-string padding、invalid document rejection、private name reservation、iterative HDF5 traversal、current metadata mutation の rollback を含む。

### Task 4: path-independent identity

exact authority は保存済みpathではなくdataset markerに追従する。
テストは hard-link alias、local soft-link read、move、same-file copy、cross-file copy、attribute-free copy、stale v2 record、同じ時刻を持つ独立dataset、public floating coordinate が fresh conversion と 1 ULP 異なる exact slice を扱う。

### Task 5: native paths and reload behavior

wrapper は、安全な relative/absolute `str` と UTF-8 `bytes` path を GWpy に渡すとき、元のpath objectを保持する。
external link と soft link の write case を検証し、対応済みの GWpy behavior は変更しない。
module reload を繰り返しても、保存した native handler を復元し、wrapper layer を増やさない。

### 主なコミット

- `209d061f1`: native HDF5 path と reload behavior を修正。
- `ff62d5c80`: path-independent exact epoch のテストを追加。
- `ffb6c4f50`: exact epoch metadata validation を補強。
- `e9bf84c3e`: nonscalar exact epoch metadata を拒否。
- `d40823c04`: exact epoch marker と HDF5 I/O を統合。
- `7664bdb78` から `9c7b9d52f`: marker codec、strict v2 sidecar、各task記録。

## 検証結果

次の結果は、現在の Task 1–5 実装で取得した。

| 検証 | 結果 |
| --- | --- |
| 公式 opt-in exact-time claim | 1 passed。復元値は `1234567890123456789`、差は 0 ns |
| opt-in環境変数なしの Checkpoint A combined command | 705 passed、1 skipped。skipは意図的にopt-inとなっているqualification node |
| Task 5後のfull exact HDF5 integration file | 590 passed |
| marker/sidecar codec file | 115 passed |
| Task 5 focused path/link/reload selection | 106 passed |
| Task 4 identity selection | 30 passed |
| existing HDF5 layout/NDScope selection | 63 passed、1 skipped。local sample fixture不在による既存skip |
| Ruff changed-file checks | passed |
| Ruff format check | passed |
| `hdf5.py` の MyPy | passed |
| `git diff --check` | passed |

公式 claim を明示的に実行したコマンドは次のとおりである。

~~~bash
rtk conda run -n gwexpy env GWEXPY_POST_RELEASE_QUALIFICATION=1 pytest tests/qualification/test_v020_release_claims.py::test_timeseries_hdf5_roundtrip_retains_exact_t0_gps_ns -vv
~~~

## Checkpoint A の確認事項

Task 1–5 は直接的な exact-time defect を閉じている。
一方で、旧 Phase 1 report の未解決事項4と5に対応する transaction requirements は残っている。

現在のsourceには次の4点が残る。

1. `_preflight_core_write` が native writer を呼び、その後 `_write_open_container` がもう一度呼ぶ。
2. pathname と file-like の staging が `_write_open_container` を使うため、disposable stage でも internal recovery link を作る。
3. 最後の `_delete_rollback` は main transaction block の外にあり、新しいdatasetが見える状態になってからcleanup exceptionを報告し得る。
4. file-like write は complete byte snapshot を保持し、`BytesIO(snapshot)` と `getvalue()` を使う。

該当箇所は `gwexpy/timeseries/io/hdf5.py` の次の定義付近である。

- `_preflight_core_write`
- `_write_open_container`
- `_filelike_snapshot`
- `_write_path_transaction`
- `_write_filelike_transaction`

Task 6–8 のsource/test変更は未着手である。
`tests/timeseries/test_hdf5_exact_t0_transactions.py` はまだ存在しない。

## 未決のスコープ判断

implementation plan は Task 6–8 を conditionally approved とし、Checkpoint A で停止するよう定めている。
次セッションは production code を編集する前に、これらを v0.2.1 に含めるか確認する。

旧 handover は transaction cleanup と file-like memory の項目を未解決の P0 requirement としているため、既存の記録は Task 6–8 を含める判断を支持する。
最終判断はユーザーが行う。

含める場合は、次の順序で進める。

1. **Task 6**: one-native-write tests を追加し、disposable staging と caller-owned handle recovery を分離して、atomic pathname staging を実装する。
2. **Task 7**: file-like全体のmemory copyをdisk-backed temporary filesとbounded chunk copyingへ置き換える。
3. **Task 8**: original HDF5 object identityを保ちながら、caller-owned File/Groupのrecovery setup、restoration、final cleanupを明示する。
4. **Task 9**: focused resource/compatibility qualificationを実行する。
5. **Task 10**: static gatesを実行し、specification reviewとquality reviewを別セッションで受ける。

各taskは、記載済みtestとsurrounding regressionがpassしてから別々にcommitする。

## 参照文書

- 現在の引継ぎ: `docs/developers/reports/report_HDF5P0CheckpointHandover_20260830_133056.md`
- 旧 Phase 1 report: `docs/developers/reports/report_Phase1QualificationHandover_20260829_171704.md`
- 承認済みdesign: `docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md`
- 承認済みimplementation plan: `docs/superpowers/plans/2026-08-29-hdf5-exact-epoch-identity.md`
- 次セッション用prompt: `docs/developers/reports/prompt_HDF5P0CheckpointContinuation_20260830_133056.md`

## 次セッションの制約

- 既存worktreeだけを使い、branch historyを維持する。
- existing commitsをreset、amend、squash、rewriteしない。
- push、tag、publish、package upload、release metadata edit、remote workflow dispatchを行わない。
- P1 bootstrapとon-demand registrationをこのHDF5 seriesに混ぜない。
- すべてのshell commandに`rtk`を付け、Python toolingは`gwexpy` conda environmentで実行する。
- 一つのtest nodeごとに、test追加、意図したassertion failure、最小実装、同じnodeのpass、既存green nodeの再確認を行う。
- unrelated user changeを保持する。
- 継続セッションでは一つのcoding workerだけを使う。
  final independent reviewは、並列workerではなく後続の別review sessionとして実施する。

## 会話表現

この作業は HDF5 metadata integrity、write restoration、memory use の改善である。
チャット更新ではこの中立的な説明を使う。
長いdiagnostic logはlocal fileに保存し、チャットではcommand、result count、短い原因だけを報告する。
過度な比喩や本作業と関係のない用語は避ける。
これにより、通常のdata-format maintenance作業中にinterfaceが会話を中断する可能性を下げる。

## 次セッションの最初の操作

1. 本report、approved design、planのCheckpoint AとTask 6を読む。
2. `rtk git status --short --branch` を実行し、HEADが`209d061f1`であることを確認する。
   handover filesが未commitなら、この2ファイルだけがuntrackedであることも確認する。
3. Checkpoint A の結果を数行で報告し、Task 6–8 のスコープ判断を一度だけ求める。
4. 承認された場合は `superpowers:executing-plans`、`superpowers:test-driven-development`、`gwexpy_conda_jobs` を使い、Task 6 だけを開始する。

## 次モデルへの引継ぎ指示

元の +165/+166 ns reproduction ではなく、Checkpoint A から再開する。
direct exact-epoch contractは0 ns差でpassしている。
新しいregression failureが具体的に示されない限り、Task 1–5をやり直さない。

source edit前に、Task 6–8をv0.2.1へ含めるか確認する。
承認後はwritten planを逐次実行し、progress messageを短く中立的に保つ。
remaining local gatesとhuman reviewが完了するまで、P0は未承認のままとする。
