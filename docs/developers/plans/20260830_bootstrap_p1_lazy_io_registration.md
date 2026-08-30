# Bootstrap P1 遅延I/O登録計画

## 目的

plain `import gwexpy`、constructor-only利用、明示的I/O利用の登録契約を分離する。

P0承認済みのHDF5 exact epoch差分には変更を加えない。

Status: planned

## 現状の確認

2026-08-30にP1 worktreeで次をfresh Python processにより確認した。

```text
import gwexpy
_bootstrapped=True
gwexpy.timeseries.io loaded=True
```

トップレベル`gwexpy.__init__`がsubpackageをeager importし、末尾で
`register_all()`を呼ぶ。

`gwexpy.timeseries`と`gwexpy.frequencyseries`はimport時にそれぞれの
`io` packageをimportし、reader、writer、identifierを登録する。

したがって、plain import、constructor-only指定でもI/O登録済み、
on-demand契約違反は同じbootstrap経路による。

## 範囲

含むもの。

- top-level importから暗黙の全登録を除去する。
- constructor登録とI/O handler登録を別のidempotentな操作にする。
- P1対象のpublic I/O入口で必要な登録を一度だけ行う。
- fresh subprocessによるimport stateとregistry stateの契約テストを追加する。
- reloadと反復I/O利用で二重登録、再帰、挙動変化がないことを検証する。

含まないもの。

- HDF5 P0承認済み差分の変更。
- format追加、registry backend統合、公開APIの新設。
- bootstrap P1以外のexact-time機能変更。

## 実装ロードマップ

### Phase 1: 登録境界のテスト

Status: planned

1. `tests/test_import_order.py`へfresh subprocessのRED nodeを追加する。

   - plain `import gwexpy`後にP1で監視するI/O packageが`sys.modules`にない。
   - plain import後に選定したGWpy registry formatが未登録である。
   - constructor-only importとconstructor利用が成功し、I/O未登録を維持する。
   - 明示的な`.read()`または`.write()`が必要なhandlerを登録する。
   - 反復利用と`importlib.reload()`でregistry entry数とhandler identityが安定する。

2. 現在のeager behaviorで各nodeがsemantic REDになることを確認する。

3. registryの監視対象は、TimeSeriesとFrequencySeriesの代表format、および
   `gwexpy.timeseries.io`と`gwexpy.frequencyseries.io`のmodule load stateとする。

### Phase 2: bootstrapとsubpackage importの分離

Status: planned

1. `gwexpy.__init__`からimplicit `register_all()`を除去する。

2. `register_all(include_io=False)`がconstructor登録だけを完了し、
   `include_io=True`がconstructorとI/Oの両方を完了するよう、bootstrap stateを
   一つのbooleanではなく契約に必要な粒度で管理する。

3. `gwexpy.timeseries`と`gwexpy.frequencyseries`のimport時I/O登録を除去する。
   constructor登録とpublic exportは維持する。

4. RED nodesを最小変更でGREENにする。

### Phase 3: explicit I/Oのon-demand登録

Status: planned

1. TimeSeriesとFrequencySeriesのpublic `.read()`／`.write()`が、registry dispatch前に
   対応I/O packageをidempotentにensureするprivate helperを通るようにする。

2. 既存のexplicit format、auto-identify、collection dispatchを対象に、
   registration後に従来と同じregistryを利用することを確認する。

3. import cycleを作らず、constructor-only pathがI/O moduleをimportしないことを
   subprocess nodeで再確認する。

### Phase 4: qualification

Status: planned

1. 新規bootstrap testsと既存のimport order、timeseries I/O、frequencyseries registrationを実行する。

2. `tests/timeseries`、`tests/io/test_gwpy_hdf5_compat.py`、
   `tests/io/test_hdf5_timeseries_family.py`、P0 focused HDF5 selectorを再実行する。

3. changed-file Ruff、format check、MyPy、`git diff --check`を実行する。

4. P1完了後、HDF5 P0を含むfull v0.2.1 qualificationを別工程で実行する。

## 受入条件

- plain `import gwexpy`はP1監視対象のI/O handlerを登録しない。
- constructor-only importとconstructor利用は成功し、I/O未登録を維持する。
- 最初のexplicit I/O利用が必要なregistrationを一度だけ実行する。
- 反復I/O利用とreloadでduplicate registration、recursion、handler identity driftがない。
- 明示format、auto-identify、collection I/Oの既存動作を維持する。
- P0のHDF5 exact-time qualificationは後続full qualificationで再確認する。

## 実行方針と見積り

- 推奨モデル：`gpt-5.6-sol`、high reasoning。
- 推奨スキル：`superpowers:test-driven-development`、`systematic-debugging`、
  `gwexpy_conda_jobs`、`lint_check`、`run_tests`。
- 見積り：90から150分、クオータはHigh。
- 変動要因：GWpy unified I/O descriptorsがon-demand importをどこで解決するか、
  collection classの独自registry dispatch、reload時の外部registry状態。

## 実行制約

- P1は`agent/v020-p1-exact-time` worktreeだけで実装する。
- `test/v020-post-release-qualification`のP0 freezeを変更しない。
- 各behaviorはRED、最小GREEN、周辺回帰、task-level static gateの順で進める。
- merge、push、tag、release、workflow dispatchは実行しない。
