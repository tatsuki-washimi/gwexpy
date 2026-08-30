# HDF5 exact-epoch P0 仕様レビュー

## 判定

**P0 specification verdict: Changes required.**

Critical は 0 件、Important は 2 件、Minor は 0 件である。

正常系の dataset-local marker、v2 sidecar authority、alias と copy、native path、pathname staging、および native writer 最大 1 回の実装は、確認した仕様とテストに整合している。

一方、file-like target の transaction envelope 外で caller position を変更する経路と、verified recovery の再作成失敗時に不完全な group を recovery artifact として報告する経路が残っている。

このため、独立仕様レビューとして P0 適合とは判定できない。

この判定は人間による最終 P0 承認ではない。

## レビュー対象

- Branch：`test/v020-post-release-qualification`
- Reviewed implementation head：`c6fc46d9a4a44cb5e9bfeab00551fca2a51c77c0`
- Comparison：`origin/main...c6fc46d9a`
- Specification：`docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md`
- Canonical plan：`docs/superpowers/plans/2026-08-29-hdf5-exact-epoch-identity.md`
- Focused implementation：指定された production 2 ファイルと test 3 ファイル
- Test evidence：指定された fresh local logs 7 件、および下記の read-only 再現

既存の `docs/developers/reviews/`、`docs/developers/reports/`、会話ログはレビュー根拠として読んでいない。

## Findings

### Important 1：file-like の private-namespace preflight が transaction envelope 外で caller position を破壊し得る

**仕様節**：`Transaction architecture / Validation`、`File-like transaction`、`Error policy`、`Test-driven implementation order / 3. Transaction atomicity`

**該当箇所**：`gwexpy/timeseries/io/hdf5.py:359`、`gwexpy/timeseries/io/hdf5.py:368`、`gwexpy/timeseries/io/hdf5.py:1766`

`write_exact` は `_write_filelike_transaction` に入る前に `_reject_private_namespace` を呼ぶ。

`append=True` の file-like target では、この preflight が caller object を `h5py.File` で開き、最後に `target.seek(position)` で位置を戻している。

この復元 `seek` が失敗すると、file-like transaction の `_RollbackError` 分類を通らず、生の例外が返る。

隔離一時データを使った read-only 再現では、旧 bytes は維持されたが caller position は `7` から `1192` に変化し、生の `OSError` が返った。

例外には `state`、`byte_state`、`position_state`、`recovery_path` がなかった。

これは「最初の target seek から最終 position 確立までを envelope に含める」という仕様、および byte state と position state を独立分類する契約に違反する。

既存の `tests/timeseries/test_hdf5_exact_t0_transactions.py:1224` は `_write_disposable_stage` 内の失敗後に position が戻ることを検証しているが、それより前の private-namespace preflight には故障を注入していない。

**Observable impact**：caller は write 失敗後に bytes が旧状態であることは確認できても、file position が変更された事実と状態分類を例外メタデータから判断できない。

さらに、preflight 本来の operation error が `finally` の position restore error に置き換わる場合がある。

**RED-first test node**：`tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_filelike_private_preflight_position_failure_is_classified`

この node では、private-resolution seam を失敗させ、caller の original-position `seek` も失敗させる file-like object を用いる。

修正前には生の `OSError` と unclassified position を観測し、修正後には original operation を保持した `_RollbackError`、`byte_state="old"`、`position_state="indeterminate"`、および仕様どおりの durable backup path を検証する。

**最小修正案**：file-like target の既存 image に対する private resolution を `_reject_private_namespace` で行わない。

root component の構文検査だけを mutation 前に残し、既存 image の resolution は、durable backup 作成後に working file を開く `_write_disposable_stage` 内の再検査へ一本化する。

これにより caller object に対する最初の `seek` が `_write_filelike_transaction` の envelope 内に入る。

### Important 2：recovery 再作成の部分失敗時に、未検証の空 group を indeterminate recovery path として報告する

**仕様節**：`Caller-owned open-container transaction`、`Error policy`、`Acceptance criteria`

**該当箇所**：`gwexpy/timeseries/io/hdf5.py:965`、`gwexpy/timeseries/io/hdf5.py:973`、`gwexpy/timeseries/io/hdf5.py:1130`、`gwexpy/timeseries/io/hdf5.py:1143`

rollback-link deletion が link を削除してから例外を送出した場合、`_remove_or_recreate_recovery` は `_prepare_handle_recovery` で verified recovery を再作成する。

再作成が group 作成後に失敗し、その partial group の cleanup も失敗すると、内側の `_RollbackError.recovery_path` は partial group の場所を表す。

現在の `_remove_or_recreate_recovery` は、その path を complete recovery であるかのように outer rollback へ返している。

続く public dataset restore も失敗すると、outer `_RollbackError` は `state="indeterminate"` と非 `None` の `recovery_path` を報告する。

隔離一時 HDF5 file でこの複合故障を注入したところ、報告された path は reopen 前にも存在したが、group の children は空で、old dataset link と sidecar snapshots を保持していなかった。

public `data` link も失われていた。

これは、artifact persistence 自体が失敗した indeterminate state では `recovery_path=None` を明示するという仕様に違反する。

既存の `tests/timeseries/test_hdf5_exact_t0_transactions.py:458` は recovery 再作成 failure と public restore failure を組み合わせているが、再作成時の partial group cleanup failure を組み合わせていない。

**Observable impact**：運用者は `_RollbackError.recovery_path` を old dataset の復旧場所として扱うが、実際には空の partial group であり、そこから旧 dataset と sidecar state を復元できない。

**RED-first test node**：`tests/timeseries/test_hdf5_exact_t0_transactions.py::test_hdf5_handle_partial_recreation_cleanup_and_public_restore_failure_reports_no_recovery_path`

この node では、commit cleanup を delete-then-raise とし、二回目の recovery group create を group 作成後に失敗させ、その partial cleanup と public relink も失敗させる。

修正前には空 group の path が返ることを RED とし、修正後には `state="indeterminate"`、`recovery_path is None`、recreation failure、partial cleanup failure、および public restore failure の全件が observable metadata に残ることを検証する。

**最小修正案**：`_remove_or_recreate_recovery` は `_prepare_handle_recovery` が正常に返した場合だけ `recreated.path` を durable recovery path として返す。

再作成中の `_RollbackError.recovery_path` は未検証 partial cleanup location なので、outer authority path へ昇格させず、nested rollback failure metadata にだけ保持する。

その後の public restoration が不完全なら、outer `_RollbackError` は `state="indeterminate"`、`recovery_path=None` とする。

## 仕様対応の確認結果

| レビュー範囲 | 確認結果 | 主な根拠 |
|---|---|---|
| dataset-local marker と v2 sidecar authority | 適合 | marker を唯一の exact authority とし、sidecar 全文を dataset 選択前に検証している。marker-only recovery、conflicting record、stale path、v1 fallback 禁止の test がある。 |
| aliases、moves、copies、external links | 適合 | hard alias、soft alias、move、同一 file と cross-file H5Ocopy、without-attrs copy、resolved external file authority を個別に検証している。 |
| path validation | 適合 | relative と absolute の `str`/UTF-8 `bytes` を元 object のまま native handler へ渡し、NUL、invalid UTF-8、empty component、`.`、`..` を mutation 前に拒否している。leaf SoftLink と external traversal の拒否も検証している。 |
| native writer 最大 1 回 | 適合 | pathname、file-like、File、Group の success と post-write failure について call count 1 を検証している。core-driver preflight は production code に存在しない。 |
| pathname atomic stage と拒否条件 | 適合 | `lstat` による non-regular と multiple-hard-link 拒否、sibling stage、append copy、fresh overwrite、mode 復元、replace failure と cleanup failure の分類を確認した。 |
| file-like bounded transaction | 不適合 | disk-backed backup/working、bounded copy、commit/rollback、cleanup warning は実装されているが、Important 1 の pre-envelope caller access が残る。 |
| caller-owned handle recovery | 不適合 | old object identity、sidecar snapshot、delete-before/after-raise、reopen recovery、成功時 cleanup は検証されているが、Important 2 の partial recreation path 分類が残る。 |
| `_RollbackError` observable metadata | 不適合 | 通常の failure matrix では operation error、rollback errors、state、recovery path、byte/position state を公開している。Important 1 では構造化例外に到達せず、Important 2 では path の意味が誤る。 |
| private recovery object bounds | 適合。ただし Important 2 の path authority を除く | ordinary success で private link なし、incomplete rollback で recovery object 最大 1 個を検証している。再現した partial group も 1 個だが、verified artifact ではない。 |
| acceptance tests の主張 | 不適合 | 広い正常系と主要な failure seam は実データ、object address、refs、sidecar raw stateまで確認している。しかし上記二つの複合 failure boundary が未検証で、現実の仕様違反を見逃している。 |

## Fresh log の確認

- Focused pytest：`780 passed in 59.05s`。skip、xfail、xpass の表示はない。
- Opt-in qualification claim：`1 passed in 0.80s`。
- Surrounding selectors：`54 passed, 1 skipped in 21.67s`。
- Surrounding の skip は read-only 再実行の `-rs` で `tests/qualification/test_v020_release_claims.py:32: post-release qualification is opt-in` と確認した。
- Focused Ruff：`All checks passed!`
- Source/test Ruff：`All checks passed!`
- Format check：`5 files already formatted`
- Changed production MyPy：`Success: no issues found in 2 source files`

fresh logs に記録されていない full-repository MyPy、docs build、physics/maintainer review は pass と推定していない。

## 追加した read-only 検証

- File-like preflight position-restore failure の一時 `BytesIO` 再現：旧 bytes 維持、position `7 -> 1192`、生の `OSError`、state metadata なし。
- Caller-owned handle の partial recovery recreation cleanup failure と public restore failure の一時 HDF5 再現：`state="indeterminate"`、非 `None` path、path は children を持たない空 group、public dataset なし。
- 隣接する既存 test 2 node：`2 passed in 0.87s`。
- Surrounding selectors の skip reason 再確認：`54 passed, 1 skipped in 21.56s`。
- `git diff origin/main...c6fc46d9a --check`：問題なし。

すべて一時メモリまたは一時 directory を使った read-only reproduction であり、implementation、tests、specification、canonical plan は変更していない。
