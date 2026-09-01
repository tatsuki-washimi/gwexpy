# v0.2.2 リリースclosure作業報告

**記録日時**: 2026-09-01 18:46:25 JST  
**対象version**: `0.2.2`  
**状態**: PyPI publication acceptance済み。GitHub ReleaseとZenodoは完了し、documentation deploymentとconda-forge伝播を継続中。  
**作業branch**: `docs/v022-release-closure`  
**基点**: release source R `2503743cf654606a5baa83c7b7e7c8b8e1e06596`

## 結論

v0.2.2の公開payloadは、tag-triggered workflowで検証したbytesとPyPIから取得したbytesが一致し、fresh-install smokeも完了している。

GitHub Release `v0.2.2`を公開し、Zenodo version DOI `10.5281/zenodo.22228340`のmetadataをreadbackした。
公開docsについては、英語版と日本語版のrelease note、v0.2.2までのdevelopment activity plot、weekly CSVを専用branchへ追加した。

conda-forgeはbot PR生成待ちである。
release直後のpolicyに従い、24時間以内は手動feedstock更新へ切り替えず監視する。

## 不変のrelease identity

| 項目 | 値 |
| --- | --- |
| reviewed source S′ | `8e023fdd355190399b1c1aab9c919c11b278aca9` |
| released source R | `2503743cf654606a5baa83c7b7e7c8b8e1e06596` |
| tag | `v0.2.2` |
| annotated tag object | `83b1916537214a4446d76d3773c57c35bb3cd6a5` |
| tagger UTC | `2026-09-01T08:45:28Z` |
| tag workflow | `33488802448` |
| wheel SHA-256 | `64e517b906366d30b96560e1149f39fa343b8e24be977bdb18dbc40868c38126` |
| sdist SHA-256 | `3448af15e417187f201f1d910e92fc11e04224607b9cb77849e6d9e172383636` |

本closure作業ではR、tag、公開payloadを変更していない。
runtime codeとscientific behaviorも変更していない。

## 完了した外部channel

### GitHub Release

- Release ID: `380369377`
- URL: <https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.2>
- 公開日時: `2026-09-01T09:29:42Z`
- 状態: latest、draftではない、prereleaseではない
- release bodyへPyPI qualification、最終digest、Zenodo DOIを記録

### Zenodo

- Record ID: `22228340`
- Version DOI: `10.5281/zenodo.22228340`
- Concept DOI: `10.5281/zenodo.19059422`
- Version: `0.2.2`
- Publication date: `2026-09-01`
- Resource type: software
- Access: open
- Archive: `tatsuki-washimi/gwexpy-v0.2.2.zip`
- Archive MD5: `76a89b2fc10cbc069640eb7c8c15397c`

## Documentation更新

- `docs/web/en/user_guide/changelog.md`へ英語release noteを追加した。
- `docs/web/ja/user_guide/changelog.md`へ日本語release noteを追加した。
- `release_notes/v0.2.2.md`をpublication後の事実とZenodo DOIへ更新した。
- `docs_redesign/about/changelog.md`のdevelopment activityをv0.2.2へ更新した。
- v0.2.2 tagまでの1,755 non-merge commitsからSVGとweekly CSVを再生成した。
- weekly CSV SHA-256は`cd72102029af78b05dcb002051365092f128df341125164e4ba7c8a96d4203e3`である。
- 日本語gettext catalogの変更messageは空翻訳・fuzzyを残していない。

## ローカル検証

次の検証は完了した。

- release facts focused tests: 9 passed
- docs tests: 179 passed、4 skipped
- Ruff check: passed
- Ruff format check: passed
- terminology check: passed
- JA/EN synchronization: heading count一致。既知のJA-only 3件、EN-only 4件をwarningとして維持
- forbidden generated artifacts: passed
- EN/JA quickstart: 各2 code block passed
- legacy docs strict Sphinx build: passed
- docs_redesign EN/JA build: notebook実行を無効化したlocal smokeが既知warning baselineでpassed
- rendered HTML readback: EN/JA release link、Zenodo DOI、SVG、CSVを確認
- `git diff --check`: passed

## 操作上の注記

- GitHub CLI readbackで未対応の`isLatest` fieldを一度指定した。REST readbackで正式状態を確認した。
- 初回Zenodo helperは旧schemaの`upload_type`を参照した。current schemaによる再readbackは成功した。
- bare `sphinx-build`がconda環境外のexecutableを解決したため失敗した。`python -m sphinx`へ固定して再実行した。
- full-site link checkには既存の無関係なbroken linkとZenodo/DOIの一時timeoutが含まれた。GitHub Release、PyPI、Zenodoのrelease identityはそれぞれのauthoritative readbackで検証した。
- 重複して起動した旧disposable Sphinx buildを、対象PIDを確認した上でTERM終了した。source treeとtracked filesへの影響はない。
- canonical public pageへのrelease link追加を決めた時点で、変更前sourceを読むfull notebook buildを中断した。最終local buildは`nb_execution_mode=off`で完了し、isolated notebook executionはPR CIの責務とした。

これらはrelease source、tag、payload、publication workflowのdefectではない。

## 未完了項目

1. documentation PRのCIを完了し、mainへ反映する。
2. Docs Pages deploymentを監視し、EN/JA release note、SVG、CSVをlive readbackする。
3. conda-forge bot PRでversion、source URL、sdist SHA-256を確認する。
4. conda-forge CI、merge、channel availability、fresh-environment smokeを確認する。
5. すべて完了後、closure manifestを最終状態へ更新する。

conda-forge bot PRが公開後24時間以内に生成されない場合は、その時点でmanual follow-upを別途判断する。

## 監査正典

機械可読なrelease closure factsは次に保存した。

`docs/developers/plans/manifests/audit-manifest-v0.2.2-release-closure.yaml`
