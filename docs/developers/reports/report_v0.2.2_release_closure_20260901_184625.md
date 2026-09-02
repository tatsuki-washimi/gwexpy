# v0.2.2 リリースclosure作業報告

**記録日時**: 2026-09-02 08:04:27 JST  
**対象version**: `0.2.2`  
**状態**: release closure完了。  
**最終監査branch**: `docs/v022-release-closure-final`  
**基点**: release source R `2503743cf654606a5baa83c7b7e7c8b8e1e06596`

## 結論

v0.2.2の公開payloadは、tag-triggered workflowで検証したbytesとPyPIから取得したbytesが一致し、fresh-install smokeも完了している。

GitHub Release `v0.2.2`を公開し、Zenodo version DOI `10.5281/zenodo.22228340`のmetadataをreadbackした。
公開docsはPR #697でmainへ反映し、GitHub Pagesの英語版と日本語版でrelease note、v0.2.2までのdevelopment activity plot、weekly CSVを確認した。
conda-forgeのbot PR #12はCI通過後にマージされ、公開channelのpackageをPython 3.12の隔離環境で検証した。

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
- PR #697は`c325f6fa0b4ee8de0ca5f6203880504a2af0f9d9`としてmainへマージした。
- GitHub Pages run `33498679298`はsuccessで完了し、gh-pages commitは`202ec1c9cbf5d570b1022d8b593deddd19efece8`である。
- live EN/JA changelog、SVG、CSVのreadbackはpassedである。

## conda-forge

- Feedstock: <https://github.com/conda-forge/gwexpy-feedstock>
- bot PR: [#12](https://github.com/conda-forge/gwexpy-feedstock/pull/12)（`gwexpy v0.2.2`）
- recipe: version `0.2.2`、build number `0`
- source: `https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.2.tar.gz`
- source SHA-256: `3448af15e417187f201f1d910e92fc11e04224607b9cb77849e6d9e172383636`
- CI: package buildとconda-forge-linterがpass
- merge: `2026-09-01T12:31:04Z`、commit `5eea7e9a7d1b15f6a8e625df33595b957ac9e8db`
- channel package: `noarch/gwexpy-0.2.2-pyhc364b38_0.conda`（`2026-09-01T12:33:10.289000Z`）
- fresh smoke: `PYTHONNOUSERSITE=1`のPython `3.12.14`環境でmetadata version、module version、import path、`TimeSeries([1, 2, 3], dt=1).shape == (3,)`を確認

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
- 初回のconda smokeでは親ユーザーsiteの`gwexpy-0.1.14.dist-info`が`importlib.metadata`の探索に混入した。`PYTHONNOUSERSITE=1`で再実行し、conda packageとmoduleがともに`0.2.2`であることを確認した。

これらはrelease source、tag、payload、publication workflowのdefectではない。

## 完了状態

GitHub Release、Zenodo、英語版と日本語版のDocs Pages、conda-forge recipeとchannel package、fresh conda installのすべてを確認した。
本closure作業によるrelease source、tag、公開payloadの変更はない。

## 監査正典

機械可読なrelease closure factsは次に保存した。

`docs/developers/plans/manifests/audit-manifest-v0.2.2-release-closure.yaml`
