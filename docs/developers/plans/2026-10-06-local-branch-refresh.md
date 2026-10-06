# ローカルbranchと作業基盤の更新

作成日：2026-10-06

Status: completed (verified: python3 -B <local-recovery>/finalize.py verify)

## Objectives & Goals

公開v0.2.5とv0.3.0 Core/Field候補を保持し、古いローカル参照と混在した作業状態を整理する。
この記録はローカル管理の完了を表し、候補の採用、公開PR、mergeやreleaseの承認を表さない。

## Detailed Roadmap

### 復旧保管

Status: completed (verified: python3 -B <local-recovery>/refresh.py prepare)

旧roadmap checkoutのHEAD、index、staged/unstaged patch、untrackedと非再生成ignoredファイルを外部archiveへ保存した。
Git bundleから独立repositoryを作り、28,565ファイルの内容とmode、statusとindexを復元照合した。
旧branchのreflogとGit設定も保存した。

### ローカル参照と作業基盤

Status: completed (verified: python3 -B <local-recovery>/finalize.py verify)

mainを`fd37e1cb9674ef218c3ebd88f92928905d64ba56`、maint/0.1を`42eec70450b867f10b7a9331c3a0217ce589c564`へfast-forwardした。
現在の作業基盤はmainから作成した`work/current-20261006`である。
旧roadmap branchと8件の休止untracked計画/監査ファイルは復元可能な保管へ移した。
Core/Fieldのactive plan、deliverablesと稼働中索引用ファイルは元のパスに保持した。
CodeGraphのSQLite索引とdaemon logはcheckout変更に伴って再生成されるキャッシュとして記録した。
その旧内容もarchiveに保存し、復元照合済みである。

### Docs試験計画

Status: completed (verified: python3 -B <local-recovery>/docs_update.py validate)

PR #717の最新remote headへローカルbranchを同期した後、2つの計画/監査ファイルを更新したlocal commitを作成した。
試験baselineは公開build-infoから確認した`3439fba41dcc644af12870d91bee794524098fb4`、パッケージはv0.2.5である。
過去のbaselineと検証記録を保持し、四名の実ユーザー試験は未実施とした。
このローカル更新はremote PRへpushしていない。

### 旧変更とFFLの判断

Status: completed (verified: python3 -B <local-recovery>/probes.py)

旧lazy import案はimport後のconstructor登録に差が出るため、現mainのon-demand登録実装との再評価候補として保管し、今回採用しない。
現mainはplain importでregistry全体をbootstrapすることを要求しない。この差だけで現行の公開契約違反とは断定しない。
rolling案はforced Bottleneckのsilent fallbackを修正する独立候補として保管した。今回の作業基盤へは混入させない。
文書、Notebook、翻訳と依存条件の旧変更は保管済み、採用未判定とする。
FFL修正は現mainに未統合であり、現行GWF処理とparallel/cache契約への再適合が必要である。
local branchとPR #625を保持し、実装採用を保留する。

## Testing & Verification Plan

旧rootの全記録対象ファイル、mode、差分、status、indexの復元照合を実行した。
Core/Field/baselineのsource IDと全記録対象ファイル、既存証跡、過去archiveのチェックサムを照合した。
残存refs、tag、remote-tracking refs、stashと実際のorigin refsを照合した。
Docs計画のJSON、fence、四名の定義、baseline、履歴保持と差分のwhitespaceを検証した。
旧案のregistry/rolling挙動は隔離checkoutで限定probeを行った。
製品変更の採用を行っていないためfull suiteは再実行していない。

## Models, Skills, and Effort

既存sessionのCodexモデルを使用し、単一エージェントで実行した。
setup_plan、using-git-worktrees、docs-ja、verification-before-completionを使用した。
隔離には一時的な独立repositoryを用い、登録worktreeを増やしていない。
見積りは30から60分、クォータ消費は中程度である。

## Remaining Work

Status: planned

rollingと最低依存条件の修正候補を別の限定変更として再評価する。
FFLを現mainへ移植し、serialとparallelの入力契約およびGWF回帰を検証する。
Docsの四名試験を実施し、source SHAごとの匿名結果を記録する。
Core性能受容、Field human adoption、真正producer fixture、solver統合と既存レビュー残件は引き続き未完了である。
外部復旧保管が唯一の保存先である間は保持する。
