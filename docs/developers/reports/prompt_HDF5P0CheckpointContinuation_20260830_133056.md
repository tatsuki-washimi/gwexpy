# 次セッション開始用プロンプト

以下の文章を新しいセッションの最初のメッセージとして使用する。

~~~text
既存のローカル作業を Checkpoint A から再開してください。

作業ツリー:
/home/washimi/.config/superpowers/worktrees/gwexpy/v020-post-release-qualification

ブランチ:
test/v020-post-release-qualification

最初に次の文書を全文読んでください。

1. docs/developers/reports/report_HDF5P0CheckpointHandover_20260830_133056.md
2. docs/superpowers/specs/2026-08-29-hdf5-exact-epoch-identity-design.md
3. docs/superpowers/plans/2026-08-29-hdf5-exact-epoch-identity.md

Tasks 1–5 はローカル実装と検証が完了しています。
公式 exact-time claim は元の整数時刻を 0 ns 差で復元します。
HEAD は 209d061f1、文書作成前の worktree は clean、origin/main より 41 commits ahead でした。

P0 は未承認です。
公開、push、tag、release、package upload、release metadata変更、remote workflow dispatch は行わないでください。
P1 bootstrap はこの作業に混ぜないでください。

Checkpoint A の未決事項は、Tasks 6–8 を v0.2.1 に含めるかどうかです。
実装を始める前に、現在残っている次の事実を数行で報告し、Task 6–8 のスコープ承認を一度だけ確認してください。

- native writer が事前確認と実処理で二度呼ばれる
- disposable stage が caller-owned handle 用の recovery link 処理も実行する
- final cleanup が main transaction block の外にある
- file-like write が complete bytes snapshot、BytesIO、getvalue を使う

承認後は Task 6 だけから始めてください。
計画記載の test node ごとに、意図した failing assertion、最小実装、同一 node の pass、既存 node の再確認を順番に行ってください。
Task 6 のローカルコミットと検証が終わるまで Task 7 へ進まないでください。

この作業は HDF5 metadata integrity、write restoration、memory use の改善です。
会話ではこの中立的な表現を使い、長いログはローカルに保存して、進捗はコマンド名、件数、短い原因だけを日本語で報告してください。
並列ワーカーやサブエージェントは使わず、一つの作業セッションで逐次進めてください。

すべての shell command に rtk を付け、Python tooling は conda の gwexpy 環境で実行してください。
既存コミットを reset、amend、squash せず、無関係な変更を保存してください。
~~~

## スコープ承認後に使う短い追記

Task 6–8 を v0.2.1 のローカル修正範囲として承認する場合は、次の一文を続けて送る。

~~~text
Tasks 6–8 を v0.2.1 のローカル修正範囲として承認します。公開操作は行わず、Task 6 の最初のテストから開始してください。
~~~
