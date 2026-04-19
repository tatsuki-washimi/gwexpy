---
name: multi_agent_orchestration
description: tmuxセッションを使ったマルチエージェント並列運用のコーディネーターガイド。GM・コンサルタント・Workerの役割分担、伝令パターン、進捗監視、コンフリクト回避の標準手順を提供する。大量タスクの分散処理・ドキュメント監査・リファクタリング等で有効。
---

# Multi-Agent Orchestration（マルチエージェント並列運用）

複数の AI エージェントを tmux セッションに分散させ、Antigravity がコーディネーターとして各エージェントの伝令・進捗管理を行う運用パターン。

---

## 役割定義

| 役割 | セッション例 | モデル推奨 | 責任範囲 |
|------|-------------|-----------|---------|
| **Coordinator（自分）** | Antigravity | – | 情報集約・伝令・ユーザー報告 |
| **GM（ゼネラルマネージャー）** | <GM-Session-ID> | GPT-4o / GPT-4.5 / Gemini Pro | 計画立案・優先順位決定・最終承認 |
| **Consultant（コンサルタント）** | <Consultant-Session-ID> | Claude Opus / Gemini Pro | 戦略アドバイス・リスク評価（実作業禁止） |
| **Worker** | <Worker-Session-ID>... | GPT-4o-mini / GPT-4 / Gemini Flash | 実際のコード・ドキュメント作業 |

---

## セッション起動パターン

```bash
# 一括起動（セッション構成はタスクに合わせて調整）
for i in <GM-ID> <Consultant-ID> <Worker-ID-Range>; do
    tmux new-session -d -s $i -c /path/to/project
done

# 確認
tmux ls
```

---

## 伝令の基本パターン

### テキスト送信（最重要）

`send-keys` で本文を送ったあと、**必ず別コマンドで `ENTER` を送る**。
1 つのコマンドに `"text" ENTER` を混在させると、モデルが interactive セッションにいる場合に改行扱いで動かないことがある。

```bash
# ステップ1: テキスト入力
tmux send-keys -t 30 "Worker 30, your mission: ..." ""

# ステップ2: 確定
tmux send-keys -t 30 ENTER
```

あるいは、`C-m`（Carriage Return）を使っても良い：

```bash
tmux send-keys -t 30 "Worker 30, your mission: ..." C-m
```

### 状態確認（モニタリング）

```bash
# -S -100 で最大100行向こうから取得
tmux capture-pane -pt 30:0 -S -100

# 全員まとめて確認
for i in 27 28 29 30 31 32; do
    tmux capture-pane -pt $i:0 -S -30
    echo "===$i==="
done
```

### 既存テキストのクリアと再送

```bash
# Ctrl-u でプロンプトをクリアしてから送信
tmux send-keys -t 31 C-u
tmux send-keys -t 31 "Worker 31, CORRECTED mission: ..." C-m
```

---

## 運用フロー

```text
1. タスク分解（Coordinator / GM）
      ↓
2. GM作戦会議：Consultantへ相談
      ↓
3. GMが正式指示を出す
      ↓
4. Coordinatorが各Workerへ伝令
      ↓
5. Coordinatorがtmuxを監視・進捗取得
      ↓
6. Coordinatorが結果をGMへ報告
      ↓ (問題あり → goto 3)
7. GMが最終承認 → リリース作業へ
```

---

## 指示の書き方ガイドライン

### Worker向け指示の必須要素

```text
Worker [番号], [MISSION / GM DECISION]:
1. タスク: 具体的なファイルパスや作業内容
2. 制約: 触ってはいけないファイル、パス
3. 参照資料: [プロジェクト固有の指示書/設計書/要件定義書]
4. 完了条件: コミットメッセージ・報告内容
```

### コンサルタントへの相談の書き方

トークン節約のため、**背景は最小限**に絞る：

```text
Consultant, [テーマ] について以下を教えてください：
- リスク: [具体的な懸念]
- 選択肢: [A] vs [B]
短い箇条書きで回答してください。
```

---

## コンフリクト回避ルール

1. **同一ファイルを同時に複数 Worker が触らない**  
   - タスク分割時にファイルスコープを明確化する
2. **Phase 2（修正）と Phase 3（新規）は書き込み先を分離**  
   - Phase 3 は `docs_internal/.../phase3_prototypes/` に隔離
   - 本番ファイルへの統合は GM 承認後のみ
3. **「コミットが先、新規作業は後」**（コンサルタントの助言）  
   - 未コミット変更が 50 件以上溜まったら即座に Worker に締め指示を出す
4. **linkcheck の外部リンク失敗は NW 制限由来が多い**  
   - release blocker にしない（CI 環境で再確認する旨をメモ）

---

## よくあるエラーと対処

| 症状 | 原因 | 対処 |
|------|------|------|
| テキストが入力されるが実行されない | `ENTER` が送られていない | `tmux send-keys -t N ENTER` を追加送信 |
| `/reviewSOMETHING` と解釈される | テキストと `/` コマンドが連結 | 前のプロンプトをクリア（`C-u`）して再送 |
| `linkcheck` が大量失敗 | sandbox のネットワーク制限 | 外部リンクは non-blocking 扱いにして CI 環境に委ねる |
| `test_hardening` がタイムアウト | fixture 生成やモデル初期化のコスト | 単体テストは `--collect-only` で確認後、CI 環境で実行 |
| Worker が `setup_plan` 等を探す | skill 名を実行ファイルと誤解 | Worker への追加説明「skill 名はコマンドではない」を送信 |

---

## セッション管理

```bash
# 不要セッションの一括削除
for i in 33 34 35; do tmux kill-session -t $i; done

# セッション一覧確認
tmux ls

# 特定セッションにアタッチ（ユーザー向け）
tmux a -t 25

# アタッチ中にデタッチ（セッションは維持したまま抜ける）
Ctrl-b d
```

---

## GM・Consultant の引継ぎ時キーポイント

- GM には **現状のコミット一覧**（`git log --oneline -20`）と **未コミット変更数**（`git status --short | wc -l`）を先に渡す。また、[プロジェクトの全容を記したドキュメント] を把握させる。
- Consultant には **一案件ごとにコンテキストをリセット**し、最小限の背景のみを渡す。
- コンサルタントが実地調査（コード読み取り等）をする際は **ツール/ブラウザ/Codex プラグイン** 等を活用させ、直接の `run_command` による破壊的操作を未然に防ぐ。

---

## 使用するタイミング

- 監査レポートが 100 件以上ある大量タスク
- Phase が複数並行する大規模ドキュメント整備
- 独立したタスクを並列処理して納期を短縮したい時
- 戦略的な技術的判断と実作業員の分離が必要な時
