---
name: phase0_exception_sweep
description: "(deprecated) 広域例外の監査・修正。exception-auditor agent に統合されました。"
---

# Phase 0 Exception Sweep — Deprecated

この skill は **exception-auditor agent** に統合されました。

広域例外（`except Exception:` / 裸の `except:` / サイレント失敗）の監査・修正は以下を使用してください:

- **canonical**: `.harness/agents/exception-auditor.md`

## 移行手順

以前このスキルを呼び出していた箇所を `exception-auditor` agent に置き換えてください。
agent はこの skill の全チェックリスト・判断ルール・出力フォーマットを引き継いでいます。
