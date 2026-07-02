---
name: triage_ci
description: CI（GitHub Actions）が失敗した際、失敗ジョブの特定からローカル再現・修正・再確認までの triage フローとして使用する。
---

# Triage CI Failures

CI パイプラインの失敗を特定し、ローカル環境で再現・修正・再確認する体系的なワークフロー。

## When to Use

- GitHub Actions の CI パイプライン（PR Fast、Extended Nightly など）が失敗したとき
- 失敗ジョブから原因を特定し、ローカルで修正を検証する必要があるとき
- 修正後に CI を再実行して確認するとき

## Workflow

### 1. 失敗の特定（`gh` CLI）

失敗した run を特定し、失敗ジョブのログを取得する:

```bash
# 現在のブランチの最新 5 run を表示（失敗状況を確認）
gh run list --branch <branch> --limit 5

# 失敗 run の詳細とログを表示
gh run view <run-id> --log-failed

# ログが長い場合は tail で末尾 100 行を抽出
gh run view <run-id> --log-failed | tail -100
```

### 2. AI 要約の活用（ci-log-summarizer）

CI 失敗が発生すると、`ci-log-summarizer` ワークフロー（`.github/workflows/ci-log-summarizer.yml`）が自動実行され、エラー抽出を行います:

- **出力先**: GitHub Actions の **Job Step Summary** に記載される
- **内容**: 失敗ログから ERROR / Traceback / FAILED などのパターンを抽出し、ファイル単位で分類
- **注意**: AI 要約は仮説であり、本ログで必ず裏取りする（特に複合エラーが疑われる場合）

Step Summary の確認方法:
- GitHub UI で run 詳細 → `Summarize logs` ジョブの Step Summary タブを確認
- または `gh run view <run-id>` で run 詳細ページの URL を確認し、Web で確認

### 3. ジョブ→gate 対応とローカル再現

失敗ジョブから対応する gate を特定し、ローカルで再現する:

**ジョブと gate の対応表・ローカル再現コマンドについては** [`../fix_errors/reference/ci.md`](../fix_errors/reference/ci.md) **を参照**

例:
- `Ruff, mypy, pytest, smoke build` → `pr-fast` gate
- `Core I/O contract gate` → `io-contract` gate

各 gate の実行方法:
```bash
python scripts/ci/run_gate.py <gate-name>
```

### 4. 修正

エラー種別ごとに対応する fix_errors reference を適用する:

- **mypy（型チェック）エラー** → [`../fix_errors/reference/mypy.md`](../fix_errors/reference/mypy.md)
- **Python 3.9 互換性エラー** → [`../fix_errors/reference/python39.md`](../fix_errors/reference/python39.md)
- **ノートブック実行エラー** → [`../fix_errors/reference/notebooks.md`](../fix_errors/reference/notebooks.md)

### 5. 再確認

修正後、該当 gate を再実行して修正を検証:

```bash
# ローカルで gate を再実行
python scripts/ci/run_gate.py <gate-name>

# push 後、GitHub Actions の run を監視（ブラウザ）
gh run watch <run-id>

# または PR チェック状況を確認
gh pr checks <pr-number>
```

### 6. 記録

修正完了後、`workflows/evidence-pack.md` に従い修正内容を記録する。

## References

- CI gate 一覧・ローカル再現コマンド: [`../fix_errors/reference/ci.md`](../fix_errors/reference/ci.md)
- Audit I/O Backend: [`../audit_io_backends/SKILL.md`](../audit_io_backends/SKILL.md)（I/O 関連 gate 失敗時）
