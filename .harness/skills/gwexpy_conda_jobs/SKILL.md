---
name: gwexpy_conda_jobs
description: gwexpyリポジトリでruff、mypy、pytestを実行する時に使う。condaのgwexpy環境を必須にし、長いジョブはtmuxバックグラウンド実行とログ保存を標準化する
---

# GWexpy Conda Jobs

`gwexpy` では `ruff` / `mypy` / `pytest` を素のシェル環境で直接実行しない。
常に `conda run -n gwexpy ...` を使い、時間のかかるジョブは `tmux` の detached session で走らせる。

## Rules

- `pip install` や `pytest: command not found` を起点に環境構築へ進まない。まず conda 環境を使う。
- `ruff` / `mypy` / `pytest` の実行は、この skill のスクリプト経由を優先する。
- ジョブはログを残す。進行確認は `tmux` かログファイルで行う。
- `pytest` は fixture 再生成でワークツリーを汚すことがある。終了後は `git status --short` を確認する。

## Quick Start

```bash
# detached session で開始
bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start ruff
bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start mypy
bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start pytest tests/test_import_order.py -q

# 一覧
bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh list

# ログ確認
bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh tail <session-name>

# tmux 接続
bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh attach <session-name>
```

## Default Commands

- `ruff`: 引数なしなら `conda run -n gwexpy ruff check .`
- `mypy`: 引数なしなら `conda run -n gwexpy mypy gwexpy`
- `pytest`: 引数なしなら `conda run -n gwexpy pytest`

追加引数を渡した場合は、それを各ツールへそのまま渡す。

## When To Use

- `run_tests` で `pytest` を実行する前
- `lint_check` で `ruff` / `mypy` を実行する前
- `finalize_work --full` 相当で検証をまとめて回す前
- エージェントが `ruff` / `mypy` / `pytest` の存在しない環境にいる可能性がある時

## Outputs

ログは `.agent/tmp/gwexpy_conda_jobs/` に保存する。

- `<session>.log`: stdout / stderr
- `latest-<tool>`: 最後に起動した session 名へのポインタ

## Notes

- `tmux` が無い環境ではこの運用は成立しない。その場合はインストール提案ではなく、ユーザーに確認する。
- 短い単発実行でも conda 環境は維持する。`tmux` を省略するかどうかは状況次第だが、このリポジトリでは基本的に `start` を優先する。
