---
name: session-start
description: セッション開始時の環境・ブランチ・依存関係チェック。作業前の事故を防ぐための Preflight Doctor。
trigger: manual
---

# Session Start Doctor

セッションを開始する前に、以下の項目を確認してください。

## When to Use
- 新しいタスクを開始する時
- 環境（Conda / Python）を切り替えた時
- 長い休憩の後に作業を再開する時

## Preflight Checks

以下のコマンドを1つ実行するだけで、環境チェックをまとめて行えます。

```bash
python scripts/preflight_doctor.py
```

`[PASS]` / `[WARN]` / `[FAIL]` の一覧と件数サマリが表示されます。FAIL が1つでもあれば exit code 1 を返します。

### オプション

| オプション | 説明 |
|---|---|
| `--env gwexpy` | 使用する conda 環境名（デフォルト: `gwexpy`） |
| `--skip-smoke` | `gwexpy.register_all()` の smoke チェックをスキップ（importに時間がかかる環境向け） |
| `--json` | machine-readable な JSON 形式で出力する |

### conda が利用できない場合（degraded モード）

`conda` コマンドが見つからない環境では、ツールチェックを現行インタープリタで実行します（WARN が表示されます）。
この状態でも ruff / mypy / pytest の存在確認は機能します。

```bash
# conda なし環境での例
python scripts/preflight_doctor.py --skip-smoke
```

## Common Failures
- **Python インタープリタの不一致**: `which python` が Conda 環境内を指していない。
- **保存忘れ**: `git diff` で意図しない変更が残っている。
- **エージェント設定の不整合**: `.agent/` と `.harness/` の両方が存在し、参照先や運用前提がずれていないか確認。

## Important Reminders
- **`gwexpy/fields/` への変更**: 物理的影響が大きいため、必ず `physics-reviewer` エージェントを使用し、完了時に Human Review を要請してください。
- **新規依存関係**: `pyproject.toml` に追加する場合は、`optional-deps-reviewer` エージェントで影響を確認してください。

## Expected Output
全てのチェックが完了したら、`setup_plan` スキルでタスクの設計に進んでください。
