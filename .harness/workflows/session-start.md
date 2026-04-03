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

以下のコマンドを実行して、環境が正しくセットアップされているか確認します。

1. **Conda 環境の確認**
   ```bash
   conda info --envs | grep "*"
   ```
   *推奨環境名: `gwexpy` または `gwex-env`*

2. **依存ツールの存在確認**
   ```bash
   conda run -n gwexpy ruff --version
   conda run -n gwexpy mypy --version
   conda run -n gwexpy pytest --version
   ```

3. **Git ステータスの確認**
   ```bash
   git status --short
   ```
   *未コミットの変更がある場合は、既存タスクとの混同を避けるためコミットまたはスタッシュを検討してください。*

4. **レジストリ初期化の前提**
   `gwexpy` の機能を使う前に、`.py` ファイルの冒頭やテスト環境で以下が実行されることを確認してください。
   ```python
   import gwexpy
   gwexpy.register_all()
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
