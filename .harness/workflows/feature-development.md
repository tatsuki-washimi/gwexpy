---
name: feature-development
description: GWexpy 新機能開発の標準ワークフロー。setup_plan → TDD実装 → lint/type → 物理検証 → PR作成の順で進める。
trigger: manual
---

# GWexpy 機能開発ワークフロー

## ステップ 1: 計画（setup_plan スキル）

```
/setup_plan
```

- `docs/developers/plans/` に計画書を作成
- 過去の計画・設計決定を確認
- 物理的影響範囲を特定

## ステップ 2: 実装（TDD）

```
/run_tests  # 既存テストがグリーンであることを確認
```

- テストを先に書く（RED）
- 最小限の実装でグリーンにする
- `gwexpy.register_all()` を確認
- `astropy.units` を全量で使用

## ステップ 3: 静的解析（gwexpy-linter エージェント）

```bash
conda run -n gwexpy ruff check --fix gwexpy/ tests/
conda run -n gwexpy ruff format gwexpy/ tests/
conda run -n gwexpy mypy gwexpy/
```

## ステップ 4: テスト実行（gwexpy-tester エージェント）

```bash
conda run -n gwexpy pytest tests/ -m "not gui and not nds and not cvmfs"
```

GUI 変更がある場合:
```bash
bash tests/run_gui_tests.sh
```

## ステップ 5: 物理検証（physics-reviewer エージェント）

`gwexpy/fields/`, `gwexpy/signal/`, `gwexpy/spectrogram/` の変更時は必須：

```
/verify_physics
```

- CRITICAL エラーはマージ禁止
- NEEDS-HUMAN-REVIEW の場合は PR に `needs-physics-review` ラベルを付ける

## ステップ 6: PR作成（finalize_work スキル）

```
/finalize_work
```

PR タイトル形式: `[AGENT:<skill>] <説明>`

必須チェックリスト:
- [ ] ruff clean
- [ ] mypy clean
- [ ] pytest PASS
- [ ] 物理検証完了（該当する場合）
- [ ] `needs-physics-review` ラベル付与（該当する場合）
