# CI 失敗対応（CI）

> **全体フロー**: gh CLI での失敗 run 特定からローカル再現・修正・再確認の完全な triage ワークフローは [`../../triage_ci/SKILL.md`](../../triage_ci/SKILL.md) を参照してください。

CI 失敗が発生した場合、ローカル環境で問題を再現・修正する方法を記載します。

## 基本フロー

1. **失敗ジョブを特定** - GitHub Actions のジョブ名から対応する gate を確認
2. **ローカルで再現** - `python scripts/ci/run_gate.py <gate>` で当該 gate を実行
3. **原因を分析** - エラーメッセージからパターンを特定（mypy, Python 3.9 互換性など）
4. **修正を適用** - `fix_errors` の各パターン（mypy.md, python39.md など）を参照
5. **検証** - gate を再実行して修正を確認
6. **コミット** - 修正を git commit

## Gate 別ローカル再現コマンド

| Gate | コマンド | 検証内容 |
|------|---------|---------|
| pr-fast | `python scripts/ci/run_gate.py pr-fast` | Ruff (lint), mypy (型), pytest (基本単体), wheel smoke test |
| io-contract | `python scripts/ci/run_gate.py io-contract` | I/O 仕様実装テスト、wheel ビルド・検証 |
| io-conformance | `python scripts/ci/run_gate.py io-conformance` | I/O 仕様適合性テスト |
| io-optional | `python scripts/ci/run_gate.py io-optional` | オプション依存モジュール (netCDF4, TDMS, オーディオ等) |
| io-network-backend | `python scripts/ci/run_gate.py io-network-backend` | ネットワーク・NDS・Kerberos 関連テスト |
| docs-notebook | `python scripts/ci/run_gate.py docs-notebook` | ドキュメントノートブック実行検証 |
| io-zarr | `python scripts/ci/run_gate.py io-zarr` | Zarr 形式リーダーテスト |
| interop-contract | `python scripts/ci/run_gate.py interop-contract` | 相互運用性仕様テスト |

## CI ジョブと Gate の対応

| GitHub Actions ジョブ | Gate | ファイル |
|------------------------|------|---------|
| Ruff, mypy, pytest, smoke build | pr-fast | .github/workflows/pr-fast.yml |
| Core I/O contract gate | io-contract | .github/workflows/pr-fast.yml |
| I/O conformance gate | io-conformance | .github/workflows/pr-fast.yml |
| Optional dependency gate | io-optional | .github/workflows/pr-fast.yml |
| Interop contract gate | interop-contract | .github/workflows/pr-fast.yml |

注: io-network-backend, docs-notebook, io-zarr ジョブは pr-fast.yml で定義されていませんが、run_gate.py でサポートされています。

## 環境前提条件

- **conda 環境**: gwexpy 環境が必要です
  ```bash
  conda run -n gwexpy python scripts/ci/run_gate.py <gate>
  ```
- **依存モジュール**: environment.yml で管理されます
- **フィクスチャ**: 一部 gate は `--fixtures` オプションで自動生成（デフォルト有効）

## トラブルシューティング

- **mypy エラー** → [reference/mypy.md](mypy.md)
- **Python 3.9 互換性エラー** → [reference/python39.md](python39.md)
- **ノートブック実行エラー** → [reference/notebooks.md](notebooks.md)
