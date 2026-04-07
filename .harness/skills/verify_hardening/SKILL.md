---
name: verify_hardening
description: 英語一次（Non-ASCII検知）、Sphinx厳格ビルド、Doctest の3要素を検証し、リポジトリの堅牢な状態を維持する
---

# Verify Hardening

リポジトリが「堅牢化（Hardened）」された状態、すなわち英語一次ポリシーが守られ、ドキュメントが正しくビルドされ、docstring 内の例が動作する状態であることを検証します。

## Quick Start

```bash
# 全ての堅牢化チェックを実行
bash .agent/skills/verify_hardening/scripts/verify_hardening.sh

# 特定の項目のみ実行（オプション）
# 非 ASCII チェックのみ
python scripts/check_non_ascii.py --root gwexpy

# Doctest のみ
conda run -n gwexpy pytest --doctest-modules gwexpy/

# Sphinx 厳格ビルドのみ
conda run -n gwexpy sphinx-build -b html -W docs docs/_build/html
```

## Checks

### 1. 英語一次ポリシー（Non-ASCII チェック）

`gwexpy/` ディレクトリ配下に CJK（日本語等）文字が含まれていないかを確認します。開発レポート等は `docs/` 配下であっても、ライブラリ本体やユーザードキュメントへの混入を防ぎます。

### 2. Sphinx 厳格ビルド (`-W`)

ドキュメントのビルド中に警告が発生した場合、それをエラーとして扱いビルドを失敗させます。リンク切れやタイポによるレンダリング不備を未然に防ぎます。

### 3. Doctest (`--doctest-modules`)

`gwexpy` パッケージ内の docstring に含まれる `>>>` で始まるコード例が、現在の実装で正しく動作するかを動的に検証します。

## When to use

- ファイルをコミット・プッシュする前の最終確認
- `finalize_work --full` の実行時（自動的に含まれます）
- 大規模なリファクタリング後のドキュメント・サンプル整合性確認

## Troubleshooting

### Doctest が失敗する

- サンプルコードに必要なインポート（`import numpy as np` 等）が含まれているか確認してください。
- インスタンス化が必要な場合は、最小限erデータで自己完結するように記述してください。

### 非 ASCII 文字が検出される

- `scripts/check_non_ascii.py` の出力結果を確認し、該当する日本語コメント等を英語に翻訳してください。
- 意図的に日本語を残す必要があるドキュメントは、`docs/web/ja/` 下に配置することでスキャンから除外されます。
