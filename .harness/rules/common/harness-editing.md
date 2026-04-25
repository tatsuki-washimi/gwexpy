# .harness/ ファイル編集ガイドライン

このドキュメントは `.harness/` 配下のファイルを編集する際のルールを定める。  
方針の背景は `docs/developers/plans/harness_policy_plan.md`（issue #190）を参照。

---

## 追跡対象ファイルの編集ルール

### 絶対パスの禁止

- `.harness/` 配下のファイルに **絶対パスを書いてはならない**。
- パスが必要な場合は `git rev-parse --show-toplevel` や `${BASH_SOURCE[0]}` 相対展開を使う。
- コミット前に `git diff .harness/` で絶対パスが混入していないことを確認する。

```bash
# NG
command: "/home/alice/projects/gwexpy/scripts/run.sh"

# OK
command: "bash -c 'repo=$(git rev-parse --show-toplevel); bash \"$repo/scripts/run.sh\"'"
```

### 個人設定値の禁止

- conda 環境名 `gwexpy` はプロジェクト標準として許容する。
- それ以外の個人固有の値（APIキー、ユーザー名、マシン固有パス）は絶対に書かない。

### hooks.json の編集

- `hooks.json` 内のコマンドは常に **ポータブル**（他の開発者・CI 環境で動作可能）に保つ。
- ローカルでのみ有効にしたい hooks は `.harness.local/hooks/hooks.json` を作成して上書きする（`.gitignore` 対象、コミット不可）。

---

## 個人設定の分離：`.harness.local/`

`.harness.local/` ディレクトリは **gitignore 対象** の個人用上書き領域である。

```
.harness.local/          ← .gitignore 対象、コミット禁止
├── hooks/
│   └── hooks.json       ← .harness/hooks/hooks.json の個人上書き
└── agents/              ← 個人実験用エージェント定義
```

`setup_symlinks.sh` を実行すると、`.harness.local/` 内の該当ディレクトリが `.harness/` より優先してシンボリックリンクされる。

```bash
# セットアップ
bash .harness/scripts/setup_symlinks.sh
```

---

## 新規ファイルの追加ルール

`.harness/` に新しいファイルを追加する場合：

1. **skills/**: `skills/<skill_name>/SKILL.md` 形式で作成し `skills/README.md` の一覧を更新する。
2. **rules/common/**: プロジェクト全体に適用されるルールのみ追加する。個人メモは不可。
3. **agents/**: エージェント定義は汎用的に書き、特定個人のローカル設定を含めない。
4. **workflows/**: チーム全員が実行可能な手順のみ記載する。

---

## チェックリスト（PR 作成前）

- [ ] `.harness/` 配下に絶対パスが含まれていない
- [ ] `.harness/` 配下に個人識別情報（ユーザー名、APIキー）が含まれていない
- [ ] `conda run -n gwexpy` 以外の環境固有値が含まれていない
- [ ] `.harness.local/` のファイルをコミットしていない
