# .harness/ 増強計画

**作成日**: 2026-04-03  
**ステータス**: ドラフト（ブレインストーミング結果）  
**目的**: gwexpy プロジェクト固有の知識・過去資産・既知の課題を `.harness/` に体系的に注入し、AI エージェントとの協働品質を向上させる。

---

## 現状の .harness/ 構成

| カテゴリ | 現在の数 | 内容 |
|---------|---------|------|
| hooks   | 4 hooks | ruff check, ruff format, physics reminder (Bash), Stop: fields 警告 |
| agents  | 3       | physics-reviewer, gwexpy-tester, gwexpy-linter |
| workflows | 2     | feature-development, release |
| rules   | 2       | physics.md, testing.md |
| skills  | 34      | 各種スキル（充実） |

---

## フェーズA: Hooks 強化

> **実装方針**: 全 hook は `.harness/hooks/hooks.json` の `hooks` オブジェクトに追記する。  
> ファイルパス抽出ロジックは既存の ruff hooks と同一パターンを流用する。  
> 既存 4 hook の順序は変えない。

### フェーズA の作業前チェック

- 対象ファイルは `.harness/hooks/hooks.json` のみとし、Phase A では Python コード本体は変更しない。
- 既存 `PostToolUse` の 3 エントリと `Stop` の 1 エントリの並び順を記録し、追加後も既存 4 hook の相対順序を維持する。
- `pyproject.toml` の `[tool.mypy] exclude` を事前確認し、mypy hook のスキップ条件を実設定
  `gwexpy/gui/reference-dtt/.*|gwexpy/gui/reference_ndscope/.*|tests/.*`
  と一致させる。
- 追加する hook はすべて既存の「JSON stdin から `tool_input.file_path` / `tool_input.path` を取り出す」パターンを流用し、新しい抽出ロジックは導入しない。
- 各追加後に必ず `python3 -m json.tool .harness/hooks/hooks.json >/dev/null` で JSON 構文を確認し、最後にまとめて手動動作確認を行う。

### フェーズA の完了条件

- `PostToolUse` に A-1 から A-4 の 4 hook、`Stop` に A-5 の 1 hook が追加されている。
- 既存の ruff lint / ruff format / physics reminder / fields warning hook の順序と挙動が変わっていない。
- A-2, A-3, A-4 は対象パターンで警告し、コメント行や非対象ファイルでは誤検知しない。
- A-1 は `tests/` と GUI 参照実装をスキップしつつ、対象 `.py` 編集時のみ `mypy` を非同期実行する。
- A-5 は Python 変更があるのに `CHANGELOG.md` が未更新な場合のみ Stop 時に表示される。
- 実装メモとして、実際に確認したコマンドと結果をこの計画に従って別途作業ログへ転記できる状態になっている。

---

### A-1. mypy PostToolUse Hook

**概要**: `.py` ファイル編集後に `mypy` をファイル単位で非同期実行する。  
**目的**: `AGENTS.md` Section 3 に必須コマンドとして記載されているにも関わらず hook が存在しない。157 件の mypy エラー（`mypy_remedy_strategy.md` 参照）の再発を防ぐ。  
**参考**:
- `.harness/hooks/hooks.json`（既存 ruff hooks のパターンを流用）
- `docs_internal/analysis/mypy_remedy_strategy.md`

#### 実装詳細

**配置**: `PostToolUse` セクション（既存 ruff-format hook の直後）  
**async**: `true` / **timeout**: `60`（mypy は ruff より重いため 60s）  
**除外**: `pyproject.toml` の `[tool.mypy] exclude` に従い `tests/` と GUI 参照実装は自動スキップ

**hooks.json エントリ**（`PostToolUse[].hooks[]` に追加）:

```json
{
  "matcher": "Edit|Write",
  "hooks": [
    {
      "type": "command",
      "command": "bash -c 'input=$(cat); file=$(printf \"%s\" \"$input\" | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get(\\\"tool_input\\\",{}).get(\\\"file_path\\\") or d.get(\\\"tool_input\\\",{}).get(\\\"path\\\",\\\"\\\"))\" 2>/dev/null); [[ \"$file\" == *.py ]] || exit 0; [[ \"$file\" == */gwexpy/gui/reference* ]] && exit 0; result=$(conda run -n gwexpy mypy \"$file\" 2>&1); if echo \"$result\" | grep -q \"error:\"; then printf \"\\n[gwexpy/mypy] %s\\n%s\\n\" \"$file\" \"$result\" >&2; fi; exit 0'",
      "async": true,
      "timeout": 60
    }
  ],
  "description": "Python ファイル編集後に mypy をファイル単位で非同期実行（gwexpy conda 環境）"
}
```

**テスト手順**:
1. 任意の `.py` ファイルに型エラーを含む行を一時追加して Edit
2. 数秒後に stderr に `[gwexpy/mypy]` プレフィックスのエラーが表示されることを確認
3. `tests/` 配下のファイルを Edit しても mypy が起動しないことを確認（`pyproject.toml` の除外設定が効いているか）

---

### A-2. `except Exception` 検出 Hook

**概要**: `.py` 編集後、`except Exception:` または裸の `except:` パターンを検出して警告する。  
**目的**: `phase0_exception_analysis.md` で特定された 17 箇所の問題の再発防止。サイレントな失敗を防ぐ。  
**参考**:
- `docs_internal/analysis/phase0_exception_analysis.md`
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`（AST スキャン手法）

#### 実装詳細

**配置**: `PostToolUse` セクション（A-1 mypy hook の直後）  
**async**: `true` / **timeout**: `10`（grep のみなので軽量）  
**検出パターン**: `except Exception[:(]` および `except:` — コメント行は除外

**hooks.json エントリ**:

```json
{
  "matcher": "Edit|Write",
  "hooks": [
    {
      "type": "command",
      "command": "bash -c 'input=$(cat); file=$(printf \"%s\" \"$input\" | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get(\\\"tool_input\\\",{}).get(\\\"file_path\\\") or d.get(\\\"tool_input\\\",{}).get(\\\"path\\\",\\\"\\\"))\" 2>/dev/null); [[ \"$file\" == *.py ]] || exit 0; result=$(grep -nE \"^[[:space:]]*(except[[:space:]]+Exception[[:space:]]*[:(]|except[[:space:]]*:)\" \"$file\" 2>/dev/null); if [ -n \"$result\" ]; then printf \"\\n[gwexpy/except-check] サイレント失敗の可能性 (%s):\\n%s\\n  → 具体的な例外型を指定し、logger.exception() でログを記録してください。\\n  参考: docs_internal/analysis/phase0_exception_analysis.md\\n\" \"$file\" \"$result\" >&2; fi; exit 0'",
      "async": true,
      "timeout": 10
    }
  ],
  "description": "Python ファイル編集後に except Exception / 裸の except を検出して警告"
}
```

**テスト手順**:
1. 任意の `.py` に `except Exception:` を含む行を追加して Edit
2. `[gwexpy/except-check]` 警告と行番号が stderr に出ることを確認
3. `except ValueError:` など具体的な例外型では警告が出ないことを確認

---

### A-3. eps/tol ハードコード検出 Hook

**概要**: `.py` 編集後、`eps=1e-`, `tol=1e-` などの魔法の数値定数を検出して警告する。  
**目的**: GW 歪みスケール（10^-21）に対して不適切な定数（例: `eps=1e-12`）の混入を防ぐ「Death Floats」問題の再発防止。  
**参考**:
- `docs_internal/analysis/phase1_dangerous_defaults.md`
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`

#### 実装詳細

**配置**: `PostToolUse` セクション（A-2 except-check hook の直後）  
**async**: `true` / **timeout**: `10`  
**検出パターン**: `(eps|tol|atol|rtol)\s*=\s*1e-[0-9]+` — `gwexpy.numerics` 経由の代入・コメント行は除外  
**スコープ**: `gwexpy/` 配下のみ（`tests/` も警告対象に含める ─ テストの前提値が間違っている場合もあるため）

**hooks.json エントリ**:

```json
{
  "matcher": "Edit|Write",
  "hooks": [
    {
      "type": "command",
      "command": "bash -c 'input=$(cat); file=$(printf \"%s\" \"$input\" | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get(\\\"tool_input\\\",{}).get(\\\"file_path\\\") or d.get(\\\"tool_input\\\",{}).get(\\\"path\\\",\\\"\\\"))\" 2>/dev/null); [[ \"$file\" == *.py ]] || exit 0; result=$(grep -nE \"(eps|tol|atol|rtol)\\\\s*=\\\\s*1e-[0-9]+\" \"$file\" 2>/dev/null | grep -v \"^[[:space:]]*#\" | grep -v \"gwexpy\\.numerics\"); if [ -n \"$result\" ]; then printf \"\\n[gwexpy/death-floats] Death Floats 候補 (%s):\\n%s\\n  GW strain スケール (~1e-21) に対して不適切な定数の可能性があります。\\n  → gwexpy.numerics の定数を使用するか、スケールを明示したコメントを追加してください。\\n  参考: docs_internal/analysis/phase1_dangerous_defaults.md\\n\" \"$file\" \"$result\" >&2; fi; exit 0'",
      "async": true,
      "timeout": 10
    }
  ],
  "description": "Python ファイル編集後に eps/tol 等のハードコード数値定数（Death Floats）を検出して警告"
}
```

**テスト手順**:
1. `gwexpy/` 配下の `.py` に `eps=1e-12` を追加して Edit
2. `[gwexpy/death-floats]` 警告と行番号が stderr に出ることを確認
3. `# eps=1e-12 は GW strain スケールに合わせた値` のようなコメント行では警告が出ないことを確認

---

### A-4. GWpy 4.0 非互換 API 検出 Hook

**概要**: `.py` 編集後、GWpy 3.x 系の廃止 API（`nproc=`, `gwpy.io.mp`, `gwpy.utils.gprint` 等）を検出して移行を促す。  
**目的**: GWpy 4.0 への移行でブレーキングチェンジが多数発生することが判明している。早期発見。  
**参考**:
- `docs_internal/tech_notes/research/GWpy4_deep-research-report.md`

#### 実装詳細

**配置**: `PostToolUse` セクション（A-3 death-floats hook の直後）  
**async**: `true` / **timeout**: `10`  
**検出パターン**（`GWpy4_deep-research-report.md` 記載の主要廃止 API）:

| パターン | 廃止理由 |
|---------|---------|
| `nproc\s*=` | `nproc` 引数は 4.0 で削除（`multiprocessing` 直接使用に変更） |
| `gwpy\.io\.mp` | `gwpy.io.mp` モジュール廃止 |
| `gwpy\.utils\.gprint` | `gprint` 関数廃止 |
| `\.fetch\(.*nproc` | `TimeSeries.fetch()` の `nproc` 引数廃止 |
| `from gwpy\.utils import.*gprint` | 同上 |

**hooks.json エントリ**:

```json
{
  "matcher": "Edit|Write",
  "hooks": [
    {
      "type": "command",
      "command": "bash -c 'input=$(cat); file=$(printf \"%s\" \"$input\" | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get(\\\"tool_input\\\",{}).get(\\\"file_path\\\") or d.get(\\\"tool_input\\\",{}).get(\\\"path\\\",\\\"\\\"))\" 2>/dev/null); [[ \"$file\" == *.py ]] || exit 0; result=$(grep -nE \"(nproc\\\\s*=|gwpy\\\\.io\\\\.mp|gwpy\\\\.utils\\\\.gprint|from gwpy\\\\.utils import.*gprint|\\.fetch\\\\(.*nproc)\" \"$file\" 2>/dev/null | grep -v \"^[[:space:]]*#\"); if [ -n \"$result\" ]; then printf \"\\n[gwexpy/gwpy4-compat] GWpy 4.0 廃止 API 候補 (%s):\\n%s\\n  → docs_internal/tech_notes/research/GWpy4_deep-research-report.md を参照して移行してください。\\n\" \"$file\" \"$result\" >&2; fi; exit 0'",
      "async": true,
      "timeout": 10
    }
  ],
  "description": "Python ファイル編集後に GWpy 4.0 廃止 API（nproc=, gwpy.io.mp 等）を検出して移行を促す"
}
```

**テスト手順**:
1. 任意の `.py` に `nproc=4` を含む行を追加して Edit
2. `[gwexpy/gwpy4-compat]` 警告が stderr に出ることを確認
3. コメントアウト行 `# nproc=4` では警告が出ないことを確認

---

### A-5. Stop: CHANGELOG 更新リマインダー

**概要**: セッション終了時に、変更があれば `CHANGELOG.md` の更新を促す。  
**目的**: リリース準備 (`roadmap_20260403.md`) で CHANGELOG 日付ズレが問題になっていた。習慣的な更新を促す。  
**参考**:
- `docs_internal/analysis/roadmap_20260403.md`
- `.harness/hooks/hooks.json`（既存 Stop hook のパターンを流用）

#### 実装詳細

**配置**: `Stop` セクション（既存 fields 警告 hook の直後）  
**async**: `false` / **timeout**: `15`  
**トリガー条件**: `git diff` / `git ls-files --others` で `.py` ファイルへの変更があり、かつ `CHANGELOG.md` が変更されていない場合

**hooks.json エントリ**（`Stop[].hooks[]` に追加）:

```json
{
  "matcher": "*",
  "hooks": [
    {
      "type": "command",
      "command": "bash -c 'repo=\"/home/washimi/work/gwexpy\"; [ -d \"$repo/.git\" ] || exit 0; changed=$(git -C \"$repo\" diff --name-only HEAD 2>/dev/null; git -C \"$repo\" diff --name-only 2>/dev/null; git -C \"$repo\" ls-files --others --exclude-standard 2>/dev/null); py_changed=$(echo \"$changed\" | grep -E \"\\.py$\" | sort -u | head -1); changelog_changed=$(echo \"$changed\" | grep -E \"^CHANGELOG\"); if [ -n \"$py_changed\" ] && [ -z \"$changelog_changed\" ]; then printf \"\\n[gwexpy/changelog-reminder]\\n  Python ファイルが変更されましたが CHANGELOG.md が更新されていません。\\n  リリースに向けて変更内容を記録することを検討してください。\\n  ヒント: /changelog-generator スキルが使えます。\\n\" >&2; fi; exit 0'",
      "async": false,
      "timeout": 15
    }
  ],
  "description": "Python ファイル変更時に CHANGELOG.md 更新を Stop 時に促す"
}
```

**テスト手順**:
1. 任意の `.py` ファイルを Edit した後、セッションを終了（Stop イベント発火）
2. `CHANGELOG.md` を変更していない状態で `[gwexpy/changelog-reminder]` が表示されることを確認
3. `CHANGELOG.md` も変更した状態では警告が出ないことを確認

---

### フェーズA 具体的な作業計画

| Task | 対応 | 編集対象 | 実施内容 | 検証 | 完了条件 |
|------|------|----------|----------|------|----------|
| 0 | 事前確認 | `.harness/hooks/hooks.json`, `pyproject.toml` | 既存 hook の並び順と `tool.mypy.exclude` を確認し、A-1 の除外条件を固定する | `python3 -m json.tool .harness/hooks/hooks.json >/dev/null` | 追加位置とスキップ条件が曖昧でない |
| 1 | A-5 | `.harness/hooks/hooks.json` の `Stop` | `fields` 警告 hook の直後に CHANGELOG reminder を追加する | Stop 発火時に `[gwexpy/changelog-reminder]` を確認 | Python 変更あり / CHANGELOG 未更新時のみ表示される |
| 2 | A-2 | `.harness/hooks/hooks.json` の `PostToolUse` | `except Exception` / 裸の `except` 検出 hook を追加する | 対象コード追加で `[gwexpy/except-check]` を確認 | 具体的な例外型では警告しない |
| 3 | A-3 | `.harness/hooks/hooks.json` の `PostToolUse` | `eps`, `tol`, `atol`, `rtol` のハードコード検出 hook を追加する | `eps=1e-12` 追加で `[gwexpy/death-floats]` を確認 | コメント行と `gwexpy.numerics` 経由文字列で誤検知しない |
| 4 | A-4 | `.harness/hooks/hooks.json` の `PostToolUse` | GWpy 4.0 廃止 API 検出 hook を追加する | `nproc=4` 追加で `[gwexpy/gwpy4-compat]` を確認 | コメント行は除外される |
| 5 | A-1 | `.harness/hooks/hooks.json` の `PostToolUse` | ruff-format hook の直後に mypy hook を追加する | 型エラー混入時に `[gwexpy/mypy]` を確認 | `tests/` と GUI 参照実装では起動しない |
| 6 | 仕上げ | `.harness/hooks/hooks.json` | JSON 妥当性、hook 順序、主要ケースの再確認を行う | `python3 -m json.tool .harness/hooks/hooks.json >/dev/null` | Phase A の 5 hook が計画通り並び、全検証結果を記録できる |

### フェーズA 実施チェックリスト

- [ ] Task 0: 既存 `PostToolUse` / `Stop` の順序を控え、A-1 の除外条件を `pyproject.toml` と突き合わせる。
- [ ] Task 1: A-5 を `Stop` セクションの既存 fields warning hook の直後に追加する。
- [ ] Task 1-Verify: Python ファイル変更あり・`CHANGELOG.md` 未変更で Stop 時に警告が出ることを確認する。
- [ ] Task 2: A-2 を `PostToolUse` の末尾に追加する。
- [ ] Task 2-Verify: `except Exception:` では警告が出て、`except ValueError:` では警告が出ないことを確認する。
- [ ] Task 3: A-3 を A-2 の直後に追加する。
- [ ] Task 3-Verify: `eps=1e-12` で警告が出て、コメント行では警告が出ないことを確認する。
- [ ] Task 4: A-4 を A-3 の直後に追加する。
- [ ] Task 4-Verify: `nproc=4` で警告が出て、コメントアウト行では警告が出ないことを確認する。
- [ ] Task 5: A-1 を既存 ruff-format hook の直後に追加する。
- [ ] Task 5-Verify: 型エラーを含む対象 `.py` の編集で警告が出て、`tests/` と `gwexpy/gui/reference-*` ではスキップされることを確認する。
- [ ] Task 6: `python3 -m json.tool .harness/hooks/hooks.json >/dev/null` を実行して JSON 妥当性を確認する。
- [ ] Task 6-Verify: 最終的な hook 順序が「既存 4 hook を維持しつつ A-1〜A-4, A-5 を追加」の形になっていることを目視確認する。

### 推奨実装順と理由

1. A-5 を先に入れる。`Stop` 側の単純な条件分岐で、失敗時の切り戻しが最も容易。
2. A-2, A-3, A-4 をこの順で入れる。いずれも `grep` ベースで軽く、警告系 hook として並べて確認しやすい。
3. A-1 を最後に入れる。`conda run -n gwexpy mypy` が最も重く、除外条件の調整も必要なため、軽量 hook の安定後に追加する。
4. 最後に JSON 構文確認と代表ケースの再実行をまとめて実施する。

### 想定所要時間

- 事前確認: 5〜10 分
- A-5, A-2, A-3, A-4 の追加と確認: 15〜25 分
- A-1 の追加と確認: 10〜20 分
- 最終確認と作業ログ整理: 5〜10 分
- 合計目安: 35〜65 分

### リスクと切り戻し方針

- `hooks.json` は単一ファイル集中変更なので、1 hook 追加ごとに JSON 構文確認を挟んで壊れた位置を即座に特定する。
- A-1 で `mypy` の誤爆や遅延が目立つ場合は、まず除外条件を `pyproject.toml` と再突合し、それでも解決しなければ A-1 のみ一時的に外して他 hook を先に確定する。
- A-2〜A-4 で誤検知が多い場合は、検出パターンを広げる前に「コメント行除外」「対象パス限定」「既知の許容パターン除外」の順で条件を絞る。
- Phase A では挙動改善よりも安全な警告導入を優先し、自動修正系 hook は追加しない。

---

## フェーズB: Rules 整備

> **実装方針**: 追加先はすべて `.harness/rules/common/` 配下の Markdown とする。  
> 既存の `physics.md` / `testing.md` と同じく、短い導入文 + セクション見出し + 箇条書き/コード例の構成に揃える。  
> フェーズBでは rule 文書の追加・更新に限定し、hook や agent 定義には手を入れない。

### フェーズB の作業前チェック

- 現在の `.harness/rules/common/` は `physics.md` と `testing.md` の 2 ファイルのみであることを確認し、新規作成 4 ファイルの配置先を固定する。
- 各 rule は「プロジェクト固有ルール」であり、分析レポートの要約コピーではなく、実装時に使う判断基準・禁止事項・推奨パターンに変換する。
- 既存 rule と同じく front matter は追加せず、`# タイトル` と短い説明文から始める。
- 参照元ドキュメントごとに、rule へ転記する内容を「禁止事項」「推奨パターン」「例」「レビュー条件」に分解して整理する。
- フェーズB完了時には、各 rule が他の rule や分析資料への参照リンクを持ち、単独で読んでも実務判断できる粒度になっていることを確認する。

### フェーズB の完了条件

- `.harness/rules/common/exception-handling.md`, `.harness/rules/common/numerical-scales.md`, `.harness/rules/common/model-assignment.md`, `.harness/rules/common/gwpy-compatibility.md` の 4 ファイルが作成されている。
- 4 ファイルとも、既存 `physics.md` / `testing.md` と整合する文体・見出し粒度になっている。
- B-1 は `except Exception` / 裸の `except` の禁止と、許容条件・ログ義務・レビュー条件を明記している。
- B-2 は GW スケール前提の定数運用、`gwexpy.numerics` 利用方針、HHT/STLT 系の特記事項を明記している。
- B-3 はモデル選択を「役割ベース」で記述しつつ、参照元の実績に基づく具体例を持っている。
- B-4 は GWpy 4.0 で壊れやすい API パターン、推奨移行先、必要テストを明記している。
- 参照元の古い分析内容をそのまま複製せず、rule として運用可能な短さと判断性に圧縮できている。

### B-1. `rules/common/exception-handling.md`

**概要**: プロジェクト固有の例外処理規則。`except Exception` 禁止、許容する例外型のホワイトリスト、ログ出力の必須化を定める。  
**目的**: phase0 分析で 17 箇所のサイレント失敗を確認。再発を規則として防ぐ。  
**参考**:
- `docs_internal/analysis/phase0_exception_analysis.md`
- `docs_internal/archive/reviews/conversation_report_20260203_180637.md`

#### 実装詳細

**作成**: `.harness/rules/common/exception-handling.md`  
**ベース**: `phase0_exception_analysis.md` の remediation strategy を rule 形式へ変換  
**必須セクション**:

- `# GWexpy Exception Handling Rules`
- 導入文
  この rule が silent failure 防止用であること、`physics.md` や `testing.md` を補完する位置づけを 2 行程度で記載
- `## Forbidden Patterns`
  `except Exception:`, 裸の `except:`, `pass` のみ、ログなしの握りつぶしを禁止
- `## Allowed Narrow Exceptions`
  Collection accessor, batch processing, IO fallback, GUI/plotting safety の代表ケースごとに許容パターンを整理
- `## Logging Requirements`
  `logger.warning(..., exc_info=True)` / `logger.exception(...)` / `warnings.warn(...)` の使い分けを明記
- `## Review Triggers`
  `gwexpy/io/`, `gwexpy/*/collections.py`, GUI フォールバックなど人手確認が必要な場面を列挙
- `## Examples`
  bad / better の短いコード例を載せる

**記載ルール**:
- 「広域例外は原則禁止、例外的に維持する場合は理由とログが必須」を最上位ルールにする
- `KeyError` と計算失敗を混同しないことを明記する
- GUI 系で broad exception を残す場合でも、無言で握りつぶさないことを明記する

---

### B-2. `rules/common/numerical-scales.md`

**概要**: GW 固有の数値スケール規則。魔法の数値禁止・`gwexpy.numerics` モジュール経由の定数使用・STLT の σ overflow 防止パターンを規則化する。  
**目的**: GW 歪み（10^-21）に対して eps/tol が 9 桁以上ズレていた「Death Floats」問題の再発防止。HHT・STLT のアルゴリズム固有パラメータ指針も含む。  
**参考**:
- `docs_internal/analysis/phase1_dangerous_defaults.md`
- `docs_internal/tech_notes/implementation/hht_implementation_notes_20260204.md`（EMD ε=0.2〜0.3）
- `docs_internal/tech_notes/implementation/stlt_implementation_notes_20260204.md`（σ overflow 対策）

#### 実装詳細

**作成**: `.harness/rules/common/numerical-scales.md`  
**ベース**: `phase1_dangerous_defaults.md` と HHT/STLT 実装メモを運用ルールへ要約  
**必須セクション**:

- `# GWexpy Numerical Scale Rules`
- 導入文
  GW strain が `~1e-21` スケールであること、固定 eps/tol の誤りが物理信号を潰すことを簡潔に記載
- `## Forbidden Magic Numbers`
  `eps=1e-12`, `tol=1e-4` のようなスケール無視定数を禁止例として示す
- `## Preferred Sources of Truth`
  `gwexpy.numerics` の定数・ヘルパーを最優先に使う方針を書く
- `## Scale-Aware Defaults`
  `None` / `'auto'` / data-relative scaling を推奨する原則を書く
- `## Algorithm-Specific Notes`
  HHT/EMD、STLT、whitening など、特記事項が必要な系だけ短く整理する
- `## Testing Expectations`
  スケール不変性テスト、極小振幅入力、NaN/Inf ガードの確認観点を書く
- `## Review Triggers`
  数値しきい値を新設・変更する場合のレビュー条件を列挙

**記載ルール**:
- 単なる「小さい値を使うな」ではなく、「データスケールに対する相対性を持たせる」が主ルールになるように書く
- `gwexpy.numerics` が未整備の箇所でも、新規の魔法定数追加を正当化しない方針を明記する
- テストでも危険なデフォルト値を固定しないことを明記する

---

### B-3. `rules/common/model-assignment.md`

**概要**: タスク種別に応じたモデル選択ガイドライン（Opus/Sonnet/Haiku の使い分け）。  
**目的**: 過去のリリース作業で実証済みのモデル割り当てパターンを明文化し、毎回の試行錯誤を削減する。  
**参考**:
- `docs_internal/archive/plans/model_assignment_v0.1.0b1.md`（実績記録）
- グローバルルール `rules/common/performance.md`（Haiku/Sonnet/Opus の基本方針）

#### 実装詳細

**作成**: `.harness/rules/common/model-assignment.md`  
**ベース**: `model_assignment_v0.1.0b1.md` の実績を、現在の harness で使いやすい役割ベース規則へ抽象化  
**必須セクション**:

- `# GWexpy Model Assignment Rules`
- 導入文
  モデル名の暗記ではなく、タスク特性に応じた割り当て判断を標準化する rule であることを記載
- `## Task Categories`
  例: 最終レビュー、ドキュメント整備、機械的コーディング、CI/統合調整、実行検証
- `## Preferred Assignment Patterns`
  各カテゴリに対して「高推論レビュー型」「バランス型」「コーディング特化型」「実行特化型」などの役割を書く
- `## Historical Examples`
  `model_assignment_v0.1.0b1.md` の実績を 3〜5 件の具体例で抜粋する
- `## Escalation Rules`
  物理判断・リリース GO/NO-GO・設計判断は高推論モデルへ寄せる条件を書く
- `## Anti-Patterns`
  単純作業に最高コストモデルを常用しない、レビューなしで実行役に最終判断をさせない、などを明記

**記載ルール**:
- モデルの固有名だけに依存しないよう、「役割」と「そのときの代表モデル例」を併記する
- この rule 単体で、誰に振るか迷ったときの初期判断ができるようにする
- グローバル `performance.md` と競合しないよう、「gwexpy での補足ルール」に留める

---

### B-4. `rules/common/gwpy-compatibility.md`

**概要**: GWpy 4.0 移行パターン集。廃止 API・新 API・移行コード例を規則として整理する。  
**目的**: `GWpy4_deep-research-report.md` で調査済みの破壊的変更を、実装時に参照できる形で保存する。  
**参考**:
- `docs_internal/tech_notes/research/GWpy4_deep-research-report.md`

#### 実装詳細

**作成**: `.harness/rules/common/gwpy-compatibility.md`  
**ベース**: `GWpy4_deep-research-report.md` の「壊れやすい API と推奨移行先」を実装ルール化  
**必須セクション**:

- `# GWexpy GWpy Compatibility Rules`
- 導入文
  GWpy 4.0 での破壊的変更が多く、互換性判断を都度調査しないための rule であることを記載
- `## Banned or Legacy Patterns`
  `gwpy.io.mp`, `nproc=`, 旧 registry import, 廃止 verbose/gprint 系を列挙
- `## Preferred Patterns`
  `default_registry`, `parallel=`, 標準ライブラリ並列化、現行 API への移行先を書く
- `## Compatibility Strategy`
  両対応が必要なときの条件分岐方針、または GWpy4 前提に寄せる判断基準を簡潔に整理する
- `## Required Tests`
  I/O 登録、読み込み、`parallel` 引数、GWpy 依存の public API 変更時に追加すべきテストを明記する
- `## Migration Notes`
  ドキュメント・CHANGELOG・リリースノートへ反映すべき観点を箇条書きで残す

**記載ルール**:
- ルール文書では「報告書の全文再掲」ではなく、禁止 API と推奨 API の対応表を中心にする
- I/O registry と `nproc -> parallel` を最重要項目として先頭に置く
- GWpy 互換性変更時はテスト追加を必須条件にする

---

### フェーズB 具体的な作業計画

| Task | 対応 | 編集対象 | 実施内容 | 検証 | 完了条件 |
|------|------|----------|----------|------|----------|
| 0 | 事前確認 | `.harness/rules/common/`, `docs_internal/analysis/*`, 既存 rule | 既存 rule の書式と、新規 4 rule の責務分割を固定する | 既存 `physics.md` / `testing.md` を読み、重複しないことを確認 | 各 rule の担当領域が曖昧でない |
| 1 | B-1 | `.harness/rules/common/exception-handling.md` | 例外処理規則を新規作成する | 見出し・禁止事項・許容条件・例が揃っていることを確認 | silent failure 防止ルールとして単独で読める |
| 2 | B-2 | `.harness/rules/common/numerical-scales.md` | 数値スケール規則を新規作成する | 魔法定数禁止、`gwexpy.numerics`、テスト期待が揃っていることを確認 | GW スケール前提の判断軸が明示されている |
| 3 | B-3 | `.harness/rules/common/model-assignment.md` | モデル割り当て規則を新規作成する | 役割分類と実績例が揃っていることを確認 | モデル名ではなく役割ベースで判断できる |
| 4 | B-4 | `.harness/rules/common/gwpy-compatibility.md` | GWpy 互換性規則を新規作成する | 禁止 API、推奨 API、必要テストが揃っていることを確認 | 互換性変更時の判断と検証が明示されている |
| 5 | 仕上げ | `.harness/rules/common/*.md` | 4 rule の相互参照と文体整合を調整する | 見出し粒度と cross-reference を確認 | `.harness/rules/common/` が一貫した rule 群になっている |

### フェーズB 実施チェックリスト

- [ ] Task 0: 既存 `physics.md` / `testing.md` の構成を基準として、新規 4 rule の責務を切り分ける。
- [ ] Task 1: `exception-handling.md` を作成し、禁止パターン・許容パターン・ログ要件・レビュー条件・例を入れる。
- [ ] Task 1-Verify: broad exception を例外的に残す条件が明記され、単なる「禁止」だけで終わっていないことを確認する。
- [ ] Task 2: `numerical-scales.md` を作成し、魔法定数禁止・スケール依存デフォルト・アルゴリズム特記事項・テスト期待を書き分ける。
- [ ] Task 2-Verify: GW strain スケールと `gwexpy.numerics` 利用方針が冒頭近くに明記されていることを確認する。
- [ ] Task 3: `model-assignment.md` を作成し、役割カテゴリ・推奨割り当て・エスカレーション・アンチパターンを整理する。
- [ ] Task 3-Verify: 特定モデル名の列挙だけでなく、役割ベースの判断が残っていることを確認する。
- [ ] Task 4: `gwpy-compatibility.md` を作成し、禁止 API と推奨 API の対応関係、必要テスト、移行メモを整理する。
- [ ] Task 4-Verify: `gwpy.io.mp`, `nproc`, registry import などの高頻度破壊ポイントが先頭側に載っていることを確認する。
- [ ] Task 5: 4 rule 間と既存 `physics.md` / `testing.md` の関係を見直し、重複と矛盾を潰す。
- [ ] Task 5-Verify: 各 file が 1 つの責務に集中し、分析資料のコピペ集になっていないことを確認する。

### 推奨実装順と理由

1. B-1 を先に作る。Phase 0 の分析結果が直接ルール化しやすく、禁止事項の軸を最初に固定できる。
2. B-2 を次に作る。数値規則は物理・アルゴリズム系ルールの中心であり、後続 rule から参照されやすい。
3. B-4 をその後に作る。外部依存互換性の禁止/推奨 API を明文化しておくと、モデル割り当て以外の技術ルール群が先に閉じる。
4. B-3 を最後に作る。技術ルール 3 本を見た後のほうが、どの作業をどのモデル役割へ振るかを整理しやすい。
5. 最後に 4 rule 全体の cross-reference と文体を揃える。

### 想定所要時間

- 事前確認と責務分割: 10〜15 分
- B-1 / B-2 作成: 20〜35 分
- B-4 / B-3 作成: 20〜35 分
- 全体の整合確認: 10〜15 分
- 合計目安: 60〜100 分

### リスクと切り戻し方針

- 分析レポートの内容をそのまま貼ると rule として長すぎるので、禁止事項・判断基準・例に圧縮できない記述は参考リンクへ逃がす。
- `physics.md` と `numerical-scales.md` が重複しやすいため、前者は物理原則、後者は数値しきい値とデフォルト戦略に責務を分ける。
- `model-assignment.md` はモデル名の陳腐化リスクがあるため、固有名詞だけでなく役割ラベルを主にし、具体例は補助情報として扱う。
- `gwpy-compatibility.md` は既存コードがすでに 4.0 対応済みの箇所も含むため、「過去に壊れた点の再発防止ルール」として記述し、過度に移行作業メモへ寄せない。

---

## フェーズC: Agents 追加

> **実装方針**: 追加先はすべて `.harness/agents/` 配下の Markdown とする。  
> 既存の `physics-reviewer.md`, `gwexpy-linter.md`, `gwexpy-tester.md` と同じく、YAML front matter + 役割説明 + チェックリスト/ワークフロー + 出力形式で統一する。  
> フェーズCでは agent 定義の追加に限定し、workflow や hook の呼び出し側までは変更しない。

### フェーズC の作業前チェック

- 現在の `.harness/agents/` は `physics-reviewer.md`, `gwexpy-linter.md`, `gwexpy-tester.md` の 3 ファイルのみであることを確認し、新規 3 agent の責務が既存 agent と衝突しないように切り分ける。
- 各 agent は「何でも屋」ではなく、1つの専門責務に絞る。既存 `physics-reviewer` は物理整合性、`gwexpy-linter` は静的解析、`gwexpy-tester` はテスト実行なので、C-1〜C-3 はその隙間を埋める役割に限定する。
- front matter は既存 agent と同じく `name`, `description`, `tools` を必須とし、利用する tool セットも必要最小限に留める。
- 各 agent は「どのファイル/変更に対して呼ぶか」「何を確認するか」「どう報告するか」を単独で理解できる内容にする。
- 既存資料の内容を agent prompt に丸写しせず、チェック項目・判断基準・出力フォーマットに要約する。

### フェーズC の完了条件

- `.harness/agents/exception-auditor.md`, `.harness/agents/numeric-scale-checker.md`, `.harness/agents/gwexpy-compatibility-checker.md` の 3 ファイルが作成されている。
- 3 agent とも、既存 agent と同じ front matter 構造を持ち、description だけで用途が判別できる。
- C-1 は broad exception / silent failure の検出と修正提案に集中し、linter や tester の責務を侵食していない。
- C-2 は数値しきい値・スケール依存デフォルト・スケール不変性確認に集中し、physics-reviewer の一般論と役割が分離されている。
- C-3 は GWpy API 互換性と外部ライブラリ依存の整合確認に集中し、単なる lint/test 実行 agent になっていない。
- 3 agent とも、結果報告の出力形式があり、親 agent がそのままレビュー結果として利用できる。

### C-1. `exception-auditor.md`

**概要**: `except Exception:` / 裸の `except:` を AST レベルで検出・修正提案する専門エージェント。phase0 の知識を事前情報として持つ。  
**目的**: phase0 分析で発見された 17 箇所の再チェック、および新規コード追加時の予防的レビュー。  
**参考**:
- `docs_internal/analysis/phase0_exception_analysis.md`
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`（AST スキャン手法）

#### 実装詳細

**作成**: `.harness/agents/exception-auditor.md`  
**ベース**: `phase0_exception_analysis.md` の remediation strategy と `prompt_phase0_1_opus.md` の観点を agent prompt 化  
**推奨 front matter**:

- `name: exception-auditor`
- `description: GWexpy 例外処理監査スペシャリスト。except Exception / bare except / pass を検出し、具体例外への絞り込みとログ方針を提案する。`
- `tools: Read, Grep, Glob, Bash`

**必須セクション**:

- 役割説明
  broad exception と silent failure の監査専門 agent であることを明記
- `## Scope`
  `gwexpy/`, `tests/`, 特に `collections.py`, `io/`, GUI fallback 周辺を対象にする条件を書く
- `## Audit Checklist`
  broad exception, bare except, pass-only handler, ログ欠如, `KeyError` と数値例外の混同を列挙
- `## Decision Rules`
  具体例外へ狭める、ログ付きで残す、再送出する、の判断基準を書く
- `## Safe Exceptions`
  GUI disconnect や optional dependency fallback など、残しうるケースを限定列挙
- `## Output Format`
  file ごとに `CRITICAL / WARNING / NOTE / RECOMMENDATION / VERDICT` を返す形式にする

**記載ルール**:
- 「自動修正する agent」ではなく、まず危険箇所の分類と修正方針提案を主責務にする
- broad exception を許容する場合でも、理由・ログ・限定条件の 3 点を要求する
- `gwexpy-linter` のような lint 全般には広げず、例外処理だけに集中させる

---

### C-2. `numeric-scale-checker.md`

**概要**: GW 歪みスケール（10^-21）を前提とした eps/tol 妥当性レビュー専門エージェント。HHT・STLT・whitening 等のアルゴリズム固有パラメータも知っている。  
**目的**: Death Floats 問題の再発防止と、新しいアルゴリズム実装時のパラメータ妥当性確認。  
**参考**:
- `docs_internal/analysis/phase1_dangerous_defaults.md`
- `docs_internal/tech_notes/implementation/hht_implementation_notes_20260204.md`
- `docs_internal/tech_notes/implementation/stlt_implementation_notes_20260204.md`

#### 実装詳細

**作成**: `.harness/agents/numeric-scale-checker.md`  
**ベース**: `phase1_dangerous_defaults.md` と HHT/STLT 実装知見を agent prompt 化  
**推奨 front matter**:

- `name: numeric-scale-checker`
- `description: GWexpy 数値スケール監査スペシャリスト。GW strain (~1e-21) 前提で eps/tol/atol/rtol とスケール依存デフォルトの妥当性をレビューする。`
- `tools: Read, Grep, Glob, Bash`

**必須セクション**:

- 役割説明
  Death Floats と scale-sensitive default の監査役であることを明記
- `## Scope`
  `gwexpy/signal/`, `gwexpy/timeseries/`, `gwexpy/frequencyseries/`, `gwexpy/types/`, 関連テストを対象にする
- `## Review Checklist`
  hardcoded eps/tol, `'auto'` / `None` defaults, relative scaling, NaN/Inf guard, scale-invariance tests を列挙
- `## Algorithm Notes`
  whitening, ICA, HHT/EMD, STLT などで見るべきポイントを書く
- `## Escalation Rules`
  物理妥当性判断が必要なら `physics-reviewer` へ、人間確認が必要なときは `needs-physics-review` 相当で返す条件を書く
- `## Output Format`
  `SCALE-RISK / TEST-GAP / RECOMMENDATION / VERDICT` を含む出力形式にする

**記載ルール**:
- 単に「数値が小さい/大きい」を見るのではなく、データスケールに相対的かどうかを主判断軸にする
- `gwexpy.numerics` 利用の有無、スケール不変性テストの有無、極小入力での破綻可能性を必ず確認させる
- `physics-reviewer` と競合しないよう、数値パラメータとテスト観点に責務を寄せる

---

### C-3. `gwexpy-compatibility-checker.md`

**概要**: GWpy バージョン互換性・オプション依存ライブラリの可用性チェック専門エージェント。GWpy 4.0 移行知識を持つ。  
**目的**: `extra_lib.md` で確認された 60+ の外部ライブラリと、GWpy 4.0 の破壊的変更への対応を支援する。  
**参考**:
- `docs_internal/tech_notes/research/GWpy4_deep-research-report.md`
- `docs_internal/tech_notes/research/extra_lib.md`

#### 実装詳細

**作成**: `.harness/agents/gwexpy-compatibility-checker.md`  
**ベース**: `GWpy4_deep-research-report.md` と `extra_lib.md` の互換性知見を agent prompt 化  
**推奨 front matter**:

- `name: gwexpy-compatibility-checker`
- `description: GWexpy 互換性監査スペシャリスト。GWpy 4.x の破壊的変更、外部ライブラリ interop、optional dependency の可用性をレビューする。`
- `tools: Read, Grep, Glob, Bash`

**必須セクション**:

- 役割説明
  GWpy 互換性と interop 可用性の監査を担当することを明記
- `## Scope`
  `gwexpy/io/`, `gwexpy/interop/`, `gwexpy/timeseries/io/`, `gwexpy/frequencyseries/`, 関連 docs を対象にする
- `## Compatibility Checklist`
  `gwpy.io.mp`, `nproc`, registry import, deprecated API, optional import fallback, version guards, Python version constraints を列挙
- `## Interop Checks`
  external library converter が GWexpy 型・unit・metadata を保つかを見る項目を書く
- `## Required Evidence`
  変更時に必要な grep、テスト、ドキュメント更新観点を列挙する
- `## Output Format`
  `BREAKING-RISK / COMPATIBILITY-GAP / TEST-NEEDED / VERDICT` 形式にする

**記載ルール**:
- 単なる依存一覧確認ではなく、「既知の壊れ方の再発防止」と「新規 interop の穴埋め」を主責務にする
- optional dependency の不在時 fallback も確認対象に含める
- `gwexpy-tester` と違って実行主体ではなく、互換性観点のレビュー主体に留める

---

### フェーズC 具体的な作業計画

| Task | 対応 | 編集対象 | 実施内容 | 検証 | 完了条件 |
|------|------|----------|----------|------|----------|
| 0 | 事前確認 | `.harness/agents/`, 既存 3 agent, 参照資料 | 既存 agent の書式と、新規 3 agent の責務分割を固定する | 既存 agent 3 本と重複しないことを確認 | 新規 agent の役割境界が曖昧でない |
| 1 | C-1 | `.harness/agents/exception-auditor.md` | 例外処理監査 agent を新規作成する | front matter, checklist, decision rules, output format が揃っていることを確認 | broad exception 監査の専門 agent として単独で使える |
| 2 | C-2 | `.harness/agents/numeric-scale-checker.md` | 数値スケール監査 agent を新規作成する | scope, review checklist, algorithm notes, escalation が揃っていることを確認 | Death Floats 監査の専門 agent として単独で使える |
| 3 | C-3 | `.harness/agents/gwexpy-compatibility-checker.md` | 互換性監査 agent を新規作成する | compatibility checklist, interop checks, required evidence が揃っていることを確認 | GWpy/interop 互換性監査 agent として単独で使える |
| 4 | 仕上げ | `.harness/agents/*.md` | 3 agent の description, tool 範囲, 出力形式の整合を調整する | 既存 agent 群と並べて命名・粒度・責務を確認 | `.harness/agents/` が一貫した agent セットになっている |

### フェーズC 実施チェックリスト

- [ ] Task 0: 既存 `physics-reviewer`, `gwexpy-linter`, `gwexpy-tester` の責務を基準として、新規 3 agent の担当範囲を切り分ける。
- [ ] Task 1: `exception-auditor.md` を作成し、scope・audit checklist・decision rules・output format を入れる。
- [ ] Task 1-Verify: broad exception の監査に集中しており、lint 全般やテスト実行まで責務が広がっていないことを確認する。
- [ ] Task 2: `numeric-scale-checker.md` を作成し、review checklist・algorithm notes・escalation rules を入れる。
- [ ] Task 2-Verify: GW strain スケール、`gwexpy.numerics`、scale-invariance テストの 3 軸が明記されていることを確認する。
- [ ] Task 3: `gwexpy-compatibility-checker.md` を作成し、compatibility checklist・interop checks・required evidence を入れる。
- [ ] Task 3-Verify: GWpy 4.0 破壊的変更と optional dependency fallback の両方が確認対象になっていることを確認する。
- [ ] Task 4: 3 agent の front matter, description, tools, output format を見直し、既存 agent 群と粒度を揃える。
- [ ] Task 4-Verify: 各 agent が 1 つの専門責務に集中し、親 agent が呼び出し条件を description だけで判断できることを確認する。

### 推奨実装順と理由

1. C-1 を先に作る。Phase 0 の分析成果が明確で、agent の観点と出力形式を決めやすい。
2. C-2 を次に作る。数値スケール監査は `physics-reviewer` と近いが、境界を先に文書化しておく価値が高い。
3. C-3 をその後に作る。GWpy 互換性と外部 interop の範囲が広いため、他 2 本の粒度を見てから調整するほうが安全。
4. 最後に 3 agent の front matter と出力形式を揃え、既存 agent 群との整合を取る。

### 想定所要時間

- 事前確認と責務分割: 10〜15 分
- C-1 / C-2 作成: 20〜35 分
- C-3 作成: 15〜30 分
- 全体整合確認: 10〜15 分
- 合計目安: 55〜95 分

### リスクと切り戻し方針

- `physics-reviewer` と `numeric-scale-checker` が重複しやすいため、前者は物理整合性全般、後者は数値しきい値とスケール検証に責務を分ける。
- `gwexpy-linter` と `exception-auditor` が近づきすぎないよう、後者は lint エラー検出ではなく設計上危険な例外パターンのレビューに寄せる。
- `gwexpy-compatibility-checker` は対象範囲が広いため、GWpy 4.0 既知変更と interop/optional dependency の 2 軸に限定し、それ以外の一般依存更新レビューは含めない。
- agent prompt が長くなりすぎると運用しづらいので、詳細な背景説明は参考資料リンクに逃がし、本文はチェック項目と出力形式中心に保つ。

---

## フェーズD: Workflows 強化・追加

> **実装方針**: workflow 定義は `.harness/workflows/` 配下の Markdown とし、既存 `feature-development.md` / `release.md` と同じく YAML front matter + ステップ列挙で統一する。  
> 各 workflow は「いつ使うか」「何を先に確認するか」「どの skill/agent/コマンドを使うか」「完了条件は何か」を 1 ファイルで完結させる。  
> フェーズDでは workflow 文書の追加・更新に限定し、hook・rule・agent 本体は直接変更しない。

### フェーズD の作業前チェック

- 現在の `.harness/workflows/` は `feature-development.md` と `release.md` の 2 ファイルのみであることを確認し、新規 2 workflow と既存 release 強化の責務を切り分ける。
- workflow は rule や agent と違い「判断基準」ではなく「実行順序とゲート」を表現するため、背景説明よりもステップ順・入力・出力・分岐条件を優先して書く。
- front matter は既存 workflow と同じく `name`, `description`, `trigger` を基本とし、description だけで発火タイミングが判断できるようにする。
- 各 workflow は、呼び出す skill や agent を明示しつつ、依存先が未整備でも破綻しないよう「未導入なら手動代替する」導線を持たせる。
- 既存 `feature-development.md` との重複を避け、D-1 は数値安全性レビュー、D-2 はリリースゲート、D-3 は負債消化の進め方に責務を固定する。

### フェーズD の完了条件

- `.harness/workflows/numerical-audit.md` と `.harness/workflows/technical-debt.md` の 2 ファイルが新規作成対象として定義されている。
- `.harness/workflows/release.md` の強化内容が、現行 release 手順に対する具体的な追加項目として明示されている。
- D-1 は silent failure / Death Floats / scale-aware test のゲートを含む。
- D-2 は version/metadata/CHANGELOG/TestPyPI/品質ゲートの順序が明示されている。
- D-3 は backlog から着手単位へ分解し、優先度付け・フェーズ分割・検証・記録まで定義している。
- 3 workflow とも、既存 agent/rule/hook とどう組み合わせるかが読み取れる。

### D-1. `workflows/numerical-audit.md` （新規）

**概要**: 新しいアルゴリズム・数値処理コードを追加する際の安全性確認手順。phase0/phase1 スタイルのゲートチェックリスト。  
**目的**: HHT・STLT・whitening 等を追加する際に、毎回 Death Floats 問題や Silent Failure を再発させないための標準手順。  
**参考**:
- `docs_internal/analysis/phase0_exception_analysis.md`
- `docs_internal/analysis/phase1_dangerous_defaults.md`
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`（手順の参考）

#### 実装詳細

**作成**: `.harness/workflows/numerical-audit.md`  
**ベース**: `prompt_phase0_1_opus.md` のフェーズ 0/1 手順と、phase0/phase1 分析の remediation を workflow 化  
**推奨 front matter**:

- `name: numerical-audit`
- `description: GWexpy 数値アルゴリズム追加・修正時の安全性監査ワークフロー。silent failure, Death Floats, scale-invariance を段階的に確認する。`
- `trigger: manual`

**必須セクション**:

- `# GWexpy Numerical Audit Workflow`
- `## Before You Start`
  対象アルゴリズム、関連モジュール、参照資料、物理レビュー要否を確認する
- `## Step 1: Risk Inventory`
  broad exception, hardcoded eps/tol, optional fallback, metadata/units, GWpy 互換性の有無を洗い出す
- `## Step 2: Unsilencing Pass`
  `except Exception` / bare except を点検し、具体例外 or ログ付き fallback へ寄せる
- `## Step 3: Scale Review`
  `gwexpy.numerics`, relative scaling, `'auto'`/`None` defaults, NaN/Inf guard を確認する
- `## Step 4: Test Design`
  scale-invariance, tiny-signal, zero/NaN/Inf, metadata preservation, regression test を定義する
- `## Step 5: Validation`
  `ruff`, `mypy`, `pytest`, 必要なら `physics-reviewer` / `numeric-scale-checker` を呼ぶ
- `## Exit Criteria`
  silent failure 不在、魔法定数不在、テスト追加、必要レビュー完了を条件化する

**記載ルール**:
- 単なるチェックリストではなく、「失敗したら次に何をするか」の分岐を書く
- D-1 は新アルゴリズム追加時の標準導線なので、HHT/STLT/whitening に閉じず一般化する
- 既存 `feature-development.md` の中の実装一般論は再掲せず、数値監査固有のゲートだけを書く

---

### D-2. `workflows/release.md` 強化

**概要**: 既存 release.md に、フェーズゲート（テスト数・ruff/mypy・メタデータ整合性）を明示的に追記する。  
**目的**: 過去のリリース作業（work_report_phase1/2）で実証されたゲート条件を標準ワークフローに組み込む。  
**参考**:
- `docs_internal/archive/plans/work_report_phase1_20260130.md`
- `docs_internal/archive/plans/work_report_phase2_20260130.md`
- `docs_internal/analysis/roadmap_20260403.md`

#### 実装詳細

**更新**: `.harness/workflows/release.md`  
**ベース**: 現行 release workflow に、phase1/phase2 の実績と roadmap の残課題をゲートとして追加  
**追加すべきセクション/内容**:

- `## Pre-Release Gates`
  `pytest`, `ruff`, `mypy`, docs build, notebook/CI 確認、coverage 低下なし、必要ラベル付けの確認
- `## Metadata Sync`
  `pyproject.toml`, `__version__`, `CITATION.cff`, `codemeta.json`, `CHANGELOG.md`, 必要なら `.zenodo.json` の整合確認
- `## Packaging Verification`
  `python -m build`, `twine check`, clean install, TestPyPI 検証などを段階化する
- `## Documentation and Install Path Check`
  README / docs が PyPI 状態や install 方法と矛盾していないか確認する
- `## Community/Governance Check`
  CoC/maintainers/contact placeholders, release note, roadmap 更新要否を確認する
- `## Go / No-Go Decision`
  どの条件でタグ付け可能か、どの条件なら止めるかを書く
- `## Rollback / Hotfix`
  既存の rollback 手順を残しつつ、yank/hotfix/version bump の判断を補う

**記載ルール**:
- 現行 release.md の「prep_release の後は自動」だけでは粗いので、公開前の確認ゲートを順番付きで明示する
- `work_report_phase1/2` の実績値を使い、CHANGELOG・metadata・twine check・docs build・CI 確認を workflow に昇格させる
- roadmap 由来の未完了事項は「毎回全部やる」ではなく、リリース blocking / non-blocking に分けて書く

---

### D-3. `workflows/technical-debt.md` （新規）

**概要**: 技術的負債を系統的に消化するためのワークフロー。backlog の優先度付け・フェーズ分割・進捗追跡の手順を定める。  
**目的**: `improvement_tasks_backlog.md` に積まれた負債を、計画的に・AI エージェントを活用しながら消化する。  
**参考**:
- `docs_internal/archive/plans/improvement_tasks_backlog.md`
- `docs_internal/archive/plans/model_assignment_v0.1.0b1.md`（フェーズ分割パターン）

#### 実装詳細

**作成**: `.harness/workflows/technical-debt.md`  
**ベース**: `improvement_tasks_backlog.md` のタスク分解パターンと `model_assignment_v0.1.0b1.md` のフェーズ分割実績を workflow 化  
**推奨 front matter**:

- `name: technical-debt`
- `description: GWexpy 技術的負債消化ワークフロー。backlog から安全な着手単位へ分解し、優先度付け・実装・検証・記録まで管理する。`
- `trigger: manual`

**必須セクション**:

- `# GWexpy Technical Debt Workflow`
- `## Step 1: Debt Intake`
  backlog 項目、発見元、対象ファイル、再現手順、リスク種別を整理する
- `## Step 2: Prioritization`
  P0/P1/P2 などの優先度、ブロッカー性、波及範囲、修正コストで分類する
- `## Step 3: Slice into Phases`
  大きい負債を 1 PR / 1 セッションで扱えるサイズへ分割する
- `## Step 4: Assign Roles`
  必要な model/agent/skill を割り当てる。レビュー役と実装役を分ける
- `## Step 5: Execute and Verify`
  `ruff`, `mypy`, `pytest`, 必要な docs/build/physics review を通す
- `## Step 6: Record Outcomes`
  backlog 更新、CHANGELOG 要否、残課題、次フェーズへの handoff を記録する
- `## Exit Criteria`
  タスク完了・部分完了・先送りのいずれで閉じるかを定義する

**記載ルール**:
- 「負債を片付ける」ではなく、「安全に小さく切って進捗を可視化する」ことを主目的にする
- backlog の全件棚卸しではなく、個別タスクを着手可能な単位に変換する workflow として書く
- `model_assignment_v0.1.0b1.md` の実績をそのまま固定化せず、役割分担パターンとして抽象化する

---

### フェーズD 具体的な作業計画

| Task | 対応 | 編集対象 | 実施内容 | 検証 | 完了条件 |
|------|------|----------|----------|------|----------|
| 0 | 事前確認 | `.harness/workflows/`, 既存 2 workflow, 参照資料 | 現行 workflow の書式と、新規 2 本 + release 強化の責務分割を固定する | 既存 `feature-development.md` / `release.md` と重複しないことを確認 | 各 workflow の担当領域が曖昧でない |
| 1 | D-1 | `.harness/workflows/numerical-audit.md` | 数値監査 workflow を新規作成する | step 順、分岐、exit criteria が揃っていることを確認 | 数値安全性監査の標準導線として単独で使える |
| 2 | D-2 | `.harness/workflows/release.md` | 既存 release workflow に phase gate を追記する | pre-release gates, metadata sync, packaging verification が揃っていることを確認 | リリース可否判断の手順が現実運用に耐える |
| 3 | D-3 | `.harness/workflows/technical-debt.md` | 技術的負債 workflow を新規作成する | prioritization, slice, assign, verify, record が揃っていることを確認 | backlog から実行単位へ落とす導線として単独で使える |
| 4 | 仕上げ | `.harness/workflows/*.md` | 3 workflow の trigger, description, step 粒度, 参照関係を整える | 既存 workflow 群と並べて命名・粒度・責務を確認 | `.harness/workflows/` が一貫した workflow セットになっている |

### フェーズD 実施チェックリスト

- [ ] Task 0: 既存 `feature-development.md` / `release.md` の構成を基準として、新規 2 workflow と release 強化の担当範囲を切り分ける。
- [ ] Task 1: `numerical-audit.md` を作成し、risk inventory・unsilencing・scale review・test design・validation・exit criteria を入れる。
- [ ] Task 1-Verify: 数値監査固有の手順に集中しており、一般的な feature 開発手順の再掲になっていないことを確認する。
- [ ] Task 2: `release.md` を強化し、pre-release gates・metadata sync・packaging verification・go/no-go を追記する。
- [ ] Task 2-Verify: CHANGELOG, metadata, twine/build, docs/install path の確認順が明示されていることを確認する。
- [ ] Task 3: `technical-debt.md` を作成し、prioritization・phase split・role assignment・verification・recording を入れる。
- [ ] Task 3-Verify: backlog の羅列ではなく、着手可能単位へ落とす運用手順になっていることを確認する。
- [ ] Task 4: 3 workflow の front matter, description, step 粒度を見直し、既存 workflow 群と整合させる。
- [ ] Task 4-Verify: 各 workflow が 1 つの運用目的に集中し、description だけでいつ使うか判断できることを確認する。

### 推奨実装順と理由

1. D-2 を先に触る。既存 `release.md` の強化は差分確認がしやすく、workflow 書式の基準にもなる。
2. D-1 を次に作る。phase0/phase1 の知見を実行順へ落としやすく、数値安全性 workflow の骨格が作りやすい。
3. D-3 をその後に作る。技術的負債 workflow は範囲が広いので、他 2 本の粒度を見てから抽象度を合わせる。
4. 最後に 3 workflow の front matter と step 粒度を揃え、既存 workflow 群との整合を取る。

### 想定所要時間

- 事前確認と責務分割: 10〜15 分
- D-2 強化: 15〜25 分
- D-1 / D-3 作成: 25〜40 分
- 全体整合確認: 10〜15 分
- 合計目安: 60〜95 分

### リスクと切り戻し方針

- `feature-development.md` と `numerical-audit.md` が重複しやすいため、前者は開発全般、後者は数値安全性ゲートに責務を分ける。
- `release.md` にチェック項目を詰め込みすぎると運用不能になるため、blocking / non-blocking を分けて書く前提にする。
- `technical-debt.md` は抽象論に流れやすいので、backlog intake → prioritization → phase split → verify → record の実行順を崩さない。
- workflow が rule や agent の説明書に寄りすぎないよう、本文は「手順と分岐」を中心に保ち、詳細判断は rule/agent 参照へ逃がす。

---

## フェーズE: Skills 移植・追加

> **実装方針**: 追加先は `.harness/skills/<skill-name>/SKILL.md` を基本とし、必要な補助資料だけ `reference/` や `scripts/` に分離する。  
> 各 skill は既存 `.harness/skills/` の構成に合わせ、短い front matter、100行前後の本文、必要最小限の reference という progressive disclosure を守る。  
> フェーズEでは「過去の prompt をそのまま保存」するのではなく、再利用可能な trigger・手順・注意点・確認方法へ変換する。

### フェーズE の作業前チェック

- 現在の `.harness/skills/` には多数の既存 skill があるため、新規 skill を増やす前に overlap を確認し、既存 skill で代替できないかを先に判断する。
- `writing-skills` と `maintain_skills` の指針に従い、description は「何をするか」ではなく「いつ使うか」を中心に書く。
- 新規 skill のディレクトリ名は既存の kebab/snake 混在命名に合わせつつ、Phase E 計画内では実際の作成パスを固定する。
- heavy な背景説明は `reference/*.md` に逃がし、SKILL.md 本文には trigger, core steps, common mistakes, expected outputs を残す。
- skill を追加したら `.harness/skills/README.md` のカテゴリ・件数・説明の同期が必要になる前提で計画を立てる。

### フェーズE の完了条件

- `.harness/skills/phase0_exception_sweep/SKILL.md` と `.harness/skills/phase1_scale_invariance/SKILL.md` の 2 skill が新規作成対象として定義されている。
- 2 skill とも、front matter の `name` / `description` が trigger-based になっている。
- E-1 は broad exception 監査から修正方針決定までの再利用手順になっており、単なる会話ログの転記ではない。
- E-2 は scale-invariance テスト設計と Death Floats 検出の再利用手順になっており、単なる一回限りの修正指示ではない。
- どちらの skill も、必要なら `reference/` に元プロンプト由来の背景を分離する設計が示されている。
- `.harness/skills/README.md` の更新が Phase E の作業範囲に含まれている。

### E-1. `skills/phase0_exception_sweep/`

**概要**: `archive/prompts/prompt_phase0_1_opus.md` の手順を skill として再利用可能な形に整理する。AST スキャン → 特定 → 修正の 3 ステップ手順。  
**目的**: phase0 作業の再現性確保と、同様の監査を将来的に定期実施できるようにする。  
**参考**:
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`
- `docs_internal/analysis/phase0_exception_analysis.md`

#### 実装詳細

**作成**:
- `.harness/skills/phase0_exception_sweep/SKILL.md`
- 必要に応じて `.harness/skills/phase0_exception_sweep/reference/phase0-patterns.md`

**ベース**: `prompt_phase0_1_opus.md` の Phase 0 手順と `phase0_exception_analysis.md` の remediation strategy を skill 化  
**推奨 front matter**:

- `name: phase0_exception_sweep`
- `description: Use when auditing or fixing broad exception handling, silent failures, or suspicious try/except fallback logic in gwexpy.`

**必須セクション**:

- `# Phase 0 Exception Sweep`
- `## Overview`
  この skill が silent failure 除去と broad exception の整理に使うものであることを簡潔に記載
- `## When to Use`
  `except Exception`, bare except, `pass`, logger なし fallback, collection accessor の誤捕捉などの trigger を列挙
- `## Core Workflow`
  1. inventory 取得
  2. context 理解
  3. remove / narrow / log のいずれかを選択
  4. focused tests 実行
- `## Decision Rules`
  broad exception を消せる条件、残す場合の logging 要件、具体例外の選び方を書く
- `## Verification`
  grep, pytest, GUI fallback などの確認観点を書く
- `## Common Mistakes`
  `KeyError` と計算失敗の混同、GUI だから何でも握りつぶす、などを明記

**reference に分離する候補**:
- 17 箇所の代表パターン一覧
- batch processing / IO fallback / GUI fallback の詳細判断例

**記載ルール**:
- E-1 は「AST スキャンして直す一般手順」を再利用可能にするので、特定ファイル名の列挙に依存しすぎない
- 修正方法は `remove`, `narrow`, `log-and-continue` の 3 択で整理する
- 既存 `review_repo` や `fix_errors` と責務が重なりすぎないよう、例外処理監査に絞る

---

### E-2. `skills/phase1_scale_invariance/`

**概要**: `archive/prompts/` の Phase 2 Codex 向けプロンプトを skill 化。スケール不変性テスト（`f(X) ≡ f(X × 10^-20)`）の設計と実行手順。  
**目的**: 新しい数値アルゴリズムを追加した際に、GW スケールでの動作を体系的に検証できるようにする。  
**参考**:
- `docs_internal/archive/prompts/`（Phase 2 向けプロンプト）
- `docs_internal/analysis/phase1_dangerous_defaults.md`

#### 実装詳細

**作成**:
- `.harness/skills/phase1_scale_invariance/SKILL.md`
- 必要に応じて `.harness/skills/phase1_scale_invariance/reference/scale-tests.md`

**ベース**: `prompt_phase2_codex.md` と `phase1_dangerous_defaults.md` の scale-invariance 手順を skill 化  
**推奨 front matter**:

- `name: phase1_scale_invariance`
- `description: Use when implementing or reviewing numerical algorithms that may break on tiny gravitational-wave scale inputs or rely on hardcoded eps/tol defaults.`

**必須セクション**:

- `# Phase 1 Scale Invariance`
- `## Overview`
  GW strain スケールで壊れない数値実装へ導く skill であることを記載
- `## When to Use`
  whitening, ICA, fitting, matrix solve, default eps/tol, `1e-X` のハードコード検出などの trigger を列挙
- `## Core Workflow`
  1. dangerous defaults inventory
  2. scale-aware default への置換
  3. tiny-signal / scaled-input test 設計
  4. regression validation
- `## Preferred Fix Patterns`
  `None` / `'auto'`, relative jitter, `slogdet`, preconditioned solve, `gwexpy.numerics` 利用を整理する
- `## Verification`
  `f(X)` と `f(X * 1e-20)` の比較、NaN/Inf guard、metadata/unit 保持、pytest ケース設計を書く
- `## Common Mistakes`
  fixed epsilon を別の magic number に置き換えるだけ、テストで大振幅しか見ない、などを明記

**reference に分離する候補**:
- whitening / ICA / MCMC / matrix math の詳細修正例
- 既知の fatal bug パターン一覧

**記載ルール**:
- E-2 は「アルゴリズム修正依頼」ではなく「スケール不変性を担保する再利用手順」としてまとめる
- `gwexpy.numerics` が未整備でも temporary local logic で前進する判断を残す
- 既存 `verify_physics` と重複しすぎないよう、物理一般論ではなく数値スケール検証に集中させる

---

### フェーズE 具体的な作業計画

| Task | 対応 | 編集対象 | 実施内容 | 検証 | 完了条件 |
|------|------|----------|----------|------|----------|
| 0 | 事前確認 | `.harness/skills/`, 既存 skill 群, 参照 prompt | 既存 skill との overlap と、新規 2 skill の責務分割を固定する | `maintain_skills` 観点で重複しないことを確認 | 新規 skill の役割境界が曖昧でない |
| 1 | E-1 | `.harness/skills/phase0_exception_sweep/` | broad exception 監査 skill を新規作成する | trigger, core workflow, decision rules, verification が揃っていることを確認 | exception audit skill として単独で使える |
| 2 | E-2 | `.harness/skills/phase1_scale_invariance/` | scale invariance skill を新規作成する | trigger, fix patterns, verification, common mistakes が揃っていることを確認 | numerical hardening skill として単独で使える |
| 3 | README同期 | `.harness/skills/README.md` | 件数、カテゴリ、説明、Quick Start への反映要否を調整する | 新規 2 skill が README から発見可能であることを確認 | skill inventory が実態と一致する |
| 4 | 仕上げ | `.harness/skills/*` | front matter, reference 分離方針, naming, 既存 skill との棲み分けを整える | `writing-skills` / `maintain_skills` の観点で見直す | `.harness/skills/` が一貫した skill ライブラリになっている |

### フェーズE 実施チェックリスト

- [ ] Task 0: 既存 `review_repo`, `fix_errors`, `verify_physics`, `maintain_skills` などとの責務重複を確認し、新規 2 skill の担当範囲を切り分ける。
- [ ] Task 1: `phase0_exception_sweep/SKILL.md` を作成し、trigger, core workflow, decision rules, verification, common mistakes を入れる。
- [ ] Task 1-Verify: 特定の一回きりの prompt ではなく、再利用可能な broad exception 監査 skill になっていることを確認する。
- [ ] Task 2: `phase1_scale_invariance/SKILL.md` を作成し、trigger, preferred fix patterns, verification, common mistakes を入れる。
- [ ] Task 2-Verify: `f(X)` と `f(X * 1e-20)` の比較観点と、`gwexpy.numerics` 利用方針が明記されていることを確認する。
- [ ] Task 3: `.harness/skills/README.md` を更新し、新規 2 skill のカテゴリと説明を反映する。
- [ ] Task 3-Verify: README の件数と一覧が実際の skill ディレクトリ構成と一致していることを確認する。
- [ ] Task 4: front matter description, reference 分離, naming を見直し、既存 skill 群と整合させる。
- [ ] Task 4-Verify: 各 skill が「いつ使うか」で発見でき、本文を読まないと workflow を実行できない形になっていることを確認する。

### 推奨実装順と理由

1. E-1 を先に作る。Phase 0 prompt は修正判断の三択が明確で、skill 化しやすい。
2. E-2 を次に作る。数値スケール検証 skill は `gwexpy.numerics` や test pattern を整理する必要があり、E-1 より少し重い。
3. 2 skill が固まってから `.harness/skills/README.md` を更新する。途中で件数やカテゴリを触ると手戻りしやすい。
4. 最後に front matter と reference 分離方針を揃え、skill ライブラリ全体としての一貫性を取る。

### 想定所要時間

- 事前確認と overlap 整理: 10〜15 分
- E-1 / E-2 作成: 25〜40 分
- README 同期: 10〜15 分
- 全体整合確認: 10〜15 分
- 合計目安: 55〜85 分

### リスクと切り戻し方針

- 過去 prompt の本文をそのまま移植すると skill ではなく「会話ログの保存」になるため、trigger と再利用手順に変換できない内容は reference へ逃がす。
- `fix_errors` や `review_repo` と役割が近づきすぎないよう、E-1 は exception audit、E-2 は scale invariance に責務を固定する。
- README 同期を忘れると新規 skill が発見されないため、Phase E では skill 本体と README 更新を同じ作業単位で扱う。
- skill 本文が長くなりすぎると発見性と運用性が落ちるので、SKILL.md は短く、詳細例だけを reference に分離する。

---

## フェーズF: 運用ガードレール拡張

> **実装方針**: フェーズFでは、A-E で整備する hooks / rules / agents / workflows / skills を「日常運用で確実に使わせる仕組み」に拡張する。  
> 単なる知識追加ではなく、セッション開始・作業中・完了前・リリース前の各時点で抜け漏れを減らすガードレールを実装対象にする。  
> 追加先は `.harness/hooks/`, `.harness/workflows/`, `.harness/agents/`, `.harness/skills/README.md` を基本とし、必要な補助ロジックだけ `.harness/scripts/` に分離する。

### フェーズF の作業前チェック

- 既存 `.harness/` の実ファイル構成を再確認し、hooks / workflows / agents / skills README のどこに追加するかを先に固定する。
- `.agent/AGENTS.md` の pre-execution checklist と audit 要件をフェーズFの主要要件として扱い、既存計画との重複ではなく運用自動化へ変換する。
- フェーズFの各項目は「警告だけで十分か」「workflow 化すべきか」「専用 agent が必要か」を先に切り分け、責務を重ねない。
- 自動判定系は false positive を避けることを優先し、最初からブロッキングにせず warning / report ベースで導入する。
- スクリプトを追加する場合は `.harness/scripts/` 配下に置き、hook や workflow から薄く呼び出す構成にする。

### フェーズF の完了条件

- F-1 から F-7 までの追加候補について、作成ファイル・責務・検証方法・導入順が明記されている。
- セッション開始時の環境確認、作業完了時の監査証跡、optional dependency 影響確認、docs/metadata 整合、導線案内、レビューラベル付与が、それぞれ独立した実装単位に分解されている。
- どの項目も「なぜ必要か」が `.agent/AGENTS.md` や既存分析資料と結びついて説明されている。
- フェーズFの項目が、A-E の既存成果を前提にしていても、段階的に独立導入できる順序になっている。
- 優先度サマリーにフェーズFが追加され、A-E との相対優先度が更新されている。

### F-1. `session-start doctor`

**概要**: セッション開始時に、環境・ブランチ状態・依存コマンド・レジストリ初期化前提を確認する workflow / skill を追加する。  
**目的**: `.agent/AGENTS.md` の pre-execution checklist を人力依存にせず、作業前の事故を減らす。  
**参考**:
- `.agent/AGENTS.md`
- `.harness/workflows/feature-development.md`
- `.harness/scripts/setup_symlinks.sh`

#### 実装詳細

**作成候補**:
- `.harness/workflows/session-start.md`
- 必要に応じて `.harness/scripts/session_start_doctor.sh`

**責務**:
- `conda` / `python` / `ruff` / `mypy` / `pytest` の存在確認
- 推奨環境名 `gwexpy` または `gwex-env` の案内
- `git status --short` による dirty worktree 検出
- `gwexpy.register_all()` または `import gwexpy` 前提の注意喚起
- 変更対象が `gwexpy/fields/` を含む場合の human review reminder

**必須セクション**:
- `When to Use`
- `Preflight Checks`
- `Common Failures`
- `Escalation`
- `Expected Output`

**記載ルール**:
- ブロッキングではなく「不足項目を並べて次アクションを返す doctor」にする
- AGENTS の原文を転載せず、実行順へ変換する
- セッション冒頭に毎回使える軽さを保つ

---

### F-2. `evidence-pack / audit manifest`

**概要**: 作業完了時に、使用 skill / commands / tests / review / changed files を JSON または YAML へまとめる workflow / skill を追加する。  
**目的**: `.agent/AGENTS.md` Section 6 の audit log 要件をハーネスで半自動化し、完了宣言の根拠を残す。  
**参考**:
- `.agent/AGENTS.md`
- `.harness/workflows/release.md`
- `.harness/agents/gwexpy-tester.md`
- `.harness/agents/physics-reviewer.md`

#### 実装詳細

**作成候補**:
- `.harness/workflows/evidence-pack.md`
- 必要に応じて `.harness/scripts/generate_audit_manifest.py`

**責務**:
- 変更ファイル一覧の収集
- 実行した確認項目の記録テンプレート生成
- `check_physics` 相当の要否フラグ
- `needs-physics-review` / `needs-release-check` 等のラベル候補出力
- PR へ貼れる短い summary と manifest 本文の分離

**出力形式**:
- `docs_internal/work_logs/` などの保存先候補を計画時点で定義
- 最低限 `skills`, `commands`, `tests`, `reviews`, `files_changed`, `known_gaps` を含む

**記載ルール**:
- 監査ログの記入負担を減らすことを主目的にする
- すべて自動収集にせず、人間が補足すべき欄を残す
- release workflow と重複しないよう、こちらは「作業証跡」、release は「出荷判定」に分ける

---

### F-3. `optional-dependency impact checker`

**概要**: optional dependency や extras に影響する差分を検出し、フォールバック・docs・テストの確認を促す hook / agent / rule を追加する。  
**目的**: 依存が揃った環境だけで成立する変更を早期に検出し、 import fallback や install 導線の崩れを防ぐ。  
**参考**:
- `docs_internal/tech_notes/research/extra_lib.md`
- `pyproject.toml`
- `.harness/agents/gwexpy-linter.md`

#### 実装詳細

**作成候補**:
- `.harness/agents/optional-deps-reviewer.md`
- `.harness/rules/common/optional-dependencies.md`
- 必要に応じて `.harness/hooks/hooks.json` への warning hook 追加

**責務**:
- `import` 追加や extras 変更を差分から検出
- fallback path / error message / install docs 更新要否を確認
- 「optional を required 扱いしていないか」をレビューする
- GUI / notebook / docs 側の導線更新を促す

**必須観点**:
- import guard
- lazy import の是非
- missing dependency 時のメッセージ
- extras 名と docs 記載の整合
- 影響テストの有無

**記載ルール**:
- dependency 追加そのものを禁止しない
- 「optional と言いながら実質必須」状態を最優先で検出する
- 科学計算系 / GUI 系 extras の両方を対象に含める

---

### F-4. `docs / notebook drift detector`

**概要**: 公開 API・install 手順・主要挙動の変更時に、README・docs・notebook の追従漏れを検知する workflow / hook を追加する。  
**目的**: コードだけ更新され、チュートリアルや運用文書が古いまま残る状態を防ぐ。  
**参考**:
- `README.md`
- `docs/`
- `.harness/workflows/feature-development.md`

#### 実装詳細

**作成候補**:
- `.harness/workflows/docs-sync.md`
- 必要に応じて `.harness/hooks/hooks.json` の Stop reminder

**責務**:
- `gwexpy/` の public API 変更時に docs 追従要否を判定
- install / usage / notebook サンプルの更新候補を列挙
- 「変更不要」の場合も理由を残す運用にする
- 実装完了前チェックとして feature workflow から参照できるようにする

**必須セクション**:
- `Change Types That Require Docs Review`
- `Docs Targets`
- `Notebook/Tutorial Checks`
- `When No Docs Change Is Acceptable`

**記載ルール**:
- 全変更で docs 更新を強制しない
- 「public API / install / tutorial semantics」の変更に集中させる
- drift 検出結果は evidence-pack に流用できるようにする

---

### F-5. `metadata consistency checker`

**概要**: バージョン、配布 metadata、引用情報、CHANGELOG の整合を確認する release 補助 agent / workflow を追加する。  
**目的**: リリース前に `pyproject.toml`, `CITATION.cff`, `codemeta.json`, `CHANGELOG.md` などのズレを検出する。  
**参考**:
- `pyproject.toml`
- `CHANGELOG.md`
- `docs_internal/analysis/roadmap_20260403.md`
- `.harness/workflows/release.md`

#### 実装詳細

**作成候補**:
- `.harness/agents/metadata-checker.md`
- `.harness/workflows/release.md` への参照追加

**責務**:
- バージョン番号、日付、配布名、主要メタデータの整合確認
- `CHANGELOG.md` エントリの有無確認
- build / twine / docs install path の確認観点を補助
- release go/no-go に必要な欠落項目を一覧化

**必須観点**:
- version sync
- citation metadata
- package metadata
- release notes / changelog
- user-facing install instructions

**記載ルール**:
- F-5 は release workflow の一部としても単独 agent としても使える構成にする
- metadata の「存在確認」だけでなく「相互整合」を見る
- packaging の詳細手順は D-2 に寄せ、F-5 は整合チェックに集中させる

---

### F-6. `task router`

**概要**: 依頼内容から、使うべき workflow / agent / skill / rule を最初に案内する軽量 workflow を追加する。  
**目的**: `.harness/` が増えた後の「何を使えばよいか分からない」を減らし、整備した資産の利用率を上げる。  
**参考**:
- `.harness/skills/README.md`
- `.harness/workflows/feature-development.md`
- `.harness/agents/*.md`

#### 実装詳細

**作成候補**:
- `.harness/workflows/task-routing.md`
- `.harness/skills/README.md` への quick routing 追記

**責務**:
- 依頼を `feature`, `physics`, `testing`, `release`, `technical debt`, `docs`, `optional deps` などへ分類
- 各分類に対して推奨 workflow / agent / rule を返す
- 複数候補がある場合は、優先順と使い分けの違いを短く示す

**必須セクション**:
- `Routing Table`
- `Primary Entry Points`
- `Escalate to Human`
- `Examples`

**記載ルール**:
- 単なる一覧表ではなく、「最初の1本」を返す router にする
- README と workflow で責務分担し、README は索引、workflow は判断手順に寄せる
- フェーズF以降の追加資産も追記しやすい拡張余地を残す

---

### F-7. `risk labeler`

**概要**: 変更ファイルや diff から、必要なレビューラベルや確認フローを提案する hook / agent を追加する。  
**目的**: physics review, release check, optional deps check, docs sync などの抜け漏れを、変更内容から先回りで可視化する。  
**参考**:
- `.agent/AGENTS.md`
- `.harness/agents/physics-reviewer.md`
- `docs_internal/analysis/phase1_dangerous_defaults.md`

#### 実装詳細

**作成候補**:
- `.harness/agents/risk-labeler.md`
- 必要に応じて `.harness/hooks/hooks.json` の Stop warning

**初期ラベル案**:
- `needs-physics-review`
- `needs-release-check`
- `needs-optional-deps-check`
- `needs-docs-sync`
- `needs-scale-invariance-check`

**判定ルール例**:
- `gwexpy/fields/` 変更 → `needs-physics-review`
- `pyproject.toml` / packaging 系変更 → `needs-release-check`
- import / extras 変更 → `needs-optional-deps-check`
- public API / README 影響 → `needs-docs-sync`
- 数値アルゴリズム変更 → `needs-scale-invariance-check`

**記載ルール**:
- 自動付与よりも「候補として提案」に留めて誤検知耐性を優先する
- 判定根拠を出力する
- evidence-pack と接続できる設計にする

---

### フェーズF 具体的な作業計画

| Task | 対応 | 編集対象 | 実施内容 | 検証 | 完了条件 |
|------|------|----------|----------|------|----------|
| 0 | 事前確認 | `.agent/AGENTS.md`, `.harness/`, 既存計画 | フェーズF各項目の責務境界と追加先を確定する | hooks / workflows / agents / skills README への割当が重複していないことを確認 | 実装単位が曖昧でない |
| 1 | F-1 | `.harness/workflows/session-start.md`, 必要なら `.harness/scripts/` | session-start doctor の preflight 手順を定義する | 環境・registry・dirty worktree・human review reminder が入っていることを確認 | セッション開始時の標準導線として使える |
| 2 | F-2 | `.harness/workflows/evidence-pack.md`, 必要なら `.harness/scripts/` | evidence-pack / audit manifest 生成手順を定義する | manifest 項目と human 補足欄が揃っていることを確認 | 作業証跡を残す標準手順になる |
| 3 | F-3 / F-7 | `.harness/agents/`, `.harness/rules/common/`, `.harness/hooks/hooks.json` | optional dependency checker と risk labeler の責務を分担して定義する | dependency / label 判定の根拠が明示されていることを確認 | 差分ベースのレビュー導線が作れる |
| 4 | F-4 / F-5 | `.harness/workflows/docs-sync.md`, `.harness/agents/metadata-checker.md`, `release.md` | docs drift と metadata consistency の確認導線を追加する | docs targets と metadata sync 観点が具体的であることを確認 | docs / release 前の抜け漏れを点検できる |
| 5 | F-6 | `.harness/workflows/task-routing.md`, `.harness/skills/README.md` | task router と quick routing 案内を追加する | 代表的な依頼から導線が一意に引けることを確認 | `.harness/` の入口が明確になる |
| 6 | 仕上げ | `docs_internal/analysis/harness_enhancement_plan.md` と各候補先 | 導入順、依存関係、README / workflow 参照関係を見直す | フェーズF全体が段階導入可能であることを確認 | 実装に移せる具体計画になる |

### フェーズF 実施チェックリスト

- [ ] Task 0: フェーズF各項目を `session start`, `in-progress warnings`, `pre-finish evidence`, `pre-release checks`, `routing` に分類し、責務を重ねない。
- [ ] Task 1: `session-start doctor` の workflow を作成し、環境・registry・dirty worktree・human review reminder を含める。
- [ ] Task 1-Verify: AGENTS の preflight をそのまま写すのではなく、実行順の checklist へ変換できていることを確認する。
- [ ] Task 2: `evidence-pack` の workflow / script を定義し、最低限の manifest schema を決める。
- [ ] Task 2-Verify: `skills`, `commands`, `tests`, `reviews`, `files_changed`, `known_gaps` を欠かさず記録できることを確認する。
- [ ] Task 3: optional dependency checker と risk labeler を agent / rule / hook のどこに置くか決め、判定根拠を明記する。
- [ ] Task 3-Verify: `import` 追加、`pyproject.toml` 変更、`gwexpy/fields/` 変更などの代表差分で期待ラベルが想定できることを確認する。
- [ ] Task 4: `docs-sync` と `metadata-checker` を定義し、release workflow との責務境界を調整する。
- [ ] Task 4-Verify: docs 更新不要ケースと metadata blocking issue が区別して書かれていることを確認する。
- [ ] Task 5: `task router` と skills README の quick routing を追加する。
- [ ] Task 5-Verify: 新規参加者が「何から使うか」を README だけで判断できることを確認する。
- [ ] Task 6: フェーズFの推奨導入順を見直し、A-E 実装後に段階追加できる形へ整理する。
- [ ] Task 6-Verify: すべてを一度に作らなくても、高優先度の F-2 / F-1 / F-7 から先行導入できることを確認する。

### 推奨実装順と理由

1. F-2 を最初に入れる。監査証跡は全フェーズ共通で効き、後続の workflow / agent の出力先にもなる。
2. F-1 を次に入れる。セッション開始時の事故を減らし、A-E の実装運用そのものを安定させる。
3. F-7 を先行導入する。レビューラベル候補があるだけで、人手レビューの見落としが減る。
4. F-3 を続ける。optional dependency は判定観点が広いので、risk labeler の骨格ができてから詳細化する。
5. F-4 と F-5 を整備する。docs / metadata 系は release workflow との境界調整が必要なため中盤で扱う。
6. F-6 を最後に入れる。router は他資産が揃ってから作る方が導線設計がぶれにくい。

### 想定所要時間

- 事前確認と責務分割: 15〜20 分
- F-1 / F-2 の計画化: 20〜30 分
- F-3 / F-7 の計画化: 20〜30 分
- F-4 / F-5 / F-6 の計画化: 25〜40 分
- 全体整合確認: 10〜15 分
- 合計目安: 90〜135 分

### リスクと切り戻し方針

- フェーズFは複数カテゴリを横断するため、1項目1責務を崩すと維持不能になる。workflow と agent の責務分離を先に固定する。
- 自動判定系は誤検知が増えると使われなくなるため、初期導入は warning / suggestion ベースに留める。
- `evidence-pack` を重く作りすぎると記録コストが上がるので、最初は最小 schema で導入し、後から項目を増やす。
- `task router` は情報の重複元になりやすいため、README を索引、workflow を判断手順、agents/skills を実体とする役割を崩さない。
- docs / metadata 系は release workflow と競合しやすいので、F-4 は docs drift、F-5 は metadata sync に責務を固定する。

---

## 実装優先度サマリー

| フェーズ | 優先度 | 理由 |
|---------|--------|------|
| A: Hooks 強化 | ★★★ | 毎回恩恵あり・設定変更のみ |
| B: Rules 整備 | ★★★ | AI の判断基準を即座に改善 |
| F: 運用ガードレール拡張 | ★★★ | セッション開始・完了・レビュー導線の抜け漏れを直接減らす |
| C: Agents 追加 | ★★  | 専門作業時に大きく貢献 |
| D: Workflows   | ★★  | リリース・負債消化に貢献 |
| E: Skills 移植 | ★   | 既存 skill が充実しているため後回し可 |

---

## 注記

- 各フェーズの詳細計画は実施時に改めて立てる
- フェーズ A・B は独立して実施可能
- フェーズ F は A-E と独立ではなく、A-D の成果を日常運用に定着させる横断フェーズとして後追い導入すると効果が高い
- フェーズ C の各エージェントも互いに独立して追加可能
- フェーズ E の skill 移植は、対応する phase0/phase1 作業を再実施する場合に合わせて行うのが効率的
