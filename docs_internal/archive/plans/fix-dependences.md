
# gwexpy 改修作業計画書
**対象**: `tatsuki-washimi/gwexpy`  
**目的**: Codex / Claude Code による安全かつ再現可能な改修を可能にする  
**作業主体**: Codex または Claude Code  
**更新日**: 2026-03-24

---

## 1. 背景

`gwexpy` は大規模ライブラリであり、公開 API・examples・notebooks・tests が比較的整っている一方で、いくつかの箇所で **import 時の登録処理** に依存している。特に `ConverterRegistry` はクラスレベルの状態を持ち、サブパッケージの `__init__.py` で constructor 登録を行う設計になっている。これにより、外部スクリプト・CI・コーディングエージェントがモジュールを読む順序によっては、未登録状態で lookup が走る余地がある。:contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

また、I/O 登録ユーティリティは `gwpy.io.registry.default_registry` へ reader / writer / identifier を登録しており、これも実質的にグローバル状態変更である。:contentReference[oaicite:2]{index=2}

Codex 側では `AGENTS.md` によるプロジェクト指示の付与が前提であり、Claude Code 側でも `CLAUDE.md` がプロジェクトメモリとして読み込まれる。Claude Code は `~/.claude/CLAUDE.md` と `./CLAUDE.md` を階層的に読み込み、`@path` で追加ファイルの import もできる。:contentReference[oaicite:3]{index=3}

---

## 2. ゴール

### 必達ゴール
1. `gwexpy` の利用に **import 順依存を残さない** か、少なくとも **明示的初期化で完全に制御可能** にする。  
2. Codex / Claude Code が、`gwexpy` の変更時に **同じ初期化手順・同じテスト手順** を踏めるようにする。  
3. 失敗時に、エージェントが自己修正しやすい **明確なエラーメッセージ** と **テスト** を整備する。  
4. プロジェクト内ドキュメント (`AGENTS.md`, `CLAUDE.md`, README) を整備し、エージェントが迷わない状態にする。

### 非ゴール
- 大規模 API 再設計
- 公開 API の全面改名
- 全モジュールのリファクタリング
- notebooks / examples の全面書き換え

---

## 3. 現状認識

### 3.1 既に良い点
- README に導入手順、extras、Quick Start、notebooks、tests が明記されている。:contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
- `TimeSeriesSignalMixin` などの API は docstring が厚く、引数・戻り値・例が比較的明確。:contentReference[oaicite:7]{index=7}
- `gwexpy` のパッケージ配置は flat layout で明示されており、エージェントが探索しやすい。:contentReference[oaicite:8]{index=8}

### 3.2 改修が必要な点
- `ConverterRegistry` は class-level singleton であり、constructor / converter をクラス変数に保持している。:contentReference[oaicite:9]{index=9}
- `gwexpy.timeseries.__init__` では import 時に constructor 登録が走る。:contentReference[oaicite:10]{index=10}
- I/O registration helper はグローバル `io_registry` に reader / writer / identifier を登録する。:contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

---

## 4. 方針

### 基本方針
- **import-time side effect を減らす**
- **初期化を明示 API に寄せる**
- **登録処理は idempotent にする**
- **失敗時メッセージは次アクションを含める**
- **エージェントに必要な情報は短く固定化する**

### エージェント運用方針
- Codex は `AGENTS.md` を参照して作業する。  
- Claude Code は `./CLAUDE.md` と必要に応じて `~/.claude/CLAUDE.md` を参照する。Claude Code では `@path` import が使えるため、共通手順ファイルを切り出して再利用可能とする。:contentReference[oaicite:13]{index=13}

---

## 5. 作業範囲

### 対象ファイル（第一段階）
- `gwexpy/interop/_registry.py`
- `gwexpy/timeseries/__init__.py`
- `gwexpy/timeseries/io/_registration.py`
- `gwexpy/__init__.py` または新規 `gwexpy/bootstrap.py`
- `README.md`
- 新規 `AGENTS.md`
- 新規 `CLAUDE.md`
- 新規 tests:
  - `tests/test_import_order.py`
  - 必要に応じて `tests/test_registry_bootstrap.py`

### 追加調査対象（第二段階）
- `gwexpy/frequencyseries/__init__.py`
- `gwexpy/spectrogram/__init__.py`
- `gwexpy/types/__init__.py`
- `gwexpy/plot/__init__.py`
- 他の import-time registration 実装

---

## 6. 実施タスク

## Phase 0: ブランチ・環境準備
### タスク
- 作業ブランチ作成
  - 例: `refactor/explicit-bootstrap-registry`
- 仮想環境作成
- editable install
- 現行テスト実行

### 完了条件
- `pip install -e .` が通る
- 現行 `python -m pytest` の結果を記録する

### 備考
README では `pip install .` と `python -m pytest` が案内されている。:contentReference[oaicite:14]{index=14} :contentReference[oaicite:15]{index=15}

---

## Phase 1: explicit bootstrap 導入
### 目的
import 時の副作用登録を減らし、明示的に初期化可能にする。

### タスク
1. `gwexpy/__init__.py` または `gwexpy/bootstrap.py` に `register_all()` を追加
2. `gwexpy.timeseries.__init__` の `register_constructor(...)` を `_register_constructors()` へ移動
3. 必要なら他サブパッケージにも同様の `_register_*()` を追加
4. `register_all()` からそれらを順に呼ぶ

### 期待成果物
- `gwexpy.register_all()` で constructor 群と I/O 群が登録される
- import しただけでは最低限の副作用にとどまる

### 完了条件
- `import gwexpy; gwexpy.register_all()` が成功する
- `ConverterRegistry.get_constructor("TimeSeries")` が bootstrap 後に成功する

---

## Phase 2: registry の堅牢化
### 目的
未初期化時の failure mode を改善し、エージェントが自己修正しやすい状態にする。

### タスク
1. `ConverterRegistry.get_constructor()` / `get_converter()` の `KeyError` 文言を改善
2. メッセージ内に次アクションを含める
   - 例: `gwexpy.register_all()` を呼ぶ
3. `register_constructor()` / `register_converter()` の idempotency 方針を決める
   - 同一登録は無害に上書き
   - 異なるオブジェクト登録は warning or exception
4. 必要なら lock を追加

### 完了条件
- 未登録時のエラーが「何が足りないか」「どう直すか」を説明する
- 重複登録が CI で不安定要因にならない

---

## Phase 3: I/O registration の整理
### 目的
I/O handler 登録も explicit bootstrap に統一する。

### タスク
1. どこで `register_timeseries_format()` が呼ばれているか全件確認
2. import 時登録があれば `register_all()` 経由へ移動
3. registration が複数回走っても安全なことを確認
4. 必要なら `register_io_formats()` を独立関数化

### 完了条件
- I/O format 登録が明示的に制御できる
- 複数回呼び出してもテストが安定する

---

## Phase 4: import-order テスト追加
### 目的
順序依存の退行を防ぐ。

### タスク
1. `tests/test_import_order.py` を追加
2. 想定シナリオ
   - `ConverterRegistry` だけ import した状態で lookup
   - `gwexpy.register_all()` 後の lookup
   - `gwexpy.timeseries` 直接 import 後の lookup
   - I/O registration の多重実行
3. 必要に応じて thread-safe / repeated bootstrap テストを追加

### 完了条件
- CI で import order regression を検出できる

---

## Phase 5: エージェント向け文書整備
### 目的
Codex / Claude Code が正しい手順で作業できるようにする。

### タスク
1. repo root に `AGENTS.md` 作成
2. repo root に `CLAUDE.md` 作成
3. README に「Agent setup / bootstrap」節を追加
4. 必要なら `docs/developers/agent_bootstrap.md` を新設
5. `CLAUDE.md` から `@docs/developers/agent_bootstrap.md` を import する構成も検討

### 記載すべき内容
- Python version
- venv 作成
- `pip install -e .`
- `gwexpy.register_all()` を先に呼ぶ
- 実行すべき tests
- 公開 API を壊さない
- examples / notebooks / tests を先に読む

### 完了条件
- Codex / Claude Code に初回プロンプトなしでも一定の正しい作業導線がある

---

## 7. Codex 用作業手順

## 7.1 事前配置
- repo root に `AGENTS.md` を置く
- そこに以下を含める
  - setup
  - bootstrap
  - test
  - public API compatibility
  - prohibited actions

## 7.2 Codex への指示テンプレート
```text
This repository uses explicit bootstrap for stable registry behavior.
Before modifying code:
1. Create and activate a Python 3.11 virtualenv
2. Run `pip install -e .`
3. Run `python -c "import gwexpy; gwexpy.register_all()"`
4. Run `python -m pytest`

Task:
- Implement explicit bootstrap for registry and IO registration
- Add import-order regression tests
- Keep public API stable
- Update README and AGENTS.md accordingly
````

## 7.3 Codex での作業単位

* 1 PR / 1 Phase を原則とする
* 推奨分割

  * PR1: bootstrap API
  * PR2: registry hardening
  * PR3: IO registration
  * PR4: tests + docs

---

## 8. Claude Code 用作業手順

## 8.1 事前配置

* repo root に `CLAUDE.md` を置く
* 必要なら `@docs/developers/agent_bootstrap.md` を import

## 8.2 `CLAUDE.md` に含めるべき最小事項

* `pip install -e .`
* `gwexpy.register_all()` を先に呼ぶ
* `python -m pytest`
* `tests/`, `notebooks/`, `README.md` を先に確認
* 公開 API は明示依頼なしに壊さない

## 8.3 Claude Code への依頼テンプレート

```text
Refactor this repository to remove import-order fragility around registry initialization.

Constraints:
- Keep public API stable
- Introduce explicit bootstrap (`gwexpy.register_all()`)
- Add import-order regression tests
- Update README and CLAUDE.md
- Keep changes small and reviewable

Before coding:
- Set up Python 3.11 env
- pip install -e .
- run current tests
```

---

## 9. 受け入れ基準

以下をすべて満たしたら完了とする。

### 機能

* `gwexpy.register_all()` が存在する
* bootstrap 後に registry lookup が安定して成功する
* I/O registration が bootstrap 経由で制御できる

### 品質

* 既存テストが通る
* 新規 import-order テストが通る
* 同一 bootstrap の複数回実行で壊れない

### ドキュメント

* README に bootstrap 手順がある
* `AGENTS.md` がある
* `CLAUDE.md` がある

### エージェント運用

* Codex / Claude Code どちらでも、初期化漏れなしで作業を開始できる

---

## 10. リスクと対策

### リスク 1: 既存ユーザーが import-time registration を暗黙前提にしている

**対策**

* 完全削除ではなく、当面は後方互換レイヤを残す
* deprecation note を README と release note に記載

### リスク 2: register_all の責務が肥大化する

**対策**

* `register_constructors()`, `register_io_formats()` などに分割し、`register_all()` は orchestration のみとする

### リスク 3: 多重登録時の挙動が曖昧

**対策**

* idempotency ポリシーを明文化
* tests で固定化

### リスク 4: Codex / Claude Code が docs を読み飛ばす

**対策**

* `AGENTS.md` / `CLAUDE.md` は短くする
* 重要コマンドを先頭に置く
* README より優先度の高い位置に配置する

---

## 11. 推奨実施順

1. Phase 0: 環境準備
2. Phase 1: explicit bootstrap
3. Phase 2: registry hardening
4. Phase 4: import-order tests
5. Phase 3: I/O registration 整理
6. Phase 5: docs / agent files

> 理由: まず registry の主問題を片付けてから、テストで固定化し、その後 I/O と文書を整備する。

---

## 12. 作業見積もり（相対）

* Phase 0: 小
* Phase 1: 中
* Phase 2: 小〜中
* Phase 3: 中
* Phase 4: 小
* Phase 5: 小

全体として **中規模改修**。
大規模 API 再設計ではなく、**初期化・登録・テスト・文書の整備** が中心。

---

## 13. 参考情報

### gwexpy 側

* `ConverterRegistry` は class-level singleton。
* `gwexpy.timeseries.__init__` で import 時登録。
* I/O registration helper は global registry へ登録。
* README には install / tests / notebooks 情報がある。 

### Claude Code

* `./CLAUDE.md`, `~/.claude/CLAUDE.md` を階層的に読む。
* `@path` import が可能。([Claude API Docs][1])

### Codex

* Codex 系のエージェントモデルは agentic coding 向けに設計されており、プロジェクト側の作業指示を明示するほど安定運用しやすい。([OpenAI Developers][2])


[1]: https://docs.anthropic.com/en/docs/claude-code/memory?utm_source=chatgpt.com "Manage Claude's memory - Anthropic"
[2]: https://developers.openai.com/api/docs/models/gpt-5.3-codex?utm_source=chatgpt.com "GPT-5.3-Codex Model | OpenAI API"
