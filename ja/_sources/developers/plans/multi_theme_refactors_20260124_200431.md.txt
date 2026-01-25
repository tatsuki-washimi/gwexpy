# 作業計画書: Multi-theme 安定化ロードマップ
**作成日時**: 2026-01-24 20:04:31
**関連タスク**: `docs/developers/plans/plan_refactor_exceptions_20260124_152245.md`

---

## 1. Objectives & Goals
1. **型チェック拡充**  
   - `pyproject.toml` の `gwexpy.types.series_matrix_*` 周りの exclude を段階的に削り、MyPy で型安全性を検証して安全な実装を示す。
   - 必要な型注釈追加、`cast`/`assert`/局所的 Any を使った制御、必要最小限の `# type: ignore` とその理由を残す。
2. **TODO／ドキュメント整合**  
   - `gwexpy/spectrogram/matrix.py` の docstring を、既存の `__reduce__`/`__setstate__` 実装に沿うように書き換え、復元されない metadata があれば再現条件と issue 参照を明示。
   - `gwexpy/spectrogram/matrix_core.py` の axis-operation TODO を「設計メモ/issue参照」に再整理し状態を明確に giữ.
3. **例外処理監査**  
   - `gwexpy/fitting/core.py`, `gwexpy/types/series_matrix_validation_mixin.py`, `gwexpy/plot/skymap.py`, `gwexpy/interop/specutils_.py` の `except Exception:` 等を具体例外やログ付きに改善しつつ挙動は変えない。

## 2. Detailed Roadmap (by Phase)
### Phase 1: Type-check Baseline (30–40分)
* Inspect `pyproject.toml` excludes under `[tool.mypy]`, un-comment one `gwexpy.types.series_matrix_*` module at a time (start with the simplest, such as `series_matrix_math` or `series_matrix_validation`).  
* Run `mypy` on the targeted module(s), capture errors, and fix via annotations/casts/asserts before proceeding to the next module.  
* Document each module’s before/after status inline in the change (e.g., comment explaining why a `cast` is needed).  
* Repeat for 1–3 modules while keeping per-phase scope manageable.

### Phase 2: TODO/Docstring Alignment (20–30分)
* Read `gwexpy/spectrogram/matrix.py` docstring; note current metadata flows inside `__reduce__`/`__setstate__`. Rewrite narrative to describe what is preserved/unrestored and remove outdated `TODO`.  
* If any metadata truly fails to round-trip, reproduce the limitation, describe reproduction steps, and convert to `See: #<issue>` instead of `TODO`.  
* For `gwexpy/spectrogram/matrix_core.py`, convert the “TODO: axis swapping support” text into a concise design note or issue reference that clarifies current limitations without claiming upcoming implementation.

### Phase 3: Exception Handling Audit (40–50分)
* For each target file, locate broad `except` clauses, analyze surrounding logic, and determine the minimal necessary set of exception types or logging.  
* Replace `except Exception:` with `(ValueError, TypeError)` etc. when obvious; otherwise wrap with `logging.exception(...)` and keep the guard minimal.  
* If an exact exception is uncertain, at least log the stack (`logging.exception`) and note the reason in a comment.  
* Leave behavior untouched by re-raising or returning fallback values consistent with prior logic.

### Phase 4: Cross-cutting Checks (15分)
* Run `ruff check .`, `mypy .`, and `pytest tests/types tests/spectrogram tests/fitting`.  
* Capture and summarize any residual failures/limitations for follow-up.

## 3. Testing & Verification Plan
- `ruff check .` (repo-wide, since policy demands this after changes).  
- `mypy .` to verify the newly re-included modules and to spot new typing regressions.  
- `pytest tests/types tests/spectrogram tests/fitting` (per instructions; run `pytest` full suite only if time/quota allows).  
- Additional targeted tests (if needed) for exception-handled paths (e.g., just run `pytest tests/fitting` again after edits to confirm nothing regressed).

## 4. Models, Recommended Skills, and Effort Estimates
### Model Selection
* **Primary**: Claude Opus 4.5 (Thinking) — best at multi-step reasoning, tailoring both type-check logic and docstring narratives while keeping physical/logical nuance.  
* **Backup**: GPT-5.1-Codex-Max for larger-scale refactor passes if the session becomes more implementation-heavy.

### Recommended Skills
1. `analyze_code` — to dissect the resource-heavy `series_matrix_*` modules and the targeted spectrogram/exception files.  
2. `lint` + `test_code` — to satisfy the mandated `ruff`/`pytest` runs.  
3. `sync_docs` — when rewriting docstrings to match implementation.  
4. `wrap_up_gwexpy` — once the three themes are complete and everything is validated.

### Effort Estimate
* **Estimated Total Time**: ~110 minutes  
* **Estimated Quota Consumption**: Medium (mix of deep reasoning + code edits).  
* **Breakdown**: Phase 1 (35min), Phase 2 (25min), Phase 3 (40min), Phase 4 (10min).  
* **Concerns**: Restoring excluded modules may uncover deep typing puzzles or require additional imports; docstring realignment must be carefully worded to avoid regressions; some broadly catching `except` blocks in `fitting/core.py` may rely on third-party behaviors, so logging/narrowing must preserve prior fallbacks.

## 5. Approval Request
May I proceed with this phased plan (type-check expansions, TODO cleanup, exception audit)? Once approved, I will begin with Phase 1 and keep you informed before moving between themes.
