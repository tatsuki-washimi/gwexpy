---
name: numerical-audit
description: GWexpy 数値アルゴリズム追加・修正時の安全性監査ワークフロー。silent failure, Death Floats, scale-invariance を段階的に確認する。
trigger: manual
---

# GWexpy Numerical Audit Workflow

Use this workflow when implementing or auditing numerical processing code in `gwexpy`.

## Phase 1: Risk Inventory
1. Identify all `try...except` blocks and check for broad exception handling.
2. Identify all numerical constants (e.g., `1e-6`) and determine if they should be scale-aware.
3. Check for external library dependencies and fallback paths.

## Phase 2: Unsilencing Exceptions
1. Replace broad `except Exception:` with narrow types or ensure `logger.exception()` is called.
2. Verify that meaningful error messages are provided for data-related failures.

## Phase 3: Scale Review
1. Replace hardcoded magic floats with scale-aware defaults (relative to data RMS or peak).
2. Integrate with `gwexpy.numerics` if applicable.
3. Ensure the algorithm can handle GW strain values (~1e-21) without underflow or catastrophic loss of precision.

## Phase 4: Test Design
1. **Scale-Invariance Test**: Verify `f(X) ≡ f(1e-20 * X)` within reasonable bounds.
2. **Tiny-Signal Test**: Run with `1e-21` magnitude data.
3. **Edge Cases**: Zero, NaN, Inf, and extremely short data lengths.

## Phase 5: Verification & Labeling
1. Run `pytest` for the modified module.
2. Run `ruff` and `mypy` on the file.
3. If risk remains, add `needs-physics-review` and `needs-scale-invariance-check` labels.

## Exit Criteria
- Silent failures are eliminated.
- Numerical constants are scale-aware.
- Scale-invariance is verified through unit tests.
