# Step 1 & 2 Integrated Summary

**Generated:** 2026-02-03
**Scope:** Initial Audit + Deep Scan + Ultra-Deep Scan

## 1. Risk Distribution

The audit phase is complete. We have identified **over 120 (approx 160 including minor formatting)** numerical risks.

| Category | Count | Primary Impact |
| :--- | :--- | :--- |
| **Unprotected Calls** | 80+ | `inv`, `solve`, `welch` etc. crashing or returning garbage on unscaled data. |
| **Formatting/Precision** | 20+ | Logs and UI showing "0.00" for valid signals ($10^{-21}$). |
| **Exception Swallowing** | 17 | Hiding all of the above errors. |
| **Dangerous Defaults** | 7 | Hardcoded `eps`/`tol` destroying signals. |
| **Hardcoded Floats** | 4 | `+ 1e-20` offset in plots, MCMC jitter. |

## 2. Validation Against Plan

The `numerical_hardening_plan.md` has been updated to reflect these realities.

* **Phase 0 (Opus)** targets the 17 Swallowing cases.
* **Phase 1 (Opus)** targets the Core Architecture (`gwexpy.numerics`) to solve the 7 Defaults.
* **Phase 2 (Codex)** targets the Unprotected Calls (Math/Algo) and Hardcoded Floats.
* **Phase 3 (Codex/Sonnet)** targets the Formatting/Precision risks.

## 3. Next Actions (Immediate)

1. **Execute Phase 0**: Unsilence the codebase.
2. **Execute Phase 1**: Establish `gwexpy.numerics`.
3. **Execute Phase 2**: Fix the "Fatal" math bugs (Whitening, MCMC).

This summary serves as the "Go/No-Go" evidence for proceeding to implementation.
