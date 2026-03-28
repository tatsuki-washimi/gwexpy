# Work Report: Numerical Hardening (Scale Invariance)

**Date**: 2026-02-03 18:06:37
**Model(s)**: Gemini 2.0 Pro (Planning/Audit), Claude Opus 4.5 (Core/Exceptions), GPT-5.2 Codex (Algo), GPT-5.1 Codex (Formatting), Sonnet/Gemini (GUI/Tests)
**Status**: ✅ Completed

---

## Summary

`gwexpy` の数値的安定性（Numerical Stability）を強化し、低振幅重力波データ ($10^{-21}$) におけるサイレントな計算失敗や精度劣化を根絶しました。 AST解析（Deep Scan）により発見された160件以上のリスクに対し、Phase 0〜3 の4段階で体系的に修正を行い、最終的に検証用テストスイートによりその効果（Scale Invariance）を証明しました。

## Changes

### Files Added/Modified

- `gwexpy/numerics/`: 新規作成。`constants.py` で `probabilistic_epsilon` などを一元管理。`scaling.py` で内部標準化を提供。
- `gwexpy/signal/preprocessing/whitening.py`: `eps=1e-12` を廃止し動的計算へ変更。
- `gwexpy/timeseries/decomposition.py`: ICA の内部標準化と `tol` の相対化。
- `gwexpy/fitting/core.py`: MCMC 初期化の絶対ジッター `+1e-8` を削除。
- `gwexpy/types/series_matrix_math.py`: `det` → `slogdet`, `inv` の保護。
- `gwexpy/gui/`: ログプロットのフロア計算を動的に変更（-200dB フラットラインの解消）。
- `gwexpy/io/`, `gwexpy/collections/*`: 17箇所の例外握りつぶし (`except Exception: pass`) を修正。
- `tests/numerics/test_scale_invariance.py`: 新規作成。検証用テストスイート。

## Test Results

- **Scale Invariance Tests**: ✅ All passed (5 cases verified)
  - `test_whitening_invariant`: $10^{-21}$ でも Identity にならず正常動作確認
  - `test_safe_log`: log plot がフラットラインにならないことを確認
- **Existing Tests**: ✅ 2579 passed (リグレッションなし)

## Resolutions

### Bugs Fixed

- **Fatal**: Whitening が $10^{-24}$ 以下の分散で無効化されるバグを修正。
- **Fatal**: GUI ログプロットで $10^{-20}$ 以下の信号が完全に消失するバグを修正。
- **Critical**: MCMC が微小パラメータに対して巨大な絶対ジッターを加え、推定を破壊するバグを修正。
- **High**: ログやUIで微小値が `"0.00"` に丸め込まれる表示バグを修正。

### Features Added

- **`gwexpy.numerics`**: 数値定数とスケーリングロジックの Single Source of Truth。
- **Scale Invariance Verification**: 今後の開発で数値安定性を担保するためのテスト基盤。

## Performance Impact

- 実行速度への影響は軽微（標準化コストのみ）。
- 信頼性向上により、ユーザーのデバッグ時間（サイレントな失敗の調査）を大幅に削減。

## Next Steps

- 今後の全PRで `gwexpy.numerics` の使用を必須とする（Linterルールの追加を検討）。
- ユーザーガイドの「Numerical Stability」セクションを周知する。
