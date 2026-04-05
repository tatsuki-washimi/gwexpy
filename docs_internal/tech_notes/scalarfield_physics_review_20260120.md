# ScalarField Physics Review

**Last Updated**: 2026-01-22T21:08+09:00

## Scope

- Targets: `ScalarField`, `FieldList`, `FieldDict`
- Focus: FFT domain transitions, axis handling, physical unit consistency
- Files reviewed: `gwexpy/fields/scalar.py`, `gwexpy/fields/collections.py`, `gwexpy/fields/base.py`, `gwexpy/types/axis.py`, `gwexpy/types/array4d.py`

## Summary

The `ScalarField` class is **mathematically and physically correct**. All core FFT workflows align with GWpy conventions, including proper axis validation, origin offset preservation, Nyquist bin handling, and real-signal validation. Collection classes enforce consistent coordinate arrays across batched fields.

## Verification Results

| カテゴリ | ステータス | 詳細 |
|---------|----------|------|
| **次元解析（単位処理)** | ✅ Pass | `astropy.units` が正しく使用されており、ドメイン遷移時に軸の単位が適切に変換される |
| **FFT正規化** | ✅ Pass | GWpy互換の one-sided spectrum 正規化が正しく実装されている（DC bin非倍増、Nyquist bin非倍増） |
| **Parseval's theorem** | ✅ Pass | 正規化係数補正後の検証でエネルギー比 1.0 (誤差 < 1%) |
| **FFT可逆性** | ✅ Pass | `fft_time()` ↔ `ifft_time()` および `fft_space()` ↔ `ifft_space()` の往復誤差 < 1e-10 |
| **軸検証** | ✅ Pass | 軸長 ≥ 2、等間隔性、単調性を検証済み |
| **軸オフセット保持** | ✅ Pass | `fft_time` 時に時間軸オフセットが保存され、`ifft_time` 時に復元される |
| **波数演算** | ✅ Pass | 角波数 `k = 2π/λ` が正しく計算されている |
| **コレクション検証** | ✅ Pass | `FieldList`/`FieldDict` が座標配列の整合性を許容差内で検証する |

## Original Findings (2026-01-20)

The following issues were identified in the initial review:

1. **High**: `fft_time`/`ifft_time` computed `dt`/`df` from the first two samples without validating axis regularity or size >= 2.
2. **Medium**: Time and space inverse transforms rebuilt axes with a zero origin, ignoring any original offset.
3. **Medium**: One-sided normalization doubled all non-DC bins; for even-length FFTs the Nyquist bin should not be doubled.
4. **Medium**: `rfft/irfft` implicitly assumed real-valued signals and Hermitian frequency data, but no validation enforced that constraint.
5. **Low**: Spatial axis monotonicity was not enforced, and `fft_space` used signed `dx` while `ifft_space` used `abs(dk)`.
6. **Low**: `FieldList`/`FieldDict` validation checked only units/domains/names, not the coordinate arrays.

## Resolution Status

All 6 issues have been **resolved**:

| Issue | Fix Location | Implementation |
|-------|-------------|----------------|
| 1. Axis validation | `_validate_axis_for_fft()` L192-221 | Length >= 2 and regularity check via `AxisDescriptor.regular` |
| 2. Origin preservation | L273-274, L379-385 | `_axis0_offset` metadata stored/restored |
| 3. Nyquist bin | L282-287 | `dft[1:-1, ...]` for even nfft, `dft[1:, ...]` for odd |
| 4. Real-signal validation | L264-268 | `np.iscomplexobj()` check raises `TypeError` |
| 5. Spatial monotonicity | L479-485, L520-522, L640-643 | `np.diff` check and signed delta handling |
| 6. Collection validation | `_AXIS_RTOL`, `_AXIS_ATOL` | `np.allclose()` comparison of axis coordinates |

## Test Coverage

- 121 tests in `tests/fields/` → All passing
- Key test files:
  - `test_scalarfield_fft_time.py`: 20 tests (normalization, reversibility, errors)
  - `test_scalarfield_fft_space.py`: 25 tests (wavenumber, reversibility, errors)
  - `test_scalarfield_collections.py`: 24 tests (validation, batch FFT)

## Verification Script

`scripts/verify_scalarfield_physics.py` performs 5 physics checks:

1. **Parseval's theorem** (energy conservation with proper normalization undoing)
2. **Time-FFT reversibility** (round-trip error < 1e-10)
3. **Spatial wavenumber calculation** (k = 2π/λ verified at peak)
4. **Spatial-FFT reversibility** (round-trip error < 1e-10)
5. **Unit consistency** (data unit preserved, axis units correct)

All checks pass.

## Notes

- **2026-01-20**: Initial review identified 6 physics/math issues
- **2026-01-20**: All 6 code fixes implemented
- **2026-01-22**: Re-verification performed; all issues confirmed resolved
- **2026-01-22**: Fixed `verify_scalarfield_physics.py` to use correct Parseval verification (undo bin-doubling before energy comparison)

## References

- GWpy `TimeSeries.fft()`: Reference for one-sided spectrum normalization
- GWpy `FrequencySeries.ifft()`: Reference for inverse normalization
- NumPy FFT documentation: `rfft`, `irfft`, `fftfreq` conventions
