# GWpy API UX Policy (2026-03-03)

## Goal

`gwpy` と同名の API については、利用者が `gwpy` と同じ呼び方で使えることを優先する。

## Rules

1. `gwpy` 同名メソッドは、`gwpy` の引数順・位置引数許容を原則として維持する。  
2. `gwexpy` 拡張引数は、既存の `gwpy` 呼び出しを壊さない形（keyword-only 追加など）で導入する。  
3. 互換レイヤを追加した箇所は、回帰テストで「gwpy流呼び出し」と「gwexpy拡張呼び出し」の両方を固定する。  
4. `gwpy` に存在しない `gwexpy` 独自 API（例: `lock_in`, `TimeSeriesDict.csd`）は、互換対象外として明示する。  
5. 並列引数（`nproc` / `parallel`）は「全体統一」ではなく「メソッドごとに gwpy 互換」を優先する。  

## Scope of This Round

- `TimeSeries.transfer_function`:
  - `gwpy` と同じ呼び順（`other, fftlength, overlap, window, average`）を受理する。
- `TimeSeriesDict/List.csd` と `coherence`:
  - `fftlength` / `overlap` の位置引数呼び出しを受理する（互換レイヤ）。

## Verification

- `tests/timeseries/test_transfer_function_compat.py`
- `tests/timeseries/test_collections_spectral_compat.py`
- `tests/timeseries/test_fft_param_compat.py`
