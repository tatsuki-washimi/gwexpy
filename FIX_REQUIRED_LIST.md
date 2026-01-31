# gwexpy 要修正リスト (Fix Required List)

動的検証（`verify_deep_compatibility.py`）により、`gwpy` との互換性や期待される動作において以下の問題が確認されました。

---

## ✅ 1. 内部属性の消失 (High Priority) - **修正済 (2026-01-31)**

- **問題**: `FrequencySeries` 等において、スライシング（`fs[10:20]`）や演算（`fs * 2`）を行うと、`gwexpy` が独自に追加した内部属性（`_gwex_fft_mode`, `_gwex_target_nfft` 等）が新しいインスタンスに引き継がれず消失する。
- **検証結果**: `verify_deep_compatibility.py` にて ~~FAIL~~ → **PASS**
- **原因**: `__array_finalize__` メソッドが実装されていないため、NumPy の配列生成メカニズムにおいて属性コピーが行われていなかった。
- **修正内容**:
  - `gwexpy/frequencyseries/frequencyseries.py` に `__array_finalize__` を実装
  - `_gwex_fft_mode`, `_gwex_target_nfft`, `_gwex_pad_left`, `_gwex_pad_right`, `_gwex_original_n` を親インスタンスからコピー
- **テスト**: `tests/test_compatibility_fixes.py::TestArrayFinalizeFrequencySeries`

---

## ✅ 2. `Plot.show()` 実行後のオブジェクト破棄 (Medium Priority) - **修正済 (2026-01-31)**

- **問題**: `Plot.show()` を実行すると、内部で `plt.close(self)` が呼ばれるため、その直後に `savefig()` 等の操作を行おうとするとエラーになる。
- **検証結果**: `verify_deep_compatibility.py` にて ~~FAIL~~ → **PASS**
- **原因**: Jupyter Notebook での二重描画を防ぐための処置（`plt.close(self)`）が無条件に適用されていた。
- **修正内容**:
  - `gwexpy/plot/plot.py` の `show()` メソッドに `close` パラメータを追加
  - デフォルト `close=True`（既存動作を維持）
  - `close=False` を指定すると、表示後も `savefig()` 等が可能
- **使用例**:
  ```python
  plot = Plot(data)
  plot.show(close=False)  # リソースを保持
  plot.savefig("output.png")  # これが動作する
  ```
- **テスト**: `tests/test_compatibility_fixes.py::TestPlotShowClose`

---

## ✅ 3. Jupyter環境での画像表示の不確実性 (Low-Medium Priority) - **対応済 (2026-01-31)**

- **問題**: `_repr_html_ = None` により二重描画は防止されるが、一部環境で画像が表示されない可能性。
- **対応内容**:
  - `gwexpy/plot/plot.py` に `_repr_png_` メソッドを明示的に実装
  - `_repr_html_` が無効でも、Jupyter が PNG フォールバックを使用して確実に表示
- **テスト**: `tests/test_compatibility_fixes.py::TestPlotReprPng`

---

## 検証結果サマリー

```
$ python verify_deep_compatibility.py

--- Test: Internal Attribute Preservation (__array_finalize__) ---
[PASS] Attribute Preservation

--- Test: Plot.show() Side Effects ---
[PASS] Plot.show() Side Effect
       Details: show(close=False) allows subsequent savefig()

--- Test: Method Signature Compatibility (whiten) ---
[PASS] Whiten Signature Match

--- Test: Mixed Unit Y-Labeling ---
[PASS] Mixed Unit Labeling
       Details: Y-Label suppressed (Safe)

========================================
SUMMARY: No critical issues found in this run.
```
