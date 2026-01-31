# gwexpy 互換性修正計画

**作成日**: 2026-01-31  
**目的**: gwpy と gwexpy のドロップイン互換性を確保するための修正

---

## 0. 背景

gwexpy は gwpy のドロップイン置き換えとして設計されているが、動的検証および静的解析の結果、以下の互換性問題が発見された：

1. **内部属性の消失** (優先度: 高)
2. **Plot.show() 後のオブジェクト破棄** (優先度: 中)
3. **Jupyter環境での画像表示の不確実性** (優先度: 低〜中)

---

## 1. 問題分析

### 1.1 内部属性の消失

| 項目         | 詳細                                                                                           |
| ------------ | ---------------------------------------------------------------------------------------------- |
| **症状**     | `fs[1:10]` や `fs * 2` の後、`_gwex_fft_mode` 等が消失                                         |
| **原因**     | `FrequencySeries` に `__array_finalize__` が未実装                                             |
| **影響**     | `ifft(mode="auto")` がデフォルト（gwpy互換）モードに戻る                                       |
| **関連属性** | `_gwex_fft_mode`, `_gwex_target_nfft`, `_gwex_pad_left`, `_gwex_pad_right`, `_gwex_original_n` |

**現状確認**:

- `FrequencySeries` は `BaseFrequencySeries` (gwpy) を継承
- 他のクラス (`Array`, `Array2D`, `ScalarField` 等) には `__array_finalize__` が実装済み
- 属性は `TimeSeries.fft()` で設定され、`FrequencySeries.ifft()` で使用される

### 1.2 Plot.show() 後のオブジェクト破棄

| 項目     | 詳細                                               |
| -------- | -------------------------------------------------- |
| **症状** | `plot.show()` 後に `plot.savefig()` でエラー       |
| **原因** | `show()` 内で無条件に `plt.close(self)` を呼び出し |
| **影響** | 「表示してから保存」ワークフローが動作しない       |

**現在の実装** (`gwexpy/plot/plot.py:701-707`):

```python
def show(self, warn=True):
    """Show the figure and close it to prevent double display."""
    import matplotlib.pyplot as plt
    plt.show()
    plt.close(self)
    return None
```

### 1.3 Jupyter環境での画像表示

| 項目     | 詳細                                                        |
| -------- | ----------------------------------------------------------- |
| **症状** | `_repr_html_ = None` により二重描画は防止されるが表示不安定 |
| **原因** | `_repr_png_` 等が未実装の場合、画像表示されない可能性       |
| **影響** | 一部のJupyter環境で画像が表示されないリスク                 |

**現在の実装** (`gwexpy/plot/plot.py:67-68`):

```python
_repr_html_ = None
```

---

## 2. 修正計画

### Phase 1: 内部属性の消失修正 (高優先度)

#### 1.1 対象クラスの特定

`__array_finalize__` を実装すべきクラス：

- `FrequencySeries` (`gwexpy/frequencyseries/frequencyseries.py`)
- `TimeSeries` (`gwexpy/timeseries/timeseries.py`) - 必要に応じて

#### 1.2 実装内容

```python
def __array_finalize__(self, obj):
    """Propagate gwexpy-specific attributes through NumPy operations."""
    super().__array_finalize__(obj)
    if obj is None:
        return

    # Copy gwexpy internal attributes (prefix: _gwex_)
    for attr in (
        "_gwex_fft_mode",
        "_gwex_target_nfft",
        "_gwex_pad_left",
        "_gwex_pad_right",
        "_gwex_original_n",
    ):
        if hasattr(obj, attr):
            setattr(self, attr, getattr(obj, attr))
```

#### 1.3 テスト計画

```python
def test_array_finalize_preserves_attributes():
    """Test that gwexpy attributes survive slicing and arithmetic."""
    from gwexpy.timeseries import TimeSeries
    ts = TimeSeries([1, 2, 3, 4], dt=1)
    fs = ts.fft(mode="transient")

    # Test slicing
    fs_slice = fs[1:3]
    assert fs_slice._gwex_fft_mode == "transient"

    # Test arithmetic
    fs_mult = fs * 2
    assert fs_mult._gwex_fft_mode == "transient"

    # Test ifft auto mode
    ts_back = fs_mult.ifft(mode="auto")
    # Should use transient mode, not gwpy mode
```

---

### Phase 2: Plot.show() の修正 (中優先度)

#### 2.1 解決案

**案A**: `show()` 後のリソース解放を遅延

```python
def show(self, warn=True, close=True):
    """Show the figure.

    Parameters
    ----------
    close : bool, default True
        If True, close the figure after showing to free resources.
        Set to False if you need to save the figure afterward.
    """
    import matplotlib.pyplot as plt
    plt.show()
    if close:
        plt.close(self)
```

**案B**: savefig()前に明示的にshow()を呼ばない運用を推奨

```python
def show(self, warn=True):
    """Show the figure and close it.

    .. warning::
        This method closes the figure. If you need to save the figure,
        call savefig() before show(), or use the `close=False` parameter.
    """
    ...
```

**推奨**: 案Aを採用（gwpyとの互換性維持）

#### 2.2 テスト計画

```python
def test_show_then_savefig():
    """Test that savefig works after show with close=False."""
    from gwexpy.plot import Plot
    from gwexpy.timeseries import TimeSeries
    import tempfile

    ts = TimeSeries([1, 2, 3], dt=1)
    plot = Plot(ts)
    plot.show(close=False)

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        plot.savefig(f.name)  # Should not raise
```

---

### Phase 3: Jupyter表示の安定化 (低〜中優先度)

#### 3.1 調査項目

1. gwpy の `_repr_png_` / `_repr_svg_` 実装を確認
2. matplotlib.Figure の repr メソッドを確認
3. 各種 Jupyter 環境でのフォールバック動作を確認

#### 3.2 解決案

**案A**: `_repr_png_` を明示的に実装

```python
def _repr_png_(self):
    """Return PNG representation for Jupyter."""
    from io import BytesIO
    buf = BytesIO()
    self.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()
```

**案B**: matplotlib.Figure のデフォルト repr を継承

```python
# _repr_html_ = None を削除し、代わりに
# _ipython_display_ で制御
def _ipython_display_(self):
    from IPython.display import display
    # ... 一度だけ表示する制御
```

#### 3.3 テスト計画

```python
def test_jupyter_repr():
    """Test that plot has working repr methods."""
    from gwexpy.plot import Plot
    from gwexpy.timeseries import TimeSeries

    ts = TimeSeries([1, 2, 3], dt=1)
    plot = Plot(ts)

    # At least one repr should work
    has_repr = (
        hasattr(plot, '_repr_png_') or
        hasattr(plot, '_repr_svg_') or
        hasattr(plot, '_repr_html_')
    )
    assert has_repr
```

---

## 3. 作業手順

### Step 1: Phase 1 実装 (約30分)

1. `FrequencySeries` に `__array_finalize__` を追加
2. 必要に応じて `TimeSeries` にも追加
3. ユニットテストの作成・実行
4. lint/mypy チェック

### Step 2: Phase 2 実装 (約20分)

1. `Plot.show()` に `close` パラメータを追加
2. docstring の更新
3. ユニットテストの作成・実行

### Step 3: Phase 3 調査・実装 (約30分)

1. gwpy および matplotlib の実装調査
2. `_repr_png_` の実装またはフォールバック戦略の決定
3. 各環境でのテスト

### Step 4: 検証・リグレッションテスト (約20分)

1. `verify_deep_compatibility.py` の再実行
2. 全体テストスイートの実行
3. ドキュメント更新

---

## 4. 推奨モデル・スキル・工数

| 項目             | 推奨                                    |
| ---------------- | --------------------------------------- |
| **LLM モデル**   | Gemini 2.5 Pro (コード修正のため)       |
| **補助スキル**   | `fix_errors`, `lint_check`, `run_tests` |
| **推定時間**     | 約 1.5〜2 時間                          |
| **推定トークン** | 中程度 (既存コードの参照多め)           |

---

## 5. 承認待ち

上記の計画について、確認・承認をお願いします：

- [ ] Phase 1 の実装方針（`__array_finalize__` の実装）
- [ ] Phase 2 の実装方針（`close` パラメータの追加）
- [ ] Phase 3 の優先度と実装方針

ご指示があれば修正計画の調整も可能です。
