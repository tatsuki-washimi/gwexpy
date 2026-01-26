# BifrequencyMap

**継承元:** `FrequencySeriesMatrix`

2つの周波数軸間の関係を表すマップ（周波数1 → 周波数2）。
通常、入力周波数（`frequency1`）から出力周波数（`frequency2`）へエネルギーが伝達される伝達関数、結合関数、散乱行列を表すために使用されます。

## 構成要素

- **周波数1 (`frequency1`)**: 入力またはソース周波数軸（列）。
- **周波数2 (`frequency2`)**: 出力またはターゲット周波数軸（行）。
- **値**: 各 (f2, f1) 点での結合強度または伝達関数スカラー。

## メソッド

### `from_points`

```python
from_points(cls, data, f2, f1, **kwargs)
```

2D配列と2つの周波数軸から `BifrequencyMap` を作成します。

パラメータ
----------
data : array-like
    形状 (len(f2), len(f1)) の2D配列。
f2 : array-like
    周波数軸2（行、出力）。
f1 : array-like
    周波数軸1（列、入力）。
**kwargs
    コンストラクタに渡される追加引数（name, unit など）。

戻り値
-------
BifrequencyMap

### `propagate`

```python
propagate(self, input_spectrum, interpolate=True, fill_value=0)
```

入力スペクトルをマップを通して伝播させ、出力スペクトルを計算します。
行列積を実行: $S_{\text{out}} = M \cdot S_{\text{in}}$

パラメータ
----------
input_spectrum : `FrequencySeries`
    入力スペクトル $S_{\text{in}}(f_1)$。
interpolate : bool, optional
    True の場合、入力スペクトルをマップの `frequency1` 軸に補間します。
fill_value : float, optional
    範囲外の補間に使用する値。

戻り値
-------
`FrequencySeries`
    投影された出力スペクトル $S_{\text{out}}(f_2)$。

### `convolute`

```python
convolute(self, input_spectrum, interpolate=True, fill_value=0)
```

マップと入力スペクトルを畳み込みます（f1に沿った積分）。
計算: $S_{\text{out}}(f_2) = \int M(f_2, f_1) S_{\text{in}}(f_1) df_1$

パラメータ
----------
input_spectrum : `FrequencySeries`
    入力スペクトル。
interpolate : bool, optional
    True の場合、入力スペクトルを補間します。

戻り値
-------
`FrequencySeries`
    周波数積分で調整された単位を持つ出力スペクトル。

### `diagonal`

```python
diagonal(self, method='mean', bins=None, absolute=False, **kwargs)
```

対角軸（$f_2 - f_1$）に沿った統計量を計算します。

パラメータ
----------
method : str, optional
    統計手法: 'mean', 'median', 'max', 'min', 'std', 'rms', 'percentile'。
    デフォルトで NaN を無視します。
bins : int or array-like, optional
    ビン数。None の場合、解像度から自動的に決定されます。
absolute : bool, optional
    True の場合、$|f_2 - f_1|$ に沿って統計を計算します。
**kwargs
    追加引数（例: `percentile` 値）。

戻り値
-------
`FrequencySeries`
    周波数差の関数としての統計量。

### `get_slice`

```python
get_slice(self, at, axis='f1', xaxis='remaining')
```

特定の周波数でマップのスライスを抽出します。

パラメータ
----------
at : float
    抽出する周波数値。
axis : str, optional
    固定する軸: 'f1' または 'f2'。
xaxis : str, optional
    結果のx軸の定義: 'remaining', 'diff', 'abs_diff' など。

戻り値
-------
`FrequencySeries`
    抽出された1Dスライス。

### `plot`

```python
plot(self, **kwargs)
```

マップを2D画像（スペクトログラム風）としてプロットします。

### `plot_lines`

```python
plot_lines(self, xaxis='f1', color='f2', num_lines=None, ax=None, cmap=None, **kwargs)
```

マップを1D線のセットとしてプロットします。

パラメータ
----------
xaxis : str, optional
    線のX軸: 'f1', 'f2', 'diff' など。
color : str, optional
    色付け/スライスのパラメータ: 'f1', 'f2'。
num_lines : int, optional
    プロットする最大線数。
ax : matplotlib Axes, optional
    プロットするAxes。
cmap : colormap, optional
    線のカラーマップ。

戻り値
-------
matplotlib Axes
