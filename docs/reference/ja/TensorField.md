# TensorField

`TensorField` は、`ScalarField` コンポーネントのコレクションとして表現されるテンソル値フィールドです。

## 概要

`TensorField` は `FieldDict` を拡張し、物理学におけるテンソル場（応力テンソル、歪みテンソル、計量摂動など）を表現します。コンポーネントはインデックスのタプル（例：ランク2テンソルの場合 `(i, j)`）によってインデックス付けされます。

## 主な機能

- **行列演算 (ランク2)**: `@` (行列積), `det()` (行列式), `trace()` (トレース), `symmetrize()` (対称化)
- **乗算のサポート**:
    - `TensorField @ VectorField -> VectorField`
    - `TensorField @ TensorField -> TensorField`
- **バッチ処理**: `fft_time_all()`, `filter_all()`, `resample_all()` など、全コンポーネントへの一括適用
- **算術演算**: スカラーとの乗算、加算、減算
- **可視化**: `plot_components()` (全成分をグリッド表示)
- **エクスポート**: NumPy との相互運用のための `to_array()` (ランク2の場合は6次元配列を出力)

## 基本的な使い方

```python
from gwexpy.fields import ScalarField, TensorField
import numpy as np
from astropy import units as u

# 応力テンソルのためのコンポーネントフィールド作成
f_pa = ScalarField(np.random.randn(100, 4, 4, 4), unit=u.Pa)

# 2x2 TensorField の作成
tf = TensorField({
    (0, 0): f_pa, (0, 1): f_pa * 0.1,
    (1, 0): f_pa * 0.1, (1, 1): f_pa * 0.8
}, rank=2)

# トレースの計算
tr = tf.trace()  # ScalarField を返す（対角成分の和）

# 行列式の計算
d = tf.det()  # 単位 Pa² の ScalarField を返す

# 対称化
tf_sym = tf.symmetrize()
```

## 行列とベクトルの相互作用

`TensorField` は `VectorField` オブジェクトに対して作用させることができます：

```python
from gwexpy.fields import VectorField

# 速度ベクトルの作成
v = VectorField({'x': ScalarField(np.ones((100, 4, 4, 4)), unit=u.m/u.s), 
                 'y': ScalarField(np.ones((100, 4, 4, 4)), unit=u.m/u.s)})

# ベクトルへのテンソルの適用（例：速度に対する応力の作用）
result_v = tf @ v  # VectorField を返す
```

## 関連項目

- [ScalarField](ScalarField.md) - 基底となるスカラー場クラス
- [VectorField](VectorField.md) - ベクトル値フィールド用
- [FieldDict](FieldDict.md) - 基底コレクションクラス
