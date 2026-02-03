# スカラーフィールド信号処理 (ScalarField Signal Processing)

このチュートリアルでは、`ScalarField` クラスを使用した高度な信号処理手法について解説します。特に、空間ドメインと波数（K）ドメインの変換、および特定地点でのデータ抽出に焦点を当てます。

## 1. 空間FFT (Spatial FFT)

`ScalarField` は、時間軸（軸0）だけでなく、空間軸（軸1, 2, 3）に対しても FFT を適用できます。これにより、物理場の波連やモード構造を解析できます。

```python
# 空間全軸を FFT して Kドメイン（波数空間）に変換
field_k = field.fft_space()

print(f"Space domains: {field_k.space_domains}")  # ('k', 'k', 'k') と表示されます
```

## 2. 波長と波数の解析

Kドメインでは、各軸の波数を取得したり、特定の波数に対応する波長を計算したりできます。

```python
# 指定した空間軸の波長を取得
wavelengths = field_k.wavelength("kx")
```

## 3. 特定座標での時系列抽出 (Point Extraction)

4次元の場から、任意の空間座標 $(x, y, z)$ における時系列データを抽出できます。指定した座標がグリッド上にない場合は、自動的に補間（Interpolation）が行われます。

```python
# 座標 (0.5, 0.5, 0.5) での時系列を抽出
ts_at_point = field.extract_points(x=0.5, y=0.5, z=0.5)

ts_at_point.plot()
```

## 4. 2次元平面の抽出 (Slice extraction)

特定の軸を固定して、2次元の断面（Plane2D）を抽出することも容易です。

```python
# z=0 の平面を抽出
plane_z0 = field.sel(z=0)
plane_z0.plot()
```

---

より詳細な使い方は、[ScalarField 入門](field_scalar_intro.ipynb) も参照してください。