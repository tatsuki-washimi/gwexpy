# Field4DDict

**継承元:** dict

`Field4D` オブジェクトの辞書型コレクション。

## 検証

`validate=True` の場合、以下の一致を検証します。
- 単位、軸名、axis0 ドメイン、空間ドメイン
- 軸座標配列（形状、単位、数値が許容誤差内で一致）

## メソッド

### `fft_time_all`

```python
fft_time_all(self, **kwargs)
```

辞書内のすべてのフィールドに対して `fft_time()` を適用します。

戻り値
-------
Field4DDict
    変換されたフィールドの辞書。

### `ifft_time_all`

```python
ifft_time_all(self, **kwargs)
```

辞書内のすべてのフィールドに対して `ifft_time()` を適用します。

戻り値
-------
Field4DDict
    変換されたフィールドの辞書。

### `fft_space_all`

```python
fft_space_all(self, **kwargs)
```

辞書内のすべてのフィールドに対して `fft_space()` を適用します。

戻り値
-------
Field4DDict
    変換されたフィールドの辞書。

### `ifft_space_all`

```python
ifft_space_all(self, **kwargs)
```

辞書内のすべてのフィールドに対して `ifft_space()` を適用します。

戻り値
-------
Field4DDict
    変換されたフィールドの辞書。
