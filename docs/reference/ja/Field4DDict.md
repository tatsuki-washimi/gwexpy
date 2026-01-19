# Field4DDict

**継承元:** dict

`Field4D` オブジェクトの辞書型コレクション。

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
