# Transform

**継承元:** object

TimeSeriesなどのオブジェクトに対する最小限の変換インターフェース。

## メソッド

### `fit`

```python
fit(self, x)
```

変換をデータに適合（学習）させます。selfを返します。

### `fit_transform`

```python
fit_transform(self, x)
```

適合と変換を一つのステップで実行します。

### `inverse_transform`

```python
inverse_transform(self, y)
```

逆変換を適用します。すべての変換がこれをサポートしているわけではありません。

### `transform`

```python
transform(self, x)
```

データを変換します。サブクラスで実装する必要があります。
