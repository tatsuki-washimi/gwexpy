# FrequencySeriesMatrix

**継承元:** SeriesMatrix

複数の FrequencySeries オブジェクトを保持する行列コンテナです。
SeriesMatrix を継承しており、インデックスでアクセスすると FrequencySeries インスタンスを返します。

## メソッド

### `MetaDataMatrix`

各要素のメタデータを含むメタデータ行列。

### `N_samples`

x軸方向のサンプル数。

### `T`

転置された配列のビュー。`self.transpose()` と同じです。

### `abs`

```python
abs(self)
```

要素ごとの絶対値（振幅）を計算します。

### `angle`

```python
angle(self, deg: bool = False)
```

要素ごとの位相角を計算します。

### `append`

```python
append(self, other, inplace=True, pad=None, gap=None, resize=True)
```

サンプル軸方向に別の行列を追加します。

### `apply_response`

```python
apply_response(self, response, inplace=False)
```

複素周波数応答を行列に適用します。GWpy にはない拡張メソッドで、複素フィルタリングや較正（キャリブレーション）をサポートします。

**パラメータ:**
- **response**: self.frequencies とアライメントされた複素周波数応答配列。

### `astype`

```python
astype(self, dtype, copy=True)
```

行列データを指定された型にキャストします。

### `channels`

各要素のチャンネル識別子の 2D 配列。

### `col_index`, `col_keys`

列インデックスの取得とすべての列キーの取得。

### `copy`

ディープコピーを作成します。

### `crop`

```python
crop(self, start=None, end=None, copy=False)
```

指定された範囲のサンプル軸で行列を切り抜きます。

### `dagger`

行列の随伴行列（エルミート共役 A†）。

### `det`

各サンプル点における行列の行列式を計算します。

### `df`

周波数間隔 (dx)。

### `diagonal`

対角要素を抽出します。

### `diff`

サンプル軸方向に離散差分を計算します。

### `f0`

開始周波数 (x0)。

### `filter`

```python
filter(self, *filt, **kwargs)
```

FrequencySeriesMatrix にフィルタを適用します（振幅のみの応答）。複素応答を適用する場合は `apply_response()` を使用してください。

### `frequencies`

周波数配列 (xindex)。

### `ifft`

```python
ifft(self)
```

この周波数ドメイン行列の逆FFTを計算し、TimeSeriesMatrix を返します。

### `imag`, `real`

虚部と実部。

### `inv`

逆行列を計算します。

### `is_compatible`, `is_contiguous`

互換性と連続性のチェック。

### `loc`

ラベルベースのインデクサ。

### `names`

各要素の名称の 2D 配列。

### `pad`

サンプル軸をパディングします。

### `plot`

プロットを作成します。

### `read`, `write`

ファイルの読み込みと書き出し。

### `units`

各要素の単位。

### `update`

サイズを変更せずに行列を更新します（ローリングバッファ）。

### `value`

基となるデータ値の numpy 配列 (3D)。

### `x0`, `xindex`, `xspan`, `xunit`

x 軸（周波数軸）のプロパティ。
