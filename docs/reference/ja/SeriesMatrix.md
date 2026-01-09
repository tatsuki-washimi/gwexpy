# SeriesMatrix

**継承元:** SeriesMatrixOps, ndarray

## メソッド

### `MetaDataMatrix`

各要素のメタデータを含むメタデータ行列。

**戻り値:**
- **MetaDataMatrix**
    各 (row, col) 要素に対応する MetaData オブジェクトの 2D 行列。

### `N_samples`

x軸方向のサンプル数。

**戻り値:**
- **int**
    サンプル軸の長さ（行列の3次元目）。

### `T`

転置された配列のビュー。

`self.transpose()` と同じです。

*( `ndarray` から継承)*

### `abs`

```python
abs(self)
```

要素ごとの絶対値を計算します。
複素数行列の場合、絶対値（振幅）を返します。

**戻り値:**
- **SeriesMatrix**
    実数値の絶対値を持つ新しい行列。

### `angle`

```python
angle(self, deg: bool = False)
```

要素ごとの位相角を計算します。
複素数行列の場合、位相角（偏角）を返します。

**パラメータ:**
- **deg** : bool, オプション
    True の場合、角度を度（degrees）で返します。デフォルトはラジアン（radians）。

**戻り値:**
- **SeriesMatrix**
    位相角を持つ新しい行列。単位は度またはラジアンに設定されます。

### `append`

```python
append(self, other, inplace=True, pad=None, gap=None, resize=True)
```

サンプル軸方向に別の行列を追加（連結）します。

**パラメータ:**
- **other** : SeriesMatrix
    追加する行列。互換性のある行/列構造を持っている必要があります。
- **inplace** : bool, オプション
    True（デフォルト）の場合、この行列をインプレースで変更します。
- **pad** : float または None, オプション
    ギャップを埋めるためのパディング値。None の場合、ギャップがあるとエラーになります。
- **gap** : str, float, または None, オプション
    ギャップの処理：'raise'（エラー）、'pad'（指定値で埋める）、'ignore'（単純な連結）、または数値による許容誤差。
- **resize** : bool, オプション
    True（デフォルト）の場合、行列のサイズ拡張を許可します。False の場合、先頭を削ることで元の長さを維持します。

**戻り値:**
- **SeriesMatrix**
    連結された行列（inplace=True の場合は self）。

### `append_exact`

```python
append_exact(self, other, inplace=False, pad=None, gap=None, tol=3.814697265625e-06)
```

厳密な連続性チェックを行いつつ、別の行列を追加します。
行列間のギャップに対して許容誤差チェックを行い、サンプル単位で厳密に連結します。

**パラメータ:**
- **other** : SeriesMatrix
    追加する行列。
- **inplace** : bool, オプション
    True の場合、この行列をインプレースで変更します。デフォルトは False。
- **tol** : float, オプション
    連続性チェックの許容誤差（デフォルト：1/2^18 ≈ 3.8e-6）。

### `astype`

```python
astype(self, dtype, copy=True)
```

行列データを指定された型にキャストします。

**パラメータ:**
- **dtype** : str または numpy.dtype
    ターゲットのデータ型。
- **copy** : bool, オプション
    True（デフォルト）の場合、型が変わらない場合でもコピーを返します。

### `channels`

各行列要素のチャンネル識別子の 2D 配列。

**戻り値:**
- **numpy.ndarray**
    チャンネル名または Channel オブジェクトの 2D オブジェクト配列。

### `col_index`

```python
col_index(self, key: Any) -> int
```

列キーに対応する整数インデックスを取得します。

**パラメータ:**
- **key** : Any
    検索する列キー。

**戻り値:**
- **int**
    列の 0 から始まるインデックス。

### `col_keys`

```python
col_keys(self) -> list[typing.Any]
```

すべての列のキー（ラベル）を取得します。

**戻り値:**
- **tuple**
    列キーのタプル（順序通り）。

### `copy`

```python
copy(self, order='C')
```

この行列のディープコピーを作成します。

**パラメータ:**
- **order** : {'C', 'F', 'A', 'K'}, オプション
    コピーのメモリレイアウト。デフォルトは 'C'（行優先）。

### `crop`

```python
crop(self, start=None, end=None, copy=False)
```

指定された範囲のサンプル軸で行列を切り抜きます。

**パラメータ:**
- **start, end** : float, Quantity, または None
    切り抜きの開始/終了値。None の場合、端まで含まれます。
- **copy** : bool, オプション
    True の場合、データのコピーを返します。それ以外の場合はビューを返します。

### `dagger`

行列の随伴行列（エルミート共役）。
転置行列の複素共役を返します。行列表記では A† と記されます。

**戻り値:**
- **SeriesMatrix**
    この行列のエルミート共役。

### `det`

```python
det(self)
```

各サンプル点における行列の行列式を計算します。

**戻り値:**
- **Series**
    行列式の値を含む Series。

### `diagonal`

```python
diagonal(self, output: str = 'list')
```

行列から対角要素を抽出します。

**パラメータ:**
- **output** : {'list', 'vector', 'matrix'}, オプション
    - 'list': Series のリストを返す（デフォルト）
    - 'vector': 列ベクトル（n x 1 行列）として返す
    - 'matrix': 対角成分以外をゼロにしたフル行列を返す

### `diff`

```python
diff(self, n=1, axis=2)
```

サンプル軸方向に n 次の離散差分を計算します。

**パラメータ:**
- **n** : int, オプション
    差分の回数。デフォルトは 1。
- **axis** : int, オプション
    2（サンプル軸）である必要があります。

### `duration`

サンプルがカバーする全期間。

### `dx`

x軸上のサンプル間のステップサイズ（間隔）。
等間隔サンプリングの場合、連続するサンプル間の一定の間隔です。タイムシリーズの場合、`1/sample_rate` に相当します。

**戻り値:**
- **dx** : `~astropy.units.Quantity`
    適切な単位を持つサンプル間隔。

### `get_index`

```python
get_index(self, key_row: Any, key_col: Any) -> tuple[int, int]
```

指定されたキーに対応する (row, col) の整数インデックスを取得します。

**戻り値:**
- **tuple of int**
    (row_index, col_index) のタプル。

### `imag`

行列の虚部。
複素数行列の場合、虚数成分のみを返します。実数行列の場合はゼロを返します。

### `inv`

```python
inv(self, swap_rowcol: bool = True)
```

各サンプル点における逆行列を計算します。

**パラメータ:**
- **swap_rowcol** : bool, オプション
    True（デフォルト）の場合、結果の行/列ラベルを入れ替えます。

### `is_compatible`

```python
is_compatible(self, other: Any) -> bool
```

互換性チェック。

### `is_compatible_exact`

```python
is_compatible_exact(self, other)
```

別の行列との厳密な互換性をチェックします。
同一の形状、xindex の値、行/列キー、および要素の単位を必要とします。

### `is_contiguous`

```python
is_contiguous(self, other, tol=3.814697265625e-06)
```

この行列が別の行列と連続しているかをチェックします（gwpy ライクなセマンティクス）。

### `is_contiguous_exact`

```python
is_contiguous_exact(self, other, tol=3.814697265625e-06)
```

厳密な形状一致を含めて連続性をチェックします。

### `keys`

```python
keys(self) -> list[tuple[typing.Any, typing.Any]]
```

行と列の両方のキーを取得します。

**戻り値:**
- **tuple**
    (row_keys, col_keys) の 2 要素タプル。

### `loc`

ラベルベースのインデクサ。
pandas の `.loc` のように、行/列キーを使用してデータ配列の値に直接アクセス（取得/設定）できます。

**例:**
```python
matrix.loc[0, 1, :] = new_values  # 行 0, 列 1 の全サンプルを設定
vals = matrix.loc['chA', 'chB', :] # チャンネル名でアクセス
```

### `names`

各行列要素の名称の 2D 配列。

### `pad`

```python
pad(self, pad_width, **kwargs)
```

行列のサンプル軸をパディングします。

**パラメータ:**
- **pad_width** : int または (int, int) のタプル
    パディングするサンプル数。

### `plot`

```python
plot(self, **kwargs)
```

`gwexpy.plot.Plot` を使用してこの SeriesMatrix をプロットします。

### `prepend`

```python
prepend(self, other, inplace=True, pad=None, gap=None, resize=True)
```

サンプル軸の先頭に別の行列を追加します。

### `read`

```python
read(source, format=None, **kwargs)
```

ファイルから SeriesMatrix を読み込みます。

**主なサポート形式:**
`dttxml`, `gbd`, `gse2`, `knet`, `li`, `lsf`, `sac`, `win32` など。

### `real`

行列の実部。

### `row_index`

```python
row_index(self, key: Any) -> int
```

行キーに対応する整数インデックスを取得します。

### `row_keys`

```python
row_keys(self) -> list[typing.Any]
```

すべての行のキーを取得します。

### `schur`

```python
schur(self, keep_rows, keep_cols=None, eliminate_rows=None, eliminate_cols=None)
```

ブロック行列の Schur 相補行列（シューア相補、シューア・コンプレメント）を計算します。
計算式： A - B @ D^(-1) @ C

### `shape3D`

行列の形状を (n_rows, n_cols, n_samples) のタプルで返します。

### `shift`

```python
shift(self, delta)
```

サンプル軸を定数オフセットでシフトします（インプレース操作）。

### `step`

```python
step(self, where='post', **kwargs)
```

行列を階段関数としてプロットします。

### `submatrix`

```python
submatrix(self, row_keys, col_keys)
```

特定の行と列を選択して部分行列を抽出します。

### `to_hdf5`

```python
to_hdf5(self, filepath, **kwargs)
```

行列を HDF5 ファイルに書き出します。

### `to_pandas`

```python
to_pandas(self, format='wide')
```

行列を pandas DataFrame に変換します。
`wide` 形式（行-列の組み合わせごとに列を作成）または `long` 形式（tidy データ）を選択可能です。

### `to_series_1Dlist`

行列を Series オブジェクトのフラットな 1D リストに変換します。

### `to_series_2Dlist`

行列を Series オブジェクトの 2D ネストリストに変換します。

### `trace`

```python
trace(self)
```

行列のトレース（対角要素の和）を各サンプル点について計算します。

### `transpose`

```python
transpose(self)
```

軸を入れ替えた配列のビューを返します（numpy.transpose と同様）。

*( `ndarray` から継承)*

### `units`

各行列要素の単位の 2D 配列（astropy.units.Unit）。

### `update`

```python
update(self, other, inplace=True, pad=None, gap=None)
```

サイズを変更せずに（ローリングバッファ方式で）行列を更新します。
`other` を追加し、元の長さを維持するために先頭をカットします。

### `value`

基となるデータ値の numpy 配列 (3D)。

### `value_at`

```python
value_at(self, x)
```

特定の x 軸位置における行列値を取得します。

### `write`

```python
write(self, target, format=None, **kwargs)
```

行列をファイル（HDF5, CSV, Parquet）に書き出します。

### `x0`

サンプル軸の開始値。

### `xarray`

サンプル軸の値の配列を返します。

### `xindex`

サンプル軸のインデックス配列（時間や周波数）。

### `xspan`

サンプル軸の全範囲を (start, end) のタプルで返します。

### `xunit`

サンプル軸の物理単位。
