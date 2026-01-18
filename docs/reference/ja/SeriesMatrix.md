# SeriesMatrix

**継承元:** RegularityMixin, InteropMixin, SeriesMatrixCoreMixin, SeriesMatrixIndexingMixin, SeriesMatrixIOMixin, SeriesMatrixMathMixin, SeriesMatrixAnalysisMixin, SeriesMatrixStructureMixin, SeriesMatrixVisualizationMixin, SeriesMatrixValidationMixin, StatisticalMethodsMixin, ndarray

複数の Series オブジェクト用の基底マトリックスクラス。

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `N_samples` | X 軸のサンプル数 |
| `dx` | サンプル間のステップサイズ |
| `x0` | サンプル軸の開始値 |
| `xindex` | サンプル軸のインデックス配列 |
| `xspan` | サンプル軸の範囲 (start, end) |
| `duration` | サンプルがカバーする時間 |
| `shape3D` | 3タプル (n_rows, n_cols, n_samples) としての形状 |
| `channel_names` | すべての要素名のフラットなリスト |
| `names` | 各要素の名前の2D配列 |
| `units` | 各要素の単位の2D配列 |
| `MetaDataMatrix` | 要素ごとのメタデータを含むマトリックス |
| `T` | 転置（行と列を交換） |
| `value` | 基盤となる numpy 配列 |

## インデックス参照

| メソッド | 説明 |
|---------|------|
| `row_index()` / `col_index()` | キーの整数インデックスを取得 |
| `row_keys()` / `col_keys()` | すべての行/列のキー（ラベル）を取得 |
| `keys()` | 行キーと列キーの両方を取得 |
| `get_index()` | キーの (row, col) 整数インデックスを取得 |
| `loc` | 直接値アクセス用のラベルベースインデクサー |
| `submatrix()` | 特定の行と列を選択してサブマトリックスを抽出 |

## 線形代数

| メソッド | 説明 |
|---------|------|
| `det()` | 各サンプル点での行列式を計算 |
| `inv()` | 各サンプル点での逆行列を計算 |
| `trace()` | 対角要素の和を計算 |
| `diagonal()` | 対角要素を抽出 |
| `schur()` | ブロック行列のシュア補行列を計算 |
| `conj()` | 複素共役 |
| `real()` / `imag()` | 実部/虚部 |

## 統計

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `median()` / `rms()` / `var()` | 統計量計算 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | サンプル軸に沿って指定範囲にクロップ |
| `append()` / `prepend()` | サンプル軸に沿って別のマトリックスを追加 |
| `append_exact()` / `prepend_exact()` | 厳密な連続性チェック付き |
| `update()` | リサイズせずに追加（ローリングバッファスタイル） |
| `interpolate()` | 新しいサンプル軸に補間 |
| `pad()` | サンプル軸に沿ってパディング |
| `shift()` | サンプル軸を定数オフセットでシフト |
| `diff()` | N 次離散差分を計算 |
| `reshape()` | マトリックスの次元を変形 |
| `copy()` | ディープコピーを作成 |
| `astype()` | データを指定した型にキャスト |

## 変換・相互運用

| メソッド | 説明 |
|---------|------|
| `to_pandas()` | pandas DataFrame に変換 |
| `to_dict()` / `to_list()` | 対応するコレクション型に変換 |
| `to_dict_flat()` | 名前から Series へのフラット辞書に変換 |
| `to_series_1Dlist()` / `to_series_2Dlist()` | Series の1D/2Dリストに変換 |
| `to_torch()` / `to_tensorflow()` / `to_jax()` / `to_cupy()` / `to_dask()` | ML フレームワークへ変換 |

## 入出力

| メソッド | 説明 |
|---------|------|
| `read()` | ファイルからマトリックスを読み込む |
| `write()` | マトリックスをファイルに書き込む |
| `to_hdf5()` / `to_zarr()` | HDF5/Zarr 形式で保存 |

## 可視化

| メソッド | 説明 |
|---------|------|
| `plot()` | gwexpy.plot.Plot でプロット |
| `step()` | ステップ関数としてプロット |

## 互換性チェック

| メソッド | 説明 |
|---------|------|
| `is_compatible()` / `is_compatible_exact()` | 互換性をチェック |
| `is_contiguous()` / `is_contiguous_exact()` | 連続性をチェック |
| `is_regular` | 等間隔グリッドかどうか |
| `value_at()` | 特定の X 位置のマトリックス値を取得 |
