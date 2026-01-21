---
name: fix_mypy
description: gwexpyのコードベースで頻発するMyPyエラーを効率的に解決するパターン集
---

# Fix MyPy Skills

`gwexpy` における静的解析エラー（MyPy）の解消に特化した知見をまとめています。

## 典型的なエラーと解決策

### 1. NumPy内部型（_NBit1等）のエラー

**症状**: `Name "_NBit1" is not defined` や、型ヒントに覚えのない内部型が含まれる。
**解決策**: `np.complexfloating` や `np.number` などの抽象的な型ヒントを、`np.complex128` や `np.floating` などの具体的な型（またはその Union）に置き換えることで、MyPy が内部表現に踏み込むのを防ぎます。

### 2. 多重継承によるメソッド衝突

**症状**: `Definition of 'mean' in base class ... is incompatible with ...`
**解決策**: 衝突しているクラス定義（特に `TimeSeriesMatrix` 等）に対し、クラス全体に `# type: ignore` を付与するか、個別のミックスイン行に `# type: ignore[misc]` を追加します。

### 3. 動的生成リストへの配列代入

**症状**: `None` で初期化された `vals[i][j]` への代入でエラー。
**解決策**: 初期化時に `list[list[Any]]` として明示的にアノテーションするか、代入時に `cast(list[list[Any]], vals)[i][j] = ...` とキャストします。

### 4. ミックスインにおける self の型

**症状**: ミックスイン内で `self` を具象クラス（`TimeSeries` 等）を期待する関数に渡すと型不一致。
**解決策**: `cast("TimeSeries", self)` を使用して、実行時の型を MyPy に伝えます。

### 5. Quantity の型絞り込み

**症状**: `is_quantity` フラグ等を使用しても、`.value` や `.unit` へのアクセスでエラーが出る。
**解決策**: `if isinstance(data, u.Quantity):` のように、`isinstance` を直接 `if` 文で使用することで、情報の損失を防ぎ、型絞り込みを確実に機能させます。
