# スカラーフィールドのスライス操作ガイド

`ScalarField` がインデクシング操作時に**4次元構造を常に維持する**挙動について説明します。これは NumPy や GWpy の標準的な挙動とは異なり、多次元物理データの整合性を保つための重要な設計です。

## なぜ 4次元 を維持するのか？

`ScalarField` は単なる配列ではなく、時間・周波数・空間（x, y）といった物理ドメインと紐付いています。
NumPy のようにスライス時に次元を削減（squeeze）してしまうと、特定の軸に紐付いたメタデータ（サンプリングレート、グリッド間隔等）との対応関係が壊れ、FFT 操作やドメイン変換が不可能になるためです。

### NumPy vs GWexpy の挙動比較

![4D Slicing comparison between NumPy and GWexpy showing dimension maintenance](/home/washimi/.gemini/antigravity/brain/389da455-0c02-483f-928d-e8f3db2746b8/scalarfield_slicing_4d_maintenance_1775634419729.png)

| 操作 | NumPy の挙動 | ScalarField の挙動 | 理由 |
| :--- | :--- | :--- | :--- |
| `field[0]` | 次元が減る (Rank Loss) | **4次元を維持** | 軸メタデータを保護するため |
| `field[:, :, :, 2]` | 3次元になる | **4次元を維持** | 座標情報の連続性を保つため |
| `field + 1.0` | パフォーマンス低下 | 高速演算 | 物理単位の一貫性チェック |

---

## 4次元構造維持のメリット

1.  **軸メタデータの完全保持**: `axis0_domain`, `space_domain` などが欠落しません。スライス後も即座に `fft()` や `plot_map()` が可能です。
2.  **物理演算の安全性**: 予期せぬ次元削減による「誤ったブロードキャスト演算」を防ぎます。
3.  **ストリーム処理の容易性**: 常に 4D であるため、パイプラインの途中で形状を気にする必要がありません。

---

## 実践的な操作例

### 1. スライシングの挙動

```python
from gwexpy.fields import ScalarField
import numpy as np

# (time, freq, x, y) = (100, 50, 10, 10)
field = ScalarField(np.zeros((100, 50, 10, 10)), ...)

# 特定の時刻の断面を取得
snapshot = field[50]
# shape は (1, 50, 10, 10) となり、時間軸の情報が残ります。

# 空間断面 (x-y 平面) を抽出
plane = field[:, :, :, 2]
# shape は (100, 50, 10, 1) となり、y軸情報が維持されます。
```

### 2. 次元を削減したい場合 (`squeeze`)

意図的に 1次元や 2次元として扱いたい場合（例：プロットや外部ライブラリへの入力）は、明示的に `.squeeze()` を呼び出します。

```python
# 特定の空間点の時系列を取得してプロット
point_ts = field[:, 2, 5, 5]      # (100, 1, 1, 1)
actual_ts = point_ts.squeeze()    # (100,) - これで TimeSeries 互換になります
```

### 3. ブロードキャスト演算の注意点

`ScalarField` は常に 4次元であるため、NumPy 配列を加減算する場合は形状を合わせる必要があります。

```python
# ❌ 悪い例: 1次元配列をそのまま足そうとする
field + np.array([1, 2, 3])  # Shape mismatch

# ✅ 良い例: 正しい次元に拡張する
calibration = np.array([1, 2, 3]).reshape(3, 1, 1, 1) # (freq, 1, 1, 1)
field + calibration
```

---

## よくある質問 (FAQ)

### Q: 常に 4次元だと、1次元の計算時に不便ではありませんか？
**A:** `ScalarField` は空間・時間の広がりを持つデータを「場」として扱うためのクラスです。単一チャンネルの単純な時系列を扱う場合は、最初から `TimeSeries` クラスを使用することをお勧めします。

### Q: `ScalarField[0, 0, 0, 0]` とスカラー抽出した場合は？
**A:** インデックスがすべてスカラーの場合は、通常の Python スカラー値または NumPy スカラーが返されます。

## 関連リンク

- {doc}`tutorials/field_scalar_intro` - ScalarField 入門チュートリアル
- {doc}`../reference/api/field` - Field モジュール API リファレンス
- {doc}`numerical_stability` - 数値安定性（4次元演算時の精度管理）
