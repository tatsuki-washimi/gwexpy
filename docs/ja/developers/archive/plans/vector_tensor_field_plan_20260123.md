# VectorField / TensorField 実装計画 (2026-01-23 15:40:00)

## 使用モデルとリソース最適化
* **推奨モデル**: Gemini 3 Pro (High)
    * 理由: 物理的・数学的な整合性が極めて重要な Vector/Tensor の実装において、高度な推論能力と広範なコンテキスト理解が必要なため。
* **リソース管理戦略**:
    * スカラー成分ごとの独立性を保ち、`ScalarField` の既存機能を最大限再利用する。
    * コンテキスト圧縮のため、一度に全ての演算を実装せず、Phase ごとに作業を完結させる。

---

## 0. 前提整理（現状認識）

### ScalarField
* **実装済・実戦投入可能**
* 主な責務：
    * 時間・周波数・空間軸の管理
    * FFT / IFFT、単位伝播
    * 切り出し・再サンプリング
    * gwpy オブジェクトとの相互運用

### VectorField / TensorField
* **NotImplementedError を投げるプレースホルダ**
* API・内部表現・数学的意味付けが未確定
* ScalarField の単純な配列拡張では済まない

👉 結論として、**ScalarField を核にした段階的拡張**が最も安全。

---

## 1. 設計方針（最重要）

### 方針 A：ScalarField を「成分 Field」の集合として扱う
Vector/Tensor を「高次 ndarray」として一気に扱うのは避ける。
* VectorField = ordered collection of ScalarField
* TensorField = ordered collection of VectorField（または ScalarField）

理由：
* gwpy 自体が基本的に scalar-valued series を前提
* 単位・FFT・窓関数・平均化は **成分ごとに完結**
* デバッグ・検証が圧倒的に容易

---

## 2. VectorField 実装計画

### 2.1 最小実装（Phase 1）

#### クラス構造
```python
class VectorField:
    components: dict[str, ScalarField]
    basis: Literal["cartesian", "custom"]
```

#### 必須仕様
* 全成分で以下が一致していることを強制
    * 時間軸 / 周波数軸
    * サンプリング周波数
    * FFT 設定
* 不一致の場合は **例外を即時送出**

#### 提供 API（最小）
* `__getitem__(component)`
* `keys()`
* `copy()`
* `to_array()`（shape = (..., n_components)）
* `norm()`（√(Σ|component|²) → ScalarField）

※ **微分・回転・発散はまだやらない**

### 2.2 演算対応（Phase 2）

#### 成分独立演算
* FFT / IFFT
* フィルタ
* 時間切り出し
→ 内部で ScalarField に委譲

#### スカラー演算
* VectorField × scalar
* scalar × VectorField

### 2.3 幾何演算（Phase 3）

ここからは「物理量としての意味」が必要。
* `dot(other)` → ScalarField
* `cross(other)` → VectorField（3成分限定）
* `project(direction)` → ScalarField

※ 単位系の整合性チェック必須

---

## 3. TensorField 実装計画

### 3.1 設計制約
Tensor は **用途依存が極端に強い**：
* 応力テンソル
* 感度テンソル
* 伝達関数行列
* 時間依存 Rank-2/Rank-3 tensor

👉 いきなり汎用 Tensor は作らない。

### 3.2 最小実装（Phase 1）

#### 構造
```python
class TensorField:
    components: dict[tuple[int, ...], ScalarField]
    rank: int
```
* `(i, j)` → ScalarField
* `(i, j, k)` → ScalarField

#### 提供 API
* `__getitem__((i, j))`
* `keys()`
* `copy()`
* `trace()` → ScalarField（rank=2）
* `symmetrize()`（rank=2）

### 3.3 行列的意味を持つ Tensor（Phase 2）
* Tensor × Vector → Vector
* Tensor × Tensor → Tensor

制約：
* 成分数・インデックスが厳密一致
* 軸ラベル（物理的意味）を optional metadata として保持

---

## 4. GUI / 可視化との関係

### 原則
* **GUI は ScalarField しか直接描画しない**

Vector/Tensor は：
* 成分選択
* ノルム表示
* 特定 contraction の結果のみ描画
これにより GUI 複雑度を爆発させない。

---

## 5. テスト戦略

### 単体テスト
* ScalarField を mock として使用
* 軸不一致時の例外確認
* 成分演算의 独立性確認

### 統合テスト
* VectorField.norm() が ScalarField と一致
* FFT → IFFT 往復での再現性

---

## 6. 実装しない（当面）
明示的に **やらないこと**を決めるのは重要。
* 自動テンソル微分
* 共変微分
* 座標変換（曲線座標）
* einsum 的な汎用 contraction
これらは **将来拡張 or 別モジュール**。

---

## 7. 実装順サマリ（ロードマップ）
1. VectorField（成分集合として）
2. norm / dot のみ
3. TensorField（rank=2 限定）
4. contraction 最小集合
5. GUI 連携（表示補助）

---

## 結論
* **現時点で VectorField / TensorField を「十分」とは言えない**
* しかし、
    * ScalarField を核に
    * 「成分集合」として厳密に実装すれば
    * gwexpy の哲学と完全に整合する
