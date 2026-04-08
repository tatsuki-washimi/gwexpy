# 数値的安定性と精度

`gwexpy` は、重力波データ解析で頻繁に扱われる極めて広いダイナミックレンジを持つデータを処理できるように設計されています。
重力波のひずみ信号（strain）は典型的には $10^{-21}$ のオーダーですが、中間処理では 1 に近い値を扱うこともあります。

科学的な正確性を保証し、数値的なアーティファクト（:term:`NaN/Inf propagation` や信号消失など）を防ぐため、`gwexpy` は堅牢な数値安定化戦略を実装しています。

## TL;DR: なぜ数値安定化が必要か

- **:term:`NaN/Inf の伝播（旧: Death Floats）` の防止**: ゼロに近い値での除算や対数計算による計算不能状態（:term:`NaN/Inf propagation`）を防ぎます。
- **微小信号の保護**: 固定の `eps` (1e-12等) による丸め誤差で、重力波信号 ($10^{-21}$) が消えるのを防ぎます。
- **可視化の改善**: プロット時に自動でダイナミックレンジを調整し、信号の細部まで見えるようにします。

### 可視化における安定化の効果

![Spectral stabilization comparison showing typical NaN artifacts vs clean GWexpy output](/home/washimi/.gemini/antigravity/brain/389da455-0c02-483f-928d-e8f3db2746b8/spectral_stabilization_comparison_1775634367572.png)

---

## 主な安定化手法と API

| 対策手法 | 対象 API | 解決する問題 | 設定のヒント |
| :--- | :--- | :--- | :--- |
| **:term:`Adaptive Whitening`** | `.whiten()` | ゼロ除算・信号埋没 | デフォルトの `eps="auto"` を推奨 |
| **:term:`Safe Log`** | `.plot()`, `.spectrogram()` | `-inf` によるプロットの穴 | `dynamic_range=200` 等で調整可能 |
| **内部標準化 (ICA)** | `ica_fit()` | 振幅依存による不収束 | 入力振幅を気にせず実行可能 |
| **相対許容誤差** | 各種数値計算 | スケール違いによる早期終了 | データの分散に基づき `tol` を自動計算 |

---

## 各機能の解説とコード例

### 1. :term:`Adaptive Whitening` (アダプティブ・ホワイトニング)

標準的なホワイトニングは、ゼロ除算を防ぐために固定の正規化パラメータ（`eps`）を使用しますが、これが大きすぎると微小な信号が埋もれます。

#### ❌ 悪い例: 固定 eps による信号消失
```python
# 1e-12 程度の固定 eps では 1e-21 の信号は 0 に丸められてしまう
whitened = data / (asd + 1e-12) 
```

#### ✅ 良い例: GWexpy の `eps="auto"`
`gwexpy` はデータのスケールに合わせて `eps` を相対的に調整し、かつ `SAFE_FLOOR` (1e-50) で特異点を防ぎます。

```python
from gwexpy.timeseries import TimeSeries
import numpy as np

data = TimeSeries(np.random.randn(1000) * 1e-21, sample_rate=1024)
whitened = data.whiten(eps="auto")  # 自動的に適切なスケーリングを適用
```

### 2. 安全な対数スケーリング (:term:`Safe Log`)

スペクトログラム等の可視化において、ゼロや極小値による `-inf` の発生を防ぎます。

#### ❌ 悪い例: 手動変換による数値エラー
```python
asd_db = 10 * np.log10(asd)  # 0 があると -inf になりプロットが崩れる
```

#### ✅ 良い例: 自動的な動的フロア適用
`gwexpy` では、データの最大値から逆算した安全なフロアを自動適用します。

```python
asd = data.asd()
plot = asd.plot()  # 内部で Safe Log が適用され、-inf のない綺麗な図になる
```

### 3. 計算機イプシロンへの配慮

パッケージ全体で使用される数値定数は、浮動小数点型（float32 vs float64）の機械精度（イプシロン）に基づいて導出されており、最適な精度を保証します。

---

## ユーザーへの推奨事項

- **手動オフセットの回避**: プロットの前に `data + 1e-20` のような恣意的な値を足す必要はありません。内部で適切に処理されます。
- **デフォルトを信頼する**: `whiten()` や `ica_fit()` のデフォルト値は、数値的な安全性を最優先に調整されています。
- **警告を確認する**: 真に不安定な操作（全区間ゼロのデータのホワイトニング等）に対しては、解決策を含む警告が出力されます。

## 関連ドキュメント

- {doc}`../reference/api/signal` — 信号処理 API リファレンス
- {doc}`validated_algorithms` — 検証済みアルゴリズムの一覧
- {doc}`glossary` — :term:`NaN/Inf propagation` 等の用語定義
