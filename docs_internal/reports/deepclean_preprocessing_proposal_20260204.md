# 提案: DeepClean 前処理パイプラインの統合

**日付**: 2026-02-04
**ステータス**: 提案中

## 1. 概要

本ドキュメントは、DeepClean v2 の前処理ロジックを、汎用的かつ再利用可能なモジュールとして `gwexpy` に統合することを提案します。現在、DeepClean の前処理（データ分割、フィルタリング、チャンネルごとのスケーリング）は、PyTorch の学習ループ（`LightningDataModule`）と強く結合しています。このロジックを `gwexpy` に抽出することで、以下のメリットが得られます。

1. **モデル非依存性**: 深層学習以外のモデル（ランダムフォレスト、XGBoostなど）でも同じ前処理を利用可能にする。
2. **使いやすさ**: ノイズ除去タスクのための前処理を「1行」で準備できるようにする。
3. **標準化**: 異なる解析パイプライン間でデータの取り扱いを一貫させる。

## 2. 提案アーキテクチャ

`gwexpy.signal.preprocessing.deepclean` モジュールを追加し、そこに `DeepCleanPreprocessor` クラスを実装することを提案します。

### 2.1 `DeepCleanPreprocessor` クラス

このクラスは `scikit-learn` の Transformer API (`fit`, `transform`) を模倣しますが、重力波時系列データ（GW TimeSeries）に特化しています。

```python
class DeepCleanPreprocessor:
    def __init__(
        self,
        sample_rate: Quantity,
        freq_low: list[float] | None = None,
        freq_high: list[float] | None = None,
        filt_order: int = 8,
        valid_frac: float = 0.0,
    ):
        ...
```

#### 主要メソッド

1. **`fit(X, y=None)`**:
    * `X` (参照チャンネル) および `y` (ターゲットチャンネル/Strain) の各チャンネルについて統計量（平均、標準偏差/中央値、MAD）を計算します。
    * 周波数帯域が指定されている場合、バンドパスフィルタの係数を設計します。

2. **`transform(X, y=None)`**:
    * バンドパスフィルタを適用します（ゼロ位相 `filtfilt`）。
    * 標準化（Z-score または Robust Scaling）を適用します。
    * 処理済みの `TimeSeriesMatrix` および `TimeSeries` を返します。

3. **`split(X, y)` -> `(X_train, y_train, X_valid, y_valid)`**:
    * `valid_frac` に基づいてデータを時系列順に分割します。
    * 分割点が整数のサンプリングポイントになるように調整します。
    * `transform` の入力としてそのまま使える分割データを返します。

### 2.2 `TimeSeriesWindowDataset` との連携

ユーザーのワークフローは以下のようになります。

```python
# 1. データ読み込み
witnesses = TimeSeriesMatrix(...)
strain = TimeSeries(...)

# 2. 前処理 (分割 -> フィルタ -> スケーリング)
preprocessor = DeepCleanPreprocessor(sample_rate=4096, valid_frac=0.2)
X_train, y_train, X_valid, y_valid = preprocessor.split(witnesses, strain)

# 学習用データ(Train)のみからスケーリング/フィルタ係数を学習
preprocessor.fit(X_train, y_train)

# 両方に適用
X_train_proc, y_train_proc = preprocessor.transform(X_train, y_train)
X_valid_proc, y_valid_proc = preprocessor.transform(X_valid, y_valid)

# 3. データセット作成 (PyTorch用)
train_ds = TimeSeriesWindowDataset(X_train_proc, labels=y_train_proc, ...)
valid_ds = TimeSeriesWindowDataset(X_valid_proc, labels=y_valid_proc, ...)
```

## 3. 実装詳細

* **フィルタリング**: `gwpy.signal.filter_design` (`butter`) を活用してフィルタを構築します。
* **スケーリング**: 既存の `standardize_matrix` のロジックを利用しますが、統計量を保持（ステートフル化）するようにラップします。
* **相互運用性**: 出力型が `gwexpy` の既存の `TimeSeries` エコシステムと互換性があることを保証します。

## 4. 作業項目

* [ ] `gwexpy/signal/preprocessing/deepclean.py` を作成する。
* [ ] `DeepCleanPreprocessor` クラスを実装する。
* [ ] DeepClean オリジナル実装と数値的に一致することを検証する単体テストを追加する。
* [ ] チュートリアルノートブック `tutorial_DeepClean_Preprocessing.ipynb` を作成する。
