# [Field Plotting] 描画機能拡充計画書 (2026-01-23 17:23:00)

`ScalarField`、`VectorField`、および `TensorField` に対する、次元を考慮した2次元/3次元可視化機能の実装計画です。新しい `FieldPlot` APIとインスタンスレベルの描画メソッドが含まれます。

## 1. コア描画エンジン (2D 断面抽出)

- **目的**: 座標値またはインデックスを指定することで、4Dフィールドから2D断面を堅牢に抽出する手法を提供します。
- **場所**: `gwexpy/fields/base.py`（または `gwexpy/plot/utils.py` のヘルパー）。
- **機能**:
    - `field.get_slice(x_axis='x', y_axis='y', **fixed_coords)`: 2D断面を返します。
    - 軸メタデータを使用した、座標からインデックスへのマッピングの自動処理。

## 2. FieldPlot クラス (API B)

- **目的**: 空間マップを管理するための専用クラス。複数のフィールドの重ね合わせを処理します。
- **場所**: `gwexpy/plot/field.py`。
- **主要メソッド**:
    - `__init__(field, ...)`: デフォルトの軸と投影法をセットアップ。
    - `add_scalar(field, cmap='viridis', mode='pcolormesh', **kwargs)`: スカラー場の断面をレンダリング。
    - `add_vector(field, mode='quiver', color='black', **kwargs)`: ベクトル場の断面をレンダリング。
        - サポートモード: `quiver`, `streamline`, `magnitude_contour`。
    - `add_tensor(field, ...)`: 未定（楕円や特定の成分を表示する予定）。

## 3. インスタンスレベルの描画 (API C)

- **目的**: フィールドオブジェクト上で直接実行できるクイックルックメソッド。
- **ScalarField**:
    - `plot(**slice_params)`: シンプルな 2D マップ。
- **VectorField**:
    - `plot_magnitude(**slice_params)`
    - `quiver(**slice_params)`
    - `streamline(**slice_params)`
- **TensorField**:
    - `plot_components(**slice_params)`: 成分マップの $3 \times 3$ グリッド表示。

## 4. アニメーションとマルチファセットのサポート

- **目的**: 4Dのダイナミクスを可視化。
- **機能**:
    - `animate(loop_axis='t', **fixed_coords)`: `matplotlib.animation.FuncAnimation` を使用。
    - `plot(col='t', col_wrap=4)`: Xarray/Seaborn に似たファセットプロット。

## 5. ロードマップ

### Phase 1: 基盤とスカラー描画
- [ ] `FieldBase` への断面抽出ロジックの実装。
- [ ] 基本的な `FieldPlot` を含む `gwexpy/plot/field.py` の作成。
- [ ] `ScalarField.plot()` の実装。

### Phase 2: ベクトル可視化
- [ ] `VectorField.quiver()` と `VectorField.streamline()` の追加。
- [ ] `FieldPlot` の `add_vector` サポートの更新。

### Phase 3: 高度な機能
- [ ] `TensorField.plot_components()` の実装。
- [ ] `field.animate()` の実装。
- [ ] ユニットテストとチュートリアルの更新。

## 使用モデルとリソース最適化

- **推奨モデル**: Gemini 2.5 Pro (M18)
- **選定理由**: 複数の新規モジュール（`field.py` 等）の設計と複雑なクラス継承（`Array4D` からの派生）を扱うため、長いコンテキストを正確に保持し、高度な推論が可能なモデルを推奨します。
- **リソース管理戦略**:
    - **コンテキスト圧縮**: 大規模な実装になるため、作業フェーズごとに完了したコードを要約し、不要なデバッグ出力を抑制してトークンを節約します。
    - **バッチ検証**: `pytest` による自動検証を細かく行い、手戻りを最小限に抑えます。

## 6. 検証計画
- **検証スクリプト**: `tests/plot/test_field_plots.py` を作成し、以下を確認：
    - 断面抽出の精度。
    - ベクトル場の描画整合性（矢印の向き）。
    - 軸ラベルの単位保持。
- **目視確認**: Jupyter Notebook を使用して、描画の見た目とインタラクティブ性を確認。
