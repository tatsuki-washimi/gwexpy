---
name: visualize_fields
description: 多次元フィールドデータ（Field4D/Matrix等）の抽出ロジックと描画APIを、物理的整合性を保ちながら実装する
---

# Visualize Fields Best Practices

多次元データ（3D, 4D, Matrix等）の可視化機能を実装する際は、メンテナンス性と物理的正確性を確保するため、以下の3レイヤー構造で設計します。

## 1. 座標・ユーティリティ層 (Infrastructure)

物理座標と配列インデックスの変換ロジックをクラス本体から切り離し、再利用可能な関数として実装します。

- **`nearest_index(axis, value)`**: 単位系（Astropy Units）の相互変換を保証した上での最近傍検索。
- **`select_value(data, mode)`**: 複素数データから `real`, `abs`, `power` 等を抽出。
- **配置**: `gwexpy/plot/utils.py` または `_coord.py`。これにより、データ型クラス（Types層）が描画ライブラリ（Matplotlib等）に直接依存することを防ぎます。

## 2. 抽出API層 (Extraction API)

可視化用の「部分集合」を生成するメソッドをデータ型クラスに追加します。

- **`extract_points` / `slice_map2d`**: 描画にそのまま使える「形」に整えたデータを返します。
- **規約**: 可能な限り元のクラス（Field4D等）を維持するか、標準的な `TimeSeries`/`FrequencySeries` を返します。

## 3. 描画API層 (Plotting API)

ユーザーが直接呼び出す描画メソッドです。

- **命名規約**: `plot_map2d`, `plot_profile`, `plot_timeseries_points` 等。
- **Spectral Visualization**:
  - `freq_space_map` (Waterfall等) は、時間軸を周波数軸に置換した2Dマップです。
  - スペクトル密度（PSD）をプロットする際は、対数スケール (`norm=LogNorm` または `set_yscale('log')`) をデフォルトで考慮します。

## 物理的整合性のチェックリスト

- [ ] **単位伝播**: `power` モード時に単位が `unit^2` になっているか？ `angle` が `rad` か？
- [ ] **座標の正確性**: 描画軸の数値と、元のデータの物理軸が `nearest_index` 等を通じて正しく対応しているか？
- [ ] **次元の維持**: スライス操作によって予期せず次元が消失（Squeeze）し、描画ロジックが壊れていないか？
- [ ] **メモリ効率**: 大規模データに対して、不必要なコピー（`copy=True`）が発生していないか？
