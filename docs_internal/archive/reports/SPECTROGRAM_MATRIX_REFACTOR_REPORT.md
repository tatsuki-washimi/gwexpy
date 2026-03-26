# SpectrogramMatrix リファクタリングレポート

## 概要
`SpectrogramMatrix` クラスを `SeriesMatrix` ベースにリファクタリングし、多次元構造 (3D/4D) のサポートと解析機能を強化しました。

## 主な変更点

1. **継承構造の変更**
   - `SeriesMatrix` を継承し、共通の行列操作機能を利用可能にしました。
   - Mixin (`SpectrogramMatrixCoreMixin`, `SpectrogramMatrixAnalysisMixin`) を導入。

2. **3D/4D データ構造のサポート**
   - 3D: `(Batch, Time, Freq)`
   - 4D: `(Row, Col, Time, Freq)`
   - `is_compatible` 等の検証ロジックをオーバーライドし、メタデータ行列 `(Row, Col)` とデータテンソルの整合性を確保。

3. **機能追加**
   - `append`, `crop`, `pad`, `interpolate` 等の解析メソッド。
   - `to_series_1Dlist`, `to_series_2Dlist` 等の変換メソッド。
   - 算術演算における `Unit` (単位) の正しい伝播。

4. **テスト強化**
   - `test_spectrogram_matrix_features.py`: 主要機能の網羅的テスト。
   - `test_sgm_extra.py`: エッジケース、Plotting、構造操作のテスト。

## 今後の課題
- **Pickle / HDF5 Support**: `SeriesMatrix` プロパティの復元に関する既知の問題（現在は `xfail`）。
- **Transpose**: 軸の意味が変わる操作に対するより良いサポート。
