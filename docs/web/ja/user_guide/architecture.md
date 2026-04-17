# アーキテクチャとデータフロー

`gwexpy` の設計思想と、内部的なデータ処理の流れについて解説します。
本パッケージは GWpy を拡張し、多チャネルの TimeSeriesMatrix 操作や 4次元物理フィールドの扱いを直感的に行えるよう設計されています。

## このページでわかること

| 項目 | 内容 |
| --- | --- |
| **ページ種別** | 設計ガイド |
| **対象読者** | コンテナ設計の考え方を知りたい利用者、内部データフローを把握したいコントリビュータ |
| **前提** | `TimeSeries` / `TimeSeriesMatrix` / `ScalarField` の基本名を知っていること |
| **こんなときに読む** | Matrix 系やフィールド API がなぜこの形なのか知りたい、理論ページや API ページへ進む前に地図が欲しい |
| **検索キーワード** | architecture, data flow, `TimeSeriesMatrix`, `ScalarField`, flattening, フィールド API |

## このページの近道

- [このページの読み方](#このページの読み方)
- [データ構造の設計思想](#データ構造の設計思想)
- [主要な解析コンポーネントの概念](#主要な解析コンポーネントの概念)
- [関連ドキュメント](#関連ドキュメント)

## このページの読み方

- `gwexpy` の内部表現や変換の考え方を掴みたい場合は、このページを先に読んでください。
- 具体的なアルゴリズムの理論は [物理モデルと解析理論](physics_models.md) を参照してください。
- 実装 API を確認したい場合は、各節末尾の API リンクから [matrix API](../reference/api/matrix.rst), [fields API](../reference/api/fields.rst), [fitting API](../reference/api/fitting.rst) へ進んでください。

## データ構造の設計思想

### 1. 行列オブジェクトの平坦化フロー

`TimeSeriesMatrix` や `FrequencySeriesMatrix` は、内部で scikit-learn 等の機械学習ライブラリと親和性の高い形式への変換を自動で行います。
通常、(チャネル数, 時間/周波数数, 行列の列数) といった 3次元的なデータを、メタデータを保持したまま一時的に 2次元へ平坦化し、計算後に元の次元と GPS タイムスタンプを復元します。これにより、高度なアルゴリズムを物理情報を失わずに適用可能です。

API 入口: [matrix API](../reference/api/matrix.rst), [timeseries API](../reference/api/timeseries.rst)

### 2. 4次元フィールド（フィールド API）モデル

`ScalarField` は (Time, Frequency, x, y) の 4次元構造を基本単位とします。
インデクシング操作において**次元を削減しない**（4次元を維持する）ことで、グリッド情報とデータの整合性を常に担保しています。

API 入口: [fields API](../reference/api/fields.rst), [スライス操作ガイド](scalarfield_slicing.md)

---

## 主要な解析コンポーネントの概念

`gwexpy` は、以下の主要コンポーネントを組み合わせて高度な解析パイプラインを構築します。個別のアルゴリズム詳細は [物理モデルと解析理論](physics_models.md) を参照してください。

- **多チャネル解析エンジン**: 環境ノイズ分離のための ICA/PCA 実装。
- **高速相関計算基盤**: 大規模観測データに対する高速コヒーレンス計算（Bruco）。
- **統計的推定・フィッティング**: GLS や MCMC を用いた物理パラメータ推定。

## 次に読む
- [物理モデルと解析理論](physics_models.md) — ICA/Bruco/MCMC などの解析理論・物理モデル詳細
- [検証済みアルゴリズム](validated_algorithms.md) — 各数値アルゴリズムの信頼性検証
- [4次元フィールドの操作詳細](scalarfield_slicing.md)
- [前提条件と規約](prerequisites_and_conventions.md) — FFT・GPS 時刻・GWpy 互換の前提を整理する
