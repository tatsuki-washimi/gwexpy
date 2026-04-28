# 更新履歴 (Changelog)

GWexpy の主な変更履歴を記載します。

## [0.1.1] - 2026-04-05
### 追加 (Added)
- `SegmentTable`、histogram/segments 補助機能、detector/time 変換、CLI、plot helper に関する公開契約の基準化ドキュメントとテストを追加。
- PyPI 公開前の release gate として、メタデータ整合性、配布物の健全性、fresh environment での wheel smoke を確認する仕組みを追加。

### 変更 (Changed)
- optional extras、ソースインストール、将来の配布チャネル切り替えに関する案内を整理。

### 既知の制限とフォローアップ (Known Limitations And Follow-Ups)
- #293 の最終段階である PyPI 公開はまだ人手実行のままです。公開インストール手順は、最初の PyPI リリースと post-publish smoke が成功するまで GitHub / ソース導入のままにします。`pip install gwexpy` への切り替えは公開後に行います。
- `conda-forge` パッケージはまだ公開されていません。#294 では staged-recipes 提出と fresh conda 環境での smoke test を継続します。
- ノイズ契約 (#278)、astro range の単位と前提 (#282)、Bruco/coupling/response ワークフロー (#284)、preprocessing / decomposition / forecasting 契約 (#288) は、現状挙動の docs/test baseline は入ったものの、方針決定が残っています。
- GUI と可視化まわりでは、payload metadata、ラベル、colorbar、plot helper の意味論、公開ドキュメントの残差 drift が継続課題です (#274, #275, #283)。GUI は引き続き experimental 扱いです。
- ローカル検証のフォローアップ #335 では、`pytest tests/ -q` を 1 プロセスで通したときに exit 139 が断続的に発生します。分割スイートでは通過していますが、単一プロセス異常終了の原因は未解明です。

## [0.1.0] - 2026-04-08
### 追加 (Added)
- ドキュメントデザインの刷新 (Task 1 & 2)。
- `SeriesMatrix` 系クラスの安定化。
- `ScalarField` による多次元データサポートの強化。

### 修正 (Fixed)
- `CITATION.cff` の追加による引用の容易化。
- 依存関係の整理とOS別インストールガイドの充実。

## 過去の履歴

詳細は [GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases) を参照してください。
