# 更新履歴 (Changelog)

GWexpy の主な変更履歴を記載します。

## [0.1.1] - 2026-04-28
### 追加 (Added)
- `SegmentTable`、histogram/segments 補助機能、detector/time 変換、CLI、plot helper に関する公開契約の基準化ドキュメントとテストを追加。
- PyPI 公開前の release gate として、メタデータ整合性、配布物の健全性、fresh environment での wheel smoke を確認する仕組みを追加。

### 変更 (Changed)
- 公開インストール手順を、core Python library の PyPI パッケージに切り替えました。conda-forge は引き続きレビュー中として扱います。

### 既知の制限とフォローアップ (Known Limitations And Follow-Ups)
- PyPI `gwexpy==0.1.1` は公開済みで、fresh install smoke test も通過しています。通常利用者は `pip install gwexpy` から開始してください。ソースインストールは、開発者や未リリース変更の検証向けです。
- `conda-forge` パッケージはまだ公開されていません。staged-recipes PR は open / CI green ですが、feedstock 作成、パッケージ公開、fresh conda 環境での smoke test が終わるまで `conda install -c conda-forge gwexpy` は公開手順として案内しません。
- ノイズ契約 (#278)、astro range の単位と前提 (#282)、BruCo/coupling/response ワークフロー (#284)、preprocessing / decomposition / forecasting 契約 (#288) は、現状挙動の docs/test baseline は入ったものの、方針決定が残っています。
- GUI と可視化まわりでは、payload metadata、ラベル、colorbar、plot helper の意味論、公開ドキュメントの残差 drift が継続課題です (#274, #275, #283)。GUI は引き続き安定性ラベル上の実験的機能です。
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
