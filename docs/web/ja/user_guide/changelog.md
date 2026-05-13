# 更新履歴 (Changelog)

GWexpy の主な変更履歴を記載します。

## [0.1.4] - 2026-05-14
### 追加 (Added)
- `to_gps()` に opt-in の `dtype=` 出力モードを追加しました。デフォルトは
  GWpy 互換のまま維持し、`dtype=float` / `dtype="float"` は通常の数値秒、
  `dtype="quantity"` は `.times` と直接比較しやすい秒の Quantity を返します。

### ドキュメント (Documentation)
- README、docs hub、roadmap、troubleshooting、footer link から、軽量なバグ報告
  と機能追加リクエストを公開 feedback form に案内するよう更新しました。
  セキュリティ報告は引き続き repository security policy に誘導します。

### テスト (Tests)
- NetCDF fixture が明示的な time coordinate を持つことを確認するカバレッジを
  追加しました (#393)。
- multi-channel の GWF list-source read と、`parallel > 1` での padded gap read
  に対する回帰テストを追加しました。

## [0.1.3] - 2026-05-12
### 修正 (Fixed)
- `TimeSeries` / `TimeSeriesDict` の multi-file GWF read を修正しました。
- `TimeSeriesMatrix.read()` の ndscope HDF5 auto-detection を修正しました。
- 公開 I/O ドキュメントと contract metadata を現行の autodetection 挙動に合わせました。
- FrequencySeries CSV fast path で元の frequency column 値を保持するようにしました。
- zarr 3 matrix round-trip coverage を改善し、timeout しやすい fixture 挙動を削除しました。
- loadable な GMT shared library がない PyGMT 環境を、import failure ではなく
  optional backend unavailable として扱うようにしました。

### 既知の問題 (Known Issues)
- bundled NetCDF fixture の一部経路では、TimeSeries reader の time-coordinate
  contract に失敗する場合があります (#393)。generated NetCDF round-trip coverage は
  引き続き通過していますが、利用するファイルでは明示的な time coordinate を確認してください。

## [0.1.2] - 2026-05-08
### 対象を絞った hotfix 範囲
- GWpy4 向け公開 I/O proxy import と GWF list/dict read 挙動の互換性修正。
- histogram HDF5、ATS/MTH5、audio、seismic、SegmentTable span CSV、FrequencySeries DTT XML の auto-identify/read-path 修正。
- この統合トラックでは #369 の最小 landing/demo import hunk のみを含みます。

## [0.1.1] - 2026-04-28
### 追加 (Added)
- `SegmentTable`、histogram/segments 補助機能、detector/time 変換、CLI、plot helper に関する公開契約の基準化ドキュメントとテストを追加。
- PyPI 公開前の release gate として、メタデータ整合性、配布物の健全性、fresh environment での wheel smoke を確認する仕組みを追加。

### 変更 (Changed)
- optional extras、ソースインストール、将来の配布チャネル切り替えに関する案内を整理。

### 既知の制限とフォローアップ (Known Limitations And Follow-Ups)
- #293 の最終段階である PyPI 公開はまだ人手実行のままです。公開インストール手順は、最初の PyPI リリースと post-publish smoke が成功するまで GitHub / ソース導入のままにします。`pip install gwexpy` への切り替えは公開後に行います。
- `conda-forge` パッケージはまだ公開されていません。#294 では staged-recipes 提出と fresh conda 環境での smoke test を継続します。
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
