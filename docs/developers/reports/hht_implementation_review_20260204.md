# HHT 実装・ドキュメント・チュートリアルの現状調査報告

`TimeSeries.hht()` および関連昨日の現状について、コード、ドキュメント、チュートリアルを調査しました。

## 1. 実装 (`gwexpy/timeseries/_spectral_special.py`)

### 問題点: EMD/EEMD パラメータの不整合

- `TimeSeries.emd()` メソッドは `sift_max_iter`, `stopping_criterion` という引数を受け取りますが、内部で `PyEMD` のインスタンスを作成する際や実行する際に**これらの値を渡していません**。
  - 結果として、ユーザーがこれらの値を指定しても **無視され、PyEMDのデフォルト値が常に使用されます**。
  - これは、「調査した実装上の注意点（停止条件 $\epsilon \approx 0.2$ 等の推奨）」をユーザーが適用しようとしても機能しないことを意味します。

### 良い点

- `hilbert_analysis()` は位相アンラップ (`unwrap_phase`) や IF の平滑化 (`if_smooth`) を適切に実装しています。
- `hht()` メソッドは、瞬時周波数のビン詰め（Binning）やエネルギー重み付け (`ia2`) を含め、調査した Spectrogram 生成の手順を正しく踏襲しています。

## 2. ドキュメント (Docstring)

- `hht()` および `emd()` の docstring は、引数の説明自体は存在しますが、上記の実装漏れにより「指定可能」と記述されているパラメータが実際には効かない状態です。
- `hilbert_kwargs` で `pad` を指定できる旨の記載があり、これは実装（`**kwargs` の転送）と整合しています。

## 3. チュートリアル (`advanced_hht.ipynb`)

- **内容が古い/低レベル**: 現在のチュートリアルは、`ts.hht()` という便利な高レベル API を使用せず、`emd()` → ループ処理 → `instantaneous_frequency()` → 散布図作成、という手順をすべて手動で行っています。
- **EEMDの活用不足**: デフォルトの `emd` (EEMDではない) を使用しており、モードミキシング対策としての EEMD の利点が示されていません。
- **可視化**: ビン詰めされた `HHTSpectrogram` ではなく、単なる散布図 (`plt.scatter`) を使用しています。

## 結論・推奨事項（今回は実施せず）

1. **バグ修正**: `TimeSeries.emd()` 内で `PyEMD` インスタンスに対し `FIXE`, `FIXE_H`, `MAX_ITERATION` 等の属性を適切に設定するよう修正が必要です。
2. **チュートリアル更新**: 手動の手順は「仕組みの理解」として残しつつ、実用的な解析として `ts.hht(output='spectrogram')` を使い、EEMD と HHTSpectrogram を活用する例を追加すべきです。
