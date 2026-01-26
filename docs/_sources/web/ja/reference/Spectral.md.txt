# スペクトル推定

gwexpy はスペクトル密度とパワースペクトル推定に厳密な単位規則を適用します。

## 単位のセマンティクス

* **PSD / ASD (密度)**:
    * デフォルトの `method='welch'` (やその他) は**パワースペクトル密度**を返します。
    * **単位**: $Unit^2 / Hz$ (例: $V^2 / Hz$)。
* **パワースペクトル**:
    * `scaling='spectrum'` を指定した場合。
    * **単位**: $Unit^2$ (例: $V^2$)。
    * これはビン幅で正規化されていない、各ビンのパワーを表します。

## バックエンド

サポートされるバックエンド (`scipy`, `lal`, `pycbc`) は、基盤となる実装に関係なく、これらの単位契約が満たされるようにラップされています。

## 主要関数

| 関数 | 説明 |
|------|------|
| `estimate_psd(ts, ...)` | NaN除外機能付きPSD推定ラッパー |
| `bootstrap_spectrogram(sgm, ...)` | 誤差棒付きロバストASD/PSD用ブートストラップリサンプリング |
| `calculate_correlation_factor(...)` | Welchオーバーラップ補正用の分散膨張係数 |

## 使用例

```python
from gwexpy.timeseries import TimeSeries
from gwexpy.spectral import estimate_psd

# 時系列データを作成
ts = TimeSeries(data, sample_rate=1024, unit='V')

# PSD を推定（密度正規化）
psd = estimate_psd(ts, fftlength=1.0)
print(psd.unit)  # V^2 / Hz

# パワースペクトルを推定
ps = estimate_psd(ts, fftlength=1.0, scaling='spectrum')
print(ps.unit)  # V^2
```

## 注意事項

* `estimate_psd()` は NaN サンプルを除外します。FFTベースの平均化はNaNを伝播させ、正規化を無効にするためです。呼び出し側はデータを事前にクリーンアップする必要があります。
* `fftlength` はデータの長さを超えてはなりません。超えると `ValueError` が発生します。
