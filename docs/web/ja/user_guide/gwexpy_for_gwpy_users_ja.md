# GWpy ユーザー向け移行ガイド (GWpy to GWexpy Migration)

GWexpy は、GWpy の基本クラス（TimeSeries 等）を継承しつつ、多チャンネル解析や信号処理の利便性を大幅に強化したライブラリです。
このガイドでは、GWpy ユーザーが GWexpy に移行する際の主な変更点と、新機能を活用したコードの簡略化例を紹介します。

| 機能・目的 | GWpy スタイル (従来) | GWexpy スタイル (推奨) |
| --- | --- | --- |
| 複数チャンネルの管理 | `TimeSeriesDict` | `TimeSeriesMatrix` (Stable) |
| 一括解析 (ASD/CSD) | ループ回し or Dict 操作 | `.asd()`, `.csd()` (Stable) |
| 深いコピー・Pickle | Pickle 不可な場合がある | 高い可搬性 (Pickle compatible) |
| 高度な信号処理 | `scipy` 等を別途呼び出し | `.hht()`, `.arima()` 等を内蔵 (Experimental) |
| 空間・多次元データ | 対応する特殊クラスなし | `ScalarField` (Experimental) |

## 1. チャンネル管理と一括解析

GWpy では `TimeSeriesDict` で複数のチャンネルを管理していましたが、GWexpy ではチャンネル情報を「行列の軸」として扱う `TimeSeriesMatrix` を推奨します。これにより、ループを書かずに一括でスペクトル解析が可能になります。

### コード例: CSD (クロススペクトル密度) の計算

**GWpy スタイル:**

```python
from gwpy.timeseries import TimeSeriesDict
tsd = TimeSeriesDict.read(cache, channels)
# CSDを計算するには、各ペアに対してループ等が必要
```

**GWexpy スタイル (安定性: :term:`Stable`):**

```python
from gwexpy.timeseries import TimeSeriesDict
tsd = TimeSeriesDict.read(cache, channels)

# 行列に変換
matrix = tsd.to_matrix()

# 全チャンネルペアの CSD を一括計算
csm = matrix.csd(fftlength=4)
csm.plot().show()
```

## 2. 信号処理のシームレスな統合

GWpy の基本メソッドに加え、SciPy や Statsmodels などの高度なアルゴリズムが基本クラスに Mixin されています。

### 主な拡張メソッド

* **フィッティング (安定性: :term:`Stable`)**: `.fit()` メソッドで、`iminuit` を用いた最小二乗法や MCMC 解析が直接実行可能です。
* **ピーク検出 (安定性: :term:`Stable`)**: `.find_peaks()` により、パルストレインや共振の特定が容易。
* **瞬時周波数解析 (安定性: :term:`Experimental`)**: `.hht()` を呼び出すだけで、Hilbert-Huang 変換による解析が行えます。
* **統計的予測 (安定性: :term:`Experimental`)**: `.arima()` により、信号の自己相関に基づいた予測やノイズ除去が行えます。

## 3. 拡張された I/O サポート

GWpy がサポートする `gwf`, `hdf5`, `ascii` 等に加え、実験現場で多用される以下のフォーマットを標準サポートしています。

* **GBD (GraphTec)**: デジタル CH の正規化や、レンジ情報に基づく count->V 換算を自動化。
* **TDMS (LabVIEW)**: NI 製ハードウェアで記録されたデータの直接読み込み。
* **WIN (地震波)**: 日本の地震観測網で標準的な WIN フォーマットのデコード。
* **Zarr / Parquet**: 大規模データの高速なクラウド/ディスク I/O。

## 4. 可搬性と互換性 (Pickle)

GWexpy は「解析結果を共有する」ことを重視しています。
`Pickle` でオブジェクトを保存した場合、**読み込み側に GWexpy がインストールされていなくても、GWpy があれば基本クラスのオブジェクトとして復元できる** 設計（GWexpy 透過 Pickle）を採用しています (安定性: :term:`Stable`)。

:::{important}
信頼できないソースからの Pickle データのロードは避けてください。
:::

## 5. 高次元データへの展開 (Field API)

空間的な広がり（センサーアレイなど）を扱う場合、`TimeSeries` を拡張した `ScalarField` を使用できます。

* **ドメイン変換 (安定性: :term:`Experimental`)**: `.fft_space()` により、時間・空間ドメインから周波数・波数ドメインへの 2 次元変換が可能です。
* **空間抽出**: 任意の位置座標における時系列を、補間を含めて 1 行で抽出できます。

---

## 次のステップ

* [クイックスタート](quickstart.md) - 実際のコードを動かしてみる
* [はじめに](getting_started.md) - ロードマップの確認
* [リファレンス](../reference/index.rst) - 各クラスの全メソッドを確認
