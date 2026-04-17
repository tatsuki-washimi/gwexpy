.. _glossary_ja:

用語集 (Glossary)
=================

GWexpy のドキュメントで使用される主要な用語や概念の定義です。

.. glossary::

   ScalarField (ja)
      (スカラー場) GWexpy の 4 次元フィールドコンテナ（time, x, y, z）。ドメイン情報と軸メタデータを保持する主要データ構造です。

   TimeSeries (ja)
      (時系列信号) 単一チャネルの時系列オブジェクト（GWpy 互換）。時刻軸と値を持つ基本データ型です。

   TimeSeriesMatrix (ja)
      (時系列行列) 多チャネルの時系列データを一括処理するための行列形式コンテナです。

   FrequencySeriesMatrix (ja)
      (周波数系列行列) 周波数ドメインの多チャネルデータを扱う行列形式コンテナです。

   FieldList (ja)
   FieldDict (ja)
      ``ScalarField`` 等の複数のフィールドを格納し、一括操作するためのコレクション型です。

   SeriesMatrix (ja)
      (シリーズ行列) 多チャネルの信号（TimeSeries または FrequencySeries）を保持するコンテナの総称です。別名が付与されている場合があります。

   ASD (ja)
      (振幅スペクトル密度) Amplitude Spectral Density。信号の振幅成分を周波数ごとに密度として表したものです。

   PSD (ja)
      (パワースペクトル密度) Power Spectral Density。信号のパワー（振幅の2乗）成分を周波数ごとに密度として表したものです。

   CSD (ja)
      (相互スペクトル密度) Cross Spectral Density。2つの信号間の相関を周波数領域で評価するための指標です。

   FFT / STFT / CWT / HHT (ja)
      各種の時間–周波数変換手法。それぞれ高速フーリエ変換、短時間フーリエ変換、連続ウェーブレット変換、ヒルベルト・ホアン変換を指します。

   Whitening (ja)
      (ホワイトニング) 信号を、そのパワースペクトルが平坦（白）になるように処理すること。ノイズ特性を正規化するために用いられます。

   Adaptive Whitening (ja)
   AD-Whitening (ja)
      (適応ホワイトニング) ノイズの局所的な統計的特性の変化に追従して、動的に正則化や均質化を行う高度なホワイトニング手法です。

   NaN/Inf propagation (ja)
      (NaN/Inf の伝播 / 旧: Death Floats) 計算途中で非数 (NaN) や無限大 (Inf) が発生し、それが後続의 計算に広がることで解析結果が破綻する現象です。

   VIF (ja)
      (分散膨張係数) Variance Inflation Factor。多重共線性の指標。回帰解析や説明変数間の相関が、推定値の分散をどの程度増大させているかを評価するために用います。

   ``BruCo`` (ja)
      BruCo はコヒーレンス／相関に基づくノイズ解析フレームワークです。実装上のキャピタライゼーションは `BruCo` または `Bruco` となりますが、ドキュメントではソースコードの定義に従います。

   Field API (ja)
      (フィールド API) ``ScalarField`` や関連する操作を提供する公開インターフェースです。

   TimePlaneTransform (ja)
      時間軸と 2 つの平面軸を持つ 3 次元変換結果を扱うコンテナです。STLT のような変換では `(time, sigma, frequency)` のような軸構造を表現できます。

   Safe Log (ja)
      (セーフログ) 対数変換時の下限処理。デフォルト値は 200 dB で、パラメータで上書き可能です。

   SegmentTable (ja)
      (セグメントテーブル) 解析で用いる区間（セグメント）を管理するテーブル。各行は `t0, t1, label, quality_flag` 等を持ちます。

   CITATION.cff (ja)
      論文やソフトウェアの引用情報を標準化されたフォーマットで記述したメタデータファイルです。

   GWOSC (ja)
      (LIGO オープンデータ) Gravitational Wave Open Science Center。LIGO、Virgo、KAGRA の観測データや重力波カタログを公開しているプラットフォームです。

   miniSEED / GWF / GBD / MTH5 / TDMS / Zarr (ja)
      GWexpy がサポートする各種入出力ファイル形式です。

   Pickle (ja)
      (Pickle — セキュリティ注意) Python のシリアライズ形式。読み込みは任意のコード実行を招く恐れがあるため、信頼できるソースからの読込に限定し、可能なら HDF5 や Zarr などの代替形式を推奨します。

   tconvert / to_gps / from_gps (ja)
      GPS 秒と UTC、datetime、ISO 文字列を相互変換するための GPS 時刻ユーティリティ関数群です。

   Leap second (ja)
      (閏秒) 地球の自転速度の変化と原子時計のずれを調整するために挿入される 1 秒。時刻変換において重要な概念です。

   GPS time (ja)
      (GPS 時刻) GPS のエポック（1980年1月6日）に同期した秒数。閏秒の影響を受けない単調増加な時刻系です。

   UTC (ja)
      (協定世界時) 原子時計に基づく世界共通の標準時。閏秒による調整が行われます。

   MCMC (ja)
      (マルコフ連鎖モンテカルロ法) ベイズ推定などで定常分布からのサンプリングを行うための数値計算法です。

   ICA / PCA (ja)
      (独立成分分析 / 主成分分析) 信号分離や次元削減を行うための統計的手法です。

   Robust ICA (ja)
      (ロバスト ICA) 外れ値やノイズに対して頑健な独立成分分析の実装です。

   ASPIRE / ICRR / LALSuite / PyCBC / Bilby (ja)
      重力波解析や関連研究で使用される外部ツールやライブラリ、機関の名称です。

   Stability Labels (ja)
      (安定性ラベル) API の成熟度を示す指標です。 **Stable** （安定版）、 **Experimental** （実験的機能）、 **Deprecated** （非推奨機能）のいずれかがラベル付けされます。

   4D Slicing (ja)
   4D Persistence (ja)
      (4次元維持) ScalarField において、インデクシング操作時に次元が削減（Rank Loss）されず、常に (Time, Freq, x, y) の 4次元構造が保たれる性質です。
