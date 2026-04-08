.. _glossary:

用語集 (Glossary)
================

GWexpy のドキュメントで使用される主要な用語や概念の定義です。

.. glossary::

   ScalarField
      (スカラー場) GWexpy の 4 次元フィールドコンテナ（time, x, y, z）。ドメイン情報と軸メタデータを保持する主要データ構造です。

   TimeSeries
      (時系列信号) 単一チャネルの時系列オブジェクト（GWpy 互換）。時刻軸と値を持つ基本データ型です。

   TimeSeriesMatrix
      (時系列行列) 複数の時系列データを一括処理するための行列形式コンテナです。

   FrequencySeriesMatrix
      (周波数系列行列) 周波数ドメインの多チャネルデータを扱う行列形式コンテナです。

   FieldList
   FieldDict
      ``ScalarField`` 等の複数のフィールドを格納し、一括操作するためのコレクション型です。

   SeriesMatrix
      (シリーズ行列) 多チャネルの信号（TimeSeries または FrequencySeries）を保持するコンテナの総称です。

   ASD
      (振幅スペクトル密度) Amplitude Spectral Density。信号の振幅成分を周波数ごとに密度として表したものです。

   PSD
      (パワースペクトル密度) Power Spectral Density。信号のパワー（振幅の2乗）成分を周波数ごとに密度として表したものです。

   CSD
      (相互スペクトル密度) Cross Spectral Density。2つの信号間の相関を周波数領域で評価するための指標です。

   Whitening
      (ホワイトニング) 信号を、そのパワースペクトルが平坦（白）になるように処理すること。ノイズ特性を正規化するために用いられます。

   Adaptive Whitening
      (適応ホワイトニング) ノイズの局所的な統計的特性の変化に追従して、動的に正則化や均質化を行う高度なホワイトニング手法です。

   NaN/Inf propagation
      (NaN/Inf の伝播 / 旧: Death Floats) 計算途中で非数 (NaN) や無限大 (Inf) が発生し、それが後続の計算に広がることで解析結果が破綻する現象です。

   VIF
      (分散膨張係数) 多重共線性の指標。回帰解析や説明変数間の相関が、推定値の分散をどの程度増大させているかを評価するために用います。

   ``BruCo``
      BruCo はコヒーレンス／相関に基づくノイズ解析フレームワークです。代表用途：多チャネルコヒーレンス解析、ノイズ源の逆推定、ブロードバンド干渉の同定。ドキュメントでは実装上のキャピタライゼーション（`BruCo` または `Bruco`）に従います。

   GPS時刻ユーティリティ関数
      ``tconvert``, ``to_gps``, ``from_gps`` など、GPS 秒と UTC、datetime、ISO 文字列を相互変換するための関数群です。

   Safe Log
      対数変換時の下限処理。デフォルト値は 200 dB。パラメータで上書き可能（例：`safe_log(level=200)`）。表示上は「Safe Log」と表記。

   SegmentTable
      (セグメントテーブル) 解析で用いる区間を管理するテーブル。各行は `t0, t1, label, quality_flag` 等を持つ。セグメント結合/フィルタリング操作が主なユースケース。

   Time-Plane Transform
      (時間—周波数平面変換) 時系列データを Q-transform や CWT 等を用いて時間–周波数平面 (Spectrogram) にマッピングする処理です。

   Pickle
      (Pickle — セキュリティ注意) Python のシリアライズ形式。読み込みは任意のコード実行を招くため、信頼できるソースからの Pickle の読込は避け、可能なら代替（HDF5/Zarr）を推奨。

   GWOSC
      (LIGO オープンデータ) Gravitational Wave Open Science Center。LIGO、Virgo、KAGRA の観測データや重力波カタログを公開しているプラットフォームです。

   Stability Labels
      (安定性ラベル) API の成熟度を示す指標です。**Stable** は安定版、**Experimental** は開発中の実験的機能、**Deprecated** は将来削除される非推奨機能を示します。
