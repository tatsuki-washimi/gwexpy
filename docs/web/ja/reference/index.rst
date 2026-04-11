リファレンス (Reference)
========================

GWexpy のクラス、関数、および API の詳細な仕様を解説します。

主要なデータ構造
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - クラス名
     - 説明
   * - :doc:`ScalarField <user_guide/scalarfield_slicing>`
     - 4次元（時間、空間）を扱う高次元データコンテナ。
   * - :class:`TimeSeriesMatrix <gwexpy.timeseries.TimeSeriesMatrix>`
     - 複数の時系列データを一括処理するための行列形式コンテナ。
   * - :class:`FrequencySeriesMatrix <gwexpy.frequencyseries.FrequencySeriesMatrix>`
     - 周波数ドメインの多チャネルデータを扱うコンテナ。

API リファレンス
----------------

各モジュールの詳細なメソッドやプロパティについては、以下のリンクを参照してください。

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🧩 API 索引
        :link: api/index
        :link-type: doc

        全てのモジュールと関数のリスト

    .. grid-item-card:: 🏗️ クラス一覧
        :link: classes
        :link-type: doc

        主要なクラスのプロパティとメソッド

.. toctree::
   :hidden:

   api/index
   classes
