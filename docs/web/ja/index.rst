:orphan:

GWexpy ドキュメントへようこそ
==============================

GWexpy は GWpy を拡張し、時系列および周波数系列データ解析のための新たなコンテナやユーティリティを提供します。

**主な特徴:**

* TimeSeries/FrequencySeries 行列対応
* 高次元フィールドAPI (ScalarField, VectorField, TensorField)
* 数値計算の安定化機能
* 実験データ解析向けユーティリティ

.. admonition:: クイックスタート
   :class: tip

   まず :doc:`インストール <user_guide/installation>` で環境構築し、次に :doc:`クイックスタート <user_guide/quickstart>` で基本的な使い方を学びましょう。


あなたに合ったガイド
--------------------

.. grid:: 3

    .. grid-item-card:: 初めての方へ
        :link: user_guide/getting_started
        :link-type: doc

        基礎から学べます

        * インストールガイド
        * クイックスタート
        * 基本チュートリアル

    .. grid-item-card:: 実験データ解析者向け
        :link: examples/index
        :link-type: doc

        実例から学べます

        * 基本的な使用例
        * 高度な信号処理
        * 実世界のケーススタディ

    .. grid-item-card:: GWpyユーザー向け
        :link: user_guide/gwexpy_for_gwpy_users_ja
        :link-type: doc

        移行情報とアップグレード

        * GWpyからの違い
        * 新機能ハイライト
        * 互換性情報


視覚的な例
----------

.. figure:: ../../_static/images/hero_plot.png
   :align: center
   :width: 90%
   :alt: GWexpyによる時系列データの可視化例

   GWexpyによる時系列データの可視化例


実例ギャラリー
--------------

代表的なケーススタディ:

.. grid:: 3

    .. grid-item-card:: ノイズバジェット解析
        :link: examples/index
        :link-type: doc
        :img-top: ../../_static/images/case_noise_budget_thumb.png

        多チャンネル相関解析とノイズ源特定

    .. grid-item-card:: 伝達関数の測定
        :link: examples/index
        :link-type: doc
        :img-top: ../../_static/images/case_transfer_function_thumb.png

        ボード線図とモデルフィッティング

    .. grid-item-card:: アクティブダンピング
        :link: examples/index
        :link-type: doc
        :img-top: ../../_static/images/case_active_damping_thumb.png

        6自由度MIMO制御シミュレーション


ユーザー別学習パス
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ユーザー層
     - まず読む
     - 次に進む
   * - 初学者
     - :doc:`インストール <user_guide/installation>`, :doc:`クイックスタート <user_guide/quickstart>`
     - :doc:`はじめに <user_guide/getting_started>`, :doc:`チュートリアル <user_guide/tutorials/index>`
   * - 実験系研究者
     - :doc:`ケーススタディ <examples/index>`
     - :doc:`高度な信号処理 <user_guide/tutorials/index>`, :doc:`APIリファレンス <reference/index>`
   * - GWpyユーザー
     - :doc:`GWpy移行ガイド <user_guide/gwexpy_for_gwpy_users_ja>`
     - :doc:`新機能チュートリアル <user_guide/tutorials/index>`


主要ドキュメント
----------------

.. toctree::
   :maxdepth: 2
   :caption: ユーザーガイド

   user_guide/installation
   user_guide/quickstart
   user_guide/getting_started
   user_guide/gwexpy_for_gwpy_users_ja

.. toctree::
   :maxdepth: 2
   :caption: チュートリアル

   user_guide/tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: ケーススタディ

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: 高度なガイド

   user_guide/numerical_stability
   user_guide/scalarfield_slicing
   user_guide/validated_algorithms

.. toctree::
   :maxdepth: 2
   :caption: リファレンス

   reference/index


次のステップ
------------

学習を進めるには:

* :doc:`はじめに <user_guide/getting_started>` - 推奨学習パスの詳細
* :doc:`チュートリアル一覧 <user_guide/tutorials/index>` - 全チュートリアル
* :doc:`実例集 <examples/index>` - 実世界のケーススタディ
* :doc:`APIリファレンス <reference/index>` - クラス・関数の詳細

**高度なトピック:**

* :doc:`スカラーフィールドのスライス操作 <user_guide/scalarfield_slicing>` - 多次元データ操作
* :doc:`数値的安定性と精度 <user_guide/numerical_stability>` - 計算の信頼性
* :doc:`検証済みアルゴリズム <user_guide/validated_algorithms>` - アルゴリズム検証レポート


言語 (Language)
---------------

* :doc:`English <../en/index>`
