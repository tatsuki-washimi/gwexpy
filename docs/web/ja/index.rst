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


主要ドキュメント
----------------

.. toctree::
   :maxdepth: 2
   :caption: ユーザーガイド

   user_guide/installation
   user_guide/quickstart
   user_guide/getting_started
   user_guide/tutorials/index
   user_guide/gwexpy_for_gwpy_users_ja

.. toctree::
   :maxdepth: 2
   :caption: 実例集 (Examples)

   examples/index

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


.. admonition:: 学習フロー
   :class: note

   **推奨される学習の流れ:**

   1. :doc:`インストール <user_guide/installation>` → 環境構築
   2. :doc:`クイックスタート <user_guide/quickstart>` → 基本操作
   3. :doc:`基本チュートリアル <user_guide/getting_started>` → データ構造理解
   4. :doc:`実例集 <examples/index>` → 応用例を確認
   5. :doc:`APIリファレンス <reference/index>` → 詳細仕様参照


言語 (Language)
---------------

* :doc:`English <../en/index>`
