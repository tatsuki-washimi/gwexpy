.. meta::
   :description: GWexpy のリファレンス入口です。API、クラス、トピック別に安定した参照先へ移動できます。

.. _reference-ja-entry:

リファレンス (Reference)
========================

**安定性:** 安定

このページは GWexpy のリファレンス入口です。モジュール別に探すか、クラス名で引くか、概念別に探すかに応じて入口を選んでください。

.. _reference-ja-entry-table:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 入口
     - 安定性
     - 用途
   * - :doc:`API リファレンス (API Reference) <api/index>`
     - 安定
     - サブシステム別にモジュールと公開関数をたどる
   * - :doc:`クラス索引 (Class Index) <classes>`
     - 安定
     - Python クラス名から個別ページを引く
   * - :doc:`トピック別参照 (Topics) <topics>`
     - 安定
     - 規約、理論、補助ページを概念別に探す

.. _reference-ja-entry-cards:

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 🧩 API リファレンス
        :link: api/index
        :link-type: doc

        モジュール別の API と関数をサブシステム単位でたどれます。

    .. grid-item-card:: 🏗️ クラス索引
        :link: classes
        :link-type: doc

        主要クラスを英名のアルファベット順で引けます。

    .. grid-item-card:: 🧭 トピック別参照
        :link: topics
        :link-type: doc

        規約、理論、補助ページを概念別にまとめています。

.. seealso::
   ハブ間の移動:

   - :doc:`api/index` でサブシステム別に API をたどる
   - :doc:`topics` で理論・規約・橋渡しページから入る
   - :doc:`../user_guide/tutorials/index` でチュートリアルから学んだあとに参照へ戻る

.. note::
   リファレンスではなく用途別ガイドから入りたい場合は、総合 index ではなく次の個別ページを起点にしてください。

   - :doc:`../user_guide/scalarfield_slicing` for `ScalarField` のスライス設計と実例
   - :doc:`../user_guide/validated_algorithms` for 高度・理論向けの検証前提と監査ベースのノート
   - :doc:`../user_guide/gwexpy_for_gwpy_users_ja` for GWpy からの移行ガイド

.. seealso::
   次に読むページ:

   - :doc:`../user_guide/tutorials/index` で notebook ベースの学習ステップをたどる
   - :doc:`api/index` でカテゴリ別の API 入口を開く
   - :doc:`topics` で理論、規約、補助資料を概念別に探す

.. toctree::
   :maxdepth: 2

   api/index
   classes
   topics
