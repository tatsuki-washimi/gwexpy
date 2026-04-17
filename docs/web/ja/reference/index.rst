リファレンス (Reference)
========================

**安定性:** Stable

このページは GWexpy のリファレンス入口です。モジュール別に探すか、クラス名で引くか、概念別に探すかに応じて入口を選んでください。

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 入口
     - 安定性
     - 用途
   * - :doc:`API リファレンス (API Reference) <api/index>`
     - Stable
     - サブシステム別にモジュールと公開関数をたどる
   * - :doc:`クラス索引 (Class Index) <classes>`
     - Stable
     - Python クラス名から個別ページを引く
   * - :doc:`トピック別参照 (Topics) <topics>`
     - Stable
     - 規約、理論、補助ページを概念別に探す

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

.. note::
   学習用の導線が必要な場合は user guide / tutorials を参照してください。`ScalarField` の設計意図や操作例は :doc:`../user_guide/scalarfield_slicing` にあります。

.. toctree::
   :maxdepth: 2

   api/index
   classes
   topics
