.. meta::
   :description: GWexpy のリファレンス入口です。API、クラス、トピック別に安定した参照先へ移動できます。

.. _reference-ja-entry:

リファレンス (Reference)
========================

.. note::
   ページ種別: リファレンス索引

**安定性:** 安定

このページは GWexpy のリファレンス入口です。モジュール別に探すか、クラス名で引くか、概念別に探すかに応じて入口を選んでください。

**対象読者:** 使いたい機能の見当がついていて、正確な API やクラス、トピック情報を引きたい利用者。
**前提知識:** GWexpy の基本用語を理解し、少なくともひとつのガイドかチュートリアルを読んでいること。
**このページの用途:** 手順学習ではなく、安定した参照先を素早く引くこと。
**検索のヒント:** API 索引, クラス索引, トピック, リファレンス, モジュール検索, クラス検索, 理論メモ

.. note::
   このページの見方:
   API リファレンスはサブシステム別の参照、クラス索引は Python クラス名からの参照、トピック別参照は規約や理論、補助資料の参照に向いています。

.. note::
   高度・理論系の入口:
   検証前提、規約、監査根拠付きの理論メモを確認したい場合は、まず :doc:`topics` を開き、その後 高度・理論向けの入口である :doc:`../user_guide/validated_algorithms` に進んでください。

.. note::
   使い方の目安:
   目的: すでに名前が分かっているオブジェクト、モジュール、話題の詳細を引くこと。
   入力: クラス名、モジュール名、または概念名。
   出力: 詳細確認の起点になる安定した参照ページ。

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
