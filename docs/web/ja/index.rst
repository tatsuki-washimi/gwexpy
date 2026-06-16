:orphan:

.. meta::
   :description: GWexpy ドキュメントの入口ページ。インストール、クイックスタート、チュートリアル、ケーススタディ、リファレンスへのリンクをまとめています。

GWexpy ドキュメント
==============================

GWexpy は GWpy を拡張し、時系列および周波数系列データ解析のための新たなコンテナや数値計算ユーティリティを提供します。

v\ |release| · Python ≥ 3.11 · 最終更新: |today|

.. raw:: html

   <section class="gw-hub-duo">
     <section class="gw-hub-panel">
       <p>Quick install</p>
       <pre><code>git clone https://github.com/tatsuki-washimi/gwexpy.git
   cd gwexpy
   python -m pip install -e .</code></pre>
     </section>
     <section class="gw-hub-panel">
       <p>3-line demo</p>
       <pre><code>from gwexpy.frequencyseries import FrequencySeriesMatrix
   fsmtx = FrequencySeriesMatrix.read("data.hdf5")
   fsmtx[2, 0].fit(model="lorentzian").plot()</code></pre>
     </section>
   </section>

.. button-ref:: user_guide/quickstart
    :ref-type: doc
    :color: primary
    :shadow:
    :expand:

    🚀 クイックスタート（5分で基本を習得）

.. note::

   不具合報告や機能追加リクエストは
   `簡易フィードバックフォーム <https://forms.gle/c8jJaf9UCs5tb5cC8>`_
   から送れます。セキュリティに関わる報告にはこのフォームを使わず、
   リポジトリのセキュリティポリシーに従ってください。

----

あなたに合ったガイド
--------------------

.. grid:: 3
    :gutter: 3
    :class-container: grid-container

    .. grid-item-card:: 🎓 初めての方へ
        :link: user_guide/getting_started
        :link-type: doc

        基礎から学びたい方向け

        * インストールガイド
        * 機能別チュートリアル

    .. grid-item-card:: 🔬 実験データ解析者向け
        :link: examples/index
        :link-type: doc

        実践的な解析例を知りたい方向け

        * 目的別ケーススタディ
        * 高度な信号処理

    .. grid-item-card:: 🔄 GWpy ユーザー向け
        :link: user_guide/gwexpy_for_gwpy_users_ja
        :link-type: doc

        GWpy から移行・併用する方向け

        * 差分レシピと互換性の入口
        * 追加 API 一覧へのリンク

.. button-ref:: examples/index
    :ref-type: doc
    :color: secondary
    :expand:

    正式な目的別ケーススタディギャラリーを見る

----

GWpy の基礎を学ぶ
-----------------

GWexpy は GWpy の上に構築されています。GWpy の基本操作は下記の公式ドキュメントを参照してください。

`gwpy.readthedocs.io/en/stable/ <https://gwpy.readthedocs.io/en/stable/>`_

----

最初に読むページの目安
----------------------

- 最短でコードを動かす: :doc:`user_guide/quickstart`
- 学習順序を決める: :doc:`user_guide/getting_started`
- GPS 時刻や FFT の前提を先に確認する: :doc:`user_guide/prerequisites_and_conventions`

----

.. toctree::
   :maxdepth: 2
   :caption: 🚀 導入 (Start)
   :hidden:

   user_guide/installation
   user_guide/quickstart
   user_guide/getting_started
   user_guide/prerequisites_and_conventions

.. toctree::
   :maxdepth: 2
   :caption: 🎓 学習 (Learn)
   :hidden:

   user_guide/tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: 📚 リファレンス (Reference)
   :hidden:

   reference/index
   user_guide/io_formats
   user_guide/interop
   user_guide/time_utilities
   user_guide/numerical_stability
   user_guide/scalarfield_slicing
   user_guide/gwexpy_for_gwpy_users_ja
   user_guide/gwpy_added_api_index_ja
   user_guide/validated_algorithms
   user_guide/architecture
   user_guide/physics_models
   user_guide/glossary

.. toctree::
   :maxdepth: 1
   :caption: ℹ️ その他 (Info)
   :hidden:

   user_guide/roadmap
   user_guide/troubleshooting
   user_guide/verification_and_quality
   user_guide/citation
   user_guide/changelog
   user_guide/license

.. toctree::
   :hidden:

   user_guide/cli
   user_guide/gui
