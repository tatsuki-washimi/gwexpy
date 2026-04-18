:orphan:

GWexpy ドキュメント
==============================

GWexpy は GWpy を拡張し、時系列および周波数系列データ解析のための新たなコンテナや数値計算ユーティリティを提供します。

v\ |release| · Python ≥ 3.9 · 最終更新: |today|

.. raw:: html

   <style>
   .gw-hub-hero {
     display: grid;
     grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
     gap: 1.25rem;
     margin: 1.5rem 0 2rem;
     align-items: stretch;
   }
   .gw-hub-panel {
     background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
     border: 1px solid #d7e3ef;
     border-radius: 20px;
     box-shadow: 0 16px 40px rgba(15, 42, 74, 0.08);
   }
   .gw-hub-copy {
     padding: 1.5rem;
   }
   .gw-hub-eyebrow {
     margin: 0 0 0.65rem;
     color: #0f6cbd;
     font-size: 0.78rem;
     font-weight: 700;
     letter-spacing: 0.08em;
     text-transform: uppercase;
   }
   .gw-hub-title {
     margin: 0;
     font-size: 2.35rem;
     line-height: 1.02;
     letter-spacing: -0.03em;
     color: #10243a;
   }
   .gw-hub-lede {
     margin: 0.9rem 0 0;
     color: #566779;
     line-height: 1.8;
   }
   .gw-hub-meta {
     display: grid;
     grid-template-columns: repeat(3, minmax(0, 1fr));
     gap: 0.75rem;
     margin-top: 1.1rem;
   }
   .gw-hub-meta div {
     padding: 0.85rem 0.9rem;
     background: rgba(255,255,255,0.82);
     border: 1px solid #d7e3ef;
     border-radius: 14px;
   }
   .gw-hub-meta dt {
     margin: 0;
     color: #61758a;
     font-size: 0.72rem;
     font-weight: 700;
     letter-spacing: 0.08em;
     text-transform: uppercase;
   }
   .gw-hub-meta dd {
     margin: 0.35rem 0 0;
     color: #10243a;
     font-size: 0.92rem;
     font-weight: 700;
   }
   .gw-hub-figure {
     padding: 1rem;
   }
   .gw-hub-figure img {
     width: 100%;
     height: auto;
     display: block;
     border-radius: 14px;
     border: 1px solid #d7e3ef;
     background: #fff;
   }
   .gw-hub-figure p {
     margin: 0.75rem 0 0;
     color: #5c6d80;
     font-size: 0.88rem;
     line-height: 1.65;
   }
   .gw-hub-duo {
     display: grid;
     grid-template-columns: repeat(2, minmax(0, 1fr));
     gap: 1rem;
     margin: 0 0 2rem;
   }
   .gw-hub-duo section {
     padding: 1.15rem 1.2rem;
   }
   .gw-hub-duo p {
     margin: 0 0 0.65rem;
     color: #0f6cbd;
     font-size: 0.76rem;
     font-weight: 700;
     letter-spacing: 0.08em;
     text-transform: uppercase;
   }
   .gw-hub-duo pre {
     margin: 0;
     padding: 0.95rem 1rem;
     border: 1px solid #d7e3ef;
     border-radius: 14px;
     background: #f8fbfe;
     overflow-x: auto;
   }
   @media (max-width: 900px) {
     .gw-hub-hero,
     .gw-hub-duo {
       grid-template-columns: 1fr;
     }
     .gw-hub-meta {
       grid-template-columns: 1fr;
     }
   }
   </style>

.. raw:: html

   <section class="gw-hub-hero">
     <div class="gw-hub-panel gw-hub-copy">
       <p class="gw-hub-eyebrow">Documentation Hub</p>
       <h2 class="gw-hub-title">どう使うかを、解析導線から選ぶ</h2>
       <p class="gw-hub-lede">
         行列コンテナ、フィールド演算、フィッティング、信号処理を
         目的別の入口にまとめたトップページです。まずはクイックスタートか、
         下の 9 枚カードから解析タスクに近い入口を選んでください。
       </p>
       <dl class="gw-hub-meta">
         <div>
           <dt>Containers</dt>
           <dd>Matrix / Field / Series</dd>
         </div>
         <div>
           <dt>Analysis</dt>
           <dd>Fitting, BrUCo, MCMC</dd>
         </div>
         <div>
           <dt>Install</dt>
           <dd>Git checkout 推奨</dd>
         </div>
       </dl>
     </div>
     <figure class="gw-hub-panel gw-hub-figure">
       <img src="/_static/images/phase3/gateway_hero_scientific.png" alt="FrequencySeriesMatrix と共振フィットの可視化">
       <p>
         FrequencySeriesMatrix の全体像と、抽出した 1 チャンネルに対する共振フィットを同時に表示。
         GWexpy の Matrix 系コンテナと解析ワークフローを一画面で示します。
       </p>
     </figure>
   </section>

.. raw:: html

   <section class="gw-hub-duo">
     <section class="gw-hub-panel">
       <p>Quick install</p>
       <pre><code>git clone https://github.com/tatsuki-washimi/gwexpy.git
   cd gwexpy
   pip install -e .</code></pre>
     </section>
     <section class="gw-hub-panel">
       <p>3-line demo</p>
       <pre><code>from gwexpy.timeseries import FrequencySeriesMatrix
   fsmtx = FrequencySeriesMatrix.read("data.hdf5")
   fsmtx[2, 0].fit(model="lorentzian").plot()</code></pre>
     </section>
   </section>

.. note::

   **ページ種別**: ドキュメント入口
   **対象読者**: 初回利用者、GWpy ユーザー、解析ワークフローを探している利用者
   **検索ヒント**: ``quickstart``, ``installation``, ``tutorials``, ``examples``, ``ScalarField``, ``TimeSeriesMatrix``

.. button-ref:: user_guide/quickstart
    :ref-type: doc
    :color: primary
    :shadow:
    :expand:

    🚀 クイックスタート（5分で基本を習得）

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
        * 基本チュートリアル

    .. grid-item-card:: 🔬 実験データ解析者向け
        :link: examples/index
        :link-type: doc

        実践的な解析例を知りたい方向け

        * 解析ケーススタディ
        * 高度な信号処理

    .. grid-item-card:: 🔄 GWpy ユーザー向け
        :link: user_guide/gwexpy_for_gwpy_users_ja
        :link-type: doc

        GWpy から移行・併用する方向け

        * 差分レシピと互換性の入口
        * 追加 API 一覧への導線

----

やりたいことから探す
--------------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 📈 時系列の可視化・解析
        :link: user_guide/tutorials/intro_timeseries
        :link-type: doc

        TimeSeries の読み込み・描画・フィルタ処理

    .. grid-item-card:: 📊 スペクトログラム解析
        :link: user_guide/tutorials/intro_spectrogram
        :link-type: doc

        STFT・Qスキャン・周波数-時間表現の生成

    .. grid-item-card:: 🌊 フィールドデータ操作
        :link: user_guide/tutorials/field_scalar_intro
        :link-type: doc

        ScalarField / VectorField / TensorField の基礎

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 🔧 ノイズバジェット（BrUCo）
        :link: user_guide/tutorials/advanced_bruco
        :link-type: doc

        各ノイズ源の寄与を分離・可視化

    .. grid-item-card:: 📐 フィッティング & MCMC
        :link: user_guide/tutorials/intro_fitting
        :link-type: doc

        GLS・ベイズフィット・MCMC サンプリング

    .. grid-item-card:: 🧮 前処理 & ML パイプライン
        :link: user_guide/tutorials/ml_preprocessing_methods
        :link-type: doc

        特徴抽出・正規化・scikit-learn との連携

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 📁 マルチフォーマット I/O
        :link: user_guide/tutorials/case_gbd_format
        :link-type: doc
        :class-card: gw-highlight

        HDF5 / GBD / Frame / Zarr など多形式対応

        **GWexpy の差別化ポイント**

    .. grid-item-card:: 🔗 他ライブラリ連携
        :link: user_guide/tutorials/case_seismic_obspy
        :link-type: doc
        :class-card: gw-highlight

        ObsPy / LAL / Finesse / PyCBC との相互運用

        **GWexpy の差別化ポイント**

    .. grid-item-card:: 🔢 行列演算（Matrix 系）
        :link: user_guide/tutorials/matrix_timeseries
        :link-type: doc

        TimeSeriesMatrix / FrequencySeriesMatrix の一括処理

----

Visual Examples
---------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card::
        :img-top: /_static/images/case_noise_budget_thumb.png
        :img-alt: ノイズバジェット例のサムネイル
        :link: user_guide/tutorials/advanced_bruco
        :link-type: doc
        :text-align: center

        ノイズバジェット

    .. grid-item-card::
        :img-top: /_static/images/case_transfer_function_thumb.png
        :img-alt: 伝達関数推定例のサムネイル
        :link: user_guide/tutorials/case_transfer_function
        :link-type: doc
        :text-align: center

        伝達関数推定

    .. grid-item-card::
        :img-top: /_static/images/case_active_damping_thumb.png
        :img-alt: アクティブダンピング例のサムネイル
        :link: user_guide/tutorials/case_active_damping
        :link-type: doc
        :text-align: center

        アクティブダンピング

----

GWpy の基礎を学ぶ
-----------------

GWexpy は GWpy の上に構築されています。GWpy の基本操作は下記の公式ドキュメントを参照してください。

`gwpy.github.io/docs/stable/ <https://gwpy.github.io/docs/stable/>`_

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

   user_guide/installation
   user_guide/quickstart
   user_guide/getting_started

.. toctree::
   :maxdepth: 2
   :caption: 📖 ガイド (Guide)

   user_guide/prerequisites_and_conventions
   user_guide/interop
   user_guide/time_utilities
   user_guide/numerical_stability
   user_guide/scalarfield_slicing
   user_guide/gwexpy_for_gwpy_users_ja
   user_guide/gwpy_added_api_index_ja

.. toctree::
   :maxdepth: 2
   :caption: 🎓 学習 (Learn)

   user_guide/tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: 🗂 フォーマット & I/O

   user_guide/io_formats

.. toctree::
   :maxdepth: 2
   :caption: 📚 リファレンス (Reference)

   reference/index
   user_guide/validated_algorithms
   user_guide/architecture
   user_guide/physics_models
   user_guide/glossary

.. toctree::
   :maxdepth: 1
   :caption: ℹ️ その他 (Info)

   user_guide/roadmap
   user_guide/troubleshooting
   user_guide/citation
   user_guide/changelog
   user_guide/license

.. toctree::
   :hidden:

   user_guide/cli
   user_guide/gui
