:orphan:

GWexpy ドキュメント
==============================

GWexpy は GWpy を拡張し、時系列および周波数系列データ解析のための新たなコンテナや数値計算ユーティリティを提供します。

v\ |release| · Python ≥ 3.9 · 最終更新: |today|

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
   user_guide/io_formats
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
