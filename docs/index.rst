GWexpy
======

.. raw:: html

   <style>
   /* ゲートウェイ専用: サイドバー非表示・フルワイド化 */
   .wy-nav-side          { display: none !important; }
   .wy-nav-content-wrap  { margin-left: 0 !important; }
   .wy-nav-content       { max-width: 900px !important; margin: 0 auto !important; }
   </style>

.. raw:: html

   <div class="gw-hero" style="
     text-align: center;
     padding: 3em 1em 2em;
   ">
     <p style="font-size:1.25em; color:#555; margin-bottom:0.5em;">
       Extended Python Toolkit for Gravitational-Wave Data Analysis
     </p>
     <p style="font-size:1.05em; color:#777; margin-top:0;">
       GWpy を拡張する多次元データ解析ライブラリ —
       行列・フィールド・フィッティング・信号処理を統合
     </p>
   </div>

.. grid:: 2
   :gutter: 3
   :class-container: gw-cta-grid

   .. grid-item::

      .. button-ref:: web/ja/index
         :ref-type: doc
         :color: primary
         :shadow:
         :expand:

         📖 日本語で始める

   .. grid-item::

      .. button-ref:: web/en/index
         :ref-type: doc
         :color: secondary
         :shadow:
         :expand:

         📖 English Docs

----

主な特徴
--------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 🔬 多次元フィールド
      :text-align: center

      ``ScalarField`` / ``VectorField`` / ``TensorField``

      空間・時間を跨ぐ多次元データを統一インタフェースで操作

   .. grid-item-card:: ⚡ 数値的安定性
      :text-align: center

      Safe Log・ゼロ除算防護・NaN 伝播検出

      科学計算に必要な堅牢な数値処理を自動確保

   .. grid-item-card:: 📊 統合解析ツール群
      :text-align: center

      BrUCo / ARIMA / Fitting / MCMC

      ノイズ解析から高度なフィッティングまで一貫して提供

----

クイックインストール
--------------------

.. code-block:: bash

   git clone https://github.com/tatsuki-washimi/gwexpy.git
   cd gwexpy && pip install -e .

最短デモ
--------

.. code-block:: python

   from gwexpy.timeseries import FrequencySeriesMatrix
   fsmtx = FrequencySeriesMatrix.read("data.hdf5")
   fsmtx.fit(model="lorentzian").plot()

.. image:: _static/images/hero_plot.png
   :alt: FrequencySeriesMatrix Fitting 出力
   :align: center
   :width: 100%

.. toctree::
   :hidden:

   web/en/index
   web/ja/index
