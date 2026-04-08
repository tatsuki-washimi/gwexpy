ケーススタディ (Case Studies)
============================

実際の解析タスクを想定した、実践的なワークフロー集です。
機能ごとの基本操作（引数、戻り値等）を学ぶ場合は、 :doc:`../user_guide/tutorials/index` を参照してください。

実践例一覧
---------

.. grid:: 1
    :gutter: 3

    .. grid-item-card:: 📈 ノイズバジェット解析 (Noise Budget)
        :link: ../user_guide/tutorials/case_noise_budget
        :link-type: doc

        *   **問題**: 観測データに含まれる主要なノイズ源を特定したい。
        *   **アプローチ**: 多チャンネルのコヒーレンス解析とスペクトル合成。
        *   **利用 API**: ``TimeSeriesMatrix``, ``PSD``, ``Coherence``.

    .. grid-item-card:: 🎛️ 伝達関数の測定とフィッティング
        :link: ../user_guide/tutorials/case_transfer_function
        :link-type: doc

        *   **問題**: システムの伝達関数を実測し、理論モデルと比較したい。
        *   **アプローチ**: 正弦波スイープや白色雑音励起による TF 測定と極零点配置。
        *   **利用 API**: ``TransferFunction``, ``Fitter``, ``BodePlot``.

    .. grid-item-card:: 🏗️ アクティブダンピング制御
        :link: ../user_guide/tutorials/case_active_damping
        :link-type: doc

        *   **問題**: 懸架系の共振を抑制するための MIMO 制御系を設計・評価したい。
        *   **アプローチ**: 状態空間モデルを用いたフィードバック制御シミュレーション。
        *   **利用 API**: ``StateSpaceMatrix``, ``ActiveControl``, ``LQR``.

    .. grid-item-card:: ✂️ 長期データのセグメント解析
        :link: ../user_guide/tutorials/case_segment_analysis
        :link-type: doc

        *   **問題**: 数日間にわたるデータから、条件を満たす区間（セグメント）のみを抽出して統計処理したい。
        *   **アプローチ**: ``SegmentTable`` を活用したデータクエリと並列処理。
        *   **利用 API**: ``SegmentTable``, ``SegmentList``, ``Fetch``.

.. toctree::
   :hidden:

   ../user_guide/tutorials/case_noise_budget
   ../user_guide/tutorials/case_transfer_function
   ../user_guide/tutorials/case_active_damping
   ../user_guide/tutorials/case_segment_analysis

